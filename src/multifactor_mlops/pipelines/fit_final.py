"""
Final Production Refit & Model Bundle module.
Fits one final production model on rows with label_end_time <= production_cutoff
and constructs an immutable model bundle (model + preprocessor + overlay spec).

The bundle is the ONLY artifact served in production (train-serving parity).
"""

import os
import json
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from src.multifactor_mlops.config.loader import load_config
from src.multifactor_mlops.features.panel import PanelDatasetBuilder
from src.multifactor_mlops.features.preprocessor import FeaturePreprocessor
from src.multifactor_mlops.labels.returns import filter_train_by_label_end
from src.multifactor_mlops.portfolio.overlay import apply_stress_overlay


class ProductionModelBundle:
    """
    Encapsulates trained model binary, feature preprocessor, feature schema,
    macro overlay spec, and data lineage.
    """

    def __init__(
        self,
        booster: Any,
        feature_names: List[str],
        config: Dict[str, Any],
        lineage_manifest: Dict[str, Any],
        preprocessor: Optional[FeaturePreprocessor] = None,
        overlay_params: Optional[Dict[str, Any]] = None
    ):
        self.booster = booster
        self.feature_names = list(feature_names)
        self.config = config
        self.lineage_manifest = lineage_manifest
        self.preprocessor = preprocessor
        self.overlay_params = dict(overlay_params or {})

    def predict(self, X_input: pd.DataFrame) -> np.ndarray:
        """
        Executes model inference enforcing strict feature ordering and presence.
        Applies the SAME preprocessor transform as training when available.
        """
        missing_feats = [f for f in self.feature_names if f not in X_input.columns]
        if missing_feats:
            raise ValueError(f"Inference error: Missing required features in input: {missing_feats}")

        if self.preprocessor is not None:
            X_ordered = self.preprocessor.transform(X_input)
        else:
            X_ordered = X_input[self.feature_names].fillna(0.0)

        if isinstance(self.booster, xgb.Booster):
            dtest = xgb.DMatrix(X_ordered)
            return self.booster.predict(dtest)
        elif isinstance(self.booster, lgb.Booster):
            return self.booster.predict(X_ordered)
        else:
            return self.booster.predict(X_ordered)

    def predict_weights(self, X_input: pd.DataFrame, macro_multiplier: Optional[pd.Series] = None) -> pd.DataFrame:
        """Cross-sectional target weights (1-day decision -> caller applies lag)."""
        preds = self.predict(X_input)
        if len(preds) != len(X_input):
            raise ValueError("predict_weights: prediction length mismatch.")
        preds_series = pd.Series(preds, index=X_input.index)
        # long/short cross-sectional split: top/bottom by pct rank
        rank_pct = preds_series.rank(pct=True, method="first")
        quantiles = int(self.overlay_params.get("quantiles", 16))
        top_pct = min(0.25, max(0.05, 1.0 / float(max(quantiles, 2))))
        weights = pd.Series(0.0, index=X_input.index, dtype=float)
        weights[rank_pct > (1.0 - top_pct)] = 1.0 / max(1, int((rank_pct > (1.0 - top_pct)).sum()))
        weights[rank_pct <= top_pct] = -1.0 / max(1, int((rank_pct <= top_pct).sum()))
        if macro_multiplier is not None:
            out = apply_stress_overlay(
                weights.to_frame("w"),
                macro_multiplier=macro_multiplier,
                params=self.overlay_params,
            )["w"]
            weights = out
        return weights.to_frame()

    def save(self, bundle_dir: str) -> str:
        os.makedirs(bundle_dir, exist_ok=True)
        model_path = os.path.join(bundle_dir, "model.bin")
        if isinstance(self.booster, xgb.Booster):
            self.booster.save_model(model_path)

        metadata = {
            "feature_names": self.feature_names,
            "config": self.config,
            "lineage": self.lineage_manifest,
            "overlay_params": self.overlay_params,
            "preprocessor": self.preprocessor.to_dict() if self.preprocessor is not None else None
        }
        meta_path = os.path.join(bundle_dir, "bundle_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return bundle_dir

    @classmethod
    def from_dir(cls, bundle_dir: str, booster_kind: str = "xgboost") -> "ProductionModelBundle":
        meta_path = os.path.join(bundle_dir, "bundle_metadata.json")
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        model_path = os.path.join(bundle_dir, "model.bin")
        booster = xgb.Booster()
        booster.load_model(model_path)
        preprocessor = (
            FeaturePreprocessor.from_dict(metadata["preprocessor"]) if metadata.get("preprocessor") else None
        )
        return cls(
            booster=booster,
            feature_names=metadata["feature_names"],
            config=metadata.get("config", {}),
            lineage_manifest=metadata.get("lineage", {}),
            preprocessor=preprocessor,
            overlay_params=metadata.get("overlay_params", {}),
        )


def fit_final_production_model(
    data_dict: Dict[str, pd.DataFrame],
    symbols: List[str],
    macro_df: Optional[pd.DataFrame] = None,
    funding_df: Optional[pd.DataFrame] = None,
    config_path: str = "parameters.json",
    production_cutoff: Optional[str] = None,
    overlay_params: Optional[Dict[str, Any]] = None,
    universe_membership_df: Optional[pd.DataFrame] = None
) -> ProductionModelBundle:
    """
    Fits one final production model on rows with label_end_time <= production_cutoff.
    """
    app_config = load_config(config_path)

    panel_builder = PanelDatasetBuilder(
        windows=app_config.features.windows,
        cross_sectional_rank=True,
        lag=app_config.label.holding_bars,
        keep_families=app_config.features.keep_families,
        inverted_features=app_config.features.inverted_features,
        use_macro_features=app_config.features.use_macro_features,
    )

    panel_df = panel_builder.build_panel_dataset(
        data_dict=data_dict,
        symbols=symbols,
        macro_df=macro_df,
        funding_df=funding_df,
        universe_membership_df=universe_membership_df
    )

    if panel_df.empty:
        raise ValueError("Cannot fit final model: Panel dataset is empty.")

    if production_cutoff:
        cutoff_dt = pd.Timestamp(production_cutoff)
        panel_df = filter_train_by_label_end(panel_df, cutoff_dt + pd.Timedelta(days=1))

    feature_cols = PanelDatasetBuilder.feature_columns(panel_df)
    if not feature_cols:
        raise ValueError("Cannot fit final model: no feature columns.")

    X_train_raw = panel_df[feature_cols]
    y_train = panel_df['target'].astype(float)

    preprocessor = FeaturePreprocessor().fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)

    xgb_hyperparams = {
        'objective': 'reg:squarederror',
        'learning_rate': app_config.model.learning_rate,
        'max_depth': app_config.model.max_depth,
        'colsample_bytree': app_config.model.colsample_bytree,
        'subsample': app_config.model.subsample,
        'random_state': app_config.model.random_state,
        'verbosity': 0
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    booster = xgb.train(xgb_hyperparams, dtrain, num_boost_round=app_config.model.num_boost_round)

    lineage = {
        "production_cutoff": production_cutoff,
        "symbol_count": len(symbols),
        "total_rows": len(panel_df),
        "feature_count": len(feature_cols)
    }

    merged_overlay = dict(overlay_params or {})
    merged_overlay.setdefault("quantiles", app_config.portfolio.quantiles)
    merged_overlay.setdefault("stress_multiplier", app_config.portfolio.stress_multiplier)

    bundle = ProductionModelBundle(
        booster=booster,
        feature_names=feature_cols,
        config=app_config.model_dump(),
        lineage_manifest=lineage,
        preprocessor=preprocessor,
        overlay_params=merged_overlay
    )

    return bundle
