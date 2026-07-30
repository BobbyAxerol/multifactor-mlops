"""
Final Production Refit & Model Bundle module.
Fits one final production model through production_cutoff and constructs an immutable model bundle.
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

class ProductionModelBundle:
    """
    Encapsulates trained model binary, feature schema, macro overlay spec, and data lineage.
    """

    def __init__(
        self,
        booster: Any,
        feature_names: List[str],
        config: Dict[str, Any],
        lineage_manifest: Dict[str, Any]
    ):
        self.booster = booster
        self.feature_names = feature_names
        self.config = config
        self.lineage_manifest = lineage_manifest

    def predict(self, X_input: pd.DataFrame) -> np.ndarray:
        """
        Executes model inference enforcing strict feature ordering and presence.
        """
        # Validate feature schema
        missing_feats = [f for f in self.feature_names if f not in X_input.columns]
        if missing_feats:
            raise ValueError(f"Inference error: Missing required features in input: {missing_feats}")

        X_ordered = X_input[self.feature_names].fillna(0.0)

        if isinstance(self.booster, xgb.Booster):
            dtest = xgb.DMatrix(X_ordered)
            return self.booster.predict(dtest)
        elif isinstance(self.booster, lgb.Booster):
            return self.booster.predict(X_ordered)
        else:
            return self.booster.predict(X_ordered)

    def save(self, bundle_dir: str) -> str:
        os.makedirs(bundle_dir, exist_ok=True)
        model_path = os.path.join(bundle_dir, "model.bin")
        if isinstance(self.booster, xgb.Booster):
            self.booster.save_model(model_path)
        
        metadata = {
            "feature_names": self.feature_names,
            "config": self.config,
            "lineage": self.lineage_manifest
        }
        meta_path = os.path.join(bundle_dir, "bundle_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return bundle_dir

def fit_final_production_model(
    data_dict: Dict[str, pd.DataFrame],
    symbols: List[str],
    macro_df: Optional[pd.DataFrame] = None,
    funding_df: Optional[pd.DataFrame] = None,
    config_path: str = "parameters.json",
    production_cutoff: Optional[str] = None
) -> ProductionModelBundle:
    """
    Fits one final production model on full historical panel up to production_cutoff.
    """
    app_config = load_config(config_path)
    
    panel_builder = PanelDatasetBuilder(
        windows=app_config.features.windows,
        cross_sectional_rank=True,
        lag=app_config.label.holding_bars
    )

    panel_df = panel_builder.build_panel_dataset(
        data_dict=data_dict,
        symbols=symbols,
        macro_df=macro_df,
        funding_df=funding_df
    )

    if panel_df.empty:
        raise ValueError("Cannot fit final model: Panel dataset is empty.")

    if production_cutoff:
        cutoff_dt = pd.Timestamp(production_cutoff)
        lag_days = app_config.label.holding_bars + 1  # next-open to next-open+H
        max_decision_dt = cutoff_dt - pd.Timedelta(days=lag_days)
        panel_df = panel_df[panel_df.index.get_level_values('Time') <= max_decision_dt]

    feature_cols = [c for c in panel_df.columns if c not in ['target', 'Symbol']]
    X_train = panel_df[feature_cols].fillna(0.0)
    y_train = panel_df['target']

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

    bundle = ProductionModelBundle(
        booster=booster,
        feature_names=feature_cols,
        config=app_config.model_dump(),
        lineage_manifest=lineage
    )

    return bundle
