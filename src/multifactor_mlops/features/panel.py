"""
Panel Dataset Builder module.
Performs point-in-time macro alignment, evidence-based feature selection
(keep_families + sign-flips), universe membership masking, and cross-sectional
percentile ranking. Zero bfill.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .asset import AssetFeatureTransformer
from .macro import MacroOverlayTransformer
from src.multifactor_mlops.labels.returns import (
    add_forward_close_labels,
    add_forward_open_labels,
)

# Columns that are NOT model features (label + timing metadata).
NON_FEATURE_COLUMNS = {'target', 'Symbol', 'decision_time', 'label_start_time', 'label_end_time'}

# Macro features are per-day constants -> zero cross-sectional IC by construction.
# They are excluded from the model feature set (still used as overlay).
MACRO_FEATURE_NAMES = {'vix_z', 'fng_z', 'dvol_z', 'stress_score', 'macro_multiplier'}

def _family_of(col: str) -> str:
    """Factor family of a feature column ('mom_rsi_7' -> 'mom_rsi')."""
    for family in ("mom_wma_dist", "retail_flow", "margin_risk", "mom_rsi", "mom", "carry"):
        if col.startswith(family + "_"):
            return family
    return col

class PanelDatasetBuilder:
    """
    Constructs cross-sectional panel dataset with point-in-time alignment and zero bfill calls.
    """

    def __init__(
        self,
        windows: List[int] = [7, 14, 30, 60, 90],
        cross_sectional_rank: bool = True,
        lag: int = 1,
        keep_families: Optional[List[str]] = None,
        inverted_features: Optional[List[str]] = None,
        use_macro_features: bool = True,
        return_type: str = "next_close_to_close"
    ):
        self.asset_transformer = AssetFeatureTransformer(windows=windows)
        self.macro_transformer = MacroOverlayTransformer()
        self.cross_sectional_rank = cross_sectional_rank
        self.lag = lag
        self.keep_families = list(keep_families) if keep_families else None
        self.inverted_features = set(inverted_features or [])
        self.use_macro_features = use_macro_features
        self.return_type = return_type

    def _make_labels(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds the canonical label + timing columns matching the QuantBT engine
        (close-to-close by default) or the research open-to-open variant.
        """
        if self.return_type == "next_open_to_open":
            if 'open' not in df.columns:
                raise ValueError("return_type=next_open_to_open requires an 'open' column.")
            feats = features_df.copy()
            feats['open'] = df['open']
            symbol_df = add_forward_open_labels(feats, holding_bars=self.lag)
            symbol_df = symbol_df.drop(columns=['open'])
            return symbol_df

        # canonical: next_close_to_close (engine-realizable)
        if 'close' not in df.columns:
            raise ValueError("return_type=next_close_to_close requires a 'close' column.")
        feats = features_df.copy()
        feats['close'] = df['close']
        symbol_df = add_forward_close_labels(feats, holding_bars=self.lag)
        symbol_df = symbol_df.drop(columns=['close'])
        return symbol_df

    def _select_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Applies evidence-based family filter and sign flips (before ranking)."""
        out = features_df.copy()
        if self.keep_families is not None:
            keep = [c for c in out.columns if _family_of(c) in self.keep_families]
            out = out[keep]
        flip = [c for c in out.columns if c in self.inverted_features]
        if flip:
            out[flip] = -out[flip]
        return out

    def build_panel_dataset(
        self,
        data_dict: Dict[str, pd.DataFrame],
        symbols: List[str],
        macro_df: Optional[pd.DataFrame] = None,
        funding_df: Optional[pd.DataFrame] = None,
        universe_membership_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Builds panel dataset with point-in-time alignment and cross-sectional ranking.
        """
        macro_features = pd.DataFrame()
        if self.use_macro_features and macro_df is not None and not macro_df.empty:
            macro_features = self.macro_transformer.transform_macro_df(macro_df)

        funding_daily = pd.DataFrame()
        if funding_df is not None and not funding_df.empty:
            funding_daily = funding_df.resample('1D').last()

        panel_list = []
        asset_feature_names = []

        for symbol in symbols:
            if symbol not in data_dict:
                continue
            df = data_dict[symbol].copy()
            if df.empty or len(df) < max(self.asset_transformer.windows):
                continue

            funding_series = None
            if not funding_daily.empty and symbol in funding_daily.columns:
                funding_series = funding_daily[symbol]

            # Generate asset-level features
            features_df = self.asset_transformer.transform_symbol(df, funding_series)
            features_df = self._select_features(features_df)
            if features_df.empty:
                continue
            if not asset_feature_names:
                asset_feature_names = list(features_df.columns)

            # Point-in-time macro merge_asof (model features only)
            if not macro_features.empty:
                feat_reset = features_df.reset_index()
                macro_reset = macro_features.reset_index()

                time_col = feat_reset.columns[0]
                macro_time_col = macro_reset.columns[0]

                merged = pd.merge_asof(
                    feat_reset.sort_values(time_col),
                    macro_reset.sort_values(macro_time_col),
                    left_on=time_col,
                    right_on=macro_time_col,
                    direction="backward"
                )
                # Drop the macro time column so it never leaks into the feature set
                if macro_time_col != time_col and macro_time_col in merged.columns:
                    merged = merged.drop(columns=[macro_time_col])
                features_df = merged.set_index(time_col)

            # Return label (engine-realizable close-to-close by default)
            # + explicit timing timestamps
            symbol_df = self._make_labels(features_df, df)
            symbol_df['Symbol'] = symbol

            # Mask universe membership if provided (point-in-time)
            if universe_membership_df is not None and symbol in universe_membership_df.columns:
                valid_mask = universe_membership_df[symbol].reindex(symbol_df.index).fillna(False)
                symbol_df = symbol_df[valid_mask]

            symbol_df = symbol_df.dropna(subset=['target'])
            if not symbol_df.empty:
                panel_list.append(symbol_df)

        if not panel_list:
            return pd.DataFrame()

        panel_df = pd.concat(panel_list)
        panel_df.index.name = 'Time'
        panel_df = panel_df.reset_index().set_index(['Time', 'Symbol']).sort_index()

        # Cross-sectional percentile ranking
        if self.cross_sectional_rank and asset_feature_names:
            panel_df[asset_feature_names] = (
                panel_df[asset_feature_names]
                .groupby(level='Time')
                .rank(pct=True)
                .fillna(0.5)
            )

        # Cross-sectional target demeanization
        target_mean = panel_df['target'].groupby(level='Time').transform('mean')
        panel_df['target'] = panel_df['target'] - target_mean

        return panel_df

    @staticmethod
    def feature_columns(panel_df: pd.DataFrame) -> List[str]:
        """Returns the model feature columns (excludes target, timing, macro)."""
        if panel_df is None or panel_df.empty:
            return []
        return [c for c in panel_df.columns
                if c not in NON_FEATURE_COLUMNS and c not in MACRO_FEATURE_NAMES]
