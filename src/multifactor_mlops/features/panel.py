"""
Panel Dataset Builder module.
Performs point-in-time merge_asof macro alignment and cross-sectional percentile ranking.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .asset import AssetFeatureTransformer
from .macro import MacroOverlayTransformer
from src.multifactor_mlops.labels.returns import calculate_next_open_to_open_returns

class PanelDatasetBuilder:
    """
    Constructs cross-sectional panel dataset with point-in-time alignment and zero bfill calls.
    """

    def __init__(
        self,
        windows: List[int] = [7, 14, 30, 60, 90],
        cross_sectional_rank: bool = True,
        lag: int = 1
    ):
        self.asset_transformer = AssetFeatureTransformer(windows=windows)
        self.macro_transformer = MacroOverlayTransformer()
        self.cross_sectional_rank = cross_sectional_rank
        self.lag = lag

    def build_panel_dataset(
        self,
        data_dict: Dict[str, pd.DataFrame],
        symbols: List[str],
        macro_df: Optional[pd.DataFrame] = None,
        funding_df: Optional[pd.DataFrame] = None,
        universe_membership_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Builds panel dataset with point-in-time macro merge_asof and cross-sectional ranking.
        """
        macro_features = pd.DataFrame()
        if macro_df is not None and not macro_df.empty:
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
            if not asset_feature_names:
                asset_feature_names = list(features_df.columns)

            # Point-in-time macro merge_asof
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
                features_df = merged.set_index(time_col)

            # Return label (Next-Open to Next-Open return)
            if 'open' in df.columns:
                target_returns = calculate_next_open_to_open_returns(df['open'], holding_bars=self.lag)
            else:
                close = df['close']
                target_returns = (close.shift(-self.lag) / close - 1.0).rename('target')

            symbol_df = pd.concat([features_df, target_returns], axis=1)
            symbol_df['Symbol'] = symbol

            # Mask universe membership if provided
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
