"""
Pure stateless Asset Feature Transformer module.
Generates multi-window asset-level factor features using observations through time t only. Zero bfill.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class AssetFeatureTransformer:
    """
    Computes Momentum, Retail Flow, Carry, and Margin Risk factors for single symbol OHLCV.
    """

    def __init__(self, windows: List[int] = [7, 14, 30, 60, 90]):
        self.windows = windows

    @staticmethod
    def _custom_wma(series: pd.Series, window: int) -> pd.Series:
        weights = np.arange(1, window + 1)
        sum_weights = weights.sum()
        return series.rolling(window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / sum_weights, raw=True
        )

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, window: int) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window, min_periods=min(3, window)).mean()

    def transform_symbol(
        self,
        kline_df: pd.DataFrame,
        funding_series: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Generates feature DataFrame for a single symbol. Zero bfill calls.
        """
        kline_df = kline_df.sort_index()
        close = kline_df['close']
        high = kline_df['high']
        low = kline_df['low']
        volume = kline_df['volume']

        if funding_series is None:
            funding_series = pd.Series(0.0, index=kline_df.index)
        else:
            funding_series = funding_series.reindex(kline_df.index).fillna(0.0)

        annualized_funding = funding_series * 3 * 365
        log_ret = np.log(close / close.shift(1))

        features_dict = {}

        for w in self.windows:
            vol = log_ret.rolling(window=w, min_periods=min(3, w)).std()
            vol_clean = vol.replace(0, np.nan)

            # A. Momentum
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=w, min_periods=min(3, w)).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=w, min_periods=min(3, w)).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            wma = self._custom_wma(close, w)
            atr = self._calculate_atr(kline_df, w)

            features_dict[f'mom_rsi_{w}'] = rsi.fillna(50.0)
            features_dict[f'mom_wma_dist_{w}'] = ((close - wma) / atr.replace(0, np.nan)).fillna(0.0)

            # B. Retail Flow (Contrarian)
            volume_ma = volume.rolling(window=w, min_periods=min(3, w)).mean()
            volume_ratio = volume / volume_ma.replace(0, np.nan)
            retail_flow = -(log_ret * (volume_ratio / vol_clean))
            features_dict[f'retail_flow_{w}'] = retail_flow.fillna(0.0)

            # C. Carry
            funding_ma = annualized_funding.rolling(window=w, min_periods=min(3, w)).mean()
            funding_anomaly = annualized_funding - funding_ma
            vol_annual = vol * np.sqrt(365)
            carry = (annualized_funding / vol_annual.replace(0, np.nan)) + funding_anomaly
            features_dict[f'carry_{w}'] = carry.fillna(0.0)

            # D. Margin Risk
            high_roll = high.rolling(window=w, min_periods=min(3, w)).max()
            low_roll = low.rolling(window=w, min_periods=min(3, w)).min()
            drawdown_from_high = (close - high_roll) / high_roll.replace(0, np.nan)
            margin_risk = (drawdown_from_high.abs() * vol_clean) / (volume_ratio.replace(0, np.nan))
            features_dict[f'margin_risk_{w}'] = margin_risk.fillna(0.0)

        features_df = pd.DataFrame(features_dict, index=kline_df.index)
        return features_df
