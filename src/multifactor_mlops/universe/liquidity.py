"""
Point-in-Time Rolling Liquidity Universe Selection.
Calculates lagged turnover L_{i, t-1} and constructs eligible universe membership without look-ahead bias.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

class PointInTimeUniverseSelector:
    """
    Selects point-in-time top liquidity universe strictly using lagged observations.
    """

    def __init__(
        self,
        top_n: int = 40,
        lookback_days: int = 30,
        min_history_days: int = 180,
        selection_lag_bars: int = 1
    ):
        self.top_n = top_n
        self.lookback_days = lookback_days
        self.min_history_days = min_history_days
        self.selection_lag_bars = selection_lag_bars

    def compute_turnover_matrix(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Computes daily dollar volume (Volume * Close) wide DataFrame.
        """
        turnover_dict = {}
        for symbol, df in data_dict.items():
            if df.empty or 'close' not in df.columns or 'volume' not in df.columns:
                continue
            dollar_vol = (df['volume'] * df['close']).rename(symbol)
            turnover_dict[symbol] = dollar_vol.sort_index()

        if not turnover_dict:
            return pd.DataFrame()

        turnover_df = pd.concat(turnover_dict.values(), axis=1, join='outer').sort_index()
        return turnover_df

    def build_universe_membership(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Builds point-in-time boolean membership DataFrame:
        U_t = TopN( L_{i, t-1} )
        where L_{i, s} = rolling_mean(Volume * Close, lookback_days).
        """
        turnover_df = self.compute_turnover_matrix(data_dict)
        if turnover_df.empty:
            return pd.DataFrame()

        # Rolling mean turnover
        rolling_turnover = turnover_df.rolling(
            window=self.lookback_days,
            min_periods=min(10, self.lookback_days)
        ).mean()

        # Lag 1 bar to guarantee point-in-time availability
        lagged_turnover = rolling_turnover.shift(self.selection_lag_bars)

        # Minimum history mask
        valid_history_counts = turnover_df.notna().cumsum()
        history_mask = valid_history_counts >= self.min_history_days

        eligible_turnover = lagged_turnover.where(history_mask, np.nan)

        # Rank cross-sectionally per timestamp
        ranks = eligible_turnover.rank(axis=1, ascending=False, method="first")
        membership_df = ranks <= self.top_n

        return membership_df.fillna(False)
