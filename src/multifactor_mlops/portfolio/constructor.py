"""
Portfolio Constructor module.
Builds signed target portfolio weights from predictions, inverse-volatility risk weights, and macro overlays.
Enforces sign preservation and strict exposure invariants.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class PortfolioInvariantError(Exception):
    """Raised when portfolio constraints or exposure limits are violated."""
    pass

class PortfolioConstructor:
    """
    Constructs long/short portfolio weights and outputs final signed weights once.
    """

    def __init__(
        self,
        quantiles: int = 20,
        allocation_cap: float = 0.25,
        portfolio_mode: str = "longshort",
        hedge_type: str = "target_weight"
    ):
        self.quantiles = quantiles
        self.allocation_cap = allocation_cap
        self.portfolio_mode = portfolio_mode
        self.hedge_type = hedge_type

    def create_cross_sectional_weights(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates cross-sectional quantile binning to produce raw market-neutral target signs/weights.
        Long top percentile (+1.0 / count), Short bottom percentile (-1.0 / count).
        """
        predictions_clean = predictions_df.replace([np.inf, -np.inf], np.nan)
        rank_pct = predictions_clean.rank(axis=1, pct=True, method="first")

        num_assets = len(predictions_df.columns)
        if num_assets <= 4:
            top_pct = 0.5
        else:
            top_pct = min(0.25, max(0.05, 1.0 / float(self.quantiles)))
        long_mask = rank_pct > (1.0 - top_pct)
        short_mask = rank_pct <= top_pct

        raw_weights = pd.DataFrame(0.0, index=predictions_df.index, columns=predictions_df.columns)
        
        # Signal Conviction Sizing: Scale target weights proportionally to ML prediction magnitude
        pos_preds = predictions_clean.clip(lower=0.0)
        neg_preds = predictions_clean.clip(upper=0.0).abs()
        
        long_weighted = (pos_preds * long_mask.astype(float)).fillna(0.0)
        short_weighted = (neg_preds * short_mask.astype(float)).fillna(0.0)

        long_sums = long_weighted.sum(axis=1).replace(0, 1.0)
        short_sums = short_weighted.sum(axis=1).replace(0, 1.0)

        long_part = long_weighted.div(long_sums, axis=0)
        short_part = -short_weighted.div(short_sums, axis=0)

        if self.portfolio_mode != "longshort":
            short_part = pd.DataFrame(0.0, index=predictions_df.index, columns=predictions_df.columns)

        return (long_part + short_part).fillna(0.0)

    @staticmethod
    def apply_calendar_holding_schedule(
        weights_df: pd.DataFrame,
        schedule: str = "daily"
    ) -> pd.DataFrame:
        """
        Applies calendar holding schedule to reduce turnover and avoid weekend volatility.
        Schedules supported: 'daily', 'calendar_3d', 'calendar_5d', 'weekly_friday_exit'.
        'weekly_friday_exit': Zeroes out positions on Saturdays and Sundays (dayofweek 5 & 6) to exit on Friday.
        """
        if schedule == "daily" or weights_df.empty:
            return weights_df.copy()

        scheduled_weights = weights_df.copy()

        if schedule == "weekly_friday_exit":
            # Zero out positions on Saturday (5) and Sunday (6)
            is_weekend = scheduled_weights.index.dayofweek.isin([5, 6])
            scheduled_weights.loc[is_weekend] = 0.0

            # Hold Monday-Thursday weights steady until Friday close
            prev_row = scheduled_weights.iloc[0].copy()
            for i in range(1, len(scheduled_weights)):
                dt = scheduled_weights.index[i]
                if dt.dayofweek in [5, 6]:
                    prev_row = pd.Series(0.0, index=scheduled_weights.columns)
                elif dt.dayofweek == 0:  # Monday rebalance
                    prev_row = scheduled_weights.iloc[i].copy()
                else:  # Tue - Fri hold Monday weight
                    if not prev_row.eq(0.0).all():
                        scheduled_weights.iloc[i] = prev_row
                    else:
                        prev_row = scheduled_weights.iloc[i].copy()
            return scheduled_weights

        step = 3 if schedule == "calendar_3d" else 5
        prev_row = scheduled_weights.iloc[0].copy()
        for i in range(1, len(scheduled_weights)):
            if i % step != 0:
                scheduled_weights.iloc[i] = prev_row
            else:
                prev_row = scheduled_weights.iloc[i].copy()

        return scheduled_weights

    def apply_risk_weights_and_constraints(
        self,
        target_signs: pd.DataFrame,
        risk_weights: pd.DataFrame,
        macro_multiplier: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Combines target signs with inverse-volatility risk weights and macro overlays.
        Produces final signed weights DIRECTLY without re-multiplying signs.
        """
        # Risk weights are non-negative magnitudes per symbol
        abs_risk_weights = risk_weights.abs().fillna(0.0)
        
        # Combine target signs (+1, -1, 0) directly with risk magnitudes
        signed_weights = target_signs * abs_risk_weights

        if macro_multiplier is not None:
            mult_aligned = macro_multiplier.reindex(signed_weights.index).fillna(1.0)
            signed_weights = signed_weights.mul(mult_aligned, axis=0)

        # Apply allocation cap
        final_weights = signed_weights.clip(lower=-self.allocation_cap, upper=self.allocation_cap)

        # Validate portfolio invariants
        self.assert_portfolio_invariants(final_weights)
        return final_weights

    def assert_portfolio_invariants(self, weights: pd.DataFrame) -> None:
        """
        Validates portfolio invariants on signed target weights.
        """
        long_sum = weights[weights > 0].sum(axis=1).min()
        short_sum = weights[weights < 0].sum(axis=1).max()
        max_weight = weights.abs().max().max()

        if long_sum < -1e-5:
            raise PortfolioInvariantError(f"Long weights sum is negative: {long_sum}")
        if short_sum > 1e-5:
            raise PortfolioInvariantError(f"Short weights sum is positive: {short_sum}")
        if max_weight > self.allocation_cap + 1e-4:
            raise PortfolioInvariantError(f"Max asset weight ({max_weight}) exceeds allocation cap ({self.allocation_cap})")

    @staticmethod
    def apply_volatility_ceiling_filter(
        weights_df: pd.DataFrame,
        data_dict: Dict[str, pd.DataFrame],
        vol_ceiling_pct: float = 0.06,
        vol_window: int = 14
    ) -> pd.DataFrame:
        """
        Idea 3 (Volatility Ceiling Risk Scaling): Reduces sizing by 50% on assets whose 14-day daily
        return volatility exceeds vol_ceiling_pct (default 6%).
        """
        if weights_df.empty or not isinstance(data_dict, dict):
            return weights_df

        closes = {}
        for sym in weights_df.columns:
            if sym in data_dict and not data_dict[sym].empty:
                df = data_dict[sym]
                c_col = 'Close' if 'Close' in df.columns else ('close' if 'close' in df.columns else None)
                if c_col:
                    closes[sym] = df[c_col]

        if not closes:
            return weights_df

        close_df = pd.DataFrame(closes).reindex(weights_df.index).ffill()
        close_df = close_df.reindex(columns=weights_df.columns)
        daily_returns = close_df.pct_change()
        rolling_vol = daily_returns.rolling(window=vol_window, min_periods=5).std()

        modified = weights_df.copy()
        high_vol_mask = (rolling_vol > vol_ceiling_pct).fillna(False)
        modified = modified.mask(high_vol_mask, modified * 0.5)

        return modified.fillna(0.0)

    @staticmethod
    def apply_ewma_volatility_ceiling_filter(
        weights_df: pd.DataFrame,
        data_dict: Dict[str, pd.DataFrame],
        vol_ceiling_pct: float = 0.04,
        ewma_fast: int = 5,
        ewma_slow: int = 20
    ) -> pd.DataFrame:
        """
        V2 Choice 1 (Dual-Window EWMA Volatility Risk Scaling):
        Computes max(EWMA_fast, EWMA_slow) volatility and scales down position sizes by 50%
        for assets whose effective volatility exceeds vol_ceiling_pct (default 4%).
        """
        if weights_df.empty or not isinstance(data_dict, dict):
            return weights_df

        closes = {}
        for sym in weights_df.columns:
            if sym in data_dict and not data_dict[sym].empty:
                df = data_dict[sym]
                c_col = 'Close' if 'Close' in df.columns else ('close' if 'close' in df.columns else None)
                if c_col:
                    closes[sym] = df[c_col]

        if not closes:
            return weights_df

        close_df = pd.DataFrame(closes).reindex(weights_df.index).ffill()
        close_df = close_df.reindex(columns=weights_df.columns)
        daily_returns = close_df.pct_change()

        ewma_fast_vol = daily_returns.ewm(span=ewma_fast, min_periods=3).std()
        ewma_slow_vol = daily_returns.ewm(span=ewma_slow, min_periods=5).std()
        max_ewma_vol = np.maximum(ewma_fast_vol, ewma_slow_vol)

        modified = weights_df.copy()
        high_vol_mask = (max_ewma_vol > vol_ceiling_pct).fillna(False)
        modified = modified.mask(high_vol_mask, modified * 0.5)

        return modified.fillna(0.0)
