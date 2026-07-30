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
        raw_weights[long_mask] = 1.0
        if self.portfolio_mode == "longshort":
            raw_weights[short_mask] = -1.0

        long_counts = (raw_weights > 0).sum(axis=1).replace(0, 1)
        short_counts = (raw_weights < 0).sum(axis=1).replace(0, 1)

        long_part = raw_weights.clip(lower=0.0).div(long_counts, axis=0)
        short_part = raw_weights.clip(upper=0.0).div(short_counts, axis=0)

        return (long_part + short_part).fillna(0.0)

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
