"""
Shared Macro Overlay Transformer module.
Calculates point-in-time macro Z-scores and Sigmoid continuous exposure multipliers. Zero bfill calls.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class MacroOverlayTransformer:
    """
    Computes Sigmoid continuous exposure scaling using past macro data up to decision time t only.
    Shared identically between validation and live serving.
    """

    def __init__(
        self,
        lookback_days: int = 120,
        min_periods: int = 30,
        min_multiplier: float = 0.2,
        max_multiplier: float = 1.0
    ):
        self.lookback_days = lookback_days
        self.min_periods = min_periods
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

    def transform_macro_df(self, raw_macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates macro rolling Z-scores and Sigmoid multiplier. Zero bfill calls.
        """
        if raw_macro_df.empty:
            return pd.DataFrame()

        macro_df = raw_macro_df.sort_index().ffill()  # ONLY ffill weekend stock gaps, NO bfill

        vix_z = (macro_df['vix'] - macro_df['vix'].rolling(self.lookback_days, min_periods=self.min_periods).mean()) / macro_df['vix'].rolling(self.lookback_days, min_periods=self.min_periods).std().replace(0, 1)
        fng_z = -(macro_df['fear_greed'] - macro_df['fear_greed'].rolling(self.lookback_days, min_periods=self.min_periods).mean()) / macro_df['fear_greed'].rolling(self.lookback_days, min_periods=self.min_periods).std().replace(0, 1)
        dvol_z = (macro_df['dvol_btc'] - macro_df['dvol_btc'].rolling(self.lookback_days, min_periods=self.min_periods).mean()) / macro_df['dvol_btc'].rolling(self.lookback_days, min_periods=self.min_periods).std().replace(0, 1)

        stress_score = (vix_z.fillna(0.0) + fng_z.fillna(0.0) + dvol_z.fillna(0.0)) / 3.0

        # Sigmoid smooth exposure scaling: 1 / (1 + exp(1.5 * (stress_score - 0.5)))
        regime_multiplier = 1.0 / (1.0 + np.exp(1.5 * (stress_score - 0.5)))
        regime_multiplier = regime_multiplier.clip(lower=self.min_multiplier, upper=self.max_multiplier)

        macro_features = pd.DataFrame({
            "vix_z": vix_z.fillna(0.0),
            "fng_z": fng_z.fillna(0.0),
            "dvol_z": dvol_z.fillna(0.0),
            "stress_score": stress_score.fillna(0.0),
            "macro_multiplier": regime_multiplier.fillna(1.0)
        }, index=macro_df.index)

        return macro_features
