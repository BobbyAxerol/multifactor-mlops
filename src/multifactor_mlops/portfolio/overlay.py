"""
Single Shared Stress Overlay module.

Used IDENTICALLY by:
  - the walk-forward backtest strategy (build_signal),
  - live serving (bundle.predict -> apply_stress_overlay).

Logic (intended semantics, see train.py legacy comment):
  - Global macro stress: sigmoid multiplier [stress floor .. 1.0] derived from
    VIX/FNG/DVOL rolling z-scores (MacroOverlayTransformer).
  - Asymmetric leg scaling by BTC 21-day momentum:
      * CRASH (BTC mom < 0): keep SHORT leg (profitable leg) at full size,
        scale LONG leg down to stress floor.
      * BULL (BTC mom >= 0): keep LONG leg at full size, scale SHORT leg down
        (short_leg_bull_multiplier) to eliminate negative carry/funding drag.
  This fixes the legacy inversion (train.py:675) where shorts were cut in crashes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def apply_stress_overlay(
    weights: pd.DataFrame,
    macro_multiplier: pd.Series,
    params: Dict[str, Any],
    btc_mom: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Applies global macro stress scaling plus asymmetric long/short leg scaling.

    Parameters
    ----------
    weights : pd.DataFrame (index=decision time, columns=symbols)
        Signed target weights (positive=long, negative=short).
    macro_multiplier : pd.Series
        Global exposure multiplier [stress floor .. 1.0] indexed by date.
    params : dict
        stress_multiplier (global floor, default 0.4),
        short_leg_bull_multiplier (default 0.6).
    btc_mom : pd.Series, optional
        BTC 21-day momentum indexed by date (crash signal).

    Returns
    -------
    pd.DataFrame
        Overlay-scaled weights, same shape as input.
    """
    if weights is None or weights.empty:
        return weights

    out = weights.copy()
    idx = out.index

    global_mult = macro_multiplier.reindex(idx).fillna(1.0)
    stress_floor = float(params.get("stress_multiplier", 0.4))
    bull_short_mult = float(params.get("short_leg_bull_multiplier", 0.6))

    long_mult = global_mult.copy()
    short_mult = global_mult.copy()

    if btc_mom is not None:
        crash = (btc_mom.reindex(idx).fillna(0.0) < 0.0).astype(float)
        # crash -> long leg scaled to stress_floor; bull -> long leg kept at 1.0
        long_leg = 1.0 - crash * (1.0 - stress_floor)
        # crash -> short leg kept at 1.0; bull -> short leg scaled to bull_short_mult
        short_leg = 1.0 - (1.0 - crash) * (1.0 - bull_short_mult)
        long_mult = global_mult * long_leg
        short_mult = global_mult * short_leg

    pos = out.where(out > 0.0, 0.0)
    neg = out.where(out < 0.0, 0.0)
    out = pos.mul(long_mult, axis=0) + neg.mul(short_mult, axis=0)
    return out.fillna(0.0)
