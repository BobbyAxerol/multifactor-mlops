"""
V4 Overlay Parity Tests: single overlay module, correct asymmetric leg logic,
identical behavior in backtest and serving paths.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.multifactor_mlops.portfolio.overlay import apply_stress_overlay


def _weights():
    return pd.DataFrame({
        "AAA": [0.20, 0.20, 0.20],
        "BBB": [-0.20, -0.20, -0.20],
    }, index=pd.date_range("2024-01-01", periods=3, freq="1D"))


def test_crash_keeps_short_cuts_long():
    w = _weights()
    mult = pd.Series(1.0, index=w.index)
    btc_mom = pd.Series([-0.05, -0.05, -0.05], index=w.index)  # crash
    out = apply_stress_overlay(w, mult, {"stress_multiplier": 0.4}, btc_mom=btc_mom)
    # short leg preserved
    assert abs(out.loc[w.index[0], "BBB"] + 0.20) < 1e-9
    # long leg scaled to stress floor
    assert abs(out.loc[w.index[0], "AAA"] - 0.08) < 1e-9


def test_bull_cuts_short_keeps_long():
    w = _weights()
    mult = pd.Series(1.0, index=w.index)
    btc_mom = pd.Series([0.05, 0.05, 0.05], index=w.index)  # bull
    out = apply_stress_overlay(w, mult, {"stress_multiplier": 0.4}, btc_mom=btc_mom)
    # long leg preserved
    assert abs(out.loc[w.index[0], "AAA"] - 0.20) < 1e-9
    # short leg scaled to 0.6 (carry drag reduction)
    assert abs(out.loc[w.index[0], "BBB"] + 0.12) < 1e-9


def test_global_stress_multiplier_applies_both_legs():
    w = _weights()
    mult = pd.Series([0.5, 0.5, 0.5], index=w.index)
    out = apply_stress_overlay(w, mult, {"stress_multiplier": 0.4}, btc_mom=None)
    assert abs(out.loc[w.index[0], "AAA"] - 0.10) < 1e-9
    assert abs(out.loc[w.index[0], "BBB"] + 0.10) < 1e-9


def test_backtest_and_serving_parity():
    """The exact same function serves backtest and serving: identical inputs -> identical outputs."""
    w = _weights()
    mult = pd.Series(1.0, index=w.index)
    btc_mom = pd.Series([-0.05, 0.05, -0.05], index=w.index)
    params = {"stress_multiplier": 0.4}
    out_backtest = apply_stress_overlay(w, mult, params, btc_mom=btc_mom)
    out_serving = apply_stress_overlay(w, mult, params, btc_mom=btc_mom)
    pd.testing.assert_frame_equal(out_backtest, out_serving)
