"""
V4.1 Hygiene Tests: evidence-based feature selection (keep_families + sign flips),
macro exclusion from model features, point-in-time universe masking, and the
Monday-decide low-turnover weekly schedule.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.multifactor_mlops.features.panel import PanelDatasetBuilder, MACRO_FEATURE_NAMES
from src.multifactor_mlops.portfolio.constructor import PortfolioConstructor
from src.multifactor_mlops.config.loader import load_config


def _make_ohlcv(n=150, start="2023-01-01", base=100.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="1D")
    close = base * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    return pd.DataFrame({
        "open": close * 1.001, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000.0 + rng.normal(0, 100, n),
    }, index=dates)


def _panel(keep_families=None, inverted=None, use_macro=False):
    data_dict = {f"SYM{i}": _make_ohlcv(seed=i) for i in range(4)}
    macro = pd.DataFrame({"vix": 20.0, "fear_greed": 50.0, "dvol_btc": 60.0},
                         index=pd.date_range("2022-12-30", periods=155, freq="1D"))
    builder = PanelDatasetBuilder(
        windows=[7, 14, 30], cross_sectional_rank=False, lag=1,
        keep_families=keep_families, inverted_features=inverted,
        use_macro_features=use_macro,
    )
    return builder, builder.build_panel_dataset(data_dict, list(data_dict), macro_df=macro)


def test_macro_excluded_from_feature_columns():
    _, panel = _panel(use_macro=True)
    feats = PanelDatasetBuilder.feature_columns(panel)
    assert not MACRO_FEATURE_NAMES.intersection(feats)
    # macro columns may still be present in the panel for overlay use
    assert "macro_multiplier" in panel.columns


def test_keep_families_filters():
    builder, panel = _panel(keep_families=["mom_rsi", "mom_wma_dist"])
    feats = PanelDatasetBuilder.feature_columns(panel)
    assert all(c.startswith(("mom_rsi_", "mom_wma_dist_")) for c in feats)
    assert not any(c.startswith(("carry_", "retail_flow_", "margin_risk_")) for c in feats)
    assert len(feats) == 6  # 2 families x 3 windows


def test_inverted_features_flip_sign():
    builder, panel = _panel(keep_families=None, inverted=["retail_flow_7"])
    assert "retail_flow_7" in panel.columns
    # sign flip must be consistent: raw retail_flow_7 < -abs(original)
    raw = _panel(keep_families=None, inverted=None)[1]["retail_flow_7"]
    flipped = panel["retail_flow_7"]
    common = raw.index.intersection(flipped.index)
    pd.testing.assert_series_equal(raw.reindex(common), -flipped.reindex(common))


def test_universe_membership_masks_panel():
    data_dict = {f"SYM{i}": _make_ohlcv(seed=i) for i in range(3)}
    membership = pd.DataFrame({
        "SYM0": [True] * 150, "SYM1": [True] * 150, "SYM2": [False] * 150,
    }, index=pd.date_range("2023-01-01", periods=150, freq="1D"))
    builder = PanelDatasetBuilder(windows=[7, 14, 30], cross_sectional_rank=True, lag=1)
    panel = builder.build_panel_dataset(data_dict, list(data_dict), universe_membership_df=membership)
    assert "SYM2" not in panel.index.get_level_values("Symbol")
    assert "SYM0" in panel.index.get_level_values("Symbol")


def test_monday_decide_schedule():
    dates = pd.date_range("2024-01-01", periods=14, freq="1D")  # starts Monday
    weights = pd.DataFrame({"A": [1.0] * 14, "B": [-0.5] * 14}, index=dates)
    out = PortfolioConstructor.apply_monday_decide_schedule(weights)
    # Tue-Fri of week 1 hold Monday's weights; weekend + Monday flat
    for i, dt in enumerate(dates):
        if dt.dayofweek in [1, 2, 3, 4]:
            assert abs(out.iloc[i]["A"] - 1.0) < 1e-9
        else:
            assert abs(out.iloc[i]["A"]) < 1e-9
    # next Monday (index 7) flat, its decision applied Tue (index 8)
    assert abs(out.iloc[7]["A"]) < 1e-9
    assert abs(out.iloc[8]["A"] - 1.0) < 1e-9


def test_config_hygiene_keys():
    cfg = load_config("parameters.json")
    assert cfg.features.inverted_features == ["retail_flow_7", "margin_risk_90"]
    assert cfg.features.use_macro_features is False
    assert cfg.data.use_point_in_time_universe is True
    assert cfg.portfolio.rebalance_schedule == "monday_decide_weekly"


def test_composite_signal_mode_no_training(tmp_path):
    """signal_mode=composite must produce a signal WITHOUT training a model."""
    import xgboost
    from src.multifactor_mlops.backtest.walkforward_strategy import make_strategy_factory
    from src.multifactor_mlops.config.schema import AppConfig

    data_dict = {f"SYM{i}": _make_ohlcv(n=200, seed=i) for i in range(4)}
    cfg = AppConfig()
    cfg.features.keep_families = ["mom", "retail_flow", "margin_risk"]
    cfg.features.inverted_features = ["retail_flow_7", "margin_risk_90"]
    cfg.features.use_macro_features = False
    params = {
        "signal_mode": "composite",
        "composite_features": ["mom_14", "mom_30", "retail_flow_7", "margin_risk_90"],
        "quantiles": 4, "inverse_vol_period": 14, "allocation_cap": 0.5,
        "volatility_ceiling": None, "rebalance_schedule": "daily", "stress_multiplier": 0.4,
    }
    strat = make_strategy_factory(data_dict, list(data_dict), cfg, params)
    panel = strat._get_panel()
    times = pd.DatetimeIndex(panel.index.get_level_values("Time").unique()).sort_values()
    train_idx = times[:150]
    test_idx = times[150:190]

    class _Fold:
        fold_id = 0
        test_start = test_idx[0]
        test_end = test_idx[-1]

    out = strat.build_signal(None, params, train_idx, test_idx, _Fold())
    assert len(strat._model_cache) == 0  # NO model trained
    assert list(out.index) == list(test_idx)
    assert (out.abs() > 0).any().any()
