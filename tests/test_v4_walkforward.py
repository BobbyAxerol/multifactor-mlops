"""
V4 QuantBT Native Walk-Forward Tests.

Validates: fold separation, output coverage, model caching (3 calls per fold),
1-bar execution lag, stitching (flat before first OOS), on SYNTHETIC data
through the REAL QuantBT engine (read-only usage).
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.multifactor_mlops.config.schema import AppConfig
from src.multifactor_mlops.backtest.wf_runner import WalkForwardQuantBTRunner, extract_equity
from src.multifactor_mlops.backtest.walkforward_strategy import make_strategy_factory


def _make_data(n=365 * 3, start="2021-01-01"):
    rng = np.random.default_rng(7)
    dates = pd.date_range(start, periods=n, freq="1D")
    out = {}
    for sym, base in [("AAA", 100.0), ("BBB", 200.0)]:
        rets = rng.normal(0.0005, 0.02, n)
        close = base * np.cumprod(1 + rets)
        df = pd.DataFrame({
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000.0,
        }, index=dates)
        out[sym] = df
    return out


def test_walkforward_folds_stitched_and_lagged():
    data_dict = _make_data()
    app_config = AppConfig()
    strategy_params = {
        "quantiles": 4, "inverse_vol_period": 14, "allocation_cap": 0.5,
        "volatility_ceiling": None, "rebalance_schedule": "daily", "stress_multiplier": 0.4,
    }
    runner = WalkForwardQuantBTRunner()
    res = runner.run(
        data_dict=data_dict, symbols=["AAA", "BBB"], app_config=app_config,
        params=strategy_params,
        split_mode="walk_forward_2022", split_frequency="yearly", window_mode="expanding",
    )

    wf = res.metadata["walk_forward"]
    assert wf["n_folds"] >= 2

    folds = wf["fold_table"]
    if isinstance(folds, pd.DataFrame):
        folds = folds.to_dict("records")
    prev_end = None
    for f in folds:
        t_start = pd.Timestamp(f["test_start"])
        t_end = pd.Timestamp(f["test_end"])
        if prev_end is not None:
            assert t_start >= prev_end  # no overlap, no gap
        prev_end = t_end

    equity_df = extract_equity(res, initial_capital=100000.0)
    assert len(equity_df) > 100
    assert abs(equity_df["equity"].iloc[0] - 100000.0) < 1e-6
    # stitched equity exists over the whole timeline
    assert equity_df["equity"].isna().sum() == 0


def test_strategy_shifts_positions_one_bar():
    """Decision at close D must NOT be traded at close D (first bar flat)."""
    data_dict = _make_data(n=400, start="2021-01-01")
    app_config = AppConfig()
    strategy_params = {
        "quantiles": 4, "inverse_vol_period": 14, "allocation_cap": 0.5,
        "volatility_ceiling": None, "rebalance_schedule": "daily", "stress_multiplier": 0.4,
    }
    strategy = make_strategy_factory(
        data_dict=data_dict, symbols=["AAA", "BBB"], app_config=app_config,
        strategy_params=strategy_params,
    )
    panel = strategy._get_panel()
    times = pd.DatetimeIndex(panel.index.get_level_values("Time").unique()).sort_values()
    train_index = times[:200]
    test_index = times[200:260]

    class _Fold:
        fold_id = 0
        test_start = test_index[0]
        test_end = test_index[-1]

    positions = strategy.build_signal(
        data=None, params=strategy_params, train_index=train_index,
        test_index=test_index, fold=_Fold(),
    )
    assert list(positions.index) == list(test_index)
    assert (positions.iloc[0] == 0.0).all()  # 1-bar lag => first row flat
    assert (positions.abs() <= 0.5 + 1e-9).all().all()  # allocation cap


def test_strategy_model_cached_across_calls():
    """Engine calls build_signal 3x per fold -> exactly ONE model fit per fold."""
    data_dict = _make_data(n=400, start="2021-01-01")
    app_config = AppConfig()
    strategy_params = {
        "quantiles": 4, "inverse_vol_period": 14, "allocation_cap": 0.5,
        "volatility_ceiling": None, "rebalance_schedule": "daily", "stress_multiplier": 0.4,
    }
    strategy = make_strategy_factory(
        data_dict=data_dict, symbols=["AAA", "BBB"], app_config=app_config,
        strategy_params=strategy_params,
    )
    panel = strategy._get_panel()
    times = pd.DatetimeIndex(panel.index.get_level_values("Time").unique()).sort_values()
    train_index = times[:200]
    test_index = times[200:260]

    class _Fold:
        fold_id = 0
        test_start = test_index[0]
        test_end = test_index[-1]

    strategy.build_signal(None, strategy_params, train_index, test_index, _Fold())
    strategy.build_signal(None, strategy_params, train_index, train_index, _Fold())
    strategy.build_signal(None, strategy_params, train_index, test_index, _Fold())
    assert len(strategy._model_cache) == 1


def test_final_requires_locked_artifacts():
    """evaluate_final must fail fast when locked artifacts are missing."""
    import importlib
    import tempfile
    from src.multifactor_mlops import pipelines
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            from src.multifactor_mlops.pipelines import evaluate_final as ef
            orig_model = ef.MODEL_CONFIG_PATH
            orig_strategy = ef.STRATEGY_CONFIG_PATH
            ef.MODEL_CONFIG_PATH = os.path.join(tmp, "missing_model.json")
            ef.STRATEGY_CONFIG_PATH = os.path.join(tmp, "missing_strategy.json")
            try:
                ef.evaluate_final_oos(log_mlflow=False, data_dir=tmp)
            finally:
                ef.MODEL_CONFIG_PATH = orig_model
                ef.STRATEGY_CONFIG_PATH = orig_strategy
