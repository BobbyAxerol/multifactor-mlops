"""
V4 Config Strictness + Cost Convention Tests.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pydantic import ValidationError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.multifactor_mlops.config.loader import load_config
from src.multifactor_mlops.backtest.quantbt_runner import QuantBTRunner


def test_canonical_config_loads_sections():
    cfg = load_config("parameters.json")
    assert cfg.validation.split_mode == "walk_forward_2024"
    assert cfg.validation.split_frequency == "quarterly"
    assert cfg.label.return_type == "next_close_to_close"
    assert cfg.model.learning_rate == 0.07
    assert cfg.backtest.fee_rate_per_fill == 0.0005
    assert cfg.backtest.use_funding is True
    assert cfg.data.allow_bfill is False


def test_invalid_split_frequency_rejected():
    bad = {
        "run": {}, "data": {}, "features": {}, "label": {},
        "model": {}, "backtest": {},
        "validation": {"split_mode": "walk_forward_2024", "split_frequency": "hourly"},
        "portfolio": {},
    }
    with pytest.raises(ValidationError):
        load_config(bad)


def test_close_to_close_label_rejected():
    bad = {
        "label": {"return_type": "close_to_close"},
    }
    with pytest.raises(ValidationError):
        load_config(bad)


def test_legacy_fee_param_still_round_trip():
    """params['fee'] (legacy round-trip) must be halved by the runner -> 0.00025/fill."""
    dates = pd.date_range("2024-01-01", periods=4, freq="1D")
    btc_df = pd.DataFrame({
        "open": [100.0, 100.0, 110.0, 110.0],
        "high": [105.0, 105.0, 115.0, 115.0],
        "low": [95.0, 95.0, 105.0, 105.0],
        "close": [100.0, 100.0, 110.0, 110.0],
        "volume": [1000.0] * 4,
    }, index=dates)
    positions = pd.DataFrame({"BTCUSDT": [0.0, 0.20, 0.0, 0.0]}, index=dates)
    params = {
        "initial_capital": 100000.0, "fee": 0.0005, "slippage": 0.0, "leverage": 1.0,
        "portfolio_mode": "longshort", "hedge_type": "target_weight", "trading_days_per_year": 365,
    }
    runner = QuantBTRunner()
    equity_df, _, _ = runner.run_backtest(positions=positions, data_dict={"BTCUSDT": btc_df}, params=params)
    assert abs(equity_df["equity"].iloc[-1] - 101989.50) < 0.01


def test_fee_rate_per_fill_is_one_way():
    """fee_rate_per_fill=0.0005 must charge 0.0005 per fill (not halved)."""
    dates = pd.date_range("2024-01-01", periods=4, freq="1D")
    btc_df = pd.DataFrame({
        "open": [100.0, 100.0, 110.0, 110.0],
        "high": [105.0, 105.0, 115.0, 115.0],
        "low": [95.0, 95.0, 105.0, 105.0],
        "close": [100.0, 100.0, 110.0, 110.0],
        "volume": [1000.0] * 4,
    }, index=dates)
    positions = pd.DataFrame({"BTCUSDT": [0.0, 0.20, 0.0, 0.0]}, index=dates)
    params = {
        "initial_capital": 100000.0, "fee_rate_per_fill": 0.0005, "slippage": 0.0, "leverage": 1.0,
        "portfolio_mode": "longshort", "hedge_type": "target_weight", "trading_days_per_year": 365,
    }
    runner = QuantBTRunner()
    equity_df, _, _ = runner.run_backtest(positions=positions, data_dict={"BTCUSDT": btc_df}, params=params)
    # one-way 0.0005: entry notional 20k -> $10, exit 22k -> $11; gain $2000
    assert abs(equity_df["equity"].iloc[-1] - 101979.00) < 0.01
