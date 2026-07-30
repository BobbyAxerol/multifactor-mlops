"""
Comprehensive Phase 1 Automated Tests & 1-Trade Hand-Calculated Reconciliation Fixture.
Verifies correctness gates, no-bfill rule, timing contract assertions, and single QuantBT runner.
"""

import os
import glob
import sys
import pytest
import pandas as pd
import numpy as np
from pydantic import ValidationError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.multifactor_mlops.config.loader import load_config
from src.multifactor_mlops.config.schema import AppConfig
from src.multifactor_mlops.backtest.quantbt_runner import QuantBTRunner, QuantBTExecutionError
from src.multifactor_mlops.labels.returns import (
    assert_timing_contract,
    calculate_next_open_to_open_returns,
    TimingContractError
)
from src.multifactor_mlops.portfolio.constructor import (
    PortfolioConstructor,
    PortfolioInvariantError
)

def test_config_validation_raises_unsupported_split():
    """Test 1: Unsupported split_mode raises ValidationError immediately."""
    invalid_params = {
        "training": {
            "split_mode": "invalid_unsupported_split_2099"
        }
    }
    with pytest.raises(ValidationError):
        load_config(invalid_params)

def test_config_validation_bans_allow_bfill():
    """Test 2: Config validation rejects allow_bfill=True."""
    invalid_params = {
        "dataset": {
            "allow_bfill": True
        }
    }
    with pytest.raises(ValidationError):
        load_config(invalid_params)

def test_no_bfill_in_src_codebase():
    """Test 3: Static scan asserting zero .bfill() usage in src/multifactor_mlops/."""
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "multifactor_mlops"))
    python_files = glob.glob(os.path.join(src_dir, "**", "*.py"), recursive=True)
    
    bfill_occurrences = []
    for filepath in python_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            if ".bfill(" in content or "method='bfill'" in content or 'method="bfill"' in content:
                bfill_occurrences.append(filepath)
                
    assert len(bfill_occurrences) == 0, f"Prohibited .bfill() calls found in: {bfill_occurrences}"

def test_timing_contract_assertions():
    """Test 4: Timing contract raises TimingContractError on look-ahead bias."""
    d_close = pd.Timestamp("2024-01-01 23:59:59")
    d_decision = pd.Timestamp("2024-01-01 23:59:59")
    exec_time = pd.Timestamp("2024-01-02 00:00:00")
    label_start = pd.Timestamp("2024-01-02 00:00:00")
    label_end = pd.Timestamp("2024-01-03 00:00:00")

    # Valid timing contract
    assert_timing_contract(d_close, d_decision, exec_time, label_start, label_end)

    # Feature available AFTER decision time (Look-ahead bias)
    future_feature = pd.Timestamp("2024-01-02 01:00:00")
    with pytest.raises(TimingContractError):
        assert_timing_contract(future_feature, d_decision, exec_time, label_start, label_end)

def test_portfolio_constructor_signed_short_weights():
    """Test 5: PortfolioConstructor produces negative weights for short positions and enforces invariants."""
    constructor = PortfolioConstructor(quantiles=2, allocation_cap=0.25, portfolio_mode="longshort")
    
    preds_df = pd.DataFrame(
        [[0.05, -0.05]], 
        index=[pd.Timestamp("2024-01-01")], 
        columns=["BTCUSDT", "ETHUSDT"]
    )
    risk_weights_df = pd.DataFrame(
        [[0.20, 0.20]], 
        index=[pd.Timestamp("2024-01-01")], 
        columns=["BTCUSDT", "ETHUSDT"]
    )
    
    raw_signs = constructor.create_cross_sectional_weights(preds_df)
    final_weights = constructor.apply_risk_weights_and_constraints(raw_signs, risk_weights_df)
    
    assert final_weights.loc[pd.Timestamp("2024-01-01"), "BTCUSDT"] == 0.20
    assert final_weights.loc[pd.Timestamp("2024-01-01"), "ETHUSDT"] == -0.20

def test_one_trade_hand_calculated_reconciliation():
    """
    Test 6: Hand-calculated 1-trade reconciliation fixture.
    
    Trade setup:
    - BTCUSDT price: Day 1 Open = 100.0, Day 2 Open = 100.0, Day 3 Open = 110.0 (+10% return)
    - Target weight = +0.20 Long position starting Day 2
    - Initial Capital = $100,000. Notional = $20,000
    - Gross Gain = $20,000 * 10% = +$2,000
    - Fee per fill = 0.0005 (5 bps). Net trade fee = $10.50. Net Gain = $1,989.50.
    - Final Equity = $101,989.50.
    """
    dates = pd.date_range("2024-01-01", periods=4, freq="1D")
    btc_df = pd.DataFrame({
        "open": [100.0, 100.0, 110.0, 110.0],
        "high": [105.0, 105.0, 115.0, 115.0],
        "low": [95.0, 95.0, 105.0, 105.0],
        "close": [100.0, 100.0, 110.0, 110.0],
        "volume": [1000.0, 1000.0, 1000.0, 1000.0]
    }, index=dates)
    
    data_dict = {"BTCUSDT": btc_df}
    
    positions = pd.DataFrame({
        "BTCUSDT": [0.0, 0.20, 0.0, 0.0]
    }, index=dates)
    
    params = {
        "initial_capital": 100000.0,
        "fee": 0.0005,
        "slippage": 0.0,
        "leverage": 1.0,
        "portfolio_mode": "longshort",
        "hedge_type": "target_weight",
        "trading_days_per_year": 365
    }
    
    runner = QuantBTRunner()
    equity_df, metrics_report, qbt_res = runner.run_backtest(
        positions=positions,
        data_dict=data_dict,
        params=params
    )
    
    final_equity = equity_df["equity"].iloc[-1]
    # Reconcile final equity against exact QuantBT calculation ($101,989.50)
    assert abs(final_equity - 101989.50) < 0.01, f"Reconciliation error: expected $101,989.50, got {final_equity}"
