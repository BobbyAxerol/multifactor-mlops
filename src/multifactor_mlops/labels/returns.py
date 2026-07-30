"""
Canonical Timing Contract & Return Labels for next-open execution.
Enforces point-in-time timing assertions and generates realizable next-open returns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

class TimingContractError(Exception):
    """Raised when timing assertions or causality constraints are violated."""
    pass

def assert_timing_contract(
    feature_available_at: pd.Timestamp,
    decision_time: pd.Timestamp,
    execution_time: pd.Timestamp,
    label_start_time: pd.Timestamp,
    label_end_time: pd.Timestamp
) -> None:
    """
    Validates explicit timing contract assertions.
    """
    if feature_available_at > decision_time:
        raise TimingContractError(
            f"Look-Ahead Bias detected! Feature available_at ({feature_available_at}) > decision_time ({decision_time})"
        )
    if execution_time <= decision_time:
        raise TimingContractError(
            f"Causality error! Execution time ({execution_time}) <= decision_time ({decision_time})"
        )
    if label_start_time != execution_time:
        raise TimingContractError(
            f"Label start mismatch! Label start ({label_start_time}) != execution_time ({execution_time})"
        )
    if label_end_time <= label_start_time:
        raise TimingContractError(
            f"Invalid label duration! Label end ({label_end_time}) <= label start ({label_start_time})"
        )

def calculate_next_open_to_open_returns(
    open_prices: pd.Series,
    holding_bars: int = 1
) -> pd.Series:
    """
    Calculates realizable Next-Open to Next-Open forward return for daily crypto bars.
    
    Formula:
        y_{i,D} = (Open_{i, D + 1 + holding_bars} / Open_{i, D + 1}) - 1
    
    Parameters
    ----------
    open_prices : pd.Series
        Series of daily Open prices sorted chronologically.
    holding_bars : int
        Number of bars position is held (default 1).

    Returns
    -------
    pd.Series
        Target forward return aligned with decision bar D.
    """
    open_prices = open_prices.sort_index()
    # Open_{D+1} is execution price
    exec_price = open_prices.shift(-1)
    # Open_{D+1+holding_bars} is exit price
    exit_price = open_prices.shift(-(1 + holding_bars))
    
    target_returns = (exit_price / exec_price - 1.0).rename("target")
    return target_returns
