"""
Canonical Timing Contract & Return Labels for next-open execution.
Enforces point-in-time timing assertions and generates realizable next-open returns.

Contract (documented):
    decision after close D -> prediction after close D -> execution open D+1 -> label open D+1 -> open D+1+H
    y = Open_{D+1+H} / Open_{D+1} - 1
Timestamps (decision_time, label_start_time, label_end_time) are taken from the ACTUAL
shifted bar index of each symbol -- never calendar arithmetic (decision_time + 1 day).
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

def add_forward_open_labels(
    symbol_df: pd.DataFrame,
    holding_bars: int = 1
) -> pd.DataFrame:
    """
    Adds open-to-open forward return label plus explicit timing timestamp columns
    taken from the ACTUAL bar index (shifted per symbol), never calendar arithmetic.

    Columns added:
        target            : Open_{D+1+H}/Open_{D+1} - 1
        decision_time     : bar D (feature available time)
        label_start_time  : bar D+1 (execution open)
        label_end_time    : bar D+1+H (exit open)
    """
    df = symbol_df.copy()
    if 'open' not in df.columns:
        raise ValueError("add_forward_open_labels requires an 'open' column (next-open contract).")
    times = pd.DatetimeIndex(df.index)
    # POSITIONAL shift (Series.shift), NOT calendar arithmetic:
    # label_start_time must be the ACTUAL next bar of the symbol.
    time_series = pd.Series(times)
    df['decision_time'] = times
    df['label_start_time'] = time_series.shift(-1).values
    df['label_end_time'] = time_series.shift(-(1 + holding_bars)).values
    df['target'] = calculate_next_open_to_open_returns(df['open'], holding_bars=holding_bars)
    df = df.dropna(subset=['target'])
    return df

def filter_train_by_label_end(
    panel: pd.DataFrame,
    test_start,
    label_end_col: str = "label_end_time"
) -> pd.DataFrame:
    """
    Purges training rows whose label horizon ends at or after test_start:
    keep rows with label_end_time < test_start only.

    This is the canonical purge used at EVERY train/test boundary.
    """
    if panel is None or panel.empty:
        return panel
    if label_end_col not in panel.columns:
        raise ValueError(f"filter_train_by_label_end requires '{label_end_col}' column in panel.")
    test_start_ts = pd.Timestamp(test_start)
    return panel[panel[label_end_col] < test_start_ts]
