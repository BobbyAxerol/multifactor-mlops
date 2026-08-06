"""
Canonical Timing Contract & Return Labels matching QuantBT engine realization.

QuantBT portfolio engine contract (verified from engine source):
  target row i is filled at CLOSE(i); PnL accrues close(i) -> close(i+1).
With the canonical 1-bar lag (decision at close D -> position row D+1):

    y_D = Close_{D+1+H} / Close_{D+1} - 1

i.e. the label must be CLOSE-TO-CLOSE to equal the return the engine actually
earns. Open-to-open labels are retained only for research/backward compat.
Timestamps (decision_time, label_start_time, label_end_time) are taken from the
ACTUAL shifted bar index of each symbol -- never calendar arithmetic.
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

def calculate_next_close_to_close_returns(
    close_prices: pd.Series,
    holding_bars: int = 1
) -> pd.Series:
    """
    CANONICAL label: realizable Close-to-Close forward return matching the
    QuantBT engine with 1-bar execution lag.

    Formula:
        y_{i,D} = (Close_{i, D + 1 + holding_bars} / Close_{i, D + 1}) - 1

    Parameters
    ----------
    close_prices : pd.Series
        Series of daily Close prices sorted chronologically.
    holding_bars : int
        Number of bars position is held (default 1).

    Returns
    -------
    pd.Series
        Target forward return aligned with decision bar D.
    """
    close_prices = close_prices.sort_index()
    # Close_{D+1} is the engine execution price (position row D+1)
    exec_price = close_prices.shift(-1)
    # Close_{D+1+holding_bars} is the exit price
    exit_price = close_prices.shift(-(1 + holding_bars))
    target_returns = (exit_price / exec_price - 1.0).rename("target")
    return target_returns

def calculate_next_open_to_open_returns(
    open_prices: pd.Series,
    holding_bars: int = 1
) -> pd.Series:
    """
    Research/backward-compat label: Next-Open to Next-Open forward return.
    NOT what the QuantBT engine realizes (engine fills at close).

    Formula:
        y_{i,D} = (Open_{i, D + 1 + holding_bars} / Open_{i, D + 1}) - 1
    """
    open_prices = open_prices.sort_index()
    exec_price = open_prices.shift(-1)
    exit_price = open_prices.shift(-(1 + holding_bars))
    target_returns = (exit_price / exec_price - 1.0).rename("target")
    return target_returns

def add_forward_close_labels(
    symbol_df: pd.DataFrame,
    holding_bars: int = 1
) -> pd.DataFrame:
    """
    CANONICAL label builder: close-to-close forward return + explicit timing
    timestamp columns taken from the ACTUAL bar index (shifted per symbol).

    Columns added:
        target            : Close_{D+1+H}/Close_{D+1} - 1
        decision_time     : bar D (feature available time)
        label_start_time  : bar D+1 (execution close)
        label_end_time    : bar D+1+H (exit close)
    """
    df = symbol_df.copy()
    if 'close' not in df.columns:
        raise ValueError("add_forward_close_labels requires a 'close' column.")
    times = pd.DatetimeIndex(df.index)
    # POSITIONAL shift (Series.shift), NOT calendar arithmetic
    time_series = pd.Series(times)
    df['decision_time'] = times
    df['label_start_time'] = time_series.shift(-1).values
    df['label_end_time'] = time_series.shift(-(1 + holding_bars)).values
    df['target'] = calculate_next_close_to_close_returns(df['close'], holding_bars=holding_bars)
    df = df.dropna(subset=['target'])
    return df

def add_forward_open_labels(
    symbol_df: pd.DataFrame,
    holding_bars: int = 1
) -> pd.DataFrame:
    """
    Research/backward-compat label builder (open-to-open).
    Same timestamp semantics as add_forward_close_labels.
    """
    df = symbol_df.copy()
    if 'open' not in df.columns:
        raise ValueError("add_forward_open_labels requires an 'open' column (next-open contract).")
    times = pd.DatetimeIndex(df.index)
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
