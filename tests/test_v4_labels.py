"""
V4 Label Contract Tests: open-to-open labels with explicit per-symbol timestamps
and canonical purge by label_end_time.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.multifactor_mlops.labels.returns import add_forward_open_labels, filter_train_by_label_end
from src.multifactor_mlops.features.panel import PanelDatasetBuilder


def _make_ohlcv(n=120, start="2024-01-01"):
    dates = pd.date_range(start, periods=n, freq="1D")
    close = np.linspace(100, 200, n)
    return pd.DataFrame({
        "open": close,
        "high": close + 2,
        "low": close - 2,
        "close": close,
        "volume": 1000.0,
    }, index=dates)


def test_label_timestamps_follow_actual_bars():
    """label_start_time == next bar, label_end_time == bar D+1+H (actual index)."""
    df = _make_ohlcv()
    out = add_forward_open_labels(df[["open"]], holding_bars=1)
    assert out["label_start_time"].iloc[0] == df.index[1]
    assert out["label_end_time"].iloc[0] == df.index[2]
    # last two bars dropped (incomplete horizon)
    assert len(out) == len(df) - 2
    # y = Open_{D+2}/Open_{D+1} - 1
    assert abs(out["target"].iloc[0] - (df["open"].iloc[2] / df["open"].iloc[1] - 1.0)) < 1e-12


def test_label_never_overlaps_feature_row():
    """Feature row D must not contain any price used by the label of row D."""
    df = _make_ohlcv()
    out = add_forward_open_labels(df[["open"]], holding_bars=1)
    exec_price = df["open"].shift(-1)
    assert out.index[0] < out["label_start_time"].iloc[0]


def test_panel_contains_timing_columns_and_target_demeaned():
    data_dict = {"AAA": _make_ohlcv(), "BBB": _make_ohlcv(n=120, start="2024-01-01")}
    builder = PanelDatasetBuilder(windows=[7, 14, 30], cross_sectional_rank=True, lag=1)
    panel = builder.build_panel_dataset(data_dict, ["AAA", "BBB"])
    assert {"target", "decision_time", "label_start_time", "label_end_time"}.issubset(panel.columns)
    # target demeaned per timestamp
    demeaned = panel["target"].groupby(level="Time").transform("mean").abs()
    assert (demeaned < 1e-9).all()


def test_purge_by_label_end_time():
    data_dict = {"AAA": _make_ohlcv(n=120, start="2023-09-01"), "BBB": _make_ohlcv(n=120, start="2023-09-01")}
    builder = PanelDatasetBuilder(windows=[7, 14, 30], cross_sectional_rank=True, lag=1)
    panel = builder.build_panel_dataset(data_dict, ["AAA", "BBB"])
    test_start = pd.Timestamp("2023-12-20")
    train = filter_train_by_label_end(panel, test_start)
    assert (train["label_end_time"] < test_start).all()
    assert len(train) < len(panel)
