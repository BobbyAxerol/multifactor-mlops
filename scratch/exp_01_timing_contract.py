"""exp_01: Label open-to-open contract + purge check on synthetic data."""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from src.multifactor_mlops.labels.returns import add_forward_open_labels, filter_train_by_label_end

dates = pd.date_range("2024-01-01", periods=30, freq="1D")
close = np.linspace(100, 130, 30)
df = pd.DataFrame({"open": close, "close": close}, index=dates)

out = add_forward_open_labels(df[["open"]], holding_bars=1)
print("rows:", len(out), "(expect 28: last 2 dropped)")
print("row0: decision=%s start=%s end=%s target=%.4f" % (
    out.index[0].date(), out["label_start_time"].iloc[0].date(),
    out["label_end_time"].iloc[0].date(), out["target"].iloc[0]))
assert out["label_start_time"].iloc[0] == dates[1]
assert out["label_end_time"].iloc[0] == dates[2]
assert len(out) == 28

purged = filter_train_by_label_end(out, "2024-01-25")
print("purged rows for test_start=2024-01-25:", len(purged), "(label_end < test_start)")
assert (purged["label_end_time"] < pd.Timestamp("2024-01-25")).all()
assert len(purged) < len(out)
print("exp_01 OK")
