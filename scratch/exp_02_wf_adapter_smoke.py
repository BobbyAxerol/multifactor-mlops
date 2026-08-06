"""exp_02: QuantBT native walk-forward adapter smoke test on synthetic data."""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from src.multifactor_mlops.config.schema import AppConfig
from src.multifactor_mlops.backtest.wf_runner import WalkForwardQuantBTRunner, extract_equity

rng = np.random.default_rng(7)
dates = pd.date_range("2021-01-01", periods=365 * 3, freq="1D")
data = {}
for sym, base in [("AAA", 100.0), ("BBB", 200.0), ("CCC", 50.0)]:
    rets = rng.normal(0.0005, 0.02, len(dates))
    close = base * np.cumprod(1 + rets)
    data[sym] = pd.DataFrame({
        "open": close * 1.001, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000.0,
    }, index=dates)

params = {
    "quantiles": 4, "inverse_vol_period": 14, "allocation_cap": 0.5,
    "volatility_ceiling": None, "rebalance_schedule": "daily", "stress_multiplier": 0.4,
}
runner = WalkForwardQuantBTRunner()
res = runner.run(data, list(data), AppConfig(), params,
                 split_mode="walk_forward_2022", split_frequency="yearly", window_mode="expanding")

wf = res.metadata["walk_forward"]
print("n_folds:", wf["n_folds"])
print("fold_table columns:", list(wf["fold_table"].columns))
equity = extract_equity(res, 100000.0)
print("equity bars:", len(equity), "start:", round(equity["equity"].iloc[0], 2),
      "end:", round(equity["equity"].iloc[-1], 2))
active = equity[equity["time"] >= pd.Timestamp("2022-01-01")]
print("active bars (OOS):", len(active))
assert wf["n_folds"] >= 2
assert abs(equity["equity"].iloc[0] - 100000.0) < 1e-6
print("exp_02 OK")
