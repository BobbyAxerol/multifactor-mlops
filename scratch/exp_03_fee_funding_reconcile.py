"""exp_03: Fee one-way + cost convention reconcile (hand-calculated)."""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from src.multifactor_mlops.backtest.quantbt_runner import QuantBTRunner

dates = pd.date_range("2024-01-01", periods=4, freq="1D")
btc_df = pd.DataFrame({
    "open": [100.0, 100.0, 110.0, 110.0],
    "high": [105.0, 105.0, 115.0, 115.0],
    "low": [95.0, 95.0, 105.0, 105.0],
    "close": [100.0, 100.0, 110.0, 110.0],
    "volume": [1000.0] * 4,
}, index=dates)
positions = pd.DataFrame({"BTCUSDT": [0.0, 0.20, 0.0, 0.0]}, index=dates)
runner = QuantBTRunner()

# one-way 0.0005/fill
p1 = {"initial_capital": 100000.0, "fee_rate_per_fill": 0.0005, "slippage": 0.0,
      "leverage": 1.0, "portfolio_mode": "longshort", "hedge_type": "target_weight",
      "trading_days_per_year": 365}
eq, _, _ = runner.run_backtest(positions, {"BTCUSDT": btc_df}, p1)
print("final equity (one-way 0.0005):", round(eq["equity"].iloc[-1], 2), "(expect 101979.00)")
assert abs(eq["equity"].iloc[-1] - 101979.00) < 0.01

# legacy round-trip fee=0.0005 -> 0.00025/fill
p2 = dict(p1); p2.pop("fee_rate_per_fill"); p2["fee"] = 0.0005
eq2, _, _ = runner.run_backtest(positions, {"BTCUSDT": btc_df}, p2)
print("final equity (legacy fee=0.0005 round-trip):", round(eq2["equity"].iloc[-1], 2), "(expect 101989.50)")
assert abs(eq2["equity"].iloc[-1] - 101989.50) < 0.01
print("exp_03 OK")
