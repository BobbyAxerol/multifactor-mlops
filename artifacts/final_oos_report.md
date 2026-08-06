# Final Outer OOS Report (single-touch)

- OOS window: 2024-01-01 -> present, split_frequency=quarterly, expanding
- Folds: 11
- QuantBT metrics (trading_days=365):
  - initial_capital: 100000.0
  - final_equity: 121797.264168
  - total_return_pct: 21.797264
  - cagr_pct: 7.902047
  - sharpe: 0.388655
  - sortino: 0.440001
  - calmar: 0.145207
  - omega: 1.059677
  - max_drawdown_pct: 54.419077
  - avg_drawdown_pct: 29.146582
  - max_dd_duration_days: 611.0
  - avg_dd_duration_days: 77.0
  - profit_factor: 1.059677
  - long_hitrate_pct: 42.944411
  - short_hitrate_pct: 44.81983
  - avg_win_pct: 1.465353
  - avg_loss_pct: -1.25206
  - expectancy_pct: -0.059601
  - num_trades: 14106.0
  - liquidated: 0.0

Per-fold Sharpe (QuantBT equity, OOS slices):

| fold start | fold end | bars | sharpe |
|---|---|---|---|
| 2024-01-01 | 2024-03-31 | 90 | 2.2128 |
| 2024-04-01 | 2024-06-30 | 90 | 1.1239 |
| 2024-07-01 | 2024-09-30 | 91 | -3.1794 |
| 2024-10-01 | 2024-12-31 | 91 | 3.888 |
| 2025-01-01 | 2025-03-31 | 89 | -2.3984 |
| 2025-04-01 | 2025-06-30 | 90 | -2.4195 |
| 2025-07-01 | 2025-09-30 | 91 | -2.4791 |
| 2025-10-01 | 2025-12-31 | 91 | 2.2227 |
| 2026-01-01 | 2026-03-31 | 89 | 0.959 |
| 2026-04-01 | 2026-06-30 | 90 | 2.7769 |
| 2026-07-01 | 2026-08-05 | 35 | -2.8432 |

Params: {"learning_rate": 0.04, "max_depth": 6, "colsample_bytree": 0.7, "subsample": 0.8, "num_boost_round": 75, "random_state": 42, "quantiles": 16, "inverse_vol_period": 42, "rebalance_schedule": "daily", "rebalance_threshold": 0.01, "allocation_cap": 0.25, "volatility_ceiling": 0.08, "stress_multiplier": 0.5}