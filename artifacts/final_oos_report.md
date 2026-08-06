# Final Outer OOS Report (single-touch)

- OOS window: 2024-01-01 -> present, split_frequency=quarterly, expanding
- Folds: 11
- QuantBT metrics (trading_days=365):
  - initial_capital: 100000.0
  - final_equity: 110348.83195
  - total_return_pct: 10.348832
  - cagr_pct: 3.871203
  - sharpe: 0.302016
  - sortino: 0.327786
  - calmar: 0.168263
  - omega: 1.055249
  - max_drawdown_pct: 23.006836
  - avg_drawdown_pct: 8.756101
  - max_dd_duration_days: 376.0
  - avg_dd_duration_days: 84.0
  - profit_factor: 1.055249
  - long_hitrate_pct: 35.704771
  - short_hitrate_pct: 37.405406
  - avg_win_pct: 0.795723
  - avg_loss_pct: -0.529111
  - expectancy_pct: -0.044817
  - num_trades: 13040.0
  - liquidated: 0.0

Per-fold Sharpe (QuantBT equity, OOS slices):

| fold start | fold end | bars | sharpe |
|---|---|---|---|
| 2024-01-01 | 2024-03-31 | 90 | 0.3735 |
| 2024-04-01 | 2024-06-30 | 90 | -0.2011 |
| 2024-07-01 | 2024-09-30 | 91 | -4.4524 |
| 2024-10-01 | 2024-12-31 | 91 | 2.2214 |
| 2025-01-01 | 2025-03-31 | 89 | -1.1585 |
| 2025-04-01 | 2025-06-30 | 90 | 1.982 |
| 2025-07-01 | 2025-09-30 | 91 | 0.4266 |
| 2025-10-01 | 2025-12-31 | 91 | 2.4746 |
| 2026-01-01 | 2026-03-31 | 89 | -0.8832 |
| 2026-04-01 | 2026-06-30 | 90 | 1.4789 |
| 2026-07-01 | 2026-08-05 | 35 | -0.5616 |

Params: {"learning_rate": 0.04, "max_depth": 6, "colsample_bytree": 0.7, "subsample": 0.8, "num_boost_round": 75, "random_state": 42, "quantiles": 10, "inverse_vol_period": 21, "rebalance_schedule": "weekly_friday_exit", "rebalance_threshold": 0.02, "allocation_cap": 0.30000000000000004, "volatility_ceiling": 0.04, "stress_multiplier": 0.6000000000000001}