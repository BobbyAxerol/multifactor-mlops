# Final Outer OOS Report (single-touch)

- OOS window: 2024-01-01 -> present, split_frequency=quarterly, expanding
- Folds: 11
- QuantBT metrics (trading_days=365):
  - initial_capital: 100000.0
  - final_equity: 66412.119535
  - total_return_pct: -33.58788
  - cagr_pct: -14.603065
  - sharpe: -0.240842
  - sortino: -0.226138
  - calmar: -0.281852
  - omega: 0.951228
  - max_drawdown_pct: 51.811118
  - avg_drawdown_pct: 26.391166
  - max_dd_duration_days: 472.0
  - avg_dd_duration_days: 133.0
  - profit_factor: 0.951228
  - long_hitrate_pct: 39.600399
  - short_hitrate_pct: 43.499123
  - avg_win_pct: 1.304609
  - avg_loss_pct: -1.012438
  - expectancy_pct: -0.04971
  - num_trades: 2956.0
  - liquidated: 0.0

Per-fold Sharpe (QuantBT equity, OOS slices):

| fold start | fold end | bars | sharpe |
|---|---|---|---|
| 2024-01-01 | 2024-03-31 | 90 | -1.9039 |
| 2024-04-01 | 2024-06-30 | 90 | -1.8267 |
| 2024-07-01 | 2024-09-30 | 91 | 1.0917 |
| 2024-10-01 | 2024-12-31 | 91 | 2.8333 |
| 2025-01-01 | 2025-03-31 | 89 | 0.8144 |
| 2025-04-01 | 2025-06-30 | 90 | -1.5275 |
| 2025-07-01 | 2025-09-30 | 91 | -4.2727 |
| 2025-10-01 | 2025-12-31 | 91 | 0.3359 |
| 2026-01-01 | 2026-03-31 | 89 | 2.6279 |
| 2026-04-01 | 2026-06-30 | 90 | -1.0004 |
| 2026-07-01 | 2026-08-05 | 35 | -2.3518 |

Params: {"learning_rate": 0.04, "max_depth": 6, "colsample_bytree": 0.7, "subsample": 0.8, "num_boost_round": 75, "random_state": 42, "quantiles": 20, "inverse_vol_period": 35, "rebalance_schedule": "weekly_friday_exit", "rebalance_threshold": 0.01, "allocation_cap": 0.35, "volatility_ceiling": 0.07, "stress_multiplier": 0.30000000000000004}