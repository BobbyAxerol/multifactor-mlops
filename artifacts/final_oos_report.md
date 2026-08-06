# Final Outer OOS Report (single-touch)

- OOS window: 2024-01-01 -> present, split_frequency=quarterly, expanding
- Folds: 11
- QuantBT metrics (trading_days=365):
  - initial_capital: 100000.0
  - final_equity: 138809.46726
  - total_return_pct: 38.809467
  - cagr_pct: 13.482751
  - sharpe: 0.568098
  - sortino: 0.607252
  - calmar: 0.401186
  - omega: 1.089793
  - max_drawdown_pct: 33.607244
  - avg_drawdown_pct: 15.443877
  - max_dd_duration_days: 611.0
  - avg_dd_duration_days: 65.0
  - profit_factor: 1.089793
  - long_hitrate_pct: 47.392564
  - short_hitrate_pct: 50.288158
  - avg_win_pct: 1.140857
  - avg_loss_pct: -1.053511
  - expectancy_pct: 0.018227
  - num_trades: 31959.0
  - liquidated: 0.0

Per-fold Sharpe (QuantBT equity, OOS slices):

| fold start | fold end | bars | sharpe |
|---|---|---|---|
| 2024-01-01 | 2024-03-31 | 90 | 2.8463 |
| 2024-04-01 | 2024-06-30 | 90 | 1.4771 |
| 2024-07-01 | 2024-09-30 | 91 | -2.1608 |
| 2024-10-01 | 2024-12-31 | 91 | 3.9537 |
| 2025-01-01 | 2025-03-31 | 89 | -0.3182 |
| 2025-04-01 | 2025-06-30 | 90 | -1.2504 |
| 2025-07-01 | 2025-09-30 | 91 | -1.8439 |
| 2025-10-01 | 2025-12-31 | 91 | 1.7511 |
| 2026-01-01 | 2026-03-31 | 89 | 0.318 |
| 2026-04-01 | 2026-06-30 | 90 | 2.1161 |
| 2026-07-01 | 2026-08-05 | 35 | -3.8011 |

Params: {"learning_rate": 0.04, "max_depth": 6, "colsample_bytree": 0.7, "subsample": 0.8, "num_boost_round": 75, "random_state": 42, "quantiles": 6, "inverse_vol_period": 28, "rebalance_schedule": "daily", "rebalance_threshold": 0.02, "allocation_cap": 0.25, "volatility_ceiling": 0.07, "stress_multiplier": 0.30000000000000004}