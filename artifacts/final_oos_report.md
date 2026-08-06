# Final Outer OOS Report (single-touch)

- OOS window: 2024-01-01 -> present, split_frequency=quarterly, expanding
- Folds: 11
- QuantBT metrics (trading_days=365):
  - initial_capital: 100000.0
  - final_equity: 42235.454453
  - total_return_pct: -57.764546
  - cagr_pct: -28.282225
  - sharpe: -0.756904
  - sortino: -0.769168
  - calmar: -0.393216
  - omega: 0.890133
  - max_drawdown_pct: 71.925366
  - avg_drawdown_pct: 45.625079
  - max_dd_duration_days: 944.0
  - avg_dd_duration_days: 472.0
  - profit_factor: 0.890133
  - long_hitrate_pct: 42.294538
  - short_hitrate_pct: 41.349293
  - avg_win_pct: 1.27359
  - avg_loss_pct: -1.262961
  - expectancy_pct: -0.202127
  - num_trades: 9836.0
  - liquidated: 0.0

Per-fold Sharpe (QuantBT equity, OOS slices):

| fold start | fold end | bars | sharpe |
|---|---|---|---|
| 2024-01-01 | 2024-03-31 | 90 | -0.7012 |
| 2024-04-01 | 2024-06-30 | 90 | 0.3258 |
| 2024-07-01 | 2024-09-30 | 91 | -5.06 |
| 2024-10-01 | 2024-12-31 | 91 | -0.3386 |
| 2025-01-01 | 2025-03-31 | 89 | -2.1293 |
| 2025-04-01 | 2025-06-30 | 90 | -4.9702 |
| 2025-07-01 | 2025-09-30 | 91 | -0.7703 |
| 2025-10-01 | 2025-12-31 | 91 | 0.3179 |
| 2026-01-01 | 2026-03-31 | 89 | 1.75 |
| 2026-04-01 | 2026-06-30 | 90 | 1.7677 |
| 2026-07-01 | 2026-08-05 | 35 | 0.3775 |

Params: {"learning_rate": 0.04, "max_depth": 6, "colsample_bytree": 0.7, "subsample": 0.8, "num_boost_round": 75, "random_state": 42, "quantiles": 16, "inverse_vol_period": 42, "rebalance_schedule": "daily", "rebalance_threshold": 0.01, "allocation_cap": 0.25, "volatility_ceiling": 0.08, "stress_multiplier": 0.5}