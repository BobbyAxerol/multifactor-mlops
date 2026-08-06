# Final Outer OOS Report (single-touch)

- OOS window: 2024-01-01 -> present, split_frequency=quarterly, expanding
- Folds: 11
- QuantBT metrics (trading_days=365):
  - initial_capital: 100000.0
  - final_equity: 148574.534915
  - total_return_pct: 48.574535
  - cagr_pct: 16.497743
  - sharpe: 0.65457
  - sortino: 0.69997
  - calmar: 0.5035
  - omega: 1.104183
  - max_drawdown_pct: 32.766106
  - avg_drawdown_pct: 14.245787
  - max_dd_duration_days: 549.0
  - avg_dd_duration_days: 61.0
  - profit_factor: 1.104183
  - long_hitrate_pct: 47.7449
  - short_hitrate_pct: 50.574651
  - avg_win_pct: 1.140857
  - avg_loss_pct: -1.053041
  - expectancy_pct: 0.025474
  - num_trades: 31959.0
  - liquidated: 0.0

Per-fold Sharpe (QuantBT equity, OOS slices):

| fold start | fold end | bars | sharpe |
|---|---|---|---|
| 2024-01-01 | 2024-03-31 | 90 | 2.9487 |
| 2024-04-01 | 2024-06-30 | 90 | 1.5458 |
| 2024-07-01 | 2024-09-30 | 91 | -2.0917 |
| 2024-10-01 | 2024-12-31 | 91 | 4.0403 |
| 2025-01-01 | 2025-03-31 | 89 | -0.2543 |
| 2025-04-01 | 2025-06-30 | 90 | -1.1441 |
| 2025-07-01 | 2025-09-30 | 91 | -1.7436 |
| 2025-10-01 | 2025-12-31 | 91 | 1.8232 |
| 2026-01-01 | 2026-03-31 | 89 | 0.3969 |
| 2026-04-01 | 2026-06-30 | 90 | 2.2143 |
| 2026-07-01 | 2026-08-05 | 35 | -3.6422 |

Params: {"learning_rate": 0.04, "max_depth": 6, "colsample_bytree": 0.7, "subsample": 0.8, "num_boost_round": 75, "random_state": 42, "quantiles": 6, "inverse_vol_period": 28, "rebalance_schedule": "daily", "rebalance_threshold": 0.02, "allocation_cap": 0.25, "volatility_ceiling": 0.07, "stress_multiplier": 0.30000000000000004}