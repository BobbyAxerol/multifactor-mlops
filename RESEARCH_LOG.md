# RESEARCH LOG — V4 (QuantBT native walk-forward, single-touch OOS)

## Baseline (contaminated, TRƯỚC khi sửa)
- HEAD `4520e9a`: Mode 4 Sharpe **1.3231** (bị selection bias + label lệch 1 bar + fee thực 0.00025/fill + funding=0). KHÔNG trust.

## V4 Canonical pipeline (mới)
- Label: open-to-open `Open_{D+1+H}/Open_{D+1} − 1`, purge `label_end_time < test_start`.
- Backtest: `QuantBTEndpoint.walk_forward(target_mode="portfolio", optimization_mode="none")`, 1-bar execution lag trong strategy adapter, model cache (fold_id, params).
- Costs: `fee_rate=0.0005` one-way/fill, slippage 1.0 bps/fill, funding thực (per-symbol daily), leverage 3.0 chỉ là margin gate (gross ~1.0).
- Tuning: dev 2022-01-01 → 2023-12-31. Stage 1 = inner purged folds, objective mean rank IC. Stage 2 = WF backtest trên dev với OOF predictions cached (không refit), robust = median − 0.5·std. Outer OOS (2024-01-01 →) chỉ chạm 1 lần ở evaluate_final.

## Kết quả đã chạy (locked artifacts trong artifacts/)

### Stage 1 — ML tuning (30 trials, dev)
- Best mean rank IC: **0.0000** → feature set hiện tại KHÔNG có cross-sectional predictive power trên dev.
- Locked: `artifacts/model_config.json` {lr 0.04, depth 6, colsample 0.7, subsample 0.8, 75 rounds}.

### OOF (dev, 8 folds)
- Overall rank IC: **-0.0031** (28084 rows) — nhất quán với IC≈0.

### Stage 2 — strategy tuning (15 trials, dev)
- Best robust score (median−0.5·std fold Sharpe): **1.584** (median của 15 trial: 0.24–1.58, spread lớn → overfit nhẹ trên dev; xem `strategy_trials.json`).
- Locked: `artifacts/strategy_config.json` {quantiles 20, inv_vol 35, weekly_friday_exit, threshold 0.01, cap 0.35, ceiling 0.07, stress 0.3}.

### FINAL OUTER OOS — SINGLE TOUCH (2024-01-01 → 2026-08-05, 11 folds quarterly, expanding)
| Metric | Giá trị |
|---|---|
| Sharpe | **-0.241** |
| CAGR | -14.60% |
| Total return | -33.59% |
| Max drawdown | 51.81% |
| Profit factor | 0.951 |
| Trades | 2956 |
| Median fold Sharpe | -1.00 |

Per-fold: 2024Q1 -1.90, Q2 -1.83, Q3 +1.09, Q4 +2.83, 2025Q1 +0.81, Q2 -1.53, Q3 -4.27, Q4 +0.34, 2026Q1 +2.63, Q2 -1.00, Q3(35 ngày) -2.35.

- MLflow run: `a9e84c2c5f104c578e12b600a55e5408`
- Report: `artifacts/final_oos_report.md`, metrics: `artifacts/final_oos_metrics.json`

## Kết luận
1. **Con số trust được duy nhất hiện tại: OOS Sharpe -0.24 (không có edge).** Con số cũ 1.32 là ảo tưởng do các lỗi đã sửa.
2. Nguyên nhân gốc: rank IC ≈ 0 của feature set (momentum/flow/carry/margin + macro z-score, percentile-ranked). Model XGBoost không học được alpha từ các feature này.
3. Không có data leakage/look-ahead trong pipeline V4 (đã test: future-mutation, purge label_end_time, output coverage, 1-bar lag, no bfill, OOS không xuất hiện trong tuning).

## Re-run commands
```bash
poetry run python -m src.multifactor_mlops.optimization.stage1_ml_tuning --trials 30
poetry run python -m src.multifactor_mlops.pipelines.generate_oof_predictions
poetry run python -m src.multifactor_mlops.optimization.stage2_strategy_tuning --trials 15
poetry run python -m src.multifactor_mlops.pipelines.evaluate_final --oos-start 2024-01-01
poetry run pytest tests/ -q
```

## Hướng tiếp theo (đề xuất, cần duyệt)
- Tìm feature mới có IC thật (thử: funding regime, cross-sectional momentum spread, HV vs IV, carry decay, turnover flow) và test trên dev TRƯỚC khi chạm OOS.
- Thử label horizon H > 1 và tần suất rebalance.
- Kiểm tra thủ công universe survivorship: hiện chọn top-40 theo dollar volume toàn mẫu.
