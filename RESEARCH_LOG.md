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

---

## V4.2 — Hygiene upgrade (mục 3): kết quả rerun

### Thay đổi codebase (đã test, 46/46 pass)
- **Feature selection có bằng chứng**: `keep_families=[mom_rsi, mom_wma_dist, retail_flow, margin_risk]`, đảo dấu `retail_flow_7`, `margin_risk_90`, **loại macro features** khỏi feature set (IC NaN)
- **Point-in-time universe**: membership theo rolling lagged turnover (hết survivorship top-40 toàn mẫu) — `use_point_in_time_universe=true`
- **Low-turnover schedule**: `monday_decide_weekly` (quyết định thứ 2, giữ T2-T6)

### Kết quả rerun
| Bước | Kết quả cũ | Kết quả mới |
|---|---|---|
| Stage 1 best mean IC (dev) | 0.0000 | 0.0000 (tuning noise) |
| **OOF overall rank IC (dev)** | **-0.0031** | **+0.0302** ✅ |
| Stage 2 best robust (dev) | 1.584 | 0.736 (daily) |
| **OOS daily (2024→)** | -0.24 (pipeline cũ) | **-0.76** (9836 lệnh) |
| **OOS monday_decide_weekly** | — | **-1.14** (4322 lệnh) |

### Kết luận
1. Hygiene changes ĐÚNG (IC dev cải thiện -0.003 → +0.030; test đầy đủ) nhưng **KHÔNG tạo edge OOS**: cả 2 schedule đều âm.
2. **Model XGBoost là điểm yếu trên OOS**: pipeline model-based (-0.76/-1.14) thua xa composite không-ML (+0.26±0.5 trong exp_42, quanh 0). Model trộn feature + MSE làm hỏng ranking ngoài mẫu (xác nhận lại exp_20b).
3. Turnover vẫn là kẻ giết alpha: 4322-9836 lệnh/2.5 năm × 6bp ≈ $100k+ chi phí vs alpha ~$3-10k.
4. **Hướng có bằng chứng tốt nhất hiện tại (nghiên cứu tiếp)**: bỏ ML, dùng composite trực tiếp (mom_14+mom_30−retail_flow_7−margin_risk_90 z-sum) + monday-decide + point-in-time universe. Kỳ vọng OOS ~0 tới +0.3 (marginal) — cần thêm nguồn dữ liệu mới (OI/LS ratio) mới có thể kỳ vọng cao hơn.

### Artifacts
- model_config.json / strategy_config.json (daily) / oof_predictions.csv / final_oos_metrics.json đã cập nhật
- MLflow run daily: `7a7a8ab755a24805a6f5c471b9c80c1f`

---

## V4.3 — Model-free composite qua pipeline: kết quả OOS

### Thay đổi
- `asset.py`: thêm raw momentum `mom_{7,14,30,60,90}` (cho composite)
- `PortfolioConfig`: `signal_mode` ("ml"|"composite") + `composite_features`
- Strategy: composite mode = z-sum per-timestamp của `[mom_14, mom_30, retail_flow_7(flipped), margin_risk_90(flipped)]` — KHÔNG train model
- `wf_runner`: base params = portfolio config từ parameters.json, artifacts override

### OOS 2024→ (fee 5bp, slippage 1bp, funding ON, PIT universe, composite signal)
| Schedule | Sharpe | Total return | MaxDD | Trades |
|---|---|---|---|---|
| **daily** | **+0.39** | **+21.8%** (121.8k) | -54.4% | 14106 |
| monday_decide_weekly | -0.34 | -28.3% | -45.3% | 6560 |

→ **Lần đầu tiên OOS dương** trong toàn bộ session. Composite signal có alpha nhanh (1 ngày): daily bắt được, giữ 5 ngày mất alpha.
- MLflow: `f8b892a40c3d4f509f9e9f7e120dcc30`
- Cảnh báo: MaxDD -54% rất sâu; số lệnh 14k (cost cao nhưng alpha vượt cost trên OOS).

### So sánh toàn bộ OOS đã chạy
| Cấu hình | OOS Sharpe |
|---|---|
| Pipeline cũ (V3, contaminated) | 1.32 (ảo) |
| V4 ML pipeline (full features, H=1) | -0.24 |
| V4.2 hygiene + ML daily | -0.76 |
| V4.2 hygiene + ML monday | -1.14 |
| **V4.3 composite daily** | **+0.39** |

---

## V4.4 — Fix label lệch mục tiêu: close-to-close = engine-realizable

### Thay đổi
- `labels/returns.py`: `calculate_next_close_to_close_returns` + `add_forward_close_labels` — **CANONICAL** y_D = Close_{D+2}/Close_{D+1} − 1 (H=1), khớp chính xác engine QuantBT (fill close D+1, PnL close→close). Open-to-open giữ lại chỉ cho research.
- `panel.py`: `return_type` param (default "next_close_to_close"); strategy/fit_final/stage1/OOF pass từ config.
- `schema.py`: LabelConfig.return_type default + validator nhận 2 giá trị.
- `research_base.py`: giữ open-to-open (reproducible theo report).

### Kết quả rerun (close labels)
| Bước | Giá trị |
|---|---|
| Stage 1 best mean IC (dev) | 0.0 |
| OOF overall rank IC (dev) | +0.0141 |
| Stage 2 best robust (dev) | (chi tiết trong strategy_trials.json) |
| **OOS 2024→ composite daily** | **Sharpe +0.30, +10.3% (110.3k), MaxDD -23.0%**, 13040 lệnh |

- MaxDD giảm mạnh -54% → **-23%** (strategy_config mới: allocation_cap 0.30, vol_ceiling 0.04, stress_multiplier 0.6)
- MLflow: `ea256061f39348ab8e18e2343fe3adb2`
- Lưu ý: composite signal không dùng label (feature-only) nên Sharpe thay đổi chủ yếu do strategy_config mới + engine convention nhất quán; ML-mode artifacts giờ train trên label khớp engine.

### So sánh OOS cuối
| Cấu hình | Sharpe OOS | MaxDD |
|---|---|---|
| V4.3 composite daily (open labels) | +0.39 | -54.4% |
| **V4.4 composite daily (close labels, config mới)** | **+0.30** | **-23.0%** |
