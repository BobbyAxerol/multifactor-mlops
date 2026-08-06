# MULTIFACTOR MLOPS — KẾ HOẠCH SỬA TOÀN DIỆN (V4, DÙNG QUANTBT NATIVE WALK-FORWARD)

> Trạng thái: **KẾ HOẠCH** — chưa implement. Mỗi phase cần sự đồng ý của user trước khi thực thi.
> Tham khảo chuẩn: `/root/bobby/pool_alpha/MLops/deep_momentum` (4-phase trustable pipeline, label open-to-open, purge `label_end_time`, locked artifacts, test anti-leakage).
> Ràng buộc cứng: **KHÔNG BAO GIỜ sửa quantbt** (`/root/bobby/pool_alpha/quantbt` READ-ONLY). Mọi thứ dùng public API `QuantBTEndpoint.walk_forward` / `WalkForwardEngine`.

---

## 0. Các nguyên tắc bất biến (mirror deep_momentum 4-phase guide)

1. **Feature tại D chỉ dùng dữ liệu ≤ close(D)**. Không bfill, không rank global, không fit_transform trên cả mẫu.
2. **Label open-to-open thực thi được**: `y = Open_{D+1+H}/Open_{D+1} − 1`, kèm cột `decision_time`, `label_start_time`, `label_end_time` (lấy từ index bar thật của từng symbol, không cộng ngày kiểu calendar).
3. **Purge mọi nơi theo `label_end_time < test_start`** — không bao giờ có label rò qua biên train/test.
4. **Tách hoàn toàn 3 vùng dữ liệu**: dev (≤ 2023-12-31) dùng cho feature-selection + ML tuning + strategy tuning (OOF); **outer OOS (2024-01-01 →) chỉ chạm ĐÚNG 1 lần** ở phase đánh giá cuối.
5. **PnL chỉ đến từ QuantBT** — `result.full_report()` / `show_metrics(trading_days=365)`. Mọi module khác không được tính Sharpe/CAGR.
6. **QuantBT native walk-forward là đường duy nhất chạy backtest**: `QuantBTEndpoint.walk_forward(strategy_class=..., target_mode="portfolio", optimization_mode="none", ...)`; `strategy` adapter chứa toàn bộ train-predict-per-fold. Bỏ hẳn fold loop tự viết (`split_data` chỉ còn phục vụ inner-fold ML tuning, không phục vụ backtest).
7. **Tham số tuned nằm ở `artifacts/*.json` (locked)**, `parameters.json` bất biến; thiếu locked artifact → fail-fast, không silent fallback.
8. **Serving = đúng code đã backtest** (cùng preprocessor + cùng overlay module + cùng model bundle).

---

## 1. Hiện trạng đã audit (các lỗi cần fix)

| # | Lỗi | Nguồn (file:line) | Mức |
|---|---|---|---|
| E1 | Label legacy `close[t+1]/close[t]−1` nhưng backtest trả `ret[t+1→t+2]` (lệch 1 bar sau shift(1) + engine execute tại close) | `util/factors.py:625`, `train.py:706`, engine close-execution | NGHIÊM TRỌNG |
| E2 | Optuna chọn params trên chính OOS window sau này được report (selection bias) | `optuna_kernel.py:80-90`, `stage1:61-67`, `stage2:91-100` | NGHIÊM TRỌNG |
| E3 | Fee: `fee=0.0005` được engine hiểu là round-trip → thực tế 0.00025/fill; tên `fee_rate_per_fill` engine không đọc | `quantbt_runner.py:62`, `quantbt/endpoint.py:313-319` | NGHIÊM TRỌNG |
| E4 | Funding cost = 0 (use_funding=False, không truyền funding vào engine) | `schema.py:107`, `quantbt_runner.py:77-89` | NGHIÊM TRỌNG |
| E5 | Short leg bị scale ngược với ý định: crash → cắt short xuống 0.4 (giết chân lời); comment ngược code | `train.py:658-675` | CAO |
| E6 | Overlay backtest ≠ serving: backtest dùng z-score sigmoid 120d, serving dùng threshold 22/30/65 + 14d MA | `train.py:653-678` vs `register_model.py:162-166` | CAO |
| E7 | Config: `parameters.json` thiếu key `model/label/validation/...` → v3 `fit_final` âm thầm dùng default lr 0.04/depth 4 thay vì 0.07/5; nhiều param code default ≠ config | `config/loader.py:26-38`, `schema.py:58-59`, `train.py:635,686,698,716` | CAO |
| E8 | Model register ≠ model backtest: backtest dùng tri-blend, MLflow register XGB fold cuối | `train.py:543,564-570` | CAO |
| E9 | Serving thiếu z-score/rank (raw features + fillna) trong khi train z-score | `register_model.py:123` vs `factors.py:640-655` | CAO |
| E10 | Exception bị nuốt thành −999 | `optuna_kernel.py:91-95`, `stage1:76-78`, `stage2:101-103` | TRUNG BÌNH |
| E11 | `subsample` được tune nhưng không dùng; `num_leaves` truyền vào XGBoost (dead); `stress_*_threshold` tune nhưng backtest không đọc | `train.py:346-352,409,529-537` | TRUNG BÌNH |
| E12 | Feature selection <2024 rò rỉ vào fold 2022-2023 của quarterly WF | `feature_analysis.py:43-44` | TRUNG BÌNH |
| E13 | `quantbt_runner.py:71-74` clip 0.99 dùng max trên toàn bộ matrix (statistic tương lai) | `quantbt_runner.py:71-74` | THẤP |
| E14 | Legacy Optuna ghi đè parameters.json | `multifactor_portfolio/optimization/optuna_kernel.py:230-231` | TRUNG BÌNH |
| E15 | Leverage 3.0 chỉ là margin gate, không ảnh hưởng PnL (config gây hiểu nhầm) | engine margin gate | THẤP (document) |
| E16 | Mode 5 full-sample = in-sample; CI/test chỉ chạy legacy helper | `train.py:235-240`, `ci.yml` | TRUNG BÌNH |

---

## 2. Kiến trúc đích

```
data (alphas_storage, sys.path) 
  → panel MultiIndex [Time, Symbol] + funding + macro
  → features (trailing) → preprocessor (fit/train-only) → labels (open-to-open + 3 cột timestamp)
  → [dev ≤2023] inner purged folds → ML tuning (rank IC) → artifacts/model_config.json
  → [dev ≤2023] OOF predictions (cached) → strategy tuning qua QuantBT WF → artifacts/strategy_config.json
  → [OOS 2024+] SINGLE-TOUCH: QuantBTEndpoint.walk_forward(strategy_class=MultiFactorStrategy, target_mode="portfolio")
      strategy.build_signal(data, params, train_index, test_index, fold):
          purge theo label_end_time < test_start
          train 1 model/fold (cache theo fold_id+params)
          predict → quantile bins → inverse-vol weights → overlay → positions DataFrame(index=test_index)
      → stitch OOS → _run_portfolio (fee_rate one-way, funding, slippage) → full_report(scope=OOS)
  → final_oos_metrics.json + RESEARCH_LOG + MLflow (git sha/branch)
  → export_bundle (model + preprocessor + overlay config) → serve dùng bundle
```

---

## 3. Các quyết định thiết kế (cần chốt trước khi code)

| Quyết định | Lựa chọn đề xuất | Lý do |
|---|---|---|
| Label contract | open-to-open, `H = holding_bars`; kèm `decision_time/label_start_time/label_end_time` từ bar thật | Khớp timing thực thi, deep_momentum chuẩn |
| Backtest engine | `QuantBTEndpoint.walk_forward(target_mode="portfolio", optimization_mode="none", split_mode="walk_forward_2024", split_frequency="quarterly", window_mode="expanding")` | Public API, stitch OOS, scope OOS-only |
| Purge trong WF | Ngay trong `build_signal`: train rows có `label_end_time >= test_start` bị loại | Engine chỉ cắt theo bar, không biết label horizon |
| Fee | Truyền `fee_rate=0.0005` (one-way per fill); bỏ tên `fee`/`fee_rate_per_fill` | Engine đọc đúng; config ngụ ý per-fill |
| Funding | `use_funding=True` + truyền funding series | Futures thật phải trả funding |
| Slippage | `slippage=0.0001` (legacy kwarg → 1.0 bps) giữ nguyên | Đã verify engine honor đúng |
| Overlay | 1 module duy nhất `src/multifactor_mlops/portfolio/overlay.py`, dùng chung backtest + serving; sửa ngược short leg; thresholds đọc từ config | Xoá train/serving skew |
| Hedge/leverage | `hedge_type="gross_exposure"`, gross ≈ 1.0; leverage chỉ còn là margin gate — **document rõ trong report** ("gross 1x, leverage không nhân return") | Trung thực với engine |
| ML tuning | Inner purged expanding folds trên dev; objective = rank IC (hoặc IS score), KHÔNG chạm OOS | Xoá E2, E12 |
| Strategy tuning | Trên OOF cached (dev), backtest qua QuantBT, robust = median − 0.5·std | deep_momentum chuẩn |
| Model chốt | XGBoost đơn (hoặc ensemble nhưng register ĐÚNG model được backtest) — chốt theo kết quả tuning | Xoá E8 |
| Config | parameters.json tái cấu trúc: `research/model/validation/strategy/execution/dataset/features/labels/training/optimization`; schema 1:1, fail-fast | Xoá E7 |
| Legacy | Không xoá `multifactor_portfolio/` (CI cũ + README), nhưng đánh dấu deprecated; entrypoint chuẩn mới nằm `src/multifactor_mlops` | Giữ CI xanh, tránh phá vỡ |
| Feature selection | Chạy trên dev ≤2023, lock `selected_features.json` trước khi vào OOS | Xoá E12 |
| Annualization | 365 ngày (giữ nguyên) | Đã thống nhất crypto |

---

## 4. Các phase (thứ tự, mỗi phase: làm gì → file → script scratch → verify)

### Phase 0 — Freeze baseline & kiểm kê
- [ ] Ghi lại trạng thái hiện tại: mode-4 số 1.3231/38.97%/12.16% (contaminated — chỉ để so sánh sau), `git rev-parse HEAD`, params hiện tại.
- [ ] Tạo `scratch/` + `scratch/README.md` (chỉ import codebase, không reimplement, không chạm OOS ngoài phase 8).
- [ ] (Tuỳ chọn, cần approval) Chạy lại mode-4 1 lần như hiện tại để chắc chắn reproduce baseline.
- **Verify**: số reproduce khớp performance_report_mode4.txt.
- **Commit**: `chore(v4): baseline freeze + scratch scaffolding`.

### Phase 1 — Label contract & purge (E1, E12)
- [ ] `src/multifactor_mlops/labels/returns.py`: chốt contract open-to-open + `decision_time/label_start_time/label_end_time` từ bar thật; bỏ nhánh close-to-close fallback (`panel.py:89`).
- [ ] `src/multifactor_mlops/features/panel.py`: purge helper `filter_train_by_label_end(train, test_start)`; đảm bảo `merge_asof` macro backward chỉ khớp ≤ cùng ngày (giữ).
- [ ] `scratch/exp_01_timing_contract.py`: trên synthetic data, in ra bảng 5 dòng: feature_t, label_start, label_end, execution fill → khẳng định không chồng lấn.
- **Verify**: `poetry run pytest tests/ -k "timing or phase1"` + exp_01.
- **Commit**: `fix(labels): open-to-open label contract with label_end_time purge`.

### Phase 2 — Config schema 1:1 & fail-fast (E7, E14)
- [ ] Tái cấu trúc `parameters.json` theo taxonomy trên; `config/schema.py` + `config/loader.py`: mọi key bắt buộc, unknown key → lỗi, không silent default; giữ test immutable hiện tại.
- [ ] `artifacts/research_config.json` (W_base/windows/H/lag, feature set), `artifacts/model_config.json`, `artifacts/strategy_config.json` — template trống để phase 6/7 điền.
- [ ] Bảng tham số: `fee_rate=0.0005` (one-way), `funding=true`, `leverage` → đổi tên ngữ nghĩa `margin_leverage` (document), bỏ `fee_rate_per_fill`, bỏ `num_leaves` khỏi XGB path, wire `subsample`.
- **Verify**: `tests/test_phase3.py::test_immutable_base_config` + new test `test_v4_config_strict.py` (thiếu key → raise).
- **Commit**: `feat(config): strict 1:1 config schema, fee_rate one-way convention`.

### Phase 3 — QuantBT walk-forward adapter (E1, E16, nền cho mọi phase sau)
- [ ] `src/multifactor_mlops/backtest/walkforward_strategy.py`:
  - `MultiFactorWalkForwardStrategy.build_signal(data, params, train_index, test_index, fold)`:
    - purge `label_end_time < test_start`; train 1 model/fold; **model cache theo (fold_id, params-key)** vì engine gọi 3 lần/fold (2 scoring + 1 final) — lần gọi `test_index == train_index` (IS scoring) cũng phải trả signal hợp lệ;
    - predict → quantile bins → inverse-vol → overlay (module Phase 5) → **positions DataFrame indexed chính xác bằng test_index, phủ 100% timestamp** (engine raise nếu thiếu).
- [ ] `src/multifactor_mlops/backtest/wf_runner.py`: wrapper `QuantBTEndpoint.walk_forward(..., target_mode="portfolio", optimization_mode="none", split_mode="walk_forward_2024", split_frequency="quarterly", window_mode="expanding", fee_rate=..., use_funding=..., slippage=...)`.
- [ ] `src/multifactor_mlops/pipelines/run_walkforward.py` (entrypoint script argparse) — script chuẩn thay `run_strategy_backtest`.
- [ ] `scratch/exp_02_wf_adapter_smoke.py`: synthetic 2 symbols, 3 folds; assert: fold.train_index.max() < fold.test_index.min(), stitched output phủ đúng OOS, ngoài OOS = 0, `result.metadata["walk_forward"]["n_folds"]` đúng.
- **Verify**: exp_02 + `tests/test_v4_walkforward.py` (coverage đủ test_index; không overlap fold; gọi nhiều lần/fold vẫn ra 1 model).
- **Commit**: `feat(backtest): QuantBT native walk-forward adapter`.

### Phase 4 — Costs đúng (E3, E4, E13)
- [ ] Bỏ clip 0.99 dựa trên max-toàn-matrix (`quantbt_runner.py:71-74`) → thay bằng: không clip (weights đã normalize ±0.5) hoặc clip theo từng row chỉ khi cần.
- [ ] Truyền `fee_rate=0.0005` (one-way), `use_funding=True`, `funding_rate=funding_df`; xác nhận qua đọc docs `portfolio_engine_v3.md` (fee_rate one-way priority).
- [ ] `scratch/exp_03_fee_funding_reconcile.py`: 1 symbol, 1 giao dịch mua 100k → khớp hand-calc: fee = notional × 0.0005, slippage 1bp, funding theo ngày; equity cuối khớp `result.equity`.
- **Verify**: exp_03 + `tests/test_v4_costs.py` (reconcile như test_phase1 hiện có nhưng theo contract mới).
- **Commit**: `fix(costs): fee_rate one-way, funding charged, no future-stat clipping`.

### Phase 5 — Overlay đúng & parity (E5, E6)
- [ ] `src/multifactor_mlops/portfolio/overlay.py`: 1 hàm duy nhất `apply_stress_overlay(weights, macro, params) -> weights` — sửa logic short leg (crash → GIỮ short, cut LONG; bull → cut short theo carry drag như ý định comment cũ); thresholds/multiplier đọc từ config; dùng chung cho backtest lẫn serving.
- [ ] Serving rewrite (Phase 9 liên kết): wrapper gọi đúng overlay.py này.
- **Verify**: `tests/test_v4_overlay_parity.py` — cùng input → backtest và serving ra cùng weights.
- **Commit**: `fix(overlay): single overlay module, correct short-leg scaling, backtest=serving`.

### Phase 6 — ML tuning đúng cách (E2, E10, E11)
- [ ] `src/multifactor_mlops/optimization/stage1_ml_tuning.py`: Optuna trên **inner purged expanding folds trong dev ≤2023**; objective = rank IC (không phải Sharpe từ backtest); IS-only; exception → raise (bỏ −999); grid nhỏ mặc định; lock `artifacts/model_config.json`.
- [ ] Wire `subsample` vào XGB params; bỏ `num_leaves` khỏi XGB; thống nhất `train_step_days` semantic.
- **Verify**: `tests/test_v4_tuning.py` — assert outer OOS (≥2024) vắng mặt trong mọi fold train/test của tuning.
- **Commit**: `fix(optimization): inner purged folds, IC objective, no OOS touch`.

### Phase 7 — OOF + Strategy tuning (E2)
- [ ] `src/multifactor_mlops/pipelines/generate_oof_predictions.py`: chạy model_config đã lock qua inner folds (dev), xuất `artifacts/oof_predictions.csv` + `fold_metrics.json` (chỉ ML metrics).
- [ ] `src/multifactor_mlops/optimization/stage2_strategy_tuning.py`: tune strategy params (quantiles, inverse_vol, overlay, rebalance) bằng cách backtest **weights dựng từ OOF** qua QuantBT WF trên dev; robust = median − 0.5·std; KHÔNG refit model; lock `artifacts/strategy_config.json`.
- **Verify**: `tests/test_v4_tuning.py::test_strategy_tuning_no_refit` + `test_oof_row_not_predicted_by_own_train_model`.
- **Commit**: `feat(optimization): OOF cached + strategy tuning on dev only`.

### Phase 8 — Outer OOS single-touch (E2, E16)
- [ ] `src/multifactor_mlops/pipelines/evaluate_final.py`: yêu cầu đủ 3 locked artifacts (thiếu → FileNotFoundError); chạy `wf_runner` với `split_mode="walk_forward_2024"`; metrics từ `result.full_report()` (scope auto = OOS-only); ghi `artifacts/final_oos_metrics.json` + `final_oos_report.md`.
- [ ] MLflow: log run với git sha/branch/dirty + full params + artifact links.
- **Verify**: `tests/test_v4_final.py` (equity bắt đầu ≥ OOS start; metrics == quantbt `show_metrics`; outer OOS chỉ chạy 1 lần — check qua fold_table).
- **Commit**: `feat(evaluation): single-touch outer OOS report`.

### Phase 9 — Bundle & serving parity (E8, E9)
- [ ] `src/multifactor_mlops/register/export_bundle.py`: fit_final trên `label_end_time <= training_cutoff` với preprocessor (clip 1/99 + median, fit train-only) + model + overlay config + metadata (feature_names, params, data lineage); `bundle.to_dict/from_dict`.
- [ ] `src/multifactor_mlops/scoring/serve.py` (hoặc register wrapper mới): predict = bundle chỉ; bỏ raw-feature path; model được register = model đúng pipeline (không phải last-fold khác model).
- **Verify**: `tests/test_v4_serving_parity.py` — bundle load → predict khớp training-time transform.
- **Commit**: `feat(serving): exported bundle parity (model+preprocessor+overlay)`.

### Phase 10 — CI & verification cuối
- [ ] `.github/workflows/ci.yml`: giữ test legacy + thêm `poetry run pytest tests/` (hoặc thay — đề xuất thêm, không thay).
- [ ] Chạy full suite + pre-commit; chạy outer OOS ĐÚNG 1 lần; ghi `RESEARCH_LOG.md` (method, locked config, kết quả, re-run command).
- [ ] So sánh với baseline contaminated (Phase 0) — kỳ vọng số thấp hơn; con số mới là baseline hợp lệ duy nhất.
- **Commit**: `chore(ci): full suite in CI + research log`.

---

## 5. Test plan (mới, bắt chước deep_momentum)

| File | Assert |
|---|---|
| `tests/test_v4_config_strict.py` | thiếu key/unknown key → raise; parameters.json bất biến |
| `tests/test_v4_labels.py` | label_start_time = bar kế tiếp thật; label_end_time = bar D+1+H; purge `label_end_time < test_start` |
| `tests/test_v4_walkforward.py` | fold train < test (không overlap); output phủ 100% test_index (engine raise nếu thiếu); 3 lần gọi/fold → 1 model duy nhất; ngoài OOS = 0 |
| `tests/test_v4_costs.py` | 1-trade reconcile: fee one-way 0.0005, slippage 1bp, funding; equity khớp engine |
| `tests/test_v4_overlay_parity.py` | overlay backtest == serving; crash giữ short, bull cắt short |
| `tests/test_v4_tuning.py` | outer OOS vắng mặt trong tuning; strategy tuning không refit model; OOF row không tự dự đoán |
| `tests/test_v4_final.py` | equity ≥ OOS start; metrics == QuantBT show_metrics; locked artifacts bắt buộc |
| `tests/test_v4_serving_parity.py` | bundle load → predict khớp training transform |

Giữ lại các test cũ có giá trị: `test_phase1` (timing contract, reconcile), `test_phase2` (future-mutation), `test_phase3` (immutable, bundle), `test_remaining_fixes`.

---

## 6. Caveats QuantBT walk-forward (đã xác minh từ source — không sửa engine)

- Engine gọi strategy **3 lần/fold** (2 lần scoring: IS với `test_index=train_index`, OOS; 1 lần final) → **bắt buộc model cache theo (fold_id, params)**; lần gọi IS-scoring phải trả signal hợp lệ trên train_index.
- Output phải là DataFrame positions **indexed bằng chính `test_index`**, phủ 100% timestamp (thiếu → raise `output index must cover every expected fold timestamp`).
- Engine **không purge**: train = mọi bar `< test_start` → purge `label_end_time` phải nằm trong `build_signal`.
- Folds tile liên tục, không overlap, không gap; expanding mode: OOS fold trước trở thành train fold sau (chấp nhận, purge vẫn đảm bảo).
- `optimization_mode="none"` + `params=` cho ML (các mode Optuna của engine re-train per trial — quá đắt cho XGBoost per-fold). Tuning ML riêng ở Phase 6 (IS-only).
- Metrics mặc định scope OOS-only (`full_report()` không tính các bar train phẳng) — chính xác cho báo cáo.
- Fee: dùng `fee_rate` (one-way) — `fee` là round-trip, engine chia 2. `funding_rate`/`use_funding` truyền qua kwargs.

---

## 7. Checklist "KHÔNG ĐƯỢC LÀM"

- ❌ Không sửa bất kỳ file nào trong `/root/bobby/pool_alpha/quantbt`.
- ❌ Không chạy Optuna/large training nếu chưa được user duyệt (kèm dự kiến thời gian).
- ❌ Không ghi đè `parameters.json` (locked artifacts là nơi duy nhất ghi params tuned).
- ❌ Không dùng `full`/full-sample split làm research score (chỉ fit_final production).
- ❌ Không nuốt exception thành −999; không bfill; không trade tại close(D) với feature(D).
- ❌ Không báo cáo số mới trước khi Phase 8 chạy xong với đủ locked artifacts.

---

## 8. Lệnh chạy sau khi hoàn tất (mẫu)

```bash
poetry run python src/multifactor_mlops/optimization/stage1_ml_tuning.py --trials 30          # → artifacts/model_config.json
poetry run python src/multifactor_mlops/pipelines/generate_oof_predictions.py                # → artifacts/oof_predictions.csv
poetry run python src/multifactor_mlops/optimization/stage2_strategy_tuning.py --trials 15   # → artifacts/strategy_config.json
poetry run python src/multifactor_mlops/pipelines/evaluate_final.py                          # → final_oos_metrics.json (SINGLE TOUCH)
poetry run python src/multifactor_mlops/register/export_bundle.py --training-cutoff 2023-12-31
poetry run pytest tests/ tests/test_v4_*.py  -q
poetry run pre-commit run --all-files
```

Mỗi phase: làm → chạy verify → commit lên `feat/multifactor-macro-features-v3` → hỏi user trước phase tiếp theo.
