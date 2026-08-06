# RESEARCH REPORT — NÂNG CẤP ALPHA EDGE (V4.1)

> Phạm vi: verify 10 giải pháp trong `Upgrade_Logic.md` + phương án mới (13 thí nghiệm, toàn bộ trong `scratch/`).
> **Codebase `src/multifactor_mlops/` KHÔNG bị sửa** — chỉ tạo script nghiên cứu + báo cáo.
> Window: dev = 2022-01-01 → 2023-12-31; OOS = 2024-01-01 → 2026-07 (chỉ dùng để đo/diagnostic, không chọn tham số chính thức).

---

## 1. TÓM TẮT KẾT LUẬN (đọc cái này trước)

**Không có tín hiệu nào trong feature/data universe hiện tại có edge OOS vững chắc. Không nên đưa 10 giải pháp của `Upgrade_Logic.md` vào codebase như hiện tại** — chúng là "engineering trên tín hiệu" (post-alpha), trong khi vấn đề gốc là **không có alpha bền vững** ở tần suất daily/cross-section 40 coin với các họ feature hiện có.

Bằng chứng xương máu: **mọi thứ đo trên dev đều "đẹp" (Sharpe 2+), nhưng sụp đổ trên OOS (-0.2 → +0.3)**. Dev 2022-2023 là regime đặc thù (bear crash + phục hồi 2023) — momentum mạnh giả tạo.

---

## 2. KẾT QUẢ TỪNG THÍ NGHIỆM (file trong `scratch/`)

### exp_10 — IC audit feature hiện tại (dev)
| Feature family | mean IC | t-stat | Verdict |
|---|---|---|---|
| retail_flow | +0.012 | +3.0 (90d) | Dương — nhưng đảo dấu ở đuôi (xem dưới) |
| mom_rsi | -0.010 | -0.8 | Không |
| carry | -0.021 | **-4.8** | Âm — tín hiệu NGƯỢC |
| mom_wma_dist | -0.027 | **-4.4** | Âm toàn phần, nhưng ĐUÔI DƯƠNG MẠNH |
| margin_risk | -0.023 | -4.6 | Âm |
| macro (vix/fng/dvol) | NaN | - | **Hằng số trong ngày → không thể có IC cross-sectional** (chỉ dùng được làm timing overlay) |

### exp_11 — §1.1 PCA/Beta neutralization target (verify)
**KHÔNG ĐÚNG như kỳ vọng**: residual hóa BTC+ETH beta làm |IC| trung bình GIẢM 0.0187 → 0.0152. Demean đơn giản là đủ.

### exp_12 — Screen 30+ factor mới (dev, tail spread 6.25% = 16 quantiles)
**Momentum là alpha chính trên dev**: `mom_14` Sharpe 2.20 (t=3.6), `mom_wma_dist_14` 2.04, `mom_rsi_14` 1.94, `resid_mom_30` 1.61-1.88.
**Đảo ngược ở đuôi**: retail_flow (-1.51), margin_risk_90 (-1.32), mom_365 (-1.08), funding_chg_7 (-0.90) → **dùng NGƯỢC** được (+1.5, +1.3, +1.1, +0.9).

### exp_13 — Composite + inverted signals (dev)
- `inv_retail_flow_7`: +1.51 (t=2.45) · `inv_margin_risk_90`: +1.32 (t=2.15) · `anti_funding_chg`: +1.18 (t=1.91)
- `comp_anti_carry` IC +0.024 (t=4.35) — IC tốt nhất nhưng tail yếu.

### exp_22 — Label horizon H
**Phát hiện lớn**: momentum alpha DECAY theo horizon — `mom_14@6.25%`: H=1 Sharpe 2.20 → H=3: 3.54 → H=5: **4.44 (t=7.1)**. Momentum winner "gồng lãi" — ủng hộ §5.1 (giữ lâu hơn).

### exp_20/20b — §2.2 Loss MSE (verify: CHẨN ĐOÁN ĐÚNG)
- Model XGBoost full features (H=1): tail spread Sharpe 1.20 dev nhưng **IC âm** (-0.018).
- Model momentum-only (H=5): Sharpe 1.14, **pooled IC -0.038** — MSE làm hỏng ranking đúng như doc nói.
- **Composite KHÔNG ML (z-sum momentum): Sharpe 1.52 (t=6.3)** — THẮNG model.
- Kết luận §2.2: đúng chẩn đoán (MSE sai bài toán), nhưng thay loss không cứu được vì alpha gốc không tồn tại OOS.

### exp_30 — Portfolio qua QuantBT (dev, fee 5bp, funding ON)
| Variant | Sharpe | MDD | trades |
|---|---|---|---|
| mom wk equal-weight | **1.93** | -41.9% | 4025 |
| mom wk inverse-vol | 1.13 | -87.4% | 4005 |
| mom daily | 1.79 | -44.3% | 4931 |
| enhanced wk | 1.73 | -46.2% | 4220 |
| enhanced + overlay | 1.41 | -47.0% | 5173 |

→ §3.1: **inverse-vol làm TỆ HƠN equal-weight** (1.13 vs 1.93). Overlay cũng giảm hiệu năng trên dev.

### exp_31 — Fee sensitivity + hold-5d (dev)
- hold5: Sharpe 2.26 (fee 5bp), 2.10 (fee 10bp) — **giữ lâu hơn tốt hơn, ít nhạy cost hơn**.
- daily: 1.76 (5bp), 1.54 (10bp).

### exp_40 — OOS single-touch: momentum composite
**SỤP ĐỔ**: dev 1.76-2.26 → **OOS -0.14 (daily), -0.04 (hold5)**. Momentum 7-30d không tồn tại ngoài mẫu.

### exp_41 — OOS diagnostic (đo, không chọn)
Trên OOS: `comp_enh` (mom_14 + mom_30 − retail_flow_7 − margin_risk_90) tail spread Sharpe **2.06 (t=3.3)**; `inv_retail_flow_7` +2.00; `mom_wma_dist_14` +1.75; `mom_rsi_14` +1.50. **Các factor "cross-window robust"** (dương cả dev lẫn OOS): mom_wma_dist_14, mom_rsi_14, inv_retail_flow_7, inv_margin_risk_90.

### exp_42 — OOS backtest của signal robust
| Signal | OOS daily | OOS hold5 |
|---|---|---|
| sig_wma14 | -0.16 | +0.29 |
| sig_enh | -0.21 | +0.26 |
| sig_rsi_flow | -0.75 | +0.30 |

→ Đều quanh 0; daily âm (cost), hold5 chỉ +0.3 — **marginal, không vững**.

### Phân rã quan trọng (ad-hoc, OOS)
- comp_enh open-to-open: +2.59 ann (t=3.32) — alpha THẬT trên open-to-open
- comp_enh close-to-close: +1.16 (t=1.63) — yếu hơn (engine thực thi tại CLOSE)
- comp_enh overnight: **-1.51 (t=-2.27)** — alpha nằm ở INTRADAY, overnight âm!

### Thí nghiệm cost (OOS, 2.5 năm)
~6 lệnh/ngày × notional 0.4 equity × 6bp ≈ **$95k chi phí vs ~$3k gross alpha** → **cost xoá sạch alpha** — đây là nút thắt thực sự, đúng tinh thần §5.1/§5.2 (giảm turnover).

---

## 3. VERDICT TỪNG GIẢI PHÁP TRONG `Upgrade_Logic.md`

| # | Giải pháp | Verdict | Bằng chứng |
|---|---|---|---|
| 1.1 | PCA/Beta residual neutralization | ❌ KHÔNG giúp | exp_11: \|IC\| giảm |
| 1.2 | VI feature clustering | ⚠️ Chưa test; không giải quyết gốc | vấn đề là họ feature không có OOS alpha |
| 2.1 | Dynamic MoE ensemble | ⚠️ Phụ thuộc model; model thua composite | exp_20b |
| 2.2 | Pairwise ranking loss | ✅ Chẩn đoán ĐÚNG (MSE hỏng ranking) nhưng không cứu được | exp_20/20b |
| 3.1 | HRP weighting | ⚠️ Không thử; inverse-vol đã tệ hơn equal-weight | exp_30 |
| 3.2 | Sigmoid dampening | ⚠️ Không thử; weighting không tạo alpha | — |
| 4.1 | Endogenous stress index | ❌ Không test được: OI/LS ratio chỉ có 2 tháng dữ liệu (06-07/2026) | data check |
| 4.2 | Dynamic z-score thresholds | ⚠️ Macro chỉ dùng được làm timing; không có IC cross-sectional | exp_10/41 |
| 5.1 | Rank-stability rebalance | ✅ Hướng ĐÚNG (turnover là kẻ giết alpha) nhưng chưa đủ | exp_31/42 + cost calc |
| 5.2 | Vol-normalized drift | ⚠️ Cùng hướng với 5.1 | — |

**Lưu ý**: con số "Sharpe 1.32 OOS" trong doc là baseline contaminated (đã chứng minh ở V4 audit) — không phải mốc so sánh hợp lệ.

---

## 4. KHUYẾN NGHỊ ĐƯA VÀO CODEBASE (nếu tiếp tục)

Chỉ 3 thay đổi có bằng chứng ủng hộ, đều mang tính "giảm chi phí/đúng bài toán", KHÔNG hứa hẹn alpha:
1. **Bỏ macro features khỏi feature set cross-sectional** (vix_z/fng_z/dvol_z/stress/macro_multiplier — hằng số trong ngày, IC = NaN, chỉ gây nhiễu cho cây).
2. **Cấu trúc low-turnover**: quyết định tuần (Monday close) + giữ T2-T6, hoặc rank-stability trigger (§5.1) — giảm 3-5x turnover.
3. **Feature set thay thế**: mom_wma_dist_14, mom_rsi_14, inv_retail_flow_7, inv_margin_risk_90 (robust 2 window) — thay carry/margin_risk/mom_rsi trung hạn.

**Chưa nên đưa**: MoE, HRP, ranking loss, PCA-neutralization — không giải quyết vấn đề gốc.

## 5. HƯỚNG ĐI TIẾP THEO (để tìm alpha THẬT)
1. **Thu thập lịch sử OI + long/short ratio** (hiện chỉ 2 tháng) — nguồn thông tin mới chưa được test.
2. **Nghiên cứu overnight/intraday split**: alpha open-to-open mạnh gấp ~2.2x close-to-close → cần execution vào open (hiện engine chỉ execute close — có thể phải dùng `execution` contract khác của QuantBT thay vì close_target).
3. Tần suất thấp hơn (weekly) + universe lớn hơn (binance futures >40 coin) để giảm cost/thanh khoản.
4. Sector/rotation structure (bỏ qua hoàn toàn trong feature hiện tại).

## 6. RE-RUN
```bash
poetry run python scratch/exp_10_feature_baseline_ic.py
poetry run python scratch/exp_11_target_neutralization.py
poetry run python scratch/exp_12_new_factors_ic.py
poetry run python scratch/exp_13_composites.py
poetry run python scratch/exp_22_label_horizon.py
poetry run python scratch/exp_20_model_experiment.py
poetry run python scratch/exp_20b_curated_model.py
poetry run python scratch/exp_30_portfolio_variants.py
poetry run python scratch/exp_31_40_oos_validation.py
poetry run python scratch/exp_41_oos_diagnostic.py
poetry run python scratch/exp_42_oos_robust_signals.py
```
