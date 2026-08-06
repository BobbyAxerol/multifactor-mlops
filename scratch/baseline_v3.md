# Baseline freeze (Phase 0) — ghi lại trạng thái TRƯỚC khi sửa

- Branch: `feat/multifactor-macro-features-v3`
- HEAD (contaminated baseline): `4520e9a`
- Các con số báo cáo CŨ (contaminated, KHÔNG trust — xem MULTIFACTOR_MLOPS_AUDIT_AND_REPAIR_PLAN.md):
  - Mode 4 (WFO quarterly 2022-2026, walk_forward_quarterly, fee=0.0005 bị engine chia đôi → 0.00025/fill, funding=0): Sharpe 1.3231, CAGR 38.97%, MaxDD 12.16%
  - Mode 5 (full-sample in-sample): Sharpe 1.5331
- Lỗi đã xác nhận trong baseline: label close-to-close lệch 1 bar so với thực thi; Optuna chọn params trên chính OOS; fee thực tế 0.00025/fill; funding = 0; short-leg overlay ngược; train/serve skew.

Sau khi hoàn tất V4, con số mới (evaluate_final) là baseline hợp lệ duy nhất.
