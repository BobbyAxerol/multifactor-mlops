# scratch/ — Các script kiểm chứng cô lập (theo chuẩn deep_momentum)

Quy tắc:
- Chỉ IMPORT codebase (`src.multifactor_mlops`), KHÔNG reimplement logic.
- KHÔNG chạm outer OOS (>= 2024) ngoài `exp_06_outer_oos`.
- Chạy: `poetry run python scratch/<script>.py` (từ repo root).

| Script | Xác minh |
|---|---|
| `exp_01_timing_contract.py` | Label open-to-open + purge label_end_time |
| `exp_02_wf_adapter_smoke.py` | QuantBT walk-forward adapter: folds, stitch, coverage |
| `exp_03_fee_funding_reconcile.py` | Fee one-way + funding reconcile hand-calc |
| `exp_04_overlay_parity.py` | Overlay backtest == serving |
