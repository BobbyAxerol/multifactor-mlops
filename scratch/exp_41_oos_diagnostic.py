"""
exp_41: OOS DIAGNOSTIC (2024-01-01 -> 2026-07) — which factors (if any) carry
cross-sectional predictive power AFTER the dev window?

Diagnostic only (no parameter selection): measures rank IC + tail spreads on the
OOS window for the full candidate set, mirroring exp_12/13.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from research_base import (load_dev_data, build_panel, raw_forward_returns,
                           daily_rank_ic, tail_spread_table, compute_new_factors)

OOS_START = "2024-01-01"
cfg, data_dict, macro_df, funding_wide = load_dev_data(end_date="2026-12-31")

panel = build_panel(data_dict, macro_df, funding_wide, H=1, ranked=False)
panel = compute_new_factors(panel, data_dict, funding_wide)
if funding_wide is not None:
    panel = panel.join(pd.DataFrame({"funding_chg_7": funding_wide.diff(7).stack()}).rename_axis(["Time", "Symbol"]), how="left")

# slice to OOS window
panel = panel[panel.index.get_level_values("Time") >= OOS_START]
raw_fwd = raw_forward_returns(data_dict, H=1)
raw_fwd = raw_fwd[raw_fwd.index.get_level_values("Time") >= OOS_START]

for col in ["retail_flow_7", "margin_risk_90", "funding_chg_7", "carry_30", "mom_365"]:
    if col in panel.columns:
        panel[f"inv_{col}"] = -panel[col]

z = panel[["mom_7", "mom_14", "mom_30", "mom_wma_dist_14", "mom_rsi_14", "resid_mom_30"]] \
    .groupby(level="Time").transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
panel["comp_mom"] = z.sum(axis=1)
panel["comp_enh"] = (panel[["mom_14", "mom_30", "inv_retail_flow_7", "inv_margin_risk_90"]]
                     .groupby(level="Time").transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))).sum(axis=1)
panel["comp_reversal"] = (panel[["range_pct", "rev_1", "overnight"]]
                          .groupby(level="Time").transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))).sum(axis=1)

feature_cols = [c for c in panel.columns if c not in {"target", "Symbol", "decision_time",
                                                      "label_start_time", "label_end_time"}]

print(f"=== OOS WINDOW {OOS_START} -> {panel.index.get_level_values('Time').max().date()} ===")
print(f"rows: {len(panel)}")

print("\n=== RANK IC (OOS) ===")
ic = daily_rank_ic(panel, feature_cols)
print(ic.round(4).head(15).to_string())
print("...")
print(ic.round(4).tail(8).to_string())

print("\n=== TAIL SPREAD (OOS) — top 20 ===")
spread = tail_spread_table(panel, feature_cols, raw_fwd, tails=(0.0625, 0.25))
print(spread.head(20).round(3).to_string())
