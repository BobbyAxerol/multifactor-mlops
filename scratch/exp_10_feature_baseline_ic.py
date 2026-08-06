"""
exp_10: Baseline IC audit of CURRENT feature families on the dev window.
Verifies: which of the existing 25+5 features actually carry cross-sectional
predictive power (rank IC) — and whether the ~0 IC finding reproduces.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

from research_base import load_dev_data, build_panel, daily_rank_ic, raw_forward_returns, spread_return

cfg, data_dict, macro_df, funding_wide = load_dev_data()
panel = build_panel(data_dict, macro_df, funding_wide, H=1, ranked=False)
raw_fwd = raw_forward_returns(data_dict, H=1)

feature_cols = PanelDatasetBuilder.feature_columns(panel) if False else [
    c for c in panel.columns if c not in {"target", "Symbol", "decision_time",
                                          "label_start_time", "label_end_time"}
]
print(f"panel rows: {len(panel)}, features: {len(feature_cols)}, symbols: {len(panel.index.get_level_values('Symbol').unique())}")
print(f"window: {panel.index.get_level_values('Time').min().date()} -> {panel.index.get_level_values('Time').max().date()}")

ic = daily_rank_ic(panel, feature_cols)
print("\n=== DAILY RANK IC (dev 2022-2023) — per feature ===")
print(ic.round(4).to_string())

print("\n=== BEST 10 by IC ===")
print(ic.head(10).round(4).to_string())

print("\n=== WORST 10 ===")
print(ic.tail(10).round(4).to_string())

# family aggregates
fam = {}
for col in feature_cols:
    fam_key = col.rsplit("_", 1)[0]
    fam.setdefault(fam_key, []).append(ic.loc[col, "ic_mean"] if col in ic.index else np.nan)
fam_df = pd.DataFrame({k: {"mean_ic": np.nanmean(v), "n": len(v), "max_ic": np.nanmax(v)} for k, v in fam.items()}).T
print("\n=== FEATURE FAMILY AGGREGATE IC ===")
print(fam_df.round(4).to_string())

# spread of best single feature vs target (using raw fwd returns)
best_feat = ic.index[0]
_, spread_sum = spread_return(panel, best_feat, raw_fwd, top_pct=0.25)
print(f"\n=== L/S SPREAD of best raw feature '{best_feat}' (25% top/bottom, before costs) ===")
print(pd.Series(spread_sum).round(4).to_string())
