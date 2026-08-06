"""
exp_22: Label horizon sensitivity (H = 1, 2, 3, 5, 10) for the top signal families.

Question: does the momentum/flow signal persist over longer holding periods?
(Upgrade_Logic pillar 5 motivation: weekly rebalance may waste alpha.)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from research_base import (load_dev_data, build_panel, raw_forward_returns,
                           tail_spread_table, daily_rank_ic, compute_new_factors)

cfg, data_dict, macro_df, funding_wide = load_dev_data()

SIGNALS = ["mom_14", "mom_wma_dist_14", "mom_rsi_14", "mom_30", "resid_mom_30"]

for H in [1, 2, 3, 5, 10]:
    panel = build_panel(data_dict, macro_df, funding_wide, H=H, ranked=False)
    raw_fwd = raw_forward_returns(data_dict, H=H)
    panel = compute_new_factors(panel, data_dict, funding_wide)
    for col in ["retail_flow_7", "margin_risk_90", "funding_chg_7", "carry_30", "mom_365"]:
        if col in panel.columns:
            panel[f"inv_{col}"] = -panel[col]
    cols = SIGNALS + [f"inv_{c}" for c in ["retail_flow_7", "margin_risk_90", "funding_chg_7", "carry_30", "mom_365"]]
    cols = [c for c in cols if c in panel.columns]
    spread = tail_spread_table(panel, cols, raw_fwd, tails=(0.0625, 0.25))
    print(f"\n===== H = {H} (fwd ret Open_{{t+1+H}}/Open_{{t+1}} - 1) =====")
    print(spread.round(3).to_string())
