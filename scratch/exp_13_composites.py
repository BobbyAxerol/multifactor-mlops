"""
exp_13: Factor correlation, composite momentum signal, and INVERTED candidates.

Aims:
  - correlation matrix of the momentum family (avoid duplicating info)
  - composite z-score signals (momentum core, inverted-flow, anti-carry)
  - verify inverted candidates (retail_flow, margin_risk_90, mom_365, funding_chg_7)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from research_base import (load_dev_data, build_panel, raw_forward_returns,
                           daily_rank_ic, tail_spread_table, compute_new_factors)

cfg, data_dict, macro_df, funding_wide = load_dev_data()
panel = build_panel(data_dict, macro_df, funding_wide, H=1, ranked=False)
raw_fwd = raw_forward_returns(data_dict, H=1)
panel = compute_new_factors(panel, data_dict, funding_wide)
if funding_wide is not None:
    panel = panel.join(pd.DataFrame({"funding_level": funding_wide.stack()}).rename_axis(["Time", "Symbol"]), how="left")
    panel = panel.join(pd.DataFrame({"funding_chg_7": funding_wide.diff(7).stack()}).rename_axis(["Time", "Symbol"]), how="left")

MOM_CORE = ["mom_7", "mom_14", "mom_30", "mom_wma_dist_14", "mom_wma_dist_30",
            "mom_rsi_14", "mom_rsi_30", "resid_mom_30", "mom_risk_adj_30"]

print("=== CORRELATION of momentum core (daily cross-sectional rank corr, mean) ===")
corrs = {}
for i in range(len(MOM_CORE)):
    for j in range(i + 1, len(MOM_CORE)):
        a, b = MOM_CORE[i], MOM_CORE[j]
        vals = []
        for t, g in panel[[a, b]].dropna().groupby(level="Time"):
            if len(g) >= 16:
                c = g[a].rank().corr(g[b].rank())
                if np.isfinite(c):
                    vals.append(c)
        corrs[f"{a} | {b}"] = float(np.mean(vals))
corr_df = pd.Series(corrs).sort_values(ascending=False)
print(corr_df.round(3).head(15).to_string())

# ---- composite signals (per-timestamp z-score sum) ----
def zscore_composite(panel, cols, weights=None):
    cols = [c for c in cols if c in panel.columns]
    z = panel[cols].groupby(level="Time").transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
    if weights is not None:
        z = z * pd.Series(weights, index=z.columns)
    return z.sum(axis=1).rename("composite")

panel["comp_mom"] = zscore_composite(panel, ["mom_7", "mom_14", "mom_30", "mom_wma_dist_14", "mom_rsi_14"])
panel["comp_anti_flow"] = zscore_composite(panel, ["retail_flow_90", "retail_flow_30"], weights={"retail_flow_90": -1.0, "retail_flow_30": -1.0})
panel["comp_anti_carry"] = zscore_composite(panel, ["carry_30", "carry_60"], weights={"carry_30": -1.0, "carry_60": -1.0})
panel["comp_anti_margin"] = zscore_composite(panel, ["margin_risk_90"], weights={"margin_risk_90": -1.0})
panel["comp_anti_funding_chg"] = zscore_composite(panel, ["funding_chg_7"], weights={"funding_chg_7": -1.0})
panel["comp_anti_mom365"] = zscore_composite(panel, ["mom_365"], weights={"mom_365": -1.0})

# inverted single factors (spread sign flip)
for col in ["retail_flow_7", "margin_risk_90", "mom_365", "funding_chg_7"]:
    if col in panel.columns:
        panel[f"inv_{col}"] = -panel[col]

COMPOSITES = ["comp_mom", "comp_anti_flow", "comp_anti_carry", "comp_anti_margin",
              "comp_anti_funding_chg", "comp_anti_mom365", "inv_retail_flow_7",
              "inv_margin_risk_90", "inv_mom_365", "inv_funding_chg_7"]

print("\n=== TAIL SPREAD — composites + inverted (dev 2022-2023, raw fwd, before costs) ===")
spread = tail_spread_table(panel, COMPOSITES, raw_fwd)
print(spread.round(4).to_string())

print("\n=== RANK IC — composites ===")
ic = daily_rank_ic(panel, COMPOSITES)
print(ic.round(4).to_string())
