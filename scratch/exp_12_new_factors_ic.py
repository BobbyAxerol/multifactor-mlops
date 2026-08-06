"""
exp_12: NEW factor screen on dev window — rank IC AND tail long-short spreads.

Candidate families (trailing data only, zero bfill):
  - raw momentum horizons 1/3/7/14/30/90/180/365d
  - risk-adjusted momentum, overnight vs intraday returns
  - range/close-position, volume z-score, skew/kurt, max drawdown
  - 1-day reversal, BTC-beta + residual momentum
  - funding level / funding change (from funding_wide)
  - current features included for comparison

Evaluation: tail spread (top vs bottom) at 25%/12.5%/6.25% — this is what
the quantile strategy actually trades (16 quantiles -> 6.25% tails).
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

print("computing new factors...")
panel = compute_new_factors(panel, data_dict, funding_wide)

# funding-based factors (level + change), long panel
if funding_wide is not None and not funding_wide.empty:
    funding_long = funding_wide.stack().rename("funding_level").to_frame()
    funding_long.index.names = ["Time", "Symbol"]
    funding_chg = funding_wide.diff(7).stack().rename("funding_chg_7").to_frame()
    funding_chg.index.names = ["Time", "Symbol"]
    panel = panel.join(funding_long, how="left").join(funding_chg, how="left")

feature_cols = [c for c in panel.columns if c not in {"target", "Symbol", "decision_time",
                                                      "label_start_time", "label_end_time"}]

print(f"\n=== RANK IC (dev 2022-2023) — all {len(feature_cols)} features ===")
ic = daily_rank_ic(panel, feature_cols)
print(ic.round(4).to_string())

print("\n=== TAIL SPREAD (top minus bottom, RAW fwd ret, before costs) — top 25 ===")
spread = tail_spread_table(panel, feature_cols, raw_fwd)
print(spread.head(25).round(4).to_string())

print("\n=== TAIL SPREAD — bottom 15 ===")
print(spread.tail(15).round(4).to_string())
