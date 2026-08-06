"""
exp_11: Target neutralization — verify whether residualizing the target on
market factors (BTC/ETH betas, cross-sectional market return) improves the
tail spread vs the current simple demeaning. (Upgrade_Logic pillar 1, solution 1.1)

Also checks: does the market factor explain the tail inversion of current features?
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from research_base import load_dev_data, build_panel, raw_forward_returns, tail_spread_table, daily_rank_ic

cfg, data_dict, macro_df, funding_wide = load_dev_data()
panel = build_panel(data_dict, macro_df, funding_wide, H=1, ranked=False)
raw_fwd = raw_forward_returns(data_dict, H=1)
rf = raw_fwd["raw_fwd_ret"]

# --- market factors: forward open-to-open returns of BTC / ETH / cross-sectional mean ---
def fwd_open_ret(sym):
    o = data_dict[sym]["open"]
    return (o.shift(-2) / o.shift(-1) - 1.0).rename(sym)

btc_fwd = fwd_open_ret("BTCUSDT")
eth_fwd = fwd_open_ret("ETHUSDT") if "ETHUSDT" in data_dict else btc_fwd

# per-symbol trailing beta vs BTC and ETH (30d, on daily open-to-open returns)
closes = {s: data_dict[s]["close"] for s in data_dict}
btc_ret = closes["BTCUSDT"].pct_change()
eth_ret = closes["ETHUSDT"].pct_change() if "ETHUSDT" in closes else btc_ret

beta_btc, beta_eth = {}, {}
for sym, c in closes.items():
    r = c.pct_change()
    beta_btc[sym] = r.rolling(30, min_periods=15).cov(btc_ret) / btc_ret.rolling(30, min_periods=15).var()
    beta_eth[sym] = r.rolling(30, min_periods=15).cov(eth_ret) / eth_ret.rolling(30, min_periods=15).var()

# target variants (long panel, aligned with raw_fwd index)
sym_map = raw_fwd.index.get_level_values("Symbol").unique()
beta_btc_al = pd.DataFrame(beta_btc).stack().reindex(rf.index).fillna(0.0)
beta_eth_al = pd.DataFrame(beta_eth).stack().reindex(rf.index).fillna(0.0)
btc_arr = btc_fwd.reindex(rf.index.get_level_values("Time")).fillna(0.0).values
eth_arr = eth_fwd.reindex(rf.index.get_level_values("Time")).fillna(0.0).values

resid = rf - beta_btc_al.values * btc_arr - beta_eth_al.values * eth_arr
resid = pd.Series(resid.values, index=rf.index, name="target")

# cross-sectional-mean-neutralized raw target (exactly what the panel does)
demean = rf.copy()
ts_mean = rf.groupby(level="Time").transform("mean")
demean = rf - ts_mean

resid_df = panel.drop(columns=["target"]).join(resid.rename("target"), how="inner")
demean_df = panel.drop(columns=["target"]).join(demean.rename("target"), how="inner")
raw_df = panel.drop(columns=["target"]).join(rf.rename("target"), how="inner")

feature_cols = [c for c in panel.columns if c not in {"target", "Symbol", "decision_time",
                                                      "label_start_time", "label_end_time"}]

print("=== RANK IC of current features vs TARGET VARIANT (demean / beta-resid / raw) ===")
ic_dem = daily_rank_ic(panel, feature_cols)
for name, df, panel_v in [("demean", demean_df, panel),
                          ("beta_resid", resid_df, panel),
                          ("raw", raw_df, panel)]:
    ic = daily_rank_ic(df, feature_cols)
    top = ic.sort_values("ic_mean", ascending=False).head(5)
    print(f"\n[{name}] top-5 by IC:")
    print(top.round(4).to_string())

print("\n=== TAIL SPREAD of strongest raw feature (retail_flow_90) per target variant ===")
print("(same signal, different target — spread uses RAW fwd returns, so identical rows)")
print("spread is target-independent; neutralization only matters for the MODEL learning.")

# What matters for the model: which target makes the FEATURE->target mapping cleaner?
# Measure: |IC(feature, resid)| vs |IC(feature, demean)| for each feature.
print("\n=== IC gain: beta_resid vs demean (absolute IC change) ===")
ic_res = daily_rank_ic(resid_df, feature_cols)
ic_dem2 = daily_rank_ic(demean_df, feature_cols)
gain = pd.DataFrame({"ic_demean": ic_dem2["ic_mean"], "ic_beta_resid": ic_res["ic_mean"]}).dropna()
gain["abs_gain"] = gain["ic_beta_resid"].abs() - gain["ic_demean"].abs()
print(gain.sort_values("abs_gain", ascending=False).head(10).round(4).to_string())
print("\nmean |IC| demean:", gain["ic_demean"].abs().mean().round(4),
      "| mean |IC| beta_resid:", gain["ic_beta_resid"].abs().mean().round(4))
