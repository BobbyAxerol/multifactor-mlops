"""
exp_42: OOS backtest of CROSS-WINDOW-ROBUST signals (positive on dev AND OOS).

Signals (from exp_12/13/41):
  - wma14: mom_wma_dist_14 (dev 2.04, OOS 1.75 tail spread)
  - enh:   mom_14 + mom_30 + inv_retail_flow_7 + inv_margin_risk_90 (OOS 2.06)
  - enh_wk: same with true 5-day holding
All measured via QuantBT with fee 5bp/slippage 1bp/funding ON, on the OOS window.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from research_base import load_dev_data, build_panel, compute_new_factors
from src.multifactor_mlops.backtest.quantbt_runner import QuantBTRunner

def make_signal(end_date, dev_syms=None):
    cfg, data_dict, macro_df, funding_wide = load_dev_data(end_date=end_date)
    panel = build_panel(data_dict, macro_df, funding_wide, H=1, ranked=False)
    panel = compute_new_factors(panel, data_dict, funding_wide)
    if funding_wide is not None:
        panel = panel.join(pd.DataFrame({"funding_chg_7": funding_wide.diff(7).stack()}).rename_axis(["Time", "Symbol"]), how="left")
    for col in ["retail_flow_7", "margin_risk_90"]:
        panel[f"inv_{col}"] = -panel[col]
    def zsum(cols):
        z = panel[cols].groupby(level="Time").transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
        return z.sum(axis=1)
    panel["sig_wma14"] = zsum(["mom_wma_dist_14"])
    panel["sig_enh"] = zsum(["mom_14", "mom_30", "inv_retail_flow_7", "inv_margin_risk_90"])
    panel["sig_rsi_flow"] = zsum(["mom_rsi_14", "inv_retail_flow_7"])
    signals = {}
    for name in ["sig_wma14", "sig_enh", "sig_rsi_flow"]:
        s = panel[[name]].rename(columns={name: "s"}).reset_index()
        signals[name] = s.pivot_table(index="Time", columns="Symbol", values="s").sort_index()
    if dev_syms is not None:
        signals = {k: v[dev_syms] for k, v in signals.items()}
    return data_dict, macro_df, funding_wide, signals

def build_weights(wide, quantiles=16):
    rank_pct = wide.rank(axis=1, pct=True, method="first")
    top_pct = min(0.25, max(0.05, 1.0 / quantiles))
    long_mask = rank_pct > (1.0 - top_pct)
    short_mask = rank_pct <= top_pct
    long_w = long_mask.astype(float).div(long_mask.astype(float).sum(axis=1).replace(0, 1.0), axis=0)
    short_w = -short_mask.astype(float).div(short_mask.astype(float).sum(axis=1).replace(0, 1.0), axis=0)
    return (long_w + short_w).fillna(0.0).clip(lower=-0.25, upper=0.25)

def hold_days(weights, k=5):
    out = weights.copy()
    for i in range(0, len(out), k):
        out.iloc[i + 1:min(i + k, len(out))] = out.iloc[i]
    return out

def run_bt(weights, data_dict, funding_wide):
    w = weights.shift(1).fillna(0.0)
    w.loc[w.index.dayofweek.isin([5, 6])] = 0.0
    funding_map = {s: (funding_wide[s].reindex(w.index).fillna(0.0) if s in funding_wide.columns
                       else pd.Series(0.0, index=w.index)) for s in w.columns}
    params = {
        "fee_rate_per_fill": 0.0005, "slippage": 0.0001, "leverage": 3.0,
        "initial_capital": 100000.0, "portfolio_mode": "longshort",
        "hedge_type": "target_weight", "trading_days_per_year": 365,
        "use_funding": True, "funding_rate": funding_map,
    }
    eq, metrics, res = QuantBTRunner().run_backtest(positions=w, data_dict=data_dict, params=params)
    rets = eq["return"]
    return {"sharpe": round(float(rets.mean() / rets.std(ddof=1) * np.sqrt(365)), 3),
            "mdd": round(float((eq["equity"] / eq["equity"].cummax() - 1).min()), 4),
            "final": round(float(eq["equity"].iloc[-1]), 0),
            "trades": int(metrics.get("Number of Trades", -1))}

# dev universe (fixed)
_, _, _, dev_signals = make_signal("2023-12-31")
dev_syms = list(dev_signals["sig_enh"].columns)

# OOS data
data_full, _, fund_full, oos_signals = make_signal("2026-12-31")
dev_syms = [s for s in dev_syms if s in oos_signals["sig_enh"].columns]
oos_signals = {k: v[dev_syms] for k, v in oos_signals.items()}
data_oos = {s: data_full[s] for s in dev_syms if s in data_full}
for name, wide in oos_signals.items():
    wide = wide.loc[wide.index >= "2024-01-01"]
    w = build_weights(wide)
    print(f"OOS {name} daily: {run_bt(w, data_oos, fund_full)}")
    print(f"OOS {name} hold5: {run_bt(hold_days(w, 5), data_oos, fund_full)}")
