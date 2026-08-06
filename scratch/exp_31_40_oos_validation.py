"""
exp_31 + exp_40: robustness + OUTER OOS single validation.

A) exp_31 (dev): fee sensitivity (5bp/10bp) and TRUE weekly rebalance
   (hold weights 5 days — no daily re-rank churn).
B) exp_40 (OOS 2024+): best variant, fixed dev universe, run ONCE.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from research_base import load_dev_data, build_panel, compute_new_factors
from src.multifactor_mlops.backtest.quantbt_runner import QuantBTRunner

# ---------- build composite signal ----------
def make_signal(end_date):
    cfg, data_dict, macro_df, funding_wide = load_dev_data(end_date=end_date)
    panel = build_panel(data_dict, macro_df, funding_wide, H=1, ranked=False)
    panel = compute_new_factors(panel, data_dict, funding_wide)
    z = panel[["mom_7", "mom_14", "mom_30", "mom_wma_dist_14", "mom_rsi_14", "resid_mom_30"]] \
        .groupby(level="Time").transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
    panel["score"] = z.sum(axis=1)
    s = panel["score"].rename("s").reset_index()
    wide = s.pivot_table(index="Time", columns="Symbol", values="s").sort_index()
    return data_dict, macro_df, funding_wide, wide

def build_weights(wide, quantiles=16):
    rank_pct = wide.rank(axis=1, pct=True, method="first")
    top_pct = min(0.25, max(0.05, 1.0 / quantiles))
    long_mask = rank_pct > (1.0 - top_pct)
    short_mask = rank_pct <= top_pct
    long_w = long_mask.astype(float).div(long_mask.astype(float).sum(axis=1).replace(0, 1.0), axis=0)
    short_w = -short_mask.astype(float).div(short_mask.astype(float).sum(axis=1).replace(0, 1.0), axis=0)
    return (long_w + short_w).fillna(0.0).clip(lower=-0.25, upper=0.25)

def hold_days(weights, k=5):
    """True holding: rebalance every k days (hold the Monday decision)."""
    out = weights.copy()
    for i in range(0, len(out), k):
        out.iloc[i + 1:min(i + k, len(out))] = out.iloc[i]
    return out

def run_bt(weights, data_dict, funding_wide, fee_rate=0.0005, name=""):
    w = weights.shift(1).fillna(0.0)
    is_weekend = w.index.dayofweek.isin([5, 6])
    w.loc[is_weekend] = 0.0
    funding_map = {}
    for s in w.columns:
        if s in funding_wide.columns:
            funding_map[s] = funding_wide[s].reindex(w.index).fillna(0.0)
        else:
            funding_map[s] = pd.Series(0.0, index=w.index)
    params = {
        "fee_rate_per_fill": fee_rate, "slippage": 0.0001, "leverage": 3.0,
        "initial_capital": 100000.0, "portfolio_mode": "longshort",
        "hedge_type": "target_weight", "trading_days_per_year": 365,
        "use_funding": True, "funding_rate": funding_map,
    }
    eq, metrics, res = QuantBTRunner().run_backtest(positions=w, data_dict=data_dict, params=params)
    rets = eq["return"]
    return {
        "sharpe": round(float(rets.mean() / rets.std(ddof=1) * np.sqrt(365)), 3),
        "mdd": round(float((eq["equity"] / eq["equity"].cummax() - 1).min()), 4),
        "final": round(float(eq["equity"].iloc[-1]), 0),
        "trades": int(metrics.get("Number of Trades", metrics.get("num_trades", -1))),
        "fee_bp": int(fee_rate * 10000),
    }

# ============ A) dev robustness ============
print("=" * 60)
print("A) DEV 2022-2023 — fee sensitivity + true 5-day holding")
print("=" * 60)
data_dev, macro_dev, fund_dev, wide_dev = make_signal("2023-12-31")
w_daily = build_weights(wide_dev)
w_hold5 = hold_days(w_daily, 5)
for fee in [0.0005, 0.001]:
    for tag, w in [("daily_rerank", w_daily), ("hold5", w_hold5)]:
        r = run_bt(w, data_dev, fund_dev, fee_rate=fee)
        print(f"  dev {tag} fee={fee}: sharpe={r['sharpe']} mdd={r['mdd']} final={r['final']} trades={r['trades']}")

# ============ B) OUTER OOS single touch ============
print("=" * 60)
print("B) OUTER OOS 2024-01-01 ->  (fixed dev top-40 universe) — SINGLE TOUCH")
print("=" * 60)
data_full, macro_full, fund_full, wide_full = make_signal("2026-12-31")
# fixed universe = dev symbols only
dev_syms = set(wide_dev.columns)
full_syms = [s for s in wide_full.columns if s in dev_syms]
wide_oos = wide_full[full_syms].loc[wide_full.index >= "2024-01-01"]
data_oos = {s: data_full[s] for s in full_syms if s in data_full}
fund_oos = fund_full
w_oos = build_weights(wide_oos)
for tag, w in [("daily_rerank", w_oos), ("hold5", hold_days(w_oos, 5))]:
    r = run_bt(w, data_oos, fund_oos, fee_rate=0.0005)
    print(f"  OOS {tag}: sharpe={r['sharpe']} mdd={r['mdd']} final={r['final']} trades={r['trades']}")
