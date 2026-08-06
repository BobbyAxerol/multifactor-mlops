"""
exp_30: PORTFOLIO variants via QuantBT on the DEV window (2022-2023).

Signal = model-free composites (winner of exp_20b). Tests:
  1. momentum composite, 16 quantiles, weekly_friday_exit, equal weight
  2. momentum composite + inverse-vol weights, weekly
  3. momentum composite, daily rebalance
  4. enhanced composite (mom - retail_flow - margin_risk90 - funding_chg), weekly
  5. enhanced composite, daily
  6. enhanced composite weekly + macro stress overlay ON

Costs: fee_rate 0.0005 one-way, slippage 1bp, funding ON. 1-bar execution lag.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from research_base import load_dev_data, build_panel, compute_new_factors
from src.multifactor_mlops.backtest.quantbt_runner import QuantBTRunner
from src.multifactor_mlops.portfolio.overlay import apply_stress_overlay
from src.multifactor_mlops.features.macro import MacroOverlayTransformer

cfg, data_dict, macro_df, funding_wide = load_dev_data()
H = 5
panel = build_panel(data_dict, macro_df, funding_wide, H=H, ranked=False)
panel = compute_new_factors(panel, data_dict, funding_wide)
if funding_wide is not None and not funding_wide.empty:
    panel = panel.join(pd.DataFrame({"funding_chg_7": funding_wide.diff(7).stack()}).rename_axis(["Time", "Symbol"]), how="left")
for col in ["retail_flow_7", "retail_flow_30", "margin_risk_90", "funding_chg_7"]:
    if col in panel.columns:
        panel[f"inv_{col}"] = -panel[col]

def zsum(cols, weights=None):
    z = panel[cols].groupby(level="Time").transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
    if weights:
        z = z * pd.Series(weights, index=z.columns)
    return z.sum(axis=1).rename("score")

panel["score_mom"] = zsum(["mom_7", "mom_14", "mom_30", "mom_wma_dist_14", "mom_rsi_14",
                           "resid_mom_30"])
panel["score_enh"] = zsum(["mom_7", "mom_14", "mom_30", "mom_wma_dist_14", "mom_rsi_14",
                           "resid_mom_30", "inv_retail_flow_7", "inv_margin_risk_90",
                           "inv_funding_chg_7"])

# weights builder
def build_weights(score_col, quantiles=16, inv_vol=False, alloc_cap=0.25):
    wide_preds = panel["score"].rename(score_col).to_frame() if False else panel[[score_col]].rename(columns={score_col: "s"}).unstack(level="Symbol")["s"] if False else None
    # simpler: pivot
    s = panel[score_col].rename("s").reset_index()
    wide = s.pivot_table(index="Time", columns="Symbol", values="s").sort_index()
    rank_pct = wide.rank(axis=1, pct=True, method="first")
    top_pct = min(0.25, max(0.05, 1.0 / quantiles))
    long_mask = rank_pct > (1.0 - top_pct)
    short_mask = rank_pct <= top_pct
    long_w = (long_mask.astype(float)).div(long_mask.astype(float).sum(axis=1).replace(0, 1.0), axis=0)
    short_w = -(short_mask.astype(float)).div(short_mask.astype(float).sum(axis=1).replace(0, 1.0), axis=0)
    weights = (long_w + short_w).fillna(0.0)
    if inv_vol:
        closes = {sym: data_dict[sym]["close"] for sym in weights.columns if sym in data_dict}
        close_df = pd.DataFrame(closes).reindex(weights.index)
        vols = close_df.pct_change().rolling(42, min_periods=20).std()
        inv_vol = (1.0 / vols.replace(0, np.nan)).fillna(0.0)
        long_part = (weights.clip(lower=0) * inv_vol)
        short_part = (weights.clip(upper=0) * inv_vol)
        weights = (long_part.div(long_part.sum(axis=1).replace(0, 1.0), axis=0)
                   + short_part.div(short_part.sum(axis=1).replace(0, 1.0), axis=0)).fillna(0.0)
    return weights.clip(lower=-alloc_cap, upper=alloc_cap)

def weekly_schedule(weights):
    out = weights.copy()
    is_weekend = out.index.dayofweek.isin([5, 6])
    out.loc[is_weekend] = 0.0
    return out

def run_variant(name, weights, overlay_mult=None):
    w = weights.shift(1).fillna(0.0)  # 1-bar execution lag
    w = weekly_schedule(w)
    if overlay_mult is not None:
        pos = w.where(w > 0, 0.0)
        neg = w.where(w < 0, 0.0)
        w = pos.mul(overlay_mult.reindex(w.index).fillna(1.0), axis=0) + neg.mul(overlay_mult.reindex(w.index).fillna(1.0), axis=0)
    funding_map = {}
    for s in w.columns:
        if s in funding_wide.columns:
            funding_map[s] = funding_wide[s].reindex(w.index).fillna(0.0)
        else:
            funding_map[s] = pd.Series(0.0, index=w.index)
    params = {
        "fee_rate_per_fill": 0.0005, "slippage": 0.0001, "leverage": 3.0,
        "initial_capital": 100000.0, "portfolio_mode": "longshort",
        "hedge_type": "target_weight", "trading_days_per_year": 365,
        "use_funding": True,
        "funding_rate": funding_map,
    }
    runner = QuantBTRunner()
    eq, metrics, res = runner.run_backtest(positions=w, data_dict=data_dict, params=params)
    sharpe = float(eq["return"].mean() / eq["return"].std(ddof=1) * np.sqrt(365))
    mdd = float((eq["equity"] / eq["equity"].cummax() - 1).min())
    trades = int(metrics.get("Number of Trades", metrics.get("num_trades", np.nan)))
    return {"sharpe": round(sharpe, 3), "mdd": round(mdd, 4),
            "final": round(float(eq["equity"].iloc[-1]), 0), "trades": trades}

if macro_df is not None and not macro_df.empty:
    macro_mult = MacroOverlayTransformer().transform_macro_df(macro_df)["macro_multiplier"]
else:
    macro_mult = None

variants = {
    "1_mom_wk_eq": build_weights("score_mom"),
    "2_mom_wk_invvol": build_weights("score_mom", inv_vol=True),
    "3_mom_daily_eq": build_weights("score_mom"),
    "4_enh_wk_eq": build_weights("score_enh"),
    "5_enh_daily_eq": build_weights("score_enh"),
}

results = {}
for name, w in variants.items():
    is_daily = "daily" in name
    if not is_daily:
        w = weekly_schedule(w)
    results[name] = run_variant(name, w)

# 6: enhanced weekly + overlay (macro multiplier scaled to stress floor 0.4)
w6 = build_weights("score_enh")
overlay_mult = macro_mult.clip(lower=0.4, upper=1.0) if macro_mult is not None else None
results["6_enh_wk_overlay"] = run_variant("6_enh_wk_overlay", w6, overlay_mult=overlay_mult)

print("\n=== QUANTBT DEV BACKTEST (2022-2023, fee 5bp, slippage 1bp, funding ON) ===")
print(pd.DataFrame(results).T.to_string())
