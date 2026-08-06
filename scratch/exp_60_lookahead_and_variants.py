"""
exp_60: (A) look-ahead verification of the current best config (V4.5 composite),
       (B) quick improvement-variant backtests on dev AND OOS.

(A) future-mutation: mutate prices AFTER cutoff T -> weights <= T must be identical.
(B) variants (base = quantiles 6, inv_vol 28, cap 0.25, ceiling 0.07, stress 0.3, daily):
    1 base            (reproduce)
    2 drift           (hysteresis: rebalance asset only if |target-current| > 0.01)
    3 voltarget       (portfolio-level vol targeting: scale to trailing 20d vol)
    4 dispersion      (flat on weak-signal days: score top-bottom spread < threshold)
    5 deepstress      (stress_multiplier 0.15 instead of 0.3)
    6 asymmetric      (long top 8% / short bottom 16%)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from research_base import load_dev_data
from src.multifactor_mlops.config.loader import load_config
from src.multifactor_mlops.data.loader import load_all_data
from src.multifactor_mlops.features.panel import PanelDatasetBuilder
from src.multifactor_mlops.features.macro import MacroOverlayTransformer
from src.multifactor_mlops.backtest.quantbt_runner import QuantBTRunner

CFG = load_config("parameters.json")


def build_signal_panel(data_dict, macro_df, funding_wide, membership=None):
    builder = PanelDatasetBuilder(
        windows=CFG.features.windows,
        cross_sectional_rank=True,
        lag=CFG.label.holding_bars,
        keep_families=CFG.features.keep_families,
        inverted_features=CFG.features.inverted_features,
        use_macro_features=CFG.features.use_macro_features,
        return_type=CFG.label.return_type,
    )
    return builder.build_panel_dataset(data_dict, list(data_dict), macro_df=macro_df,
                                       funding_df=funding_wide,
                                       universe_membership_df=membership)


def composite_wide(panel, features=None):
    features = features or ["mom_14", "mom_30", "retail_flow_7", "margin_risk_90"]
    z = panel[features].groupby(level="Time").transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
    score = z.sum(axis=1).rename("s").reset_index()
    return score.pivot_table(index="Time", columns="Symbol", values="s").sort_index()


def build_weights(wide, quantiles=6, long_q=None, short_q=None, inv_vol_period=28,
                  alloc_cap=0.25, vol_ceiling=0.07, closes=None):
    long_q = long_q or quantiles
    short_q = short_q or quantiles
    rank_pct = wide.rank(axis=1, pct=True, method="first")
    long_mask = rank_pct > (1.0 - 1.0 / long_q)
    short_mask = rank_pct <= (1.0 / short_q)
    # Conviction sizing (pipeline parity): weight proportional to |score| within each leg
    pos_preds = wide.clip(lower=0.0)
    neg_preds = wide.clip(upper=0.0).abs()
    long_weighted = (pos_preds * long_mask.astype(float)).fillna(0.0)
    short_weighted = (neg_preds * short_mask.astype(float)).fillna(0.0)

    if closes is not None:
        vols = closes.pct_change().rolling(inv_vol_period, min_periods=15).std()
        inv_vol = (1.0 / vols.replace(0, np.nan)).fillna(0.0)
        long_weighted = long_weighted * inv_vol
        short_weighted = short_weighted * inv_vol

    long_w = long_weighted.div(long_weighted.sum(axis=1).replace(0, 1.0), axis=0)
    short_w = -short_weighted.div(short_weighted.sum(axis=1).replace(0, 1.0), axis=0)
    weights = (long_w + short_w).fillna(0.0)

    if vol_ceiling is not None and closes is not None:
        high_vol = (closes.pct_change().rolling(14, min_periods=5).std() > vol_ceiling).fillna(False)
        weights = weights.mask(high_vol, weights * 0.5)

    return weights.clip(lower=-alloc_cap, upper=alloc_cap)


def apply_overlay(weights, macro_df, btc_close, stress_mult=0.3):
    mult = MacroOverlayTransformer().transform_macro_df(macro_df)["macro_multiplier"]
    btc_mom = btc_close.pct_change(21)
    crash = (btc_mom.reindex(weights.index).fillna(0.0) < 0.0).astype(float)
    long_leg = 1.0 - crash * (1.0 - stress_mult)
    short_leg = 1.0 - (1.0 - crash) * (1.0 - 0.6)
    g = mult.reindex(weights.index).fillna(1.0)
    pos = weights.where(weights > 0, 0.0)
    neg = weights.where(weights < 0, 0.0)
    return (pos.mul(g * long_leg, axis=0) + neg.mul(g * short_leg, axis=0)).fillna(0.0)


def variant_positions(weights, variant, closes=None, score_wide=None, btc_close=None, macro_df=None,
                       stress_mult=0.3, long_q=None, short_q=None):
    w = weights.shift(1).fillna(0.0)  # 1-bar execution lag
    if variant == "drift":
        out = w.copy()
        for i in range(1, len(out)):
            prev = out.iloc[i - 1]
            tgt = w.iloc[i]
            keep = tgt.copy()
            changed = (tgt - prev).abs() > 0.02
            keep[~changed] = prev[~changed]
            out.iloc[i] = keep
        return out
    if variant == "voltarget":
        rets = closes.pct_change().reindex(w.index)
        port_ret = (w.shift(0) * rets).sum(axis=1)
        vol20 = port_ret.rolling(20, min_periods=10).std()
        scale = (0.025 / vol20.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
        return w.mul(scale, axis=0)
    if variant == "dispersion":
        spread = score_wide.apply(lambda r: r.nlargest(6).mean() - r.nsmallest(6).mean(), axis=1)
        thr = spread.rolling(60, min_periods=20).median()
        flat = (spread < thr).fillna(False)
        out = w.copy()
        out[flat] = 0.0
        return out
    if variant == "deepstress":
        return apply_overlay(w, macro_df, btc_close, stress_mult=0.15)
    if variant == "nooverlay":
        return w
    if variant == "asymmetric":
        # rebuild weights with long 8% / short 16% legs, then overlay stress 0.3
        rank_pct = score_wide.rank(axis=1, pct=True, method="first")
        long_mask = rank_pct > (1.0 - 0.08)
        short_mask = rank_pct <= 0.16
        pos_preds = score_wide.clip(lower=0.0)
        neg_preds = score_wide.clip(upper=0.0).abs()
        lw = (pos_preds * long_mask.astype(float)).fillna(0.0)
        sw = (neg_preds * short_mask.astype(float)).fillna(0.0)
        if closes is not None:
            vols = closes.pct_change().rolling(28, min_periods=15).std()
            inv = (1.0 / vols.replace(0, np.nan)).fillna(0.0)
            lw, sw = lw * inv, sw * inv
        lw = lw.div(lw.sum(axis=1).replace(0, 1.0), axis=0)
        sw = -sw.div(sw.sum(axis=1).replace(0, 1.0), axis=0)
        w2 = (lw + sw).fillna(0.0).clip(lower=-0.25, upper=0.25)
        return apply_overlay(w2, macro_df, btc_close, stress_mult=stress_mult).shift(1).fillna(0.0)
    return w


def run_bt(positions, data_dict, funding_wide):
    funding_map = {s: (funding_wide[s].reindex(positions.index).fillna(0.0) if s in funding_wide.columns
                       else pd.Series(0.0, index=positions.index)) for s in positions.columns}
    params = {
        "fee_rate_per_fill": 0.0005, "slippage": 0.0001, "leverage": 3.0,
        "initial_capital": 100000.0, "portfolio_mode": "longshort",
        "hedge_type": "target_weight", "trading_days_per_year": 365,
        "use_funding": True, "funding_rate": funding_map,
    }
    eq, metrics, res = QuantBTRunner().run_backtest(positions=positions, data_dict=data_dict, params=params)
    rets = eq["return"]
    return {
        "sharpe": round(float(rets.mean() / rets.std(ddof=1) * np.sqrt(365)), 3),
        "mdd": round(float((eq["equity"] / eq["equity"].cummax() - 1).min()), 4),
        "final": round(float(eq["equity"].iloc[-1]), 0),
        "trades": int(metrics.get("Number of Trades", -1)),
    }


# ================= (A) LOOK-AHEAD VERIFICATION =================
print("=" * 70)
print("(A) LOOK-AHEAD CHECK — future prices must not change past weights")
print("=" * 70)
cfg = load_config("parameters.json")
data_full, macro_full, funding_full, membership_full = load_all_data(cfg)
panel_full = build_signal_panel(data_full, macro_full, pd.DataFrame(funding_full) if funding_full else None,
                                 membership=membership_full)
wide_full = composite_wide(panel_full)
closes_full = pd.DataFrame({s: data_full[s]["close"] for s in wide_full.columns}).reindex(wide_full.index)
w_full = build_weights(wide_full, closes=closes_full)

cutoff = pd.Timestamp("2025-06-30")
data_mut = {s: df.copy() for s, df in data_full.items()}
for s in data_mut:
    df = data_mut[s]
    mask = df.index > cutoff
    df.loc[mask, "close"] *= 2.0
    df.loc[mask, "high"] *= 2.0
    df.loc[mask, "low"] *= 2.0
    df.loc[mask, "open"] *= 2.0
panel_mut = build_signal_panel(data_mut, macro_full, pd.DataFrame(funding_full) if funding_full else None,
                                membership=membership_full)
wide_mut = composite_wide(panel_mut)
closes_mut = pd.DataFrame({s: data_mut[s]["close"] for s in wide_mut.columns}).reindex(wide_mut.index)
w_mut = build_weights(wide_mut, closes=closes_mut)

common = w_full.index[w_full.index <= cutoff]
diff = (w_full.loc[common] - w_mut.reindex(common).fillna(0.0)).abs().max().max()
print(f"max |weight diff| for rows <= {cutoff.date()}: {diff:.2e}")
print("VERDICT:", "NO LOOK-AHEAD (weights identical before cutoff)" if diff < 1e-9 else "LOOK-AHEAD DETECTED!")

# ================= (B) IMPROVEMENT VARIANTS =================
print("=" * 70)
print("(B) VARIANTS — dev (2022-23) and OOS (2024+)")
print("=" * 70)

btc_close_full = data_full["BTCUSDT"]["close"]

def run_window(end_date, start_oos=None):
    data_dict, macro_df, funding_dict, membership = load_all_data(cfg, end_date=end_date)
    panel = build_signal_panel(data_dict, macro_df, pd.DataFrame(funding_dict) if funding_dict else None,
                               membership=membership)
    wide = composite_wide(panel)
    if start_oos is not None:
        wide = wide.loc[wide.index >= start_oos]
    closes = pd.DataFrame({s: data_dict[s]["close"] for s in wide.columns}).reindex(wide.index)
    funding_wide = pd.DataFrame(funding_dict) if funding_dict else None
    btc_close = data_dict["BTCUSDT"]["close"]
    weights = build_weights(wide, closes=closes)  # WITHOUT overlay (base applies stress 0.3)
    rows = {}
    for variant in ["base", "deepstress", "nooverlay", "drift", "voltarget", "dispersion", "asymmetric"]:
        if variant == "base":
            w_v = apply_overlay(weights, macro_df, btc_close, stress_mult=0.3)
            pos = w_v.shift(1).fillna(0.0)
        elif variant in ("drift", "voltarget", "dispersion"):
            w_v = apply_overlay(weights, macro_df, btc_close, stress_mult=0.3)
            pos = variant_positions(w_v, variant, closes=closes, score_wide=wide)
        elif variant == "asymmetric":
            pos = variant_positions(weights, variant, closes=closes, score_wide=wide,
                                    btc_close=btc_close, macro_df=macro_df)
        else:
            pos = variant_positions(weights, variant, closes=closes, btc_close=btc_close, macro_df=macro_df)
        rows[variant] = run_bt(pos, {s: data_dict[s] for s in wide.columns if s in data_dict}, funding_wide)
    return pd.DataFrame(rows).T

dev = run_window("2023-12-31")
print("\n--- DEV 2022-2023 ---")
print(dev.to_string())
oos = run_window("2026-12-31", start_oos="2024-01-01")
print("\n--- OOS 2024+ ---")
print(oos.to_string())
