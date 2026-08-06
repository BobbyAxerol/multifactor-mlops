"""
Shared research infrastructure (scratch-only).

Imports the codebase but NEVER modifies it. All experiments measure on the
DEV window (2022-01-01 .. dev_end) unless explicitly stated. Outer OOS
(>= 2024-01-01) is ONLY touched by the final validation experiment.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.multifactor_mlops.config.loader import load_config
from src.multifactor_mlops.data.loader import load_all_data
from src.multifactor_mlops.features.panel import PanelDatasetBuilder
from src.multifactor_mlops.labels.returns import calculate_next_open_to_open_returns

DEV_START = "2022-01-01"
DEV_END = "2023-12-31"
CURRENT_FEATURE_FAMILIES = ["mom_rsi", "mom_wma_dist", "retail_flow", "carry", "margin_risk"]


def load_dev_data(end_date=DEV_END):
    cfg = load_config("parameters.json")
    data_dict, macro_df, funding_dict = load_all_data(cfg, end_date=end_date)
    funding_wide = pd.DataFrame(funding_dict) if funding_dict else None
    return cfg, data_dict, macro_df, funding_wide


def build_panel(data_dict, macro_df, funding_wide=None, H=1, ranked=False):
    """Panel with current features (raw or cross-sectionally ranked) + target."""
    builder = PanelDatasetBuilder(
        windows=[7, 14, 30, 60, 90],
        cross_sectional_rank=ranked,
        lag=H,
        return_type="next_open_to_open",  # research convention (documented in report)
    )
    return builder.build_panel_dataset(
        data_dict, list(data_dict.keys()), macro_df=macro_df, funding_df=funding_wide
    )


def raw_forward_returns(data_dict, H=1):
    """Raw (non-demeaned) open-to-open forward returns, long panel indexed (Time, Symbol)."""
    parts = []
    for sym, df in data_dict.items():
        if "open" not in df.columns:
            continue
        r = calculate_next_open_to_open_returns(df["open"], holding_bars=H).rename("raw_fwd_ret")
        frame = r.to_frame()
        frame["Symbol"] = sym
        parts.append(frame.dropna())
    out = pd.concat(parts)
    out.index.name = "Time"
    return out.reset_index().set_index(["Time", "Symbol"]).sort_index()


def daily_rank_ic(panel, columns, target="target", min_symbols=8):
    """
    Per-day cross-sectional Spearman rank IC, pooled across the window.
    Returns DataFrame indexed by feature: ic_mean / ic_ir / tstat / n_days / pct_pos.
    """
    out = {}
    times = panel.index.get_level_values("Time")
    for col in columns:
        sub = panel[[col, target]].dropna()
        if len(sub) < 50:
            out[col] = {"ic_mean": np.nan, "ic_ir": np.nan, "tstat": np.nan,
                        "n_days": 0, "pct_pos": np.nan}
            continue
        ics = []
        for t, g in sub.groupby(level="Time"):
            if len(g) >= min_symbols:
                r1 = g[col].rank()
                r2 = g[target].rank()
                c = r1.corr(r2)
                if np.isfinite(c):
                    ics.append(c)
        ics = np.array(ics)
        n = len(ics)
        if n < 20:
            out[col] = {"ic_mean": np.nan, "ic_ir": np.nan, "tstat": np.nan,
                        "n_days": n, "pct_pos": np.nan}
            continue
        mean = float(ics.mean())
        std = float(ics.std(ddof=1)) if n > 1 else 0.0
        out[col] = {
            "ic_mean": mean,
            "ic_ir": mean / std if std > 0 else np.nan,
            "tstat": mean / std * np.sqrt(n) if std > 0 else np.nan,
            "n_days": n,
            "pct_pos": float((ics > 0).mean()),
        }
    return pd.DataFrame(out).T.sort_values("ic_mean", ascending=False)


def spread_return(panel, signal_col, raw_fwd, top_pct=0.25, min_symbols=16, fwd_series=None):
    """
    Daily top/bottom quantile long-short spread using RAW forward returns
    (market neutral, before costs). Aligns signal/raw-fwd rows exactly.
    fwd_series: optional custom long forward-return series (indexed Time,Symbol)
                to measure against (e.g., close-to-close instead of open-to-open).
    """
    if fwd_series is not None:
        fwd_map = fwd_series
    else:
        fwd_map = raw_fwd["raw_fwd_ret"]
    rows = []
    for t, g in panel.groupby(level="Time"):
        if len(g) < min_symbols * 2:
            continue
        syms = g.index.get_level_values("Symbol")
        rf = fwd_map.loc[t].reindex(syms)
        g2 = g.copy()
        g2["rf"] = rf.values
        g2 = g2.dropna(subset=[signal_col, "rf"])
        if len(g2) < min_symbols * 2:
            continue
        n_top = max(1, int(len(g2) * top_pct))
        ranked = g2[signal_col].rank(method="first")
        rows.append((t, g2.loc[ranked > len(g2) - n_top, "rf"].mean()
                        - g2.loc[ranked <= n_top, "rf"].mean()))
    spread = pd.Series(dict(rows), name="spread").sort_index()
    summary = {
        "daily_mean": float(spread.mean()),
        "daily_std": float(spread.std(ddof=1)),
        "ann_mean": float(spread.mean() * 365),
        "sharpe_ann": float(spread.mean() / spread.std(ddof=1) * np.sqrt(365)),
        "tstat": float(spread.mean() / spread.std(ddof=1) * np.sqrt(len(spread))),
        "n_days": int(len(spread)),
    }
    return spread, summary


def tail_spread_table(panel, signal_cols, raw_fwd, tails=(0.25, 0.125, 0.0625), min_symbols=16):
    """Long-short spread (raw fwd returns) at several tail depths per signal."""
    out = {}
    for col in signal_cols:
        for tail in tails:
            _, s = spread_return(panel, col, raw_fwd, top_pct=tail, min_symbols=min_symbols)
            out[f"{col}@{tail:.4f}"] = {
                "sharpe_ann": s["sharpe_ann"],
                "ann_mean": s["ann_mean"],
                "tstat": s["tstat"],
                "n_days": s["n_days"],
            }
    return pd.DataFrame(out).T.sort_values("sharpe_ann", ascending=False)


def compute_new_factors(panel, data_dict, funding_wide=None):
    """
    Computes NEW candidate factors (long panel, trailing data only) and joins
    them into the panel. Zero bfill. All rolling windows look at the past.
    """
    closes, opens, highs, lows, vols = {}, {}, {}, {}, {}
    for sym, df in data_dict.items():
        if df.empty or "close" not in df.columns:
            continue
        closes[sym] = df["close"].sort_index()
        opens[sym] = df["open"].sort_index()
        highs[sym] = df["high"].sort_index()
        lows[sym] = df["low"].sort_index()
        vols[sym] = df["volume"].sort_index()

    btc_ret = closes["BTCUSDT"].pct_change() if "BTCUSDT" in closes else closes[sorted(closes)[0]].pct_change()

    factor_dict = {}
    for sym in closes:
        c = closes[sym]
        o = opens[sym]
        h = highs[sym]
        l = lows[sym]
        v = vols[sym]
        ret = c.pct_change()
        log_ret = np.log(c / c.shift(1))
        f = {}
        for w in [1, 3, 7, 14, 30, 90, 180, 365]:
            f[f"mom_{w}"] = np.log(c / c.shift(w))
        vol14 = log_ret.rolling(14).std()
        f["mom_risk_adj_30"] = f["mom_30"] / vol14.replace(0, np.nan)
        f["overnight"] = o / c.shift(1) - 1.0
        f["intraday"] = c / o - 1.0
        f["range_pct"] = (h - l) / c
        f["close_pos"] = (c - l) / (h - l).replace(0, np.nan)
        f["volume_z_7"] = (v - v.rolling(7).mean()) / v.rolling(7).std().replace(0, np.nan)
        f["skew_30"] = ret.rolling(30).skew()
        f["kurt_30"] = ret.rolling(30).kurt()
        f["maxdd_30"] = c / c.rolling(30).max() - 1.0
        f["rev_1"] = -ret.shift(1)  # 1-day reversal
        # BTC beta residual momentum (trailing 30d)
        beta30 = ret.rolling(30).cov(btc_ret) / btc_ret.rolling(30).var()
        f["beta_btc_30"] = beta30
        f["resid_mom_30"] = f["mom_30"] - beta30 * np.log(c.shift(0) / c.shift(30)).pipe(lambda s: btc_ret.rolling(30).apply(lambda x: np.log(1 + x).sum(), raw=True))
        factor_dict[sym] = pd.DataFrame(f, index=c.index)

    all_f = pd.concat(factor_dict)
    all_f.index = all_f.index.set_names(["Symbol", "Time"]).reorder_levels(["Time", "Symbol"])
    all_f = all_f.sort_index()

    joined = panel.join(all_f, how="left")
    return joined


def long_short_trade_test(panel, signal_cols, data_dict, raw_fwd, top_pct=0.25):
    """Reports spread statistics for each signal column (equal-weight long-short)."""
    out = {}
    for col in signal_cols:
        _, s = spread_return(panel, col, raw_fwd, top_pct=top_pct)
        out[col] = s
    return pd.DataFrame(out).T.sort_values("sharpe_ann", ascending=False)
