"""
exp_20b: curated momentum-only model (H=5) vs model-free composite signal (H=5).

Learning from exp_20: full feature mixing dilutes the momentum alpha. Test:
  A) XGBoost trained ONLY on momentum family features, label H=5
  B) model-free: per-timestamp z-score composite of momentum factors, no ML
Both measured by tail spread on raw fwd returns at H=5.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import xgboost as xgb

from research_base import (load_dev_data, build_panel, raw_forward_returns, compute_new_factors)
from src.multifactor_mlops.labels.returns import filter_train_by_label_end
from src.multifactor_mlops.optimization.folds import PurgedExpandingFoldBuilder

cfg, data_dict, macro_df, funding_wide = load_dev_data()
H = 5
panel = build_panel(data_dict, macro_df, funding_wide, H=H, ranked=False)
panel = compute_new_factors(panel, data_dict, funding_wide)
raw_fwd = raw_forward_returns(data_dict, H=H)

MOM_FEATURES = ["mom_7", "mom_14", "mom_30", "mom_90", "mom_wma_dist_14", "mom_wma_dist_30",
                "mom_rsi_14", "mom_rsi_30", "resid_mom_30", "mom_risk_adj_30"]

def tail_spread_of_signal(signal, raw_fwd_ret):
    rows = []
    for t, g in signal.to_frame("s").dropna().groupby(level="Time"):
        if len(g) < 16:
            continue
        syms = g.index.get_level_values("Symbol")
        rf = raw_fwd_ret.loc[t].reindex(syms)
        g2 = g.copy()
        g2["rf"] = rf.values
        g2 = g2.dropna(subset=["s", "rf"])
        if len(g2) < 16:
            continue
        k = max(1, int(len(g2) * 0.0625))
        r = g2["s"].rank(method="first")
        rows.append(g2.loc[r > len(g2) - k, "rf"].mean() - g2.loc[r <= k, "rf"].mean())
    s = np.array(rows)
    return {"sharpe": float(s.mean() / s.std(ddof=1) * np.sqrt(365 / H)),
            "ann": float(s.mean() * 365 / H),
            "tstat": float(s.mean() / s.std(ddof=1) * np.sqrt(len(s))),
            "n_days": int(len(s))}

# B) model-free composite
z = panel[MOM_FEATURES].groupby(level="Time").transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
comp = z.sum(axis=1)
print("=== B) MODEL-FREE momentum composite (z-sum), H=5 ===")
print(pd.Series(tail_spread_of_signal(comp, raw_fwd["raw_fwd_ret"])).round(4).to_string())

# A) XGBoost on momentum-only, H=5
folds = PurgedExpandingFoldBuilder(start_date="2022-01-01", end_date="2023-12-31",
                                   frequency="quarterly").build_folds()
pred_parts = []
for fold in folds:
    test_start = fold["test_start"]
    test_end = fold["test_end"]
    train_df = filter_train_by_label_end(panel, test_start)
    test_mask = ((panel.index.get_level_values("Time") >= test_start)
                 & (panel.index.get_level_values("Time") < test_end))
    test_df = panel[test_mask]
    if train_df.empty or test_df.empty:
        continue
    X = train_df[MOM_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train_df["target"].astype(float)
    booster = xgb.train({
        "objective": "reg:squarederror", "learning_rate": 0.05, "max_depth": 4,
        "colsample_bytree": 0.5, "subsample": 0.8, "random_state": 42, "verbosity": 0,
    }, xgb.DMatrix(X, label=y), num_boost_round=100)
    Xt = test_df[MOM_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fold_df = test_df[["target"]].copy()
    fold_df["pred"] = booster.predict(xgb.DMatrix(Xt))
    pred_parts.append(fold_df)

oof = pd.concat(pred_parts)
print("\n=== A) XGBoost momentum-only, H=5 ===")
print(pd.Series(tail_spread_of_signal(oof["pred"], raw_fwd["raw_fwd_ret"])).round(4).to_string())
print("pooled IC:", round(float(oof["pred"].rank().corr(oof["target"].rank())), 4))

# correlation between model pred and composite
aligned = oof["pred"].to_frame("pred").join(comp.rename("comp"), how="inner")
print("\ncorr(model pred rank, composite rank):",
      round(float(aligned["pred"].rank().corr(aligned["comp"].rank())), 3))
