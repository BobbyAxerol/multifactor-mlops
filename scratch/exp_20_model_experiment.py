"""
exp_20: MODEL experiment — current feature set + H=1 vs research feature set + H=5.

Trains per-fold XGBoost (purged quarterly folds on dev 2022-2023), measures:
  - prediction rank IC vs demeaned target (same horizon as label)
  - TAIL SPREAD of predictions (raw fwd returns at the label horizon)
This is the decisive test for the "what to train on" question.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import xgboost as xgb

from research_base import (load_dev_data, build_panel, raw_forward_returns,
                           daily_rank_ic, compute_new_factors)
from src.multifactor_mlops.labels.returns import filter_train_by_label_end
from src.multifactor_mlops.optimization.folds import PurgedExpandingFoldBuilder

cfg, data_dict, macro_df, funding_wide = load_dev_data()

BASE_ML = {"learning_rate": 0.05, "max_depth": 4, "colsample_bytree": 0.5,
           "subsample": 0.8, "num_boost_round": 100, "random_state": 42}

NEW_FEATURES = ["mom_7", "mom_14", "mom_30", "mom_90", "resid_mom_30", "mom_risk_adj_30",
                "inv_retail_flow_7", "inv_retail_flow_30", "inv_retail_flow_90",
                "inv_margin_risk_90", "inv_funding_chg_7", "inv_carry_30", "inv_mom_365",
                "overnight", "volume_z_7"]

def run_model_experiment(H, feature_suffix, panel, raw_fwd, folds, ml_params):
    feature_cols = [c for c in panel.columns if c not in {"target", "Symbol", "decision_time",
                                                          "label_start_time", "label_end_time"}]
    feature_cols = [c for c in feature_cols if c in panel.columns]
    pred_parts = []
    ics = []
    for fold in folds:
        test_start = fold["test_start"]
        test_end = fold["test_end"]
        train_df = filter_train_by_label_end(panel, test_start)
        test_mask = ((panel.index.get_level_values("Time") >= test_start)
                     & (panel.index.get_level_values("Time") < test_end))
        test_df = panel[test_mask]
        if train_df.empty or test_df.empty:
            continue
        X = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = train_df["target"].astype(float)
        booster = xgb.train({
            "objective": "reg:squarederror",
            "learning_rate": ml_params["learning_rate"],
            "max_depth": ml_params["max_depth"],
            "colsample_bytree": ml_params["colsample_bytree"],
            "subsample": ml_params["subsample"],
            "random_state": ml_params["random_state"],
            "verbosity": 0,
        }, xgb.DMatrix(X, label=y), num_boost_round=ml_params["num_boost_round"])
        Xt = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        preds = booster.predict(xgb.DMatrix(Xt))
        fold_df = test_df[["target"]].copy()
        fold_df["pred"] = preds
        fold_df["fold_id"] = fold["fold_id"]
        pred_parts.append(fold_df)
        ic = fold_df["pred"].rank().corr(fold_df["target"].rank())
        ics.append(ic if np.isfinite(ic) else 0.0)

    oof = pd.concat(pred_parts)
    # tail spread vs RAW fwd returns at horizon H (aligned)
    raw_fwd_H = raw_fwd["raw_fwd_ret"]
    spread_rows = []
    for t, g in oof.groupby(level="Time"):
        if len(g) < 16:
            continue
        syms = g.index.get_level_values("Symbol")
        rf = raw_fwd_H.loc[t].reindex(syms)
        g2 = g.copy()
        g2["rf"] = rf.values
        g2 = g2.dropna(subset=["pred", "rf"])
        if len(g2) < 16:
            continue
        k = max(1, int(len(g2) * 0.0625))
        r = g2["pred"].rank(method="first")
        spread_rows.append(g2.loc[r > len(g2) - k, "rf"].mean() - g2.loc[r <= k, "rf"].mean())
    spread = np.array(spread_rows)
    overall_ic = oof["pred"].rank().corr(oof["target"].rank())
    return {
        "mean_fold_ic": float(np.mean(ics)),
        "pooled_ic": float(overall_ic),
        "tail_spread_sharpe": float(spread.mean() / spread.std(ddof=1) * np.sqrt(365 / H)),
        "tail_spread_ann": float(spread.mean() * 365 / H),
        "tail_spread_tstat": float(spread.mean() / spread.std(ddof=1) * np.sqrt(len(spread))),
        "n_days": int(len(spread)),
    }, oof

folds = PurgedExpandingFoldBuilder(start_date="2022-01-01", end_date="2023-12-31",
                                   frequency="quarterly").build_folds()

results = {}
for H, tag in [(1, "A_current_H1"), (5, "B_research_H5")]:
    panel = build_panel(data_dict, macro_df, funding_wide, H=H, ranked=False)
    panel = compute_new_factors(panel, data_dict, funding_wide)
    for col in ["retail_flow_7", "retail_flow_30", "retail_flow_90", "margin_risk_90",
                "funding_chg_7", "carry_30", "mom_365"]:
        if col in panel.columns:
            panel[f"inv_{col}"] = -panel[col]
    raw_fwd = raw_forward_returns(data_dict, H=H)
    if tag == "A_current_H1":
        drop = [c for c in NEW_FEATURES if c in panel.columns]
        panel = panel.drop(columns=drop)
    stats, oof = run_model_experiment(H, tag, panel, raw_fwd, folds, BASE_ML)
    results[tag] = stats
    print(f"\n[{tag}] features in model: {sum(1 for c in panel.columns if c not in {'target','Symbol','decision_time','label_start_time','label_end_time'})}")
    print(pd.Series(stats).round(4).to_string())

print("\n=== SUMMARY ===")
print(pd.DataFrame(results).T.round(4).to_string())
