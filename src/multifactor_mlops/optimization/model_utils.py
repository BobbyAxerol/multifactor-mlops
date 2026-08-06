"""
Shared XGBoost model training/prediction helpers.

Used by: walk-forward strategy, OOF generation, Stage 1 ML tuning,
final production fit. Guarantees the SAME fitting convention everywhere.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Optional


def build_xgb_params(ml_params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "learning_rate": float(ml_params.get("learning_rate", 0.07)),
        "max_depth": int(ml_params.get("max_depth", 5)),
        "colsample_bytree": float(ml_params.get("colsample_bytree", 0.3)),
        "subsample": float(ml_params.get("subsample", 0.8)),
        "random_state": int(ml_params.get("random_state", 42)),
        "verbosity": 0,
    }


def train_xgb_model(
    train_df: pd.DataFrame,
    feature_cols,
    ml_params: Dict[str, Any],
) -> xgb.Booster:
    """Trains an XGBoost regressor on panel rows (target demeaned by panel builder)."""
    X = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train_df["target"].astype(float)
    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train(
        build_xgb_params(ml_params),
        dtrain,
        num_boost_round=int(ml_params.get("num_boost_round", 100)),
    )
    return booster


def predict_xgb_model(booster: xgb.Booster, rows_df: pd.DataFrame, feature_cols) -> np.ndarray:
    X = rows_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dtest = xgb.DMatrix(X)
    return booster.predict(dtest)


def rank_ic(preds: pd.Series, target: pd.Series) -> float:
    """Cross-sectional rank IC (Spearman) per common timestamp, averaged."""
    df = pd.DataFrame({"pred": preds, "target": target}).dropna()
    if len(df) < 10:
        return 0.0
    return float(df["pred"].rank().corr(df["target"].rank()))
