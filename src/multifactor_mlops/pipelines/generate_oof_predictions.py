"""
Generate Out-Of-Fold (OOF) predictions on the DEVELOPMENT window.

For each purged fold in [start_date, end_date):
  - train one XGBoost model with the LOCKED artifacts/model_config.json on
    rows with label_end_time < fold.test_start (canonical purge),
  - predict the fold test segment.

Outputs:
  artifacts/oof_predictions.csv   (Time, Symbol, pred, target, fold_id)
  artifacts/fold_metrics.json     (rank IC per fold + overall)
"""

import os
import json
import argparse

import numpy as np
import pandas as pd

from src.multifactor_mlops.config.loader import load_config
from src.multifactor_mlops.data.loader import load_all_data
from src.multifactor_mlops.features.panel import PanelDatasetBuilder
from src.multifactor_mlops.labels.returns import filter_train_by_label_end
from src.multifactor_mlops.optimization.folds import PurgedExpandingFoldBuilder
from src.multifactor_mlops.optimization.model_utils import train_xgb_model, predict_xgb_model, rank_ic

MODEL_CONFIG_PATH = "artifacts/model_config.json"


def generate_oof_predictions(
    config_path: str = "parameters.json",
    model_config_path: str = MODEL_CONFIG_PATH,
    dev_end: str = "2023-12-31",
    inner_start: str = "2022-01-01",
    frequency: str = "quarterly",
    data_dir: str = "./data",
    output_dir: str = "artifacts",
):
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(
            f"Missing locked {model_config_path}. Run Stage 1 ML tuning first. Fail-fast, no silent defaults."
        )
    with open(model_config_path, "r") as f:
        ml_params = json.load(f)

    app_config = load_config(config_path)
    data_dict, macro_df, funding_dict = load_all_data(
        app_config, data_dir=data_dir, end_date=dev_end
    )

    builder = PanelDatasetBuilder(
        windows=app_config.features.windows,
        cross_sectional_rank=True,
        lag=app_config.label.holding_bars,
    )
    panel = builder.build_panel_dataset(
        data_dict=data_dict,
        symbols=list(data_dict.keys()),
        macro_df=macro_df,
        funding_df=pd.DataFrame(funding_dict) if funding_dict else None,
    )
    if panel.empty:
        raise ValueError("generate_oof_predictions: empty panel.")
    feature_cols = PanelDatasetBuilder.feature_columns(panel)
    times = pd.DatetimeIndex(panel.index.get_level_values("Time").unique()).sort_values()

    folds = PurgedExpandingFoldBuilder(
        start_date=inner_start, end_date=dev_end, frequency=frequency
    ).build_folds()

    records = []
    fold_metrics = []
    for fold in folds:
        test_start = fold["test_start"]
        test_end = fold["test_end"]
        train_df = filter_train_by_label_end(panel, test_start)
        test_mask = (
            (panel.index.get_level_values("Time") >= test_start)
            & (panel.index.get_level_values("Time") < test_end)
        )
        test_df = panel[test_mask]
        if train_df.empty or test_df.empty:
            print(f"[OOF] Skip fold {fold['fold_id']} (empty train/test)")
            continue

        booster = train_xgb_model(train_df, feature_cols, ml_params)
        preds = predict_xgb_model(booster, test_df, feature_cols)
        fold_df = test_df[["target"]].copy()
        fold_df["pred"] = preds
        fold_df["fold_id"] = fold["fold_id"]
        fold_ic = rank_ic(fold_df["pred"], fold_df["target"])
        fold_metrics.append({"fold_id": fold["fold_id"], "rank_ic": fold_ic, "rows": len(fold_df)})
        records.append(fold_df.reset_index())

    if not records:
        raise ValueError("generate_oof_predictions: no OOF records produced.")
    oof_df = pd.concat(records, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    oof_path = os.path.join(output_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)

    overall_ic = rank_ic(oof_df["pred"], oof_df["target"])
    metrics = {
        "folds": fold_metrics,
        "overall_rank_ic": overall_ic,
        "model_config": ml_params,
        "dev_window": [inner_start, dev_end],
    }
    metrics_path = os.path.join(output_dir, "fold_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OOF] Saved {len(oof_df)} rows -> {oof_path}")
    print(f"[OOF] Overall rank IC: {overall_ic:.4f}")
    return oof_df, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-end", default="2023-12-31")
    parser.add_argument("--inner-start", default="2022-01-01")
    parser.add_argument("--frequency", default="quarterly")
    parser.add_argument("--model-config", default=MODEL_CONFIG_PATH)
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()
    generate_oof_predictions(
        model_config_path=args.model_config,
        dev_end=args.dev_end,
        inner_start=args.inner_start,
        frequency=args.frequency,
        data_dir=args.data_dir,
    )
