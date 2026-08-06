"""
Stage 1 Dedicated ML Model Tuning Module (canonical, no leakage).

- Inner PURGED expanding folds on the DEVELOPMENT window ONLY
  (default 2022-01-01 -> 2023-12-31). Outer OOS (>= 2024-01-01) is NEVER touched.
- Objective: mean cross-sectional rank IC over inner folds (pure ML quality,
  NO portfolio backtest, NO Sharpe).
- Exceptions RAISE (no -999 masking).
- Best params are persisted to artifacts/model_config.json (immutable lock).

Usage:
    poetry run python src/multifactor_mlops/optimization/stage1_ml_tuning.py --trials 30
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import QMCSampler

from src.multifactor_mlops.config.loader import load_config
from src.multifactor_mlops.data.loader import load_all_data
from src.multifactor_mlops.features.panel import PanelDatasetBuilder
from src.multifactor_mlops.labels.returns import filter_train_by_label_end
from src.multifactor_mlops.optimization.folds import PurgedExpandingFoldBuilder
from src.multifactor_mlops.optimization.model_utils import train_xgb_model, predict_xgb_model, rank_ic

MODEL_CONFIG_PATH = "artifacts/model_config.json"
TRIALS_PATH = "artifacts/model_calibration_trials.json"


class PureMLStage1Optimizer:
    """
    Stage 1 ML Hyperparameter Optimizer on purged inner folds (dev window only).
    """

    def __init__(self, config_path: str = "parameters.json"):
        self.config_path = config_path
        self.app_config = load_config(config_path)

    def prepare(self, dev_end: str, inner_start: str, frequency: str, data_dir: str):
        data_dict, macro_df, funding_dict, membership_df = load_all_data(
            self.app_config, data_dir=data_dir, end_date=dev_end
        )
        builder = PanelDatasetBuilder(
            windows=self.app_config.features.windows,
            cross_sectional_rank=True,
            lag=self.app_config.label.holding_bars,
            keep_families=self.app_config.features.keep_families,
            inverted_features=self.app_config.features.inverted_features,
            use_macro_features=self.app_config.features.use_macro_features,
        )
        panel = builder.build_panel_dataset(
            data_dict=data_dict,
            symbols=list(data_dict.keys()),
            macro_df=macro_df,
            universe_membership_df=membership_df,
        )
        if panel.empty:
            raise ValueError("Stage1: empty panel.")
        feature_cols = PanelDatasetBuilder.feature_columns(panel)
        folds = PurgedExpandingFoldBuilder(
            start_date=inner_start, end_date=dev_end, frequency=frequency
        ).build_folds()
        return panel, feature_cols, folds

    def run_stage1_objective(
        self,
        trial: optuna.Trial,
        panel: pd.DataFrame,
        feature_cols,
        folds,
        base_ml: dict,
    ) -> float:
        ml_params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, step=0.01),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.8, step=0.1),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
            "num_boost_round": trial.suggest_int("num_boost_round", 50, 200, step=25),
            "random_state": base_ml.get("random_state", 42),
        }

        fold_ics = []
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
                continue
            booster = train_xgb_model(train_df, feature_cols, ml_params)
            preds = predict_xgb_model(booster, test_df, feature_cols)
            fold_ics.append(rank_ic(pd.Series(preds), test_df["target"]))

        if not fold_ics:
            raise ValueError("Stage1 trial: no valid folds produced IC.")
        return float(np.mean(fold_ics))

    def optimize(
        self,
        n_trials: int = 30,
        dev_end: str = "2023-12-31",
        inner_start: str = "2022-01-01",
        frequency: str = "quarterly",
        data_dir: str = "./data",
        output_dir: str = "artifacts",
        storage_uri: str = "sqlite:///artifacts/optuna/stage1.db",
    ) -> tuple:
        panel, feature_cols, folds = self.prepare(dev_end, inner_start, frequency, data_dir)
        base_ml = self.app_config.model.model_dump()

        if storage_uri:
            os.makedirs(os.path.dirname(storage_uri.replace("sqlite:///", "")), exist_ok=True)
        study = optuna.create_study(
            direction="maximize",
            sampler=QMCSampler(scramble=True, seed=42),
            study_name="stage1_pure_ml_tuning",
            storage=storage_uri,
            load_if_exists=True,
        )

        def objective(trial):
            return self.run_stage1_objective(trial, panel, feature_cols, folds, base_ml)

        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_params["random_state"] = base_ml["random_state"]
        best_score = study.best_value

        os.makedirs(output_dir, exist_ok=True)
        model_config_path = os.path.join(output_dir, "model_config.json")
        with open(model_config_path, "w") as f:
            json.dump(best_params, f, indent=2)

        trials = [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials if t.value is not None
        ]
        with open(os.path.join(output_dir, "model_calibration_trials.json"), "w") as f:
            json.dump(trials, f, indent=2)

        print(f"\n[Stage 1 Complete] Best mean rank IC: {best_score:.4f}")
        print(f"[Stage 1] Locked -> {model_config_path}")
        print(json.dumps(best_params, indent=2))
        return best_params, best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--dev-end", default="2023-12-31")
    parser.add_argument("--inner-start", default="2022-01-01")
    parser.add_argument("--frequency", default="quarterly")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()
    PureMLStage1Optimizer().optimize(
        n_trials=args.trials,
        dev_end=args.dev_end,
        inner_start=args.inner_start,
        frequency=args.frequency,
        data_dir=args.data_dir,
    )
