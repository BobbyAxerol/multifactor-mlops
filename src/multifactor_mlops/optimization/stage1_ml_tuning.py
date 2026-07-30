"""
Stage 1 Dedicated ML Model Tuning Module.

Evaluates Machine Learning hyperparameters (learning_rate, max_depth, colsample_bytree,
subsample, num_boost_round, train_step_days) strictly on ML statistical predictive quality
(Purged Cross-Sectional Rank IC & IC-IR) across native QuantBT walk-forward folds.

ZERO portfolio backtesting or trade execution is performed in Stage 1.
Uses QMCSampler (Sobol sequence) to avoid TPE local minima traps.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import optuna

from src.multifactor_mlops.config import load_config
from src.multifactor_mlops.data.historical_adapter import HistoricalDataAdapter

class PureMLStage1Optimizer:
    """
    Stage 1 ML Hyperparameter Optimizer relying 100% on QuantBT native walk-forward folds
    and pure ML predictive quality without portfolio backtesters or trade simulation.
    """

    def __init__(self, config_path: str = "parameters.json"):
        self.config_path = config_path
        self.app_config = load_config(config_path)

    def run_stage1_objective(
        self,
        trial: optuna.Trial,
        raw_dict: Dict[str, pd.DataFrame],
        all_dates: pd.Index,
        base_params: Dict[str, Any]
    ) -> float:
        """
        Evaluates trial ML parameters across QuantBT walk-forward folds purely on Rank IC & IC-IR.
        """
        lr = trial.suggest_float("learning_rate", 0.01, 0.10, step=0.01)
        max_depth = trial.suggest_int("max_depth", 2, 6)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 0.8, step=0.1)
        subsample = trial.suggest_float("subsample", 0.5, 1.0, step=0.1)
        num_boost_round = trial.suggest_int("num_boost_round", 50, 200, step=25)
        train_step_days = trial.suggest_int("train_step_days", 1, 5)

        trial_params = base_params.copy()
        trial_params.update({
            "learning_rate": lr,
            "max_depth": max_depth,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            "num_boost_round": num_boost_round,
            "train_step_days": train_step_days,
            "split_mode": "walk_forward_2023"
        })

        try:
            from multifactor_portfolio.training.train import generate_walk_forward_target_weights
            _, _, _, _, ml_metrics = generate_walk_forward_target_weights(
                data_dict=raw_dict,
                params=trial_params,
                local_data_dir="./data",
                all_dates=all_dates
            )

            rank_ic = float(ml_metrics.get("ml_rank_ic", 0.0))
            ic_ir = float(ml_metrics.get("ml_ic_ir", 0.0))
            accuracy = float(ml_metrics.get("ml_accuracy", 0.5))

            # Stage 1 Objective Score: Rank IC + 0.5 * IC-IR + (Accuracy - 0.5)
            score = rank_ic + 0.5 * ic_ir + (accuracy - 0.5)
            return score
        except Exception as e:
            print(f"Notice: Stage 1 Trial {trial.number} failed with error: {e}")
            return -999.0

    def optimize(
        self,
        raw_dict: Dict[str, pd.DataFrame],
        n_trials: int = 30,
        output_dir: str = "artifacts/models"
    ) -> Tuple[Dict[str, Any], float]:
        """
        Runs Stage 1 ML optimization using QuantBT native walk-forward folds and QMCSampler.
        Saves immutable artifact artifacts/models/fixed_ml_params.json.
        """
        all_dates = pd.Index([])
        for df in raw_dict.values():
            all_dates = all_dates.union(df.index)
        all_dates = pd.DatetimeIndex(sorted(all_dates))

        with open(self.config_path, 'r') as f:
            base_params = json.load(f)

        print(f"[Stage 1 ML Tuning] Starting Optuna Search with QMC Sampler over QuantBT Walk-Forward Folds...")

        # Quasi-Monte Carlo (Sobol) Sampler avoids TPE local minima traps
        sampler = optuna.samplers.QMCSampler(scramble=True, seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler, study_name="stage1_pure_ml_tuning")

        def objective(trial):
            return self.run_stage1_objective(trial, raw_dict, all_dates, base_params)

        study.optimize(objective, n_trials=n_trials)

        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value

        os.makedirs(output_dir, exist_ok=True)
        artifact_path = os.path.join(output_dir, "fixed_ml_params.json")
        with open(artifact_path, "w") as f:
            json.dump(best_params, f, indent=2)

        print(f"\n[Stage 1 Pure ML Complete] Best ML Hyperparameters saved to: {artifact_path}")
        print(f"Best Stage 1 Score: {best_score:.4f}")
        print("Best ML Hyperparameters:", json.dumps(best_params, indent=2))
        return best_params, best_score

if __name__ == "__main__":
    adapter = HistoricalDataAdapter()
    data = adapter.load_ohlcv_data()
    optimizer = PureMLStage1Optimizer()
    best_ml_params, best_score = optimizer.optimize(raw_dict=data, n_trials=30)
