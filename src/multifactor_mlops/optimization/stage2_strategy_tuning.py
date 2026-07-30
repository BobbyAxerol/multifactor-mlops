"""
Stage 2 Strategy & Risk Overlay Tuning Module.

Locks machine learning parameters from Stage 1 (fixed_ml_params.json) and optimizes
portfolio construction, risk weighting, and macro risk overlay parameters using Mode 4 (mode_4_is_only_robust).
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import optuna

from src.multifactor_mlops.config import load_config
from src.multifactor_mlops.data.historical_adapter import HistoricalDataAdapter
from src.multifactor_mlops.backtest.quantbt_runner import QuantBTRunner

class Stage2StrategyOptimizer:
    """
    Optuna optimizer for Stage 2 strategy, position sizing, and risk overlay hyperparameters.
    """

    def __init__(self, config_path: str = "parameters.json", fixed_ml_path: str = "artifacts/models/fixed_ml_params.json"):
        self.config_path = config_path
        self.fixed_ml_path = fixed_ml_path
        self.app_config = load_config(config_path)
        self.runner = QuantBTRunner(quantbt_repo_path=self.app_config.data.quantbt_repo_path)

    def load_fixed_ml_params(self) -> Dict[str, Any]:
        """
        Loads fixed ML model parameters from Stage 1.
        """
        if os.path.exists(self.fixed_ml_path):
            with open(self.fixed_ml_path, "r") as f:
                fixed_ml = json.load(f)
                print(f"[Stage 2] Loaded Fixed ML Parameters from {self.fixed_ml_path}:", fixed_ml)
                return fixed_ml
        else:
            print(f"[Stage 2 Notice] {self.fixed_ml_path} not found. Using default fixed ML parameters.")
            return {
                "learning_rate": 0.03,
                "max_depth": 5,
                "colsample_bytree": 0.6,
                "subsample": 0.6,
                "num_boost_round": 200,
                "train_step_days": 3
            }

    def run_stage2_objective(
        self,
        trial: optuna.Trial,
        raw_dict: Dict[str, pd.DataFrame],
        base_params: Dict[str, Any],
        fixed_ml_params: Dict[str, Any]
    ) -> float:
        """
        Evaluates trial strategy parameters strictly on Mode 4 IS folds via QuantBTRunner.
        """
        quantiles = trial.suggest_int("quantiles", 4, 20, step=2)
        inverse_vol_period = trial.suggest_int("inverse_vol_period", 14, 42, step=7)
        rebalance_schedule = trial.suggest_categorical("rebalance_schedule", ["calendar_3d", "calendar_5d", "weekly_friday_exit"])
        rebalance_thresh = trial.suggest_float("rebalance_threshold", 0.01, 0.05, step=0.01)
        allocation_cap = trial.suggest_float("allocation_cap", 0.10, 0.35, step=0.05)
        stress_vix = trial.suggest_float("stress_vix_threshold", 18.0, 30.0, step=2.0)
        stress_fng = trial.suggest_float("stress_fng_threshold", 20.0, 40.0, step=5.0)
        stress_dvol = trial.suggest_float("stress_dvol_threshold", 50.0, 75.0, step=5.0)
        stress_mult = trial.suggest_float("stress_multiplier", 0.2, 0.8, step=0.1)

        trial_params = base_params.copy()
        trial_params.update(fixed_ml_params)
        trial_params.update({
            "quantiles": quantiles,
            "inverse_vol_period": inverse_vol_period,
            "rebalance_schedule": rebalance_schedule,
            "rebalance_threshold": rebalance_thresh,
            "allocation_cap": allocation_cap,
            "stress_vix_threshold": stress_vix,
            "stress_fng_threshold": stress_fng,
            "stress_dvol_threshold": stress_dvol,
            "stress_multiplier": stress_mult,
            "split_mode": "train_test_split_2024",
            "optimization_mode": "mode_4_is_only_robust",
            "scoring_backend": "endpoint",
            "backend": "native_portfolio"
        })

        try:
            from multifactor_portfolio.training.train import run_strategy_backtest
            _, equity_df, metrics = run_strategy_backtest(
                data_dict=raw_dict,
                params=trial_params,
                local_data_dir="./data"
            )

            qbt_sharpe = float(metrics.get("sharpe_ratio", 0.0))
            return qbt_sharpe
        except Exception as e:
            print(f"Notice: Stage 2 Trial {trial.number} failed with error: {e}")
            return -999.0

    def optimize(
        self,
        raw_dict: Dict[str, pd.DataFrame],
        n_trials: int = 30,
        output_dir: str = "artifacts/models"
    ) -> Tuple[Dict[str, Any], float]:
        """
        Runs Stage 2 Optuna optimization for strategy parameters using Mode 4.
        Saves best parameters to output_dir/best_strategy_params.json.
        """
        fixed_ml_params = self.load_fixed_ml_params()

        with open(self.config_path, 'r') as f:
            base_params = json.load(f)

        study = optuna.create_study(direction="maximize", study_name="stage2_strategy_tuning")

        def objective(trial):
            return self.run_stage2_objective(trial, raw_dict, base_params, fixed_ml_params)

        study.optimize(objective, n_trials=n_trials)

        best_trial = study.best_trial
        best_strategy = best_trial.params
        best_sharpe = best_trial.value

        combined_best = fixed_ml_params.copy()
        combined_best.update(best_strategy)

        os.makedirs(output_dir, exist_ok=True)
        artifact_path = os.path.join(output_dir, "best_strategy_params.json")
        with open(artifact_path, "w") as f:
            json.dump(combined_best, f, indent=2)

        print(f"\n[Stage 2 Complete] Best Strategy Parameters saved to: {artifact_path}")
        print(f"Best Mode 4 QuantBT Sharpe Ratio: {best_sharpe:.4f}")
        print("Best Strategy Parameters:", json.dumps(combined_best, indent=2))
        return combined_best, best_sharpe

if __name__ == "__main__":
    adapter = HistoricalDataAdapter()
    data = adapter.load_ohlcv_data()
    optimizer = Stage2StrategyOptimizer()
    best_params, best_sharpe = optimizer.optimize(raw_dict=data, n_trials=30)
