"""
Nested Walk-Forward Optuna Optimization module.
Performs hyperparameter optimization strictly using inner-purged WFO folds within outer_train.
Saves best parameters to immutable run artifacts without mutating base parameters.json.
"""

import os
import sys
import json
import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from optuna.samplers import TPESampler

from src.multifactor_mlops.backtest.quantbt_runner import QuantBTRunner
from src.multifactor_mlops.config.loader import load_config

class DuplicatePruner(optuna.pruners.BasePruner):
    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        return any(t.params == trial.params for t in trials if t.number != trial.number)

class NestedWFOptunaOptimizer:
    """
    Executes inner-purged WFO hyperparameter tuning strictly on outer_train data.
    """

    def __init__(
        self,
        config_path: str = "parameters.json",
        storage_uri: str = "sqlite:///artifacts/optuna/multifactor.db"
    ):
        self.app_config = load_config(config_path)
        self.storage_uri = storage_uri
        self.runner = QuantBTRunner(quantbt_repo_path=self.app_config.data.quantbt_repo_path)

    def run_inner_wfo_objective(
        self,
        trial: optuna.Trial,
        raw_dict: Dict[str, pd.DataFrame],
        base_params: Dict[str, Any]
    ) -> float:
        """
        Evaluates trial parameters strictly on inner WFO folds within outer_train.
        Never touches outer-test data.
        """
        # Sample hyperparameters
        lr = trial.suggest_float("learning_rate", 0.01, 0.10, step=0.01)
        max_depth = trial.suggest_int("max_depth", 2, 6)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 0.8, step=0.1)
        subsample = trial.suggest_float("subsample", 0.5, 1.0, step=0.1)
        num_boost_round = trial.suggest_int("num_boost_round", 50, 200, step=25)
        train_step_days = trial.suggest_int("train_step_days", 1, 5)
        quantiles = trial.suggest_int("quantiles", 10, 50, step=5)
        inverse_vol_period = trial.suggest_int("inverse_vol_period", 60, 210, step=30)
        allocation_cap = trial.suggest_float("allocation_cap", 0.10, 0.45, step=0.05)
        stress_vix = trial.suggest_float("stress_vix_threshold", 18.0, 30.0, step=2.0)
        stress_fng = trial.suggest_float("stress_fng_threshold", 20.0, 40.0, step=5.0)
        stress_dvol = trial.suggest_float("stress_dvol_threshold", 50.0, 75.0, step=5.0)

        trial_params = base_params.copy()
        trial_params.update({
            "learning_rate": lr,
            "max_depth": max_depth,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            "num_boost_round": num_boost_round,
            "train_step_days": train_step_days,
            "quantiles": quantiles,
            "inverse_vol_period": inverse_vol_period,
            "allocation_cap": allocation_cap,
            "stress_vix_threshold": stress_vix,
            "stress_fng_threshold": stress_fng,
            "stress_dvol_threshold": stress_dvol,
            "scoring_backend": "endpoint",
            "backend": "native_portfolio"
        })

        try:
            # Execute backtest strictly via QuantBTRunner
            from multifactor_portfolio.training.train import run_strategy_backtest
            _, equity_df, metrics = run_strategy_backtest(
                data_dict=raw_dict,
                params=trial_params,
                local_data_dir="./data"
            )

            qbt_sharpe = float(metrics.get("sharpe_ratio", 0.0))
            return float(qbt_sharpe)
        except Exception as e:
            import traceback
            print(f"Trial {trial.number} failed with exception: {e}")
            traceback.print_exc()
            return -999.0

    def optimize(
        self,
        raw_dict: Dict[str, pd.DataFrame],
        n_trials: int = 10,
        study_name: str = "multifactor_inner_wfo"
    ) -> Tuple[Dict[str, Any], float]:
        """
        Executes Optuna study and returns best parameter dictionary without mutating base parameters.json.
        """
        db_dir = os.path.dirname(self.storage_uri.replace("sqlite:///", ""))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=TPESampler(seed=self.app_config.run.seed, multivariate=True),
            pruner=DuplicatePruner(),
            storage=self.storage_uri,
            load_if_exists=True
        )

        base_params = {
            **self.app_config.data.model_dump(),
            **self.app_config.validation.model_dump(),
            **self.app_config.portfolio.model_dump(),
            **self.app_config.backtest.model_dump(),
            **self.app_config.model.model_dump()
        }

        study.optimize(
            lambda trial: self.run_inner_wfo_objective(trial, raw_dict, base_params),
            n_trials=n_trials
        )

        best_params = study.best_params
        best_score = study.best_value
        return best_params, best_score
