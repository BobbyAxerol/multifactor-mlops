import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")
import operator
import numpy as np
import pandas as pd
import optuna
import contextlib

# Add strategy path to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from multifactor_portfolio.training.train import load_ohlcv_data, run_strategy_backtest

class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    try:
        yield
    finally:
        sys.stdout = save_stdout

class EarlyStoppingCallback(object):
    def __init__(self, early_stopping_rounds: int, direction: str = "maximize") -> None:
        self.early_stopping_rounds = early_stopping_rounds
        self._iter = 0
        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            raise ValueError(f"invalid direction: {direction}")
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1
        if self._iter >= self.early_stopping_rounds:
            study.stop()

def logging_callback(study, frozen_trial, strategy_name, config_info=""):
    log_dir = "/root/bobby/pool_alpha/alphas_storage/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{strategy_name}_optuna_log.txt")
    with open(log_path, "a") as f:
        f.write(f"Trial {frozen_trial.number}: Params = {str(frozen_trial.params)}, Value = {str(frozen_trial.value)}\n")

class DuplicatePruner(optuna.pruners.BasePruner):
    def __init__(self):
        self.trial_params = set()
    
    def prune(self, study, trial):
        params = {k: v for k, v in trial.params.items()}
        params_key = frozenset(params.items())
        if params_key in self.trial_params:
            return True
        self.trial_params.add(params_key)
        return False

def optimize_parameters(
    raw_dict,
    param_ranges,
    strategy_name,
    n_trials=50,
    fixed_params=None,
    use_fixed_params=False
):
    local_data_dir = os.path.join(project_root, "data")
    
    def objective(trial):
        params = {}
        for param_name, spec in param_ranges.items():
            if use_fixed_params and fixed_params and param_name in fixed_params:
                params[param_name] = fixed_params[param_name]
                continue
            if isinstance(spec, tuple) and len(spec) == 3:
                low, high, step = spec
                if all(isinstance(x, int) for x in (low, high, step)):
                    params[param_name] = trial.suggest_int(param_name, int(low), int(high), step=int(step))
                else:
                    params[param_name] = trial.suggest_float(param_name, float(low), float(high), step=float(step))
            elif isinstance(spec, list):
                params[param_name] = trial.suggest_categorical(param_name, spec)
        
        if use_fixed_params and fixed_params:
            for k, v in fixed_params.items():
                if k not in params:
                    params[k] = v

        try:
            with nostdout():
                _, equity_df, metrics = run_strategy_backtest(
                    data_dict=raw_dict,
                    params=params,
                    local_data_dir=local_data_dir
                )
            
            qbt_sharpe = float(metrics.get("sharpe_ratio", 0.0))
            rank_ic = float(metrics.get("ml_rank_ic", 0.0))
            opt_mode = params.get('optimization_mode', 'mode_4_is_only_robust')
            
            # Mode 4 is_only_robust: QuantBT IS Sharpe + 0.5 * Rank IC - Sub-period dispersion penalty
            if opt_mode in ['mode_4_is_only_robust', 'mode_5_full_robust'] or params.get('mode_4_is_only_robust', {}).get('active'):
                rets = equity_df['return'].values
                n_chunks = 6 if opt_mode == 'mode_4_is_only_robust' else 8
                if len(rets) > 60:
                    chunks = np.array_split(rets, n_chunks)
                    sub_sharpes = []
                    for chunk in chunks:
                        if len(chunk) > 10 and np.std(chunk) > 1e-8:
                            s = (np.mean(chunk) / np.std(chunk)) * np.sqrt(365)
                        else:
                            s = -1.0
                        sub_sharpes.append(s)
                    dispersion_penalty = params.get('mode_4_is_only_robust', {}).get('dispersion_penalty', 0.5)
                    robust_score = qbt_sharpe + 0.5 * rank_ic - dispersion_penalty * np.std(sub_sharpes)
                    return float(robust_score)
            
            return qbt_sharpe
        except Exception as e:
            return -999.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=DuplicatePruner()
    )
    early_stopping = EarlyStoppingCallback(early_stopping_rounds=100)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[
            lambda s, t: logging_callback(s, t, strategy_name, f"Split Mode: {fixed_params.get('split_mode', 'full')}"),
            early_stopping
        ]
    )
    return study.best_params, study.best_value

def main():
    param_path = os.path.join(project_root, "parameters.json")
    with open(param_path) as f:
        pars = json.load(f)
        
    features_args = pars.get('features', {})
    dataset_args = pars.get('dataset', {})
    training_args = pars.get('training', {})
    
    all_params = {**features_args, **dataset_args, **training_args}
    asset_class = all_params.get("asset_class", "crypto")
    
    data_path = all_params.get("data_path", "/root/bobby/pool_alpha/alphas_storage/_get_data")
    raw_dict = load_ohlcv_data(data_path)
    if not raw_dict:
        print("Failed to load data.")
        sys.exit(1)
        
    # Multi-dimensional Search Space (Group 1 Narrow Model + Group 2, 3, 4 Structural Params)
    param_ranges = {
        # Group 1: Model Hyperparameters (Narrow anti-overfit range: depth 2-4)
        "learning_rate": (0.01, 0.08, 0.01),
        "max_depth": (2, 4, 1),
        "num_boost_round": (40, 120, 10),
        "colsample_bytree": (0.2, 0.5, 0.1),

        # Group 2: Data & Universe Engine (Expanded Universe up to 80 coins)
        "top_n_symbols": [30, 40, 50, 60, 80],
        "quantiles": (10, 40, 5),
        "train_step_days": [1, 2, 3, 5],

        # Group 3: Portfolio Risk & Capital Allocation
        "inverse_vol_period": [60, 90, 120, 180],
        "allocation_cap": (0.10, 0.35, 0.05),

        # Group 4: Macro Regime Protection Overlay
        "stress_vix_threshold": (18.0, 26.0, 2.0),
        "stress_fng_threshold": (20.0, 35.0, 5.0),
        "stress_dvol_threshold": (55.0, 70.0, 5.0),
        "stress_multiplier": (0.3, 0.7, 0.1)
    }
    
    fixed = {
        "asset_class": asset_class,
        "universe_name": all_params.get("universe_name", "binance_daily"),
        "fee": all_params.get("fee", 0.0005),
        "slippage": all_params.get("slippage", 0.0001),
        "leverage": all_params.get("leverage", 3.0),
        "lag": all_params.get("lag", 1),
        "trading_days_per_year": all_params.get("trading_days_per_year", 365),
        "split_mode": all_params.get("split_mode", "train_test_split_2024"),
        "scoring_backend": "endpoint",
        "optimization_mode": "mode_4_is_only_robust"
    }
    
    print("Starting Optuna optimization...")
    n_trials = pars.get('optimization', {}).get('n_trials', 20)
    best_params, best_sharpe = optimize_parameters(
        raw_dict=raw_dict,
        param_ranges=param_ranges,
        strategy_name="multifactor_portfolio",
        n_trials=n_trials,
        fixed_params=fixed,
        use_fixed_params=True
    )
    
    print(f"Best Params found: {best_params} with Sharpe: {best_sharpe}")
    
    # Save best parameters back to json
    for k, v in best_params.items():
        if k in pars["features"]:
            pars["features"][k] = v
        elif k in pars["dataset"]:
            pars["dataset"][k] = v
        elif k in pars["training"]:
            pars["training"][k] = v
            
    with open(param_path, "w") as f:
        json.dump(pars, f, indent=2)
    print("Successfully updated parameters.json with optimized parameters.")

if __name__ == "__main__":
    main()
