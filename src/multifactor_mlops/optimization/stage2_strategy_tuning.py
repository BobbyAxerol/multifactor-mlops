"""
Stage 2 Strategy & Risk Overlay Tuning Module (canonical, no leakage).

- Locks Stage 1 ML params (artifacts/model_config.json) and OOF predictions
  (artifacts/oof_predictions.csv) -> NO model refitting during tuning.
- Optimizes portfolio construction / risk / overlay params by running the
  QuantBT native walk-forward backtest on the DEVELOPMENT window ONLY
  (2022 -> 2023-12-31) with pre-computed OOF predictions.
- Objective: robust score = median(fold Sharpe) - 0.5 * std(fold Sharpe),
  where per-fold Sharpe is computed from the QuantBT stitched equity curve
  sliced to each fold's OOS window (QuantBT is the only PnL source).
- Exceptions RAISE (no -999 masking). Best params -> artifacts/strategy_config.json.

Usage:
    poetry run python src/multifactor_mlops/optimization/stage2_strategy_tuning.py --trials 15
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import optuna

from src.multifactor_mlops.config.loader import load_config
from src.multifactor_mlops.data.loader import load_all_data
from src.multifactor_mlops.backtest.wf_runner import WalkForwardQuantBTRunner, extract_equity

MODEL_CONFIG_PATH = "artifacts/model_config.json"
OOF_PATH = "artifacts/oof_predictions.csv"
STRATEGY_CONFIG_PATH = "artifacts/strategy_config.json"
STRATEGY_TRIALS_PATH = "artifacts/strategy_trials.json"


def _fold_sharpes(equity_df: pd.DataFrame, folds_meta) -> list:
    """Per-fold Sharpe from the QuantBT equity curve sliced to fold test windows."""
    out = []
    eq = equity_df.set_index("time")
    if isinstance(folds_meta, pd.DataFrame):
        folds_meta = folds_meta.to_dict("records")
    for f in folds_meta:
        t_start = pd.Timestamp(f.get("test_start") or f.get("start"))
        t_end = pd.Timestamp(f.get("test_end") or f.get("end"))
        if t_start.tz is not None:
            t_start = t_start.tz_localize(None)
        if t_end.tz is not None:
            t_end = t_end.tz_localize(None)
        seg = eq[(eq.index >= t_start) & (eq.index < t_end)]
        if len(seg) < 10:
            continue
        rets = seg["return"].dropna()
        if rets.std() == 0 or len(rets) < 10:
            continue
        out.append(float(rets.mean() / rets.std() * np.sqrt(365)))
    return out


class Stage2StrategyOptimizer:
    def __init__(self, config_path: str = "parameters.json"):
        self.config_path = config_path
        self.app_config = load_config(config_path)
        self.runner = WalkForwardQuantBTRunner(quantbt_repo_path=self.app_config.data.quantbt_repo_path)

    def _require(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required locked artifact {path}. Fail-fast, no silent defaults.")

    def prepare(self, dev_end: str, inner_start: str, data_dir: str):
        self._require(MODEL_CONFIG_PATH)
        self._require(OOF_PATH)
        with open(MODEL_CONFIG_PATH, "r") as f:
            self.fixed_ml = json.load(f)

        data_dict, macro_df, funding_dict, membership_df = load_all_data(
            self.app_config, data_dir=data_dir, end_date=dev_end
        )
        oof = pd.read_csv(OOF_PATH, parse_dates=["Time"])
        oof_wide = oof.pivot_table(index="Time", columns="Symbol", values="pred").sort_index()
        return data_dict, macro_df, funding_dict, oof_wide, membership_df

    def run_stage2_objective(self, trial: optuna.Trial, data_dict, macro_df, funding_dict, oof_wide, membership_df) -> float:
        strategy_params = {
            "quantiles": trial.suggest_int("quantiles", 4, 20, step=2),
            "inverse_vol_period": trial.suggest_int("inverse_vol_period", 14, 42, step=7),
            "rebalance_schedule": trial.suggest_categorical(
                "rebalance_schedule", ["monday_decide_weekly", "daily", "calendar_3d", "weekly_friday_exit"]
            ),
            "rebalance_threshold": trial.suggest_float("rebalance_threshold", 0.01, 0.05, step=0.01),
            "allocation_cap": trial.suggest_float("allocation_cap", 0.10, 0.35, step=0.05),
            "volatility_ceiling": trial.suggest_float("volatility_ceiling", 0.04, 0.10, step=0.01),
            "stress_multiplier": trial.suggest_float("stress_multiplier", 0.2, 0.8, step=0.1),
        }
        params = {**self.fixed_ml, **strategy_params}
        funding_rate = {s: funding_dict[s] for s in data_dict if s in funding_dict} or 0.0
        funding_wide = pd.DataFrame(funding_dict) if funding_dict else None

        qbt_res = self.runner.run(
            data_dict=data_dict,
            symbols=list(data_dict.keys()),
            app_config=self.app_config,
            params=params,
            macro_df=macro_df,
            funding_rate=funding_rate,
            funding_wide=funding_wide,
            universe_membership_df=membership_df,
            split_mode="walk_forward_2022",
            split_frequency="quarterly",
            window_mode="expanding",
            predictions_cache=oof_wide,
        )
        equity_df = extract_equity(qbt_res, initial_capital=self.app_config.backtest.initial_capital)
        wf_meta = qbt_res.metadata.get("walk_forward", {}) if hasattr(qbt_res, "metadata") else {}
        folds_meta = wf_meta.get("fold_table", [])
        fold_sharpes = _fold_sharpes(equity_df, folds_meta)
        if not fold_sharpes:
            raise ValueError("Stage2 trial: no fold Sharpe computed.")
        return float(np.median(fold_sharpes) - 0.5 * np.std(fold_sharpes))

    def optimize(
        self,
        n_trials: int = 20,
        dev_end: str = "2023-12-31",
        inner_start: str = "2022-01-01",
        data_dir: str = "./data",
        output_dir: str = "artifacts",
        storage_uri: str = "sqlite:///artifacts/optuna/stage2.db",
    ) -> tuple:
        data_dict, macro_df, funding_dict, oof_wide, membership_df = self.prepare(dev_end, inner_start, data_dir)

        if storage_uri:
            os.makedirs(os.path.dirname(storage_uri.replace("sqlite:///", "")), exist_ok=True)
        study = optuna.create_study(
            direction="maximize",
            study_name="stage2_strategy_tuning",
            storage=storage_uri,
            load_if_exists=True,
        )

        def objective(trial):
            return self.run_stage2_objective(trial, data_dict, macro_df, funding_dict, oof_wide, membership_df)

        study.optimize(objective, n_trials=n_trials)

        best_strategy = study.best_params
        combined = {**self.fixed_ml, **best_strategy}

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "strategy_config.json"), "w") as f:
            json.dump(combined, f, indent=2)
        trials = [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials if t.value is not None
        ]
        with open(os.path.join(output_dir, "strategy_trials.json"), "w") as f:
            json.dump(trials, f, indent=2)

        print(f"\n[Stage 2 Complete] Best robust score: {study.best_value:.4f}")
        print(f"[Stage 2] Locked -> {STRATEGY_CONFIG_PATH}")
        print(json.dumps(combined, indent=2))
        return combined, study.best_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--dev-end", default="2023-12-31")
    parser.add_argument("--inner-start", default="2022-01-01")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()
    Stage2StrategyOptimizer().optimize(
        n_trials=args.trials,
        dev_end=args.dev_end,
        inner_start=args.inner_start,
        data_dir=args.data_dir,
    )
