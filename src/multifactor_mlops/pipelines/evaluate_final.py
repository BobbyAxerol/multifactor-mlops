"""
Final Single-Touch Outer OOS Evaluation (canonical).

Requires locked artifacts: artifacts/model_config.json + artifacts/strategy_config.json.
Runs the QuantBT native walk-forward backtest from outer OOS start (default
2024-01-01, quarterly, expanding) on the FULL dataset, with per-fold model
training inside the strategy adapter. The outer OOS window is touched EXACTLY
ONCE here -- it is excluded from Stage 1 and Stage 2 tuning.

Outputs:
  artifacts/final_oos_metrics.json
  artifacts/final_oos_report.md
"""

import os
import json
import argparse

import numpy as np
import pandas as pd

from src.multifactor_mlops.config.loader import load_config
from src.multifactor_mlops.data.loader import load_all_data
from src.multifactor_mlops.backtest.wf_runner import (
    WalkForwardQuantBTRunner,
    extract_metrics,
    extract_equity,
    QuantBTExecutionError,
)
from src.multifactor_mlops.tracking.mlflow_logger import MLOpsRunLogger

MODEL_CONFIG_PATH = "artifacts/model_config.json"
STRATEGY_CONFIG_PATH = "artifacts/strategy_config.json"
OOS_METRICS_PATH = "artifacts/final_oos_metrics.json"
OOS_REPORT_PATH = "artifacts/final_oos_report.md"


def _per_fold_sharpes(equity_df: pd.DataFrame, folds_meta) -> list:
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
        rets = seg["return"].dropna()
        if len(rets) < 10 or rets.std() == 0:
            continue
        out.append({
            "test_start": str(t_start.date()),
            "test_end": str(t_end.date()),
            "sharpe": round(float(rets.mean() / rets.std() * np.sqrt(365)), 4),
            "bars": int(len(rets)),
        })
    return out


def evaluate_final_oos(
    config_path: str = "parameters.json",
    oos_start: str = "2024-01-01",
    split_frequency: str = "quarterly",
    data_dir: str = "./data",
    output_dir: str = "artifacts",
    log_mlflow: bool = True,
):
    for p in (MODEL_CONFIG_PATH, STRATEGY_CONFIG_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing locked artifact {p}. Fail-fast, no silent defaults.")

    with open(MODEL_CONFIG_PATH, "r") as f:
        model_config = json.load(f)
    with open(STRATEGY_CONFIG_PATH, "r") as f:
        strategy_config = json.load(f)

    app_config = load_config(config_path)
    data_dict, macro_df, funding_dict = load_all_data(app_config, data_dir=data_dir)
    funding_rate = {s: funding_dict[s] for s in data_dict if s in funding_dict} or 0.0
    funding_wide = pd.DataFrame(funding_dict) if funding_dict else None

    params = {**model_config, **strategy_config}
    runner = WalkForwardQuantBTRunner(quantbt_repo_path=app_config.data.quantbt_repo_path)

    qbt_res = runner.run(
        data_dict=data_dict,
        symbols=list(data_dict.keys()),
        app_config=app_config,
        params=params,
        macro_df=macro_df,
        funding_rate=funding_rate,
        funding_wide=funding_wide,
        split_mode=f"walk_forward_{oos_start[:4]}",
        split_frequency=split_frequency,
        window_mode="expanding",
    )

    metrics = extract_metrics(qbt_res, trading_days=app_config.backtest.trading_days_per_year)
    equity_df = extract_equity(qbt_res, initial_capital=app_config.backtest.initial_capital)

    wf_meta = qbt_res.metadata.get("walk_forward", {}) if hasattr(qbt_res, "metadata") else {}
    fold_sharpes = _per_fold_sharpes(equity_df, wf_meta.get("fold_table", []))
    oos_equity = equity_df[equity_df["time"] >= pd.Timestamp(oos_start)]

    report = {
        "oos_start": oos_start,
        "split_frequency": split_frequency,
        "window_mode": "expanding",
        "n_folds": len(wf_meta.get("fold_table", [])),
        "quantbt_metrics": {k: round(float(v), 6) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
        "per_fold_sharpe": fold_sharpes,
        "median_fold_sharpe": round(float(np.median([f["sharpe"] for f in fold_sharpes])), 4) if fold_sharpes else None,
        "oos_equity_start": float(oos_equity["equity"].iloc[0]) if len(oos_equity) else None,
        "oos_equity_end": float(oos_equity["equity"].iloc[-1]) if len(oos_equity) else None,
        "oos_bars": int(len(oos_equity)),
        "params": params,
        "config_snapshot": app_config.model_dump(),
        "git_sha": os.popen("git rev-parse --short HEAD").read().strip() if os.path.isdir(".git") else None,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "final_oos_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# Final Outer OOS Report (single-touch)",
        "",
        f"- OOS window: {oos_start} -> present, split_frequency={split_frequency}, expanding",
        f"- Folds: {report['n_folds']}",
        f"- QuantBT metrics (trading_days=365):",
    ]
    for k, v in report["quantbt_metrics"].items():
        md_lines.append(f"  - {k}: {v}")
    md_lines.append("")
    md_lines.append("Per-fold Sharpe (QuantBT equity, OOS slices):")
    md_lines.append("")
    md_lines.append("| fold start | fold end | bars | sharpe |")
    md_lines.append("|---|---|---|---|")
    for f in fold_sharpes:
        md_lines.append(f"| {f['test_start']} | {f['test_end']} | {f['bars']} | {f['sharpe']} |")
    md_lines.append("")
    md_lines.append("Params: " + json.dumps(params))
    with open(os.path.join(output_dir, "final_oos_report.md"), "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n[Final OOS] Report -> {os.path.join(output_dir, 'final_oos_report.md')}")
    print(json.dumps(report["quantbt_metrics"], indent=2))

    if log_mlflow:
        try:
            import mlflow
            from dotenv import load_dotenv
            load_dotenv()
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
            mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "multifactor_portfolio"))
            with mlflow.start_run(run_name=f"v4_oos_{oos_start}") as run:
                for k, v in report["quantbt_metrics"].items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, float(v))
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_artifact(os.path.join(output_dir, "final_oos_metrics.json"))
                mlflow.log_artifact(os.path.join(output_dir, "final_oos_report.md"))
                print(f"[MLflow] run_id={run.info.run_id}")
        except Exception as e:
            print(f"[MLflow] logging skipped: {e}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oos-start", default="2024-01-01")
    parser.add_argument("--frequency", default="quarterly")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    evaluate_final_oos(
        oos_start=args.oos_start,
        split_frequency=args.frequency,
        data_dir=args.data_dir,
        log_mlflow=not args.no_mlflow,
    )
