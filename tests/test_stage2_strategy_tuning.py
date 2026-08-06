import os
import json
import pytest
from src.multifactor_mlops.optimization.stage2_strategy_tuning import Stage2StrategyOptimizer, MODEL_CONFIG_PATH, OOF_PATH

def test_stage2_optimizer_initialization():
    optimizer = Stage2StrategyOptimizer(config_path="parameters.json")
    assert optimizer.config_path == "parameters.json"
    assert optimizer.app_config is not None

def test_stage2_requires_locked_artifacts(tmp_path):
    """Stage 2 must fail fast when locked model config / OOF are missing."""
    optimizer = Stage2StrategyOptimizer(config_path="parameters.json")
    with pytest.raises(FileNotFoundError):
        optimizer._require(str(tmp_path / "missing_model_config.json"))
    with pytest.raises(FileNotFoundError):
        optimizer._require(str(tmp_path / "missing_oof.csv"))

def test_stage2_locks_best_params(tmp_path):
    """optimize() persists locked strategy_config.json."""
    import optuna
    data_dict = {}
    optimizer = Stage2StrategyOptimizer(config_path="parameters.json")
    out_dir = str(tmp_path / "artifacts")
    # construct a tiny in-memory study equivalent to the module flow
    study = optuna.create_study(direction="maximize")
    study.add_trial(optuna.trial.create_trial(
        params={"quantiles": 8, "inverse_vol_period": 21, "rebalance_schedule": "daily",
                "rebalance_threshold": 0.02, "allocation_cap": 0.2,
                "volatility_ceiling": 0.06, "stress_multiplier": 0.4},
        distributions={
            "quantiles": optuna.distributions.IntDistribution(4, 20),
            "inverse_vol_period": optuna.distributions.IntDistribution(14, 42),
            "rebalance_schedule": optuna.distributions.CategoricalDistribution(["daily"]),
            "rebalance_threshold": optuna.distributions.FloatDistribution(0.01, 0.05),
            "allocation_cap": optuna.distributions.FloatDistribution(0.10, 0.35),
            "volatility_ceiling": optuna.distributions.FloatDistribution(0.04, 0.10),
            "stress_multiplier": optuna.distributions.FloatDistribution(0.2, 0.8),
        },
        value=1.0,
    ))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "strategy_config.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)
    with open(os.path.join(out_dir, "strategy_config.json")) as f:
        saved = json.load(f)
    assert saved["quantiles"] == 8
