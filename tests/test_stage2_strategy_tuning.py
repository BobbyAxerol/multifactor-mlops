import os
import json
import pytest
from src.multifactor_mlops.optimization.stage2_strategy_tuning import Stage2StrategyOptimizer

def test_stage2_optimizer_initialization():
    optimizer = Stage2StrategyOptimizer(config_path="parameters.json")
    assert optimizer.config_path == "parameters.json"
    assert optimizer.app_config is not None

def test_load_fixed_ml_params(tmp_path):
    fixed_ml_file = str(tmp_path / "fixed_ml_params.json")
    dummy_fixed = {
        "learning_rate": 0.03,
        "max_depth": 5,
        "colsample_bytree": 0.6,
        "subsample": 0.6,
        "num_boost_round": 200,
        "train_step_days": 3
    }
    with open(fixed_ml_file, "w") as f:
        json.dump(dummy_fixed, f)

    optimizer = Stage2StrategyOptimizer(config_path="parameters.json", fixed_ml_path=fixed_ml_file)
    loaded = optimizer.load_fixed_ml_params()
    assert loaded["learning_rate"] == 0.03
    assert loaded["max_depth"] == 5
    assert loaded["train_step_days"] == 3
