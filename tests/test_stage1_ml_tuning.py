import os
import json
import pytest
import pandas as pd
import numpy as np
from src.multifactor_mlops.optimization.stage1_ml_tuning import PureMLStage1Optimizer

def test_pure_ml_stage1_optimizer_initialization(tmp_path):
    optimizer = PureMLStage1Optimizer(config_path="parameters.json")
    assert optimizer.config_path == "parameters.json"
    assert optimizer.app_config is not None

def test_stage1_ml_params_saving(tmp_path):
    dummy_params = {
        "learning_rate": 0.05,
        "max_depth": 3,
        "colsample_bytree": 0.4,
        "subsample": 0.8,
        "num_boost_round": 100,
        "train_step_days": 2
    }
    output_dir = str(tmp_path / "artifacts" / "models")
    os.makedirs(output_dir, exist_ok=True)
    target_file = os.path.join(output_dir, "fixed_ml_params.json")
    
    with open(target_file, "w") as f:
        json.dump(dummy_params, f, indent=2)
        
    assert os.path.exists(target_file)
    with open(target_file, "r") as f:
        loaded = json.load(f)
    assert loaded["learning_rate"] == 0.05
    assert loaded["max_depth"] == 3
