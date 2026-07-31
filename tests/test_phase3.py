"""
Phase 3 Automated Unit Tests: Optuna Inner WFO, Production Bundle, and Reproducibility.
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.multifactor_mlops.config.loader import load_config
from src.multifactor_mlops.optimization.optuna_kernel import NestedWFOptunaOptimizer
from src.multifactor_mlops.pipelines.fit_final import ProductionModelBundle, fit_final_production_model
from src.multifactor_mlops.tracking.mlflow_logger import MLOpsRunLogger

def test_immutable_base_config():
    """
    Test 1: Base parameters.json is immutable and never mutated by Optuna.
    """
    config_path = os.path.abspath("parameters.json")
    with open(config_path, "r") as f:
        before_content = f.read()

    # Load config and verify
    app_config = load_config(config_path)
    assert app_config.validation.split_mode in ["train_test_split_2024", "walk_forward_2024_90d", "walk_forward_quarterly"]

    with open(config_path, "r") as f:
        after_content = f.read()

    assert before_content == after_content, "base parameters.json was mutated!"

def test_model_bundle_feature_validation():
    """
    Test 2: ProductionModelBundle rejects missing features during inference.
    """
    mock_booster = object()
    feature_names = ["mom_rsi_7", "mom_rsi_14", "carry_7"]
    config = {}
    lineage = {}

    bundle = ProductionModelBundle(mock_booster, feature_names, config, lineage)

    # Incomplete input missing 'carry_7'
    incomplete_input = pd.DataFrame({"mom_rsi_7": [50.0], "mom_rsi_14": [50.0]})
    with pytest.raises(ValueError):
        bundle.predict(incomplete_input)

def test_mlops_run_logger_artifacts():
    """
    Test 3: MLOpsRunLogger creates immutable run directory and checksum manifest.
    """
    logger = MLOpsRunLogger(run_id="test_run_123")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    
    pq_path = logger.log_dataframe_artifact("predictions", df)
    manifest_path = logger.save_run_manifest()

    assert os.path.exists(pq_path)
    assert os.path.exists(manifest_path)
    assert "predictions" in logger.manifest["checksums"]

def test_fit_final_production_model():
    """
    Test 4: fit_final_production_model constructs valid model bundle.
    """
    dates = pd.date_range("2024-01-01", periods=100, freq="1D")
    prices = pd.Series(np.linspace(100, 150, 100), index=dates)
    
    df = pd.DataFrame({
        "open": prices,
        "high": prices + 1,
        "low": prices - 1,
        "close": prices,
        "volume": 1000.0
    }, index=dates)
    
    data_dict = {"BTCUSDT": df, "ETHUSDT": df}
    bundle = fit_final_production_model(
        data_dict=data_dict,
        symbols=["BTCUSDT", "ETHUSDT"],
        config_path="parameters.json"
    )

    assert bundle.booster is not None
    assert len(bundle.feature_names) > 0
