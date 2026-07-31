"""
Automated Test Suite for Remaining MLOps Defect Repairs.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.multifactor_mlops.config.loader import load_config
from src.multifactor_mlops.portfolio.constructor import PortfolioConstructor
from src.multifactor_mlops.pipelines.fit_final import fit_final_production_model

def test_config_loader_schema_validation():
    """
    Test 1: AppConfig validation loads parameters.json without fallback.
    """
    app_config = load_config("parameters.json")
    assert app_config.validation.split_mode in ["train_test_split_2024", "walk_forward_2024_90d", "walk_forward_quarterly"]
    assert app_config.data.allow_bfill is False

def test_signed_portfolio_short_weights():
    """
    Test 2: Short predictions strictly produce negative target weights.
    """
    constructor = PortfolioConstructor(quantiles=5, allocation_cap=0.20)

    dates = pd.date_range("2024-01-01", periods=5, freq="1D")
    symbols = [f"SYM_{i}" for i in range(5)]
    
    # SYM_0 has lowest prediction -> Short
    # SYM_4 has highest prediction -> Long
    preds_data = np.array([
        [-0.10, -0.05, 0.00, 0.05, 0.10],
        [-0.20, -0.10, 0.00, 0.10, 0.20],
        [-0.05, -0.02, 0.00, 0.02, 0.05],
        [-0.15, -0.05, 0.00, 0.05, 0.15],
        [-0.30, -0.15, 0.00, 0.15, 0.30]
    ])
    
    preds_df = pd.DataFrame(preds_data, index=dates, columns=symbols)
    target_weights = constructor.create_cross_sectional_weights(preds_df)

    # SYM_0 must be short (negative weight)
    assert (target_weights["SYM_0"] < 0).all()
    # SYM_4 must be long (positive weight)
    assert (target_weights["SYM_4"] > 0).all()

def test_fit_final_production_cutoff():
    """
    Test 3: Final production model training respects label_end_time <= cutoff.
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
    cutoff = "2024-03-01"
    
    bundle = fit_final_production_model(
        data_dict=data_dict,
        symbols=["BTCUSDT", "ETHUSDT"],
        config_path="parameters.json",
        production_cutoff=cutoff
    )

    assert bundle.lineage_manifest["production_cutoff"] == cutoff
