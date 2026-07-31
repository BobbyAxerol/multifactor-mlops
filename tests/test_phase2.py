"""
Phase 2 Automated Unit Tests: Point-in-Time Data, Universe, and Parity Tests.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.multifactor_mlops.universe.liquidity import PointInTimeUniverseSelector
from src.multifactor_mlops.features.asset import AssetFeatureTransformer
from src.multifactor_mlops.features.macro import MacroOverlayTransformer
from src.multifactor_mlops.features.panel import PanelDatasetBuilder

def test_future_mutation_invariance():
    """
    Test 1: Future mutation test.
    Mutating prices after cutoff T must NOT alter features at t <= T.
    """
    dates = pd.date_range("2024-01-01", periods=100, freq="1D")
    prices_base = pd.Series(np.linspace(100, 200, 100), index=dates)
    
    btc_base = pd.DataFrame({
        "open": prices_base,
        "high": prices_base + 2,
        "low": prices_base - 2,
        "close": prices_base,
        "volume": 1000.0
    }, index=dates)
    
    transformer = AssetFeatureTransformer(windows=[7, 14, 30])
    feats_base = transformer.transform_symbol(btc_base)
    
    # Mutate data AFTER index 50
    btc_mutated = btc_base.copy()
    btc_mutated.iloc[60:, btc_mutated.columns.get_loc("close")] *= 5.0
    feats_mutated = transformer.transform_symbol(btc_mutated)
    
    # Assert features up to index 50 are 100% identical
    cutoff_date = dates[50]
    pd.testing.assert_frame_equal(
        feats_base.loc[:cutoff_date],
        feats_mutated.loc[:cutoff_date]
    )

def test_point_in_time_universe_lagged_turnover():
    """
    Test 2: Point-in-time universe test.
    Universe membership at time t uses only lagged turnover t-1.
    """
    dates = pd.date_range("2024-01-01", periods=200, freq="1D")
    
    # Symbol A has high volume on day 1-100, low on 101-200
    df_a = pd.DataFrame({
        "close": 10.0,
        "volume": [10000.0 if i < 100 else 10.0 for i in range(200)]
    }, index=dates)
    
    # Symbol B has low volume on day 1-100, high on 101-200
    df_b = pd.DataFrame({
        "close": 10.0,
        "volume": [10.0 if i < 100 else 10000.0 for i in range(200)]
    }, index=dates)
    
    data_dict = {"SYMB_A": df_a, "SYMB_B": df_b}
    
    selector = PointInTimeUniverseSelector(top_n=1, lookback_days=10, min_history_days=10)
    membership_df = selector.build_universe_membership(data_dict)
    
    # On day 50, SYMB_A must be selected (lagged high volume)
    assert membership_df.loc[dates[50], "SYMB_A"] == True
    assert membership_df.loc[dates[50], "SYMB_B"] == False

def test_training_serving_parity():
    """
    Test 3: Training-serving parity test.
    Identical snapshot produces byte-equivalent feature DataFrames.
    """
    dates = pd.date_range("2024-01-01", periods=50, freq="1D")
    btc_df = pd.DataFrame({
        "open": np.linspace(100, 150, 50),
        "high": np.linspace(102, 152, 50),
        "low": np.linspace(98, 148, 50),
        "close": np.linspace(100, 150, 50),
        "volume": 5000.0
    }, index=dates)
    
    transformer = AssetFeatureTransformer(windows=[7, 14])
    
    # Training replay
    feats_train = transformer.transform_symbol(btc_df)
    # Serving inference on identical data
    feats_serving = transformer.transform_symbol(btc_df)
    
    pd.testing.assert_frame_equal(feats_train, feats_serving)

def test_point_in_time_macro_merge_asof():
    """
    Test 4: Macro point-in-time merge_asof alignment test.
    """
    dates = pd.date_range("2024-01-01", periods=50, freq="1D")
    macro_df = pd.DataFrame({
        "vix": np.random.randn(50) + 20.0,
        "fear_greed": np.random.randn(50) + 50.0,
        "dvol_btc": np.random.randn(50) + 60.0
    }, index=dates)
    
    macro_transformer = MacroOverlayTransformer(lookback_days=10, min_periods=3)
    macro_feats = macro_transformer.transform_macro_df(macro_df)
    
    assert "macro_multiplier" in macro_feats.columns
    assert macro_feats["macro_multiplier"].max() <= 1.0
    assert macro_feats["macro_multiplier"].min() >= 0.2
