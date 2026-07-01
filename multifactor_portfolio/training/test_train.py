import pytest
import numpy as np
import pandas as pd
from multifactor_portfolio.util.factors import CrossSectionalFactorEngine

def test_custom_wma():
    engine = CrossSectionalFactorEngine(symbols=['TEST'], quantiles=5)
    prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    wma = engine._custom_wma(prices, 3)
    # The first window-1 values should be NaN
    assert np.isnan(wma.iloc[0])
    assert np.isnan(wma.iloc[1])
    # The 3rd value (index 2): (1.0*1 + 2.0*2 + 3.0*3) / (1+2+3) = (1+4+9)/6 = 14/6 = 2.3333...
    assert abs(wma.iloc[2] - 2.333333333) < 1e-6

def test_retail_flow_proxy():
    # Make a synthetic dataframe
    dates = pd.date_range(start='2025-01-01', periods=10)
    df = pd.DataFrame({
        'close': [10.0, 10.1, 10.2, 10.1, 10.3, 10.4, 10.5, 10.6, 10.5, 10.7],
        'volume': [1000, 1100, 1200, 950, 1300, 1400, 1500, 1600, 1450, 1800]
    }, index=dates)
    
    retail = CrossSectionalFactorEngine.calculate_retail_flow_proxy(df, volume_period=5)
    assert len(retail) == 10
    # First 4 elements should be NaN due to shift(1) or rolling std
    assert np.isnan(retail.iloc[0])
    assert np.isnan(retail.iloc[3])
    # The rest should be valid float numbers
    assert not np.isnan(retail.iloc[5])

def test_cross_sectional_binning():
    engine = CrossSectionalFactorEngine(symbols=['A', 'B', 'C', 'D'], quantiles=4)
    # 3 days, 4 symbols
    dates = pd.date_range(start='2025-01-01', periods=3)
    factor_data = pd.DataFrame({
        'A': [1.0, 2.0, 3.0],
        'B': [2.0, 1.0, 4.0],
        'C': [3.0, 4.0, 1.0],
        'D': [4.0, 3.0, 2.0]
    }, index=dates)
    
    weights = engine.create_cross_sectional_bins(factor_data)
    assert weights.shape == (3, 4)
    # Check neutrality: row sum should be close to 0
    row_sums = weights.sum(axis=1)
    for s in row_sums:
        assert abs(s) < 1e-6

def test_split_data():
    from multifactor_portfolio.training.train import split_data
    # 2 years of daily data
    dates = pd.date_range(start='2021-01-01', end='2023-12-31')
    df = pd.DataFrame({'close': np.random.randn(len(dates))}, index=dates)
    data_dict = {'BTCUSDT': df}
    
    split_info = split_data(data_dict, split_mode='walk_forward_2022', target_window=10)
    assert split_info['mode'] == 'walk_forward'
    folds = split_info['folds']
    assert len(folds) >= 2  # should have 2022 and 2023 folds
    for fold in folds:
        assert 'train' in fold
        assert 'test' in fold
        assert isinstance(fold['label'], str)
