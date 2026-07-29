from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class PortfolioBacktestResult:
    portfolio_returns: pd.Series
    component_returns: pd.DataFrame
    transaction_costs: pd.Series
    lag: int

    def split(self, start_date, end_date) -> PortfolioBacktestResult:
        return PortfolioBacktestResult(
            portfolio_returns=self.portfolio_returns[start_date:end_date],
            component_returns=self.component_returns[start_date:end_date],
            transaction_costs=self.transaction_costs[start_date:end_date],
            lag=self.lag,
        )

def get_underlying_price_df(klines_dict: dict[str, pd.DataFrame], target_symbols: list[str]) -> pd.DataFrame:
    """
    Extract and merge Close prices into a wide DataFrame for target symbols.
    """
    price_series = {}
    CLOSE_COLUMNS = ['Close', 'close', '4'] 

    for symbol in target_symbols:
        if symbol not in klines_dict:
            continue
            
        df = klines_dict[symbol]
        found_close_col = None
        for col_name in CLOSE_COLUMNS:
            if col_name in df.columns:
                found_close_col = col_name
                break
        
        if found_close_col:
            series = df[found_close_col].rename(symbol)
            if isinstance(series.index, pd.MultiIndex):
                series = series.droplevel(series.index.names[1:]) 
            price_series[symbol] = series.sort_index()

    if not price_series:
        print("Warning: No close price found for target symbols.")
        return pd.DataFrame()

    underlying_df = pd.concat(price_series.values(), axis=1, join='outer')
    try:
        underlying_df.index = pd.to_datetime(underlying_df.index)
    except Exception:
        pass
        
    return underlying_df.sort_index()

def calculate_inverse_volatility_weighting(
    underlying: pd.DataFrame, weights: pd.DataFrame, period: int
) -> pd.DataFrame:
    """
    Calculate Side-Normalized Inverse Volatility Weights (+0.50 Long, -0.50 Short).
    """
    stds = underlying.rolling(period, min_periods=10).std().replace(0, np.nan)
    inv_stds = 1.0 / stds
    inv_stds = inv_stds.fillna(0.0)

    long_mask = weights > 0
    short_mask = weights < 0

    long_weights = inv_stds.where(long_mask, 0.0)
    long_sum = long_weights.sum(axis=1).replace(0, 1.0)
    long_normalized = long_weights.div(long_sum, axis=0) * 0.50

    short_weights = inv_stds.where(short_mask, 0.0)
    short_sum = short_weights.sum(axis=1).replace(0, 1.0)
    short_normalized = -short_weights.div(short_sum, axis=0) * 0.50

    final_portfolio_weights = long_normalized + short_normalized
    return final_portfolio_weights.fillna(0.0)

def backtest_portfolio(
    weights: pd.DataFrame,
    underlying: pd.DataFrame,
    transaction_cost: float,
    lag: int,
) -> PortfolioBacktestResult:
    """
    Runs the portfolio backtest using weights and underlying price.
    """
    # 1. Calculate daily returns from Close price
    underlying_returns = underlying.pct_change()

    # 2. Align indexes and columns
    common_index = weights.index.intersection(underlying_returns.index)
    common_index = common_index[1:]  # skip the first row (NaN from pct_change)
    
    underlying_returns = underlying_returns.loc[common_index]
    weights = weights.loc[common_index]

    # Align columns to match weights columns
    if not weights.columns.equals(underlying_returns.columns):
        common_cols = [c for c in weights.columns if c in underlying_returns.columns]
        weights = weights[common_cols]
        underlying_returns = underlying_returns[common_cols]

    assert weights.columns.equals(underlying_returns.columns), "Symbols must match between Weights and Returns"
    
    weights = weights.fillna(0.0).ffill().copy()

    # 3. Transaction costs calculation
    delta_pos = weights.diff(1).abs().fillna(0.0)
    costs = transaction_cost * delta_pos
    
    # 4. Portfolio returns calculation
    returns = (underlying_returns * weights.shift(lag)) - costs
    portfolio_returns = returns.sum(axis="columns")
    transaction_costs = costs.sum(axis="columns")
    
    return PortfolioBacktestResult(
        portfolio_returns=portfolio_returns,
        component_returns=returns,
        transaction_costs=transaction_costs,
        lag=lag,
    )
