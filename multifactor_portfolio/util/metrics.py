import numpy as np
import pandas as pd

def calculate_performance_metrics(equity_df: pd.DataFrame, trading_days_per_year: float = 365.0) -> dict:
    """
    Computes performance metrics (Sharpe ratio, CAGR, max drawdown).
    """
    if equity_df.empty or len(equity_df) < 2:
        return {
            'sharpe_ratio': 0.0,
            'cagr': 0.0,
            'max_drawdown': 0.0
        }
        
    initial_val = equity_df['equity'].iloc[0]
    final_val = equity_df['equity'].iloc[-1]
    
    # Calculate years based on actual timeline
    total_days = (equity_df['time'].iloc[-1] - equity_df['time'].iloc[0]).days
    years = total_days / 365.25 if total_days > 0 else 0.0
    
    cagr = (final_val / initial_val) ** (1.0 / years) - 1.0 if years > 0 and final_val > 0 else 0.0
    
    daily_returns = equity_df['return']
    ann_vol = daily_returns.std() * np.sqrt(trading_days_per_year)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0
    
    cum_max = equity_df['equity'].cummax()
    drawdowns = (equity_df['equity'] / cum_max) - 1.0
    max_dd = drawdowns.min()
    
    return {
        'sharpe_ratio': round(float(sharpe), 4),
        'cagr': round(float(cagr), 4),
        'max_drawdown': round(float(max_dd), 4)
    }

def to_drawdown(prices: pd.Series) -> pd.Series:
    running_max = prices.cummax()
    return (prices - running_max) / running_max

def rebase(prices: pd.Series) -> pd.Series:
    """Rebase a price series to 1.0"""
    if prices.empty:
        return prices
    return prices / prices.iloc[0]
