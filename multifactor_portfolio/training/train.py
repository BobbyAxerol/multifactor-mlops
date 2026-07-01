import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
from dotenv import load_dotenv
load_dotenv()

# Add _get_data to path
sys.path.append('/root/bobby/pool_alpha/alphas_storage/_get_data')

from multifactor_portfolio.util.factors import CrossSectionalFactorEngine
from multifactor_portfolio.util.data_collector import download_missing_data
from multifactor_portfolio.util.rebalance import (
    get_underlying_price_df, 
    calculate_inverse_volatility_weighting, 
    backtest_portfolio
)
from multifactor_portfolio.util.metrics import calculate_performance_metrics

def load_ohlcv_data(data_path: str, start_date: str = '2020-01-01') -> Dict[str, pd.DataFrame]:
    """
    Loads daily OHLCV data using the unified data loader from _get_data,
    or falls back to reading CSV files from the specified local data path.
    """
    try:
        from data_loader import CryptoDailyMatrix
        print("Loading daily OHLCV from CryptoDailyMatrix...")
        loader = CryptoDailyMatrix()
        data_dict = loader.load_ohlcv(
            symbols=None,
            start_date=start_date,
            check_val=False
        )
        if data_dict:
            print(f"Successfully loaded OHLCV for {len(data_dict)} symbols.")
            return data_dict
    except Exception as e:
        print(f"Error loading via data_loader: {e}. Falling back to local files...")
        
    data_dict = {}
    if os.path.isdir(data_path):
        for f in os.listdir(data_path):
            if f.endswith('.csv') or f.endswith('.csv.gz'):
                symbol = f.split('.')[0]
                try:
                    df = pd.read_csv(os.path.join(data_path, f))
                    if 'date' in df.columns:
                        df = df.set_index('date')
                    elif 'time' in df.columns:
                        df = df.set_index('time')
                    df.index = pd.to_datetime(df.index)
                    data_dict[symbol] = df
                except Exception as ex:
                    print(f"Failed to read local file {f}: {ex}")
    return data_dict

def split_data(
    data_dict: Dict[str, pd.DataFrame],
    split_mode: str = 'full',
    target_window: int = 30
) -> Dict:
    """
    Identifies walk-forward test folds and returns dates.
    """
    all_dates = pd.Index([])
    for df in data_dict.values():
        all_dates = all_dates.union(df.index)
    all_dates = pd.to_datetime(all_dates).sort_values()
    
    if split_mode == 'full' or all_dates.empty:
        return {
            "mode": "single",
            "folds": [{"train": data_dict, "test": data_dict, "label": "full"}],
            "all_dates": all_dates
        }
        
    folds = []
    
    if split_mode == 'walk_forward_2022':
        years = sorted([y for y in all_dates.year.unique() if y >= 2022])
        for y in years:
            test_dates = all_dates[all_dates.year == y]
            if not test_dates.empty:
                train_end = test_dates.min() - pd.Timedelta(days=target_window)
                train_dict = {sym: df[df.index <= train_end] for sym, df in data_dict.items()}
                test_dict = {sym: df[df.index.year == y] for sym, df in data_dict.items()}
                folds.append({"train": train_dict, "test": test_dict, "label": str(y)})
                
    elif split_mode == 'walk_forward_semi_yearly':
        dates_2022 = all_dates[all_dates >= pd.Timestamp('2022-01-01')]
        unique_periods = sorted(list(set((t.year, (t.month - 1) // 6 + 1) for t in dates_2022)))
        for y, h in unique_periods:
            test_dates = all_dates[(all_dates.year == y) & (((all_dates.month - 1) // 6 + 1) == h)]
            if not test_dates.empty:
                train_end = test_dates.min() - pd.Timedelta(days=target_window)
                train_dict = {sym: df[df.index <= train_end] for sym, df in data_dict.items()}
                test_dict = {sym: df[(df.index.year == y) & (((df.index.month - 1) // 6 + 1) == h)] for sym, df in data_dict.items()}
                folds.append({"train": train_dict, "test": test_dict, "label": f"{y}-H{h}"})
                
    elif split_mode == 'walk_forward_quarterly':
        dates_2022 = all_dates[all_dates >= pd.Timestamp('2022-01-01')]
        unique_periods = sorted(list(set((t.year, (t.month - 1) // 3 + 1) for t in dates_2022)))
        for y, q in unique_periods:
            test_dates = all_dates[(all_dates.year == y) & (((all_dates.month - 1) // 3 + 1) == q)]
            if not test_dates.empty:
                train_end = test_dates.min() - pd.Timedelta(days=target_window)
                train_dict = {sym: df[df.index <= train_end] for sym, df in data_dict.items()}
                test_dict = {sym: df[(df.index.year == y) & (((df.index.month - 1) // 3 + 1) == q)] for sym, df in data_dict.items()}
                folds.append({"train": train_dict, "test": test_dict, "label": f"{y}-Q{q}"})
                
    return {
        "mode": "walk_forward",
        "folds": folds,
        "all_dates": all_dates
    }

def generate_walk_forward_target_weights(
    data_dict: Dict[str, pd.DataFrame], 
    params: dict,
    local_data_dir: str,
    all_dates: pd.Index
) -> Tuple[pd.DataFrame, pd.Timestamp, List[str]]:
    """
    Generates out-of-sample portfolio weights fold-by-fold to prevent lookahead bias.
    """
    split_mode = params.get('split_mode', 'full')
    top_n = params.get('top_n_symbols', 40)
    
    if split_mode == 'full':
        temp_engine = CrossSectionalFactorEngine(symbols=[], quantiles=params.get('quantiles', 20))
        
        # Standardize klines_dict to MultiIndex panel
        kline_list = []
        for symbol, df in data_dict.items():
            temp_df = df.copy()
            temp_df['Symbol'] = symbol
            if not isinstance(temp_df.index, pd.DatetimeIndex):
                temp_df.index = pd.to_datetime(temp_df.index)
            temp_df.index.name = 'time'
            kline_list.append(temp_df.set_index('Symbol', append=True).reorder_levels(['Symbol', 'time']))
            
        klines_all = pd.concat(kline_list).sort_index() if kline_list else pd.DataFrame()
        factor_mcap_proxy = temp_engine.calculate_market_cap_proxy(klines_all, top_n=top_n)
        target_symbols = temp_engine.symbols
        
        start_date = str(klines_all.index.get_level_values('time').min().strftime('%Y-%m-%d'))
        futures_data = download_missing_data(
            symbols=target_symbols,
            data_dir=local_data_dir,
            start_date=start_date
        )
        
        klines_dict = {sym: data_dict[sym] for sym in target_symbols if sym in data_dict}
        engine = CrossSectionalFactorEngine(symbols=target_symbols, quantiles=params.get('quantiles', 20))
        
        final_weights_df = engine.run_factor_engine(
            klines_dict=klines_dict,
            ls_ratio_df=futures_data['ls_ratio'],
            oi_df=futures_data['open_interest'],
            funding_df=futures_data['funding'],
            momentum_params={
                'ma_length': params.get('ma_length', 20),
                'rsi_lower': params.get('rsi_lower', 50),
                'rsi_upper': params.get('rsi_upper', 60)
            },
            retail_params={
                'volume_period': params.get('volume_period', 20)
            },
            carry_window=params.get('carry_window', 60),
            risk_window=params.get('risk_window', 30),
            top_n_symbols=top_n
        )
        return final_weights_df, all_dates[0], target_symbols

    # Walk-forward mode
    split_info = split_data(data_dict, split_mode=split_mode, target_window=params.get('carry_window', 60))
    folds = split_info['folds']
    
    if not folds:
        # Fallback
        params_copy = params.copy()
        params_copy['split_mode'] = 'full'
        return generate_walk_forward_target_weights(data_dict, params_copy, local_data_dir, all_dates)
        
    all_active_symbols = set()
    fold_weights_list = []
    
    print(f"Executing strategy over {len(folds)} walk-forward folds...")
    for fold in folds:
        label = fold['label']
        print(f"--- Fold {label} ---")
        train_dict = fold['train']
        test_dict = fold['test']
        
        # Calculate OOS period dates
        test_dates = pd.Index([])
        for df in test_dict.values():
            test_dates = test_dates.union(df.index)
        test_dates = pd.DatetimeIndex(sorted(test_dates))
        
        test_start, test_end = test_dates.min(), test_dates.max()
        
        # Select active universe using data up to test_end (strictly historical or OOS endpoint)
        temp_engine = CrossSectionalFactorEngine(symbols=[], quantiles=params.get('quantiles', 20))
        kline_list = []
        for symbol, df in train_dict.items():
            temp_df = df.copy()
            temp_df['Symbol'] = symbol
            if not isinstance(temp_df.index, pd.DatetimeIndex):
                temp_df.index = pd.to_datetime(temp_df.index)
            temp_df.index.name = 'time'
            kline_list.append(temp_df.set_index('Symbol', append=True).reorder_levels(['Symbol', 'time']))
            
        klines_all_fold = pd.concat(kline_list).sort_index() if kline_list else pd.DataFrame()
        factor_mcap_proxy = temp_engine.calculate_market_cap_proxy(klines_all_fold, top_n=top_n)
        target_symbols = temp_engine.symbols
        all_active_symbols.update(target_symbols)
        
        # Fetch / load missing data up to test_end
        start_date = str(klines_all_fold.index.get_level_values('time').min().strftime('%Y-%m-%d'))
        futures_data = download_missing_data(
            symbols=target_symbols,
            data_dir=local_data_dir,
            start_date=start_date
        )
        
        # Slice futures data to test_end
        ls_ratio_fold = futures_data['ls_ratio'].loc[:test_end]
        oi_fold = futures_data['open_interest'].loc[:test_end]
        funding_fold = futures_data['funding'].loc[:test_end]
        
        klines_dict_fold = {sym: df[df.index <= test_end] for sym, df in data_dict.items() if sym in target_symbols}
        
        # Instantiate engine and calculate weights
        engine = CrossSectionalFactorEngine(symbols=target_symbols, quantiles=params.get('quantiles', 20))
        final_weights_fold = engine.run_factor_engine(
            klines_dict=klines_dict_fold,
            ls_ratio_df=ls_ratio_fold,
            oi_df=oi_fold,
            funding_df=funding_fold,
            momentum_params={
                'ma_length': params.get('ma_length', 20),
                'rsi_lower': params.get('rsi_lower', 50),
                'rsi_upper': params.get('rsi_upper', 60)
            },
            retail_params={
                'volume_period': params.get('volume_period', 20)
            },
            carry_window=params.get('carry_window', 60),
            risk_window=params.get('risk_window', 30),
            top_n_symbols=top_n
        )
        
        # Slice weights for out-of-sample test period only
        common_fold_dates = test_dates.intersection(final_weights_fold.index)
        oos_weights_fold = final_weights_fold.loc[common_fold_dates]
        fold_weights_list.append(oos_weights_fold)
        
    # Combine out-of-sample weights
    oos_weights_df = pd.concat(fold_weights_list, axis=0).sort_index().fillna(0.0)
    first_test_start = folds[0]['test'][next(iter(folds[0]['test']))].index.min()
    
    return oos_weights_df, first_test_start, list(all_active_symbols)

def run_strategy_backtest(
    data_dict: Dict[str, pd.DataFrame],
    params: dict,
    local_data_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Executes the complete strategy, fully supporting Walk-Forward Out-Of-Sample validation.
    """
    # 1. Gather all unique timestamps
    all_dates = pd.Index([])
    for df in data_dict.values():
        all_dates = all_dates.union(df.index)
    all_dates = pd.DatetimeIndex(sorted(all_dates))
    
    # 2. Generate target weights (Out-of-sample fold-by-fold or full timeline)
    target_weights_df, eval_start_date, target_symbols = generate_walk_forward_target_weights(
        data_dict=data_dict,
        params=params,
        local_data_dir=local_data_dir,
        all_dates=all_dates
    )
    
    # 3. Filter timeline to evaluation period (OOS evaluation start onwards)
    eval_dates = all_dates[all_dates >= eval_start_date]
    target_weights_df = target_weights_df.reindex(eval_dates).fillna(0.0)
    
    # 4. Get Close price DataFrame for active symbols
    underlying_price_df = get_underlying_price_df(data_dict, target_symbols).reindex(eval_dates).ffill().bfill()
    
    # Calculate daily returns
    underlying_returns = underlying_price_df.pct_change()
    
    # 5. Calculate Risk weights via Inverse Volatility
    print("Calculating inverse volatility risk weighting...")
    risk_weights = calculate_inverse_volatility_weighting(
        underlying=underlying_returns, 
        weights=target_weights_df, 
        period=120 
    )
    
    # Final portfolio weights
    portfolio_weights = target_weights_df.mul(risk_weights, axis="columns")
    allocation_cap = params.get('allocation_cap', 0.2)
    portfolio_weights = portfolio_weights.clip(lower=-allocation_cap, upper=allocation_cap)
    
    # 6. Run backtest
    print("Running portfolio backtest...")
    backtest_result = backtest_portfolio(
        weights=portfolio_weights,
        underlying=underlying_price_df,
        transaction_cost=params.get('fee', 0.0005),
        lag=params.get('lag', 1)
    )
    
    # Build equity curves DataFrame
    portfolio_equity = (1.0 + backtest_result.portfolio_returns).cumprod()
    
    equity_df = pd.DataFrame({
        'time': backtest_result.portfolio_returns.index,
        'equity': portfolio_equity.values,
        'return': backtest_result.portfolio_returns.values
    }).reset_index(drop=True)
    
    # Calculate overall metrics
    metrics = calculate_performance_metrics(
        equity_df, 
        trading_days_per_year=params.get('trading_days_per_year', 365)
    )
    
    return portfolio_weights, equity_df, metrics
