import os
import sys
import json
import pandas as pd
import numpy as np
import xgboost as xgb
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
    backtest_portfolio,
    PortfolioBacktestResult
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
    
    if split_mode.startswith('walk_forward_') and not split_mode.endswith('yearly') and not split_mode.endswith('quarterly'):
        try:
            start_year = int(split_mode.split('_')[-1])
        except ValueError:
            start_year = 2022
        years = sorted([y for y in all_dates.year.unique() if y >= start_year])
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
) -> Tuple[pd.DataFrame, pd.Timestamp, List[str], Any]:
    """
    Generates out-of-sample portfolio weights fold-by-fold using XGBoost models to predict returns.
    """
    split_mode = params.get('split_mode', 'full')
    top_n = params.get('top_n_symbols', 40)
    windows = [7, 14, 30, 60, 90]
    
    # Load selected features
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    selected_path = os.path.join(project_root, "multifactor_portfolio", "research", "selected_features.json")
    if os.path.exists(selected_path):
        with open(selected_path) as f:
            selected_features = json.load(f)
    else:
        # Fallback to all features if not analyzed yet
        selected_features = []
        for w in windows:
            selected_features.extend([f'mom_rsi_{w}', f'mom_wma_dist_{w}', f'retail_flow_{w}', f'carry_{w}', f'margin_risk_{w}'])

    model_type = params.get('model_type', 'xgboost')
    
    xgb_hyperparams = {
        'objective': 'reg:squarederror',
        'learning_rate': params.get('learning_rate', 0.05),
        'max_depth': int(params.get('max_depth', 4)),
        'colsample_bytree': params.get('colsample_bytree', 0.3),
        'verbosity': 0
    }
    num_boost_round = int(params.get('num_boost_round', 100))

    if split_mode == 'full':
        temp_engine = CrossSectionalFactorEngine(symbols=[], quantiles=params.get('quantiles', 20))
        kline_list = []
        for symbol, df in data_dict.items():
            temp_df = df.copy()
            temp_df['Symbol'] = symbol
            temp_df.index.name = 'time'
            kline_list.append(temp_df.set_index('Symbol', append=True).reorder_levels(['Symbol', 'time']))
            
        klines_all = pd.concat(kline_list).sort_index() if kline_list else pd.DataFrame()
        _ = temp_engine.calculate_market_cap_proxy(klines_all, top_n=top_n)
        target_symbols = temp_engine.symbols
        
        start_date = str(klines_all.index.get_level_values('time').min().strftime('%Y-%m-%d'))
        futures_data = download_missing_data(
            symbols=target_symbols,
            data_dir=local_data_dir,
            start_date=start_date
        )
        
        from multifactor_portfolio.util.macro_collector import download_macro_features
        print("Downloading global macro features...")
        macro_df = download_macro_features(local_data_dir, start_date=start_date)
        
        panel_df = CrossSectionalFactorEngine.prepare_panel_dataset(
            data_dict=data_dict,
            funding_df=futures_data['funding'],
            symbols=target_symbols,
            macro_df=macro_df,
            windows=windows,
            lag=params.get('lag', 1)
        )
        
        # Apply Step Sampling to training set if configured (to avoid serial autocorrelation)
        train_step_days = params.get('train_step_days', 1)
        if train_step_days > 1 and not panel_df.empty:
            unique_dates = sorted(panel_df.index.get_level_values('Time').unique())
            sampled_dates = unique_dates[::train_step_days]
            panel_df = panel_df[panel_df.index.get_level_values('Time').isin(sampled_dates)]
            print(f"Downsampled full-timeline panel from {len(unique_dates)} to {len(sampled_dates)} dates using step {train_step_days} days.")
            
        X = panel_df[selected_features].fillna(0.0)
        y = panel_df['target']
        
        print(f"Training full-timeline {model_type} model...")
        if model_type == 'lightgbm':
            import lightgbm as lgb
            dtrain = lgb.Dataset(X, label=y)
            lgb_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': params.get('learning_rate', 0.05),
                'max_depth': int(params.get('max_depth', 4)),
                'num_leaves': int(params.get('num_leaves', 15)),
                'feature_fraction': params.get('colsample_bytree', 0.3),
                'verbosity': -1
            }
            bst = lgb.train(lgb_params, dtrain, num_boost_round=num_boost_round)
            preds = bst.predict(X)
        else:
            dtrain = xgb.DMatrix(X, label=y)
            bst = xgb.train(xgb_hyperparams, dtrain, num_boost_round=num_boost_round)
            preds = bst.predict(dtrain)
            
        predicted_returns_df = pd.Series(preds, index=panel_df.index).unstack(level='Symbol').fillna(0.0)
        
        engine_bin = CrossSectionalFactorEngine(symbols=target_symbols, quantiles=params.get('quantiles', 20))
        final_weights_df = engine_bin.create_cross_sectional_bins(predicted_returns_df)
        
        return final_weights_df, all_dates[0], target_symbols, bst

    # Walk-forward mode
    split_info = split_data(data_dict, split_mode=split_mode, target_window=max(windows))
    folds = split_info['folds']
    
    if not folds:
        params_copy = params.copy()
        params_copy['split_mode'] = 'full'
        return generate_walk_forward_target_weights(data_dict, params_copy, local_data_dir, all_dates)
        
    from multifactor_portfolio.util.macro_collector import download_macro_features
    min_date = all_dates.min().strftime('%Y-%m-%d')
    print("Downloading global macro features...")
    global_macro_df = download_macro_features(local_data_dir, start_date=min_date)

    all_active_symbols = set()
    fold_weights_list = []
    last_bst = None
    
    print(f"Executing strategy over {len(folds)} walk-forward folds...")
    for fold in folds:
        label = fold['label']
        print(f"--- Fold {label} ---")
        train_dict = fold['train']
        test_dict = fold['test']
        
        test_dates = pd.Index([])
        for df in test_dict.values():
            test_dates = test_dates.union(df.index)
        test_dates = pd.DatetimeIndex(sorted(test_dates))
        
        test_start, test_end = test_dates.min(), test_dates.max()
        
        # 1. Determine active universe using train set
        temp_engine = CrossSectionalFactorEngine(symbols=[], quantiles=params.get('quantiles', 20))
        kline_list = []
        for symbol, df in train_dict.items():
            temp_df = df.copy()
            temp_df['Symbol'] = symbol
            temp_df.index.name = 'time'
            kline_list.append(temp_df.set_index('Symbol', append=True).reorder_levels(['Symbol', 'time']))
            
        klines_all_fold = pd.concat(kline_list).sort_index() if kline_list else pd.DataFrame()
        _ = temp_engine.calculate_market_cap_proxy(klines_all_fold, top_n=top_n)
        target_symbols = temp_engine.symbols
        all_active_symbols.update(target_symbols)
        
        # 2. Get funding rates up to test_end
        start_date = str(klines_all_fold.index.get_level_values('time').min().strftime('%Y-%m-%d'))
        futures_data = download_missing_data(
            symbols=target_symbols,
            data_dir=local_data_dir,
            start_date=start_date
        )
        
        funding_fold = futures_data['funding'].loc[:test_end]
        macro_fold = global_macro_df.loc[:test_end]
        train_end = train_dict[next(iter(train_dict))].index.max()
        
        # 3. Prepare Train panel dataset
        panel_train = CrossSectionalFactorEngine.prepare_panel_dataset(
            data_dict=train_dict,
            funding_df=funding_fold,
            symbols=target_symbols,
            macro_df=macro_fold.loc[:train_end],
            windows=windows,
            lag=params.get('lag', 1)
        )
        
        # Apply Step Sampling to training set if configured (to avoid serial autocorrelation)
        train_step_days = params.get('train_step_days', 1)
        if train_step_days > 1 and not panel_train.empty:
            unique_train_dates = sorted(panel_train.index.get_level_values('Time').unique())
            sampled_train_dates = unique_train_dates[::train_step_days]
            panel_train = panel_train[panel_train.index.get_level_values('Time').isin(sampled_train_dates)]
            print(f"Downsampled training panel from {len(unique_train_dates)} to {len(sampled_train_dates)} dates using step {train_step_days} days.")
            
        X_train = panel_train[selected_features].fillna(0.0)
        y_train = panel_train['target']
        
        # Train
        if model_type == 'lightgbm':
            import lightgbm as lgb
            dtrain = lgb.Dataset(X_train, label=y_train)
            lgb_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': params.get('learning_rate', 0.05),
                'max_depth': int(params.get('max_depth', 4)),
                'num_leaves': int(params.get('num_leaves', 15)),
                'feature_fraction': params.get('colsample_bytree', 0.3),
                'verbosity': -1
            }
            bst = lgb.train(lgb_params, dtrain, num_boost_round=num_boost_round)
        else:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            bst = xgb.train(xgb_hyperparams, dtrain, num_boost_round=num_boost_round)
        last_bst = bst
        
        # 4. Prepare Test panel dataset (OOS dates)
        test_dict_fold = {sym: df[df.index <= test_end] for sym, df in data_dict.items() if sym in target_symbols}
        panel_test = CrossSectionalFactorEngine.prepare_panel_dataset(
            data_dict=test_dict_fold,
            funding_df=funding_fold,
            symbols=target_symbols,
            macro_df=macro_fold,
            windows=windows,
            lag=params.get('lag', 1)
        )
        
        # Slice test dates
        panel_test = panel_test.loc[test_start:test_end]
        
        if panel_test.empty:
            continue
            
        X_test = panel_test[selected_features].fillna(0.0)
        
        # Predict
        if model_type == 'lightgbm':
            preds = bst.predict(X_test)
        else:
            dtest = xgb.DMatrix(X_test)
            preds = bst.predict(dtest)
        
        # Unstack predictions
        predicted_returns_df = pd.Series(preds, index=panel_test.index).unstack(level='Symbol').fillna(0.0)
        
        # Cross-sectional binning on predictions
        engine_bin = CrossSectionalFactorEngine(symbols=target_symbols, quantiles=params.get('quantiles', 20))
        final_weights_fold = engine_bin.create_cross_sectional_bins(predicted_returns_df)
        fold_weights_list.append(final_weights_fold)
        
    # Combine out-of-sample weights
    oos_weights_df = pd.concat(fold_weights_list, axis=0).sort_index().fillna(0.0)
    first_test_start = folds[0]['test'][next(iter(folds[0]['test']))].index.min()
    
    return oos_weights_df, first_test_start, list(all_active_symbols), last_bst

def run_strategy_backtest(
    data_dict: Dict[str, pd.DataFrame],
    params: dict,
    local_data_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Executes the complete strategy, fully supporting Walk-Forward Out-Of-Sample validation.
    """
    all_dates = pd.Index([])
    for df in data_dict.values():
        all_dates = all_dates.union(df.index)
    all_dates = pd.DatetimeIndex(sorted(all_dates))
    
    target_weights_df, eval_start_date, target_symbols, bst = generate_walk_forward_target_weights(
        data_dict=data_dict,
        params=params,
        local_data_dir=local_data_dir,
        all_dates=all_dates
    )
    
    eval_dates = all_dates[all_dates >= eval_start_date]
    target_weights_df = target_weights_df.reindex(eval_dates).fillna(0.0)
    
    underlying_price_df = get_underlying_price_df(data_dict, target_symbols).reindex(eval_dates).ffill().bfill()
    underlying_returns = underlying_price_df.pct_change()
    
    print("Calculating inverse volatility risk weighting...")
    risk_weights = calculate_inverse_volatility_weighting(
        underlying=underlying_returns, 
        weights=target_weights_df, 
        period=params.get('inverse_vol_period', 120)
    )
    
    portfolio_weights = risk_weights.copy()
    
    # PA 5.1: Sigmoid Continuous Exposure Scaling
    try:
        from multifactor_portfolio.util.macro_collector import download_macro_features
        print("Applying PA 5.1 Sigmoid Continuous Macro-Regime Risk Overlay...")
        start_dt_str = eval_dates.min().strftime('%Y-%m-%d')
        end_dt_str = eval_dates.max().strftime('%Y-%m-%d')
        macro_df = download_macro_features(local_data_dir, start_date=start_dt_str, end_date=end_dt_str)
        if not macro_df.empty:
            macro_df = macro_df.reindex(portfolio_weights.index).ffill().bfill()
            
            vix_z = (macro_df['vix'] - macro_df['vix'].rolling(120, min_periods=30).mean()) / macro_df['vix'].rolling(120, min_periods=30).std().replace(0, 1)
            fng_z = -(macro_df['fear_greed'] - macro_df['fear_greed'].rolling(120, min_periods=30).mean()) / macro_df['fear_greed'].rolling(120, min_periods=30).std().replace(0, 1)
            dvol_z = (macro_df['dvol_btc'] - macro_df['dvol_btc'].rolling(120, min_periods=30).mean()) / macro_df['dvol_btc'].rolling(120, min_periods=30).std().replace(0, 1)
            
            stress_score = (vix_z.fillna(0.0) + fng_z.fillna(0.0) + dvol_z.fillna(0.0)) / 3.0
            
            # Sigmoid smooth exposure scaling: 1 / (1 + exp(1.5 * (stress_score - 0.5)))
            regime_multiplier = 1.0 / (1.0 + np.exp(1.5 * (stress_score - 0.5)))
            regime_multiplier = regime_multiplier.clip(lower=0.2, upper=1.0)
            
            portfolio_weights = portfolio_weights.mul(regime_multiplier, axis=0)
            print(f"Applied PA 5.1 Sigmoid Exposure Scaling. Mean exposure multiplier: {regime_multiplier.mean():.4f}")
    except Exception as e:
        print(f"Warning: Failed to apply PA 5.1 Sigmoid Macro Risk Overlay: {e}")
        
    allocation_cap = params.get('allocation_cap', 0.2)
    portfolio_weights = portfolio_weights.clip(lower=-allocation_cap, upper=allocation_cap)
    
    print("Running portfolio backtest...")
    use_quantbt = params.get('scoring_backend') == 'endpoint' or params.get('backend') == 'quantbt'
    quantbt_executed = False
    
    if use_quantbt:
        try:
            sys.path.append(params.get('quantbt_repo_path', '/root/bobby/pool_alpha/quantbt'))
            sys.path.append('/root/bobby/pool_alpha')
            from quantbt.endpoint import QuantBTEndpoint
            
            # Anti-Look-Ahead Bias: Enforce 1-bar execution lag
            scaled_weights = portfolio_weights.shift(1).fillna(0.0)
            if float(scaled_weights.abs().sum(axis=1).max()) >= 0.999:
                scaled_weights = scaled_weights * 0.99
                
            bt_engine = QuantBTEndpoint.portfolio(
                portfolio_mode=params.get('portfolio_mode', 'longshort'),
                backend=params.get('backend', 'native_portfolio'),
                hedge_type=params.get('hedge_type', 'target_weight'),
                initial_capital=params.get('initial_capital', 100000.0),
                leverage=params.get('leverage', 1.0),
                asset_type=params.get('asset_class', 'crypto'),
                use_funding=params.get('use_funding', False),
                fee=params.get('fee', 0.0005) * 2.0,
                slippage=params.get('slippage', 0.0001),
                contract_size=1.0,
                report_level="minimal"
            )
            qbt_res = bt_engine.backtest(positions=scaled_weights, data=data_dict)
            qbt_equity = getattr(qbt_res, "daily_equity", None)
            if qbt_equity is not None and len(qbt_equity) > 0:
                qbt_rets = qbt_equity.pct_change().fillna(0.0)
            else:
                qbt_rets = getattr(qbt_res, "daily_returns", getattr(qbt_res, "portfolio_returns", None))
                
            if qbt_rets is not None and len(qbt_rets) > 0:
                backtest_result = PortfolioBacktestResult(
                    portfolio_returns=qbt_rets,
                    component_returns=pd.DataFrame(),
                    transaction_costs=pd.Series(0.0, index=qbt_rets.index),
                    lag=params.get('lag', 1)
                )
                quantbt_executed = True
                print("QuantBT Endpoint backtest executed successfully.")
        except Exception as q_err:
            print(f"Warning: QuantBT Endpoint execution failed ({q_err}). Falling back to native vector backtest...")

    if not quantbt_executed:
        backtest_result = backtest_portfolio(
            weights=portfolio_weights,
            underlying=underlying_price_df,
            transaction_cost=params.get('fee', 0.0005),
            lag=params.get('lag', 1)
        )
    
    # Filter returns strictly to eval_dates to eliminate pre-evaluation zero padding
    p_returns = backtest_result.portfolio_returns.copy()
    if p_returns.index.tz is not None:
        p_returns.index = p_returns.index.tz_localize(None)
        
    clean_eval_dates = eval_dates.tz_localize(None) if eval_dates.tz is not None else eval_dates
    eval_returns = p_returns.reindex(clean_eval_dates).fillna(0.0)
    portfolio_equity = (1.0 + eval_returns).cumprod()
    
    equity_df = pd.DataFrame({
        'time': clean_eval_dates,
        'equity': portfolio_equity.values,
        'return': eval_returns.values
    }).reset_index(drop=True)
    
    metrics = calculate_performance_metrics(
        equity_df, 
        trading_days_per_year=params.get('trading_days_per_year', 365)
    )
    
    # Attach model for MLflow registration
    metrics['__bst__'] = bst
    metrics['__selected_features__'] = target_symbols
    
    return portfolio_weights, equity_df, metrics
