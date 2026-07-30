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

def calculate_ml_evaluation_metrics(y_true: pd.Series, y_pred: np.ndarray, index: pd.Index) -> Dict[str, float]:
    """
    Computes ML Directional Classification & Rank IC metrics.
    """
    y_true_vals = y_true.values if hasattr(y_true, 'values') else y_true
    
    # Directional Sign Accuracy
    y_true_sign = np.where(y_true_vals > 0, 1, 0)
    y_pred_sign = np.where(y_pred > 0, 1, 0)
    
    acc = np.mean(y_true_sign == y_pred_sign)
    
    # Precision, Recall, F1
    tp = np.sum((y_true_sign == 1) & (y_pred_sign == 1))
    fp = np.sum((y_true_sign == 0) & (y_pred_sign == 1))
    fn = np.sum((y_true_sign == 1) & (y_pred_sign == 0))
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    # R2 & MSE
    ss_res = np.sum((y_true_vals - y_pred) ** 2)
    ss_tot = np.sum((y_true_vals - np.mean(y_true_vals)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    mse = np.mean((y_true_vals - y_pred) ** 2)
    
    # Daily Rank IC (Spearman correlation per daily timestamp)
    from scipy.stats import spearmanr
    eval_df = pd.DataFrame({'true': y_true_vals, 'pred': y_pred}, index=index)
    daily_ic = []
    time_level = 'Time' if 'Time' in eval_df.index.names else eval_df.index.names[0]
    for dt, group in eval_df.groupby(level=time_level):
        if len(group) >= 5 and group['true'].std() > 1e-8 and group['pred'].std() > 1e-8:
            ic, _ = spearmanr(group['true'], group['pred'])
            if not np.isnan(ic):
                daily_ic.append(ic)
                
    mean_ic = float(np.mean(daily_ic)) if daily_ic else 0.0
    std_ic = float(np.std(daily_ic)) if daily_ic else 0.0
    ic_ir = mean_ic / std_ic if std_ic > 1e-8 else 0.0
    
    return {
        'ml_accuracy': round(float(acc), 4),
        'ml_precision': round(float(prec), 4),
        'ml_recall': round(float(rec), 4),
        'ml_f1_score': round(float(f1), 4),
        'ml_rank_ic': round(float(mean_ic), 4),
        'ml_ic_ir': round(float(ic_ir), 4),
        'ml_r2_score': round(float(r2), 4),
        'ml_mse': round(float(mse), 6)
    }

def save_backtest_report_and_chart(equity_df: pd.DataFrame, metrics: dict, params: dict, output_dir: str):
    """
    Saves performance_report.txt and performance_chart.png in output_dir whenever backtest is run.
    """
    import os
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save performance_report.txt
    txt_path = os.path.join(output_dir, "performance_report.txt")
    with open(txt_path, "w") as f:
        f.write("=== MULTI-FACTOR MACRO STRATEGY PERFORMANCE REPORT ===\n\n")
        f.write(f"Model Type: {params.get('model_type', params.get('features', {}).get('model_type', 'xgboost'))}\n")
        f.write(f"Learning Rate: {params.get('learning_rate', params.get('features', {}).get('learning_rate', 0.04))}\n")
        f.write(f"Max Depth: {params.get('max_depth', params.get('features', {}).get('max_depth', 4))}\n")
        f.write(f"Num Boost Rounds: {params.get('num_boost_round', params.get('features', {}).get('num_boost_round', 100))}\n")
        f.write(f"Leverage: {params.get('leverage', 3.0)}\n")
        f.write(f"Inverse Vol Period: {params.get('inverse_vol_period', 60)}\n")
        f.write(f"Quantiles: {params.get('quantiles', 20)}\n\n")
        f.write("--- PORTFOLIO METRICS ---\n")
        f.write(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.4f}\n")
        f.write(f"CAGR: {metrics.get('cagr', 0.0)*100:.2f}%\n")
        f.write(f"Max Drawdown: {metrics.get('max_drawdown', 0.0)*100:.2f}%\n\n")
        f.write("--- ML MODEL EVALUATION METRICS ---\n")
        f.write(f"Sign Direction Accuracy: {metrics.get('ml_accuracy', 0.0)*100:.2f}%\n")
        f.write(f"Precision: {metrics.get('ml_precision', 0.0)*100:.2f}%\n")
        f.write(f"Recall: {metrics.get('ml_recall', 0.0)*100:.2f}%\n")
        f.write(f"F1-Score: {metrics.get('ml_f1_score', 0.0):.4f}\n")
        f.write(f"Rank IC (Spearman): {metrics.get('ml_rank_ic', 0.0):.4f}\n")
        f.write(f"IC IR: {metrics.get('ml_ic_ir', 0.0):.4f}\n")
        f.write(f"R2 Explanatory Power: {metrics.get('ml_r2_score', 0.0)*100:.2f}%\n")
        f.write(f"MSE: {metrics.get('ml_mse', 0.0):.6f}\n")
        
    # 2. Save performance_chart.png
    chart_path = os.path.join(output_dir, "performance_chart.png")
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False, gridspec_kw={'height_ratios': [2.5, 1]})
        
        ax1.plot(equity_df['time'], equity_df['equity'], label=f'Strategy Equity (Sharpe: {metrics.get("sharpe_ratio", 0.0):.2f}, CAGR: {metrics.get("cagr", 0.0)*100:.1f}%)', color='#00d26a', linewidth=2.0)
        ax1.set_title('Multi-Factor Macro Strategy Performance', fontsize=14, fontweight='bold', pad=12)
        ax1.set_ylabel('Equity (Rebased 1.0)', fontsize=11, fontweight='bold')
        ax1.legend(loc='upper left', frameon=True, facecolor='#1e1e1e', labelcolor='white')
        ax1.set_facecolor('#141414')

        cum_max = equity_df['equity'].cummax()
        dd = (equity_df['equity'] / cum_max) - 1.0
        ax2.fill_between(equity_df['time'], dd, 0, color='#ff4d4d', alpha=0.5, label=f'Max Drawdown ({metrics.get("max_drawdown", 0.0)*100:.1f}%)')
        ax2.set_title('Drawdown Profile', fontsize=12, fontweight='bold', pad=8)
        ax2.set_ylabel('Drawdown', fontsize=11, fontweight='bold')
        ax2.legend(loc='lower left', frameon=True, facecolor='#1e1e1e', labelcolor='white')
        ax2.set_facecolor('#141414')

        fig.patch.set_facecolor('#0a0a0a')
        ax1.tick_params(colors='white')
        ax2.tick_params(colors='white')
        ax1.yaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')
        ax1.title.set_color('white')
        ax2.title.set_color('white')

        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        print(f"Automatically saved report to: {txt_path}")
        print(f"Automatically saved chart to: {chart_path}")
    except Exception as err:
        print(f"Warning: Failed to generate performance chart: {err}")

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
                
    elif split_mode.startswith('train_test_split_'):
        try:
            split_year = int(split_mode.split('_')[-1])
        except ValueError:
            split_year = 2024
        split_date = pd.Timestamp(f"{split_year}-01-01")
        test_dates = all_dates[all_dates >= split_date]
        if not test_dates.empty:
            train_end = split_date - pd.Timedelta(days=target_window)
            train_dict = {sym: df[df.index <= train_end] for sym, df in data_dict.items()}
            test_dict = {sym: df[df.index >= split_date] for sym, df in data_dict.items()}
            folds.append({"train": train_dict, "test": test_dict, "label": f"{split_year}_OOS"})
                
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
        
        times_list = [t for t in klines_all.index.get_level_values('time') if pd.notna(t)] if not klines_all.empty else []
        start_date = pd.Timestamp(min(times_list)).strftime('%Y-%m-%d') if times_list else "2020-01-01"
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
        
        ml_metrics = calculate_ml_evaluation_metrics(y, preds, panel_df.index)
        
        engine_bin = CrossSectionalFactorEngine(symbols=target_symbols, quantiles=params.get('quantiles', 20))
        final_weights_df = engine_bin.create_cross_sectional_bins(predicted_returns_df)
        
        return final_weights_df, all_dates[0], target_symbols, bst, ml_metrics

    # Walk-forward mode
    split_info = split_data(data_dict, split_mode=split_mode, target_window=max(windows))
    folds = split_info['folds']
    
    if not folds:
        if split_mode != 'full':
            raise ValueError(f"Unsupported or empty split_mode '{split_mode}'. Cannot generate walk-forward folds.")
        params_copy = params.copy()
        params_copy['split_mode'] = 'full'
        return generate_walk_forward_target_weights(data_dict, params_copy, local_data_dir, all_dates)
        
    from multifactor_portfolio.util.macro_collector import download_macro_features
    clean_all_dates_list = [d for d in all_dates if pd.notna(d)]
    min_date = pd.Timestamp(min(clean_all_dates_list)).strftime('%Y-%m-%d') if clean_all_dates_list else "2020-01-01"
    print("Downloading global macro features...")
    global_macro_df = download_macro_features(local_data_dir, start_date=min_date)

    all_active_symbols = set()
    fold_weights_list = []
    fold_y_true = []
    fold_y_pred = []
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
        times_fold_list = [t for t in klines_all_fold.index.get_level_values('time') if pd.notna(t)] if not klines_all_fold.empty else []
        start_date = pd.Timestamp(min(times_fold_list)).strftime('%Y-%m-%d') if times_fold_list else "2020-01-01"
        futures_data = download_missing_data(
            symbols=target_symbols,
            data_dir=local_data_dir,
            start_date=start_date
        )
        
        funding_fold = futures_data['funding'].loc[:test_end]
        macro_fold = global_macro_df.loc[:test_end]
        valid_train_dates = [df.index.max() for df in train_dict.values() if not df.empty and len(df.index) > 0]
        train_end = max(valid_train_dates) if valid_train_dates else test_end
        
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
            
        if panel_train.empty:
            print(f"Warning: Fold {label} panel_train is empty. Skipping fold.")
            continue
            
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
        
        y_test = panel_test['target']
        fold_y_true.append(y_test)
        fold_y_pred.append(pd.Series(preds, index=panel_test.index))
        
        # Unstack predictions
        predicted_returns_df = pd.Series(preds, index=panel_test.index).unstack(level='Symbol').fillna(0.0)
        
        # Cross-sectional binning on predictions
        engine_bin = CrossSectionalFactorEngine(symbols=target_symbols, quantiles=params.get('quantiles', 20))
        final_weights_fold = engine_bin.create_cross_sectional_bins(predicted_returns_df)
        fold_weights_list.append(final_weights_fold)
        
    # Combine out-of-sample weights
    if not fold_weights_list:
        raise ValueError("No valid walk-forward folds produced non-empty target weights.")
    oos_weights_df = pd.concat(fold_weights_list, axis=0).sort_index().fillna(0.0)
    first_test_start = folds[0]['test'][next(iter(folds[0]['test']))].index.min()
    
    if fold_y_true and fold_y_pred:
        concat_y_true = pd.concat(fold_y_true)
        concat_y_pred = pd.concat(fold_y_pred)
        ml_metrics = calculate_ml_evaluation_metrics(concat_y_true, concat_y_pred.values, concat_y_true.index)
    else:
        ml_metrics = {}
        
    return oos_weights_df, first_test_start, list(all_active_symbols), last_bst, ml_metrics

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
    
    target_weights_df, eval_start_date, target_symbols, bst, ml_metrics = generate_walk_forward_target_weights(
        data_dict=data_dict,
        params=params,
        local_data_dir=local_data_dir,
        all_dates=all_dates
    )
    
    eval_dates = all_dates[all_dates >= eval_start_date]
    target_weights_df = target_weights_df.reindex(eval_dates).fillna(0.0)
    
    # Raw underlying prices (ONLY forward fill for weekend/holiday gaps, NO bfill artificial backfilling)
    full_underlying_price_df = get_underlying_price_df(data_dict, target_symbols).ffill()
    full_underlying_returns = full_underlying_price_df.pct_change()
    underlying_price_df = full_underlying_price_df.reindex(eval_dates).ffill()
    
    print("Calculating inverse volatility risk weighting...")
    full_risk_weights = calculate_inverse_volatility_weighting(
        underlying=full_underlying_returns, 
        weights=target_weights_df.reindex(all_dates).fillna(0.0), 
        period=params.get('inverse_vol_period', 120)
    )
    risk_weights = full_risk_weights.reindex(eval_dates).fillna(0.0)
    portfolio_weights = risk_weights.copy()
    
    # PA 5.1: Sigmoid Continuous Exposure Scaling
    try:
        from multifactor_portfolio.util.macro_collector import download_macro_features
        print("Applying PA 5.1 Sigmoid Continuous Macro-Regime Risk Overlay...")
        clean_all_dates = [d for d in all_dates if pd.notna(d)]
        clean_eval_dates = [d for d in eval_dates if pd.notna(d)]
        min_hist_date = pd.Timestamp(min(clean_all_dates)).strftime('%Y-%m-%d') if clean_all_dates else "2020-01-01"
        max_hist_date = pd.Timestamp(max(clean_eval_dates)).strftime('%Y-%m-%d') if clean_eval_dates else "2026-07-29"
        macro_df = download_macro_features(local_data_dir, start_date=min_hist_date, end_date=max_hist_date)
        if not macro_df.empty:
            # ONLY ffill weekend macro gaps, NO bfill artificial backfilling into past
            macro_df = macro_df.reindex(all_dates).ffill()
            
            vix_z = (macro_df['vix'] - macro_df['vix'].rolling(120, min_periods=30).mean()) / macro_df['vix'].rolling(120, min_periods=30).std().replace(0, 1)
            fng_z = -(macro_df['fear_greed'] - macro_df['fear_greed'].rolling(120, min_periods=30).mean()) / macro_df['fear_greed'].rolling(120, min_periods=30).std().replace(0, 1)
            dvol_z = (macro_df['dvol_btc'] - macro_df['dvol_btc'].rolling(120, min_periods=30).mean()) / macro_df['dvol_btc'].rolling(120, min_periods=30).std().replace(0, 1)
            
            # Regime-Aware Macro Stress: Scale down exposure ONLY during high volatility + negative market momentum (crash regimes)
            btc_returns = full_underlying_returns['BTCUSDT'] if 'BTCUSDT' in full_underlying_returns.columns else full_underlying_returns.mean(axis=1)
            btc_mom = btc_returns.rolling(30, min_periods=10).mean()
            crash_filter = (btc_mom < 0.0).astype(float).reindex(all_dates).fillna(0.5)

            stress_score = (vix_z.fillna(0.0) + fng_z.fillna(0.0) + dvol_z.fillna(0.0)) / 3.0
            
            # Exposure multiplier decreases ONLY when stress is high AND market is in crash mode
            effective_stress = stress_score * crash_filter
            regime_multiplier = 1.0 / (1.0 + np.exp(1.5 * (effective_stress - 0.5)))
            stress_mult_floor = params.get('stress_multiplier', 0.4)
            regime_multiplier = regime_multiplier.clip(lower=stress_mult_floor, upper=1.0)
            
            eval_multiplier = regime_multiplier.reindex(eval_dates).fillna(1.0)
            portfolio_weights = portfolio_weights.mul(eval_multiplier, axis=0)
            print(f"Applied Regime-Aware Sigmoid Exposure Scaling. Mean exposure multiplier: {eval_multiplier.mean():.4f}")
    except Exception as e:
        print(f"Warning: Failed to apply PA 5.1 Sigmoid Macro Risk Overlay: {e}")
        
    allocation_cap = params.get('allocation_cap', 0.15)
    portfolio_weights = portfolio_weights.clip(lower=-allocation_cap, upper=allocation_cap)
    
    # Rebalance Drift Threshold Filter: Only rebalance if weight drift >= threshold (reduces trade turnover friction)
    rebalance_thresh = params.get('rebalance_threshold', 0.03)
    if rebalance_thresh > 0.0 and not portfolio_weights.empty:
        filtered_weights = portfolio_weights.copy()
        prev_row = filtered_weights.iloc[0].copy()
        for idx in range(1, len(filtered_weights)):
            curr_row = filtered_weights.iloc[idx].copy()
            drift = (curr_row - prev_row).abs()
            no_rebalance_mask = drift < rebalance_thresh
            curr_row[no_rebalance_mask] = prev_row[no_rebalance_mask]
            filtered_weights.iloc[idx] = curr_row
            prev_row = curr_row
        portfolio_weights = filtered_weights
        print(f"Applied Rebalance Drift Threshold Filter ({rebalance_thresh*100:.1f}%).")

    print("Running portfolio backtest via QuantBT Endpoint...")
    from src.multifactor_mlops.backtest import QuantBTRunner, QuantBTExecutionError
    from src.multifactor_mlops.config import load_config
    
    # Anti-Look-Ahead Bias: Enforce 1-bar execution lag
    scaled_weights = portfolio_weights.shift(1).fillna(0.0)
    
    runner = QuantBTRunner(quantbt_repo_path=params.get('quantbt_repo_path', '/root/bobby/pool_alpha/quantbt'))
    equity_df, qbt_metrics_report, qbt_res = runner.run_backtest(
        positions=scaled_weights,
        data_dict=data_dict,
        params=params
    )
    
    metrics = calculate_performance_metrics(
        equity_df, 
        trading_days_per_year=params.get('trading_days_per_year', 365)
    )
    if qbt_metrics_report:
        metrics.update(qbt_metrics_report)
    metrics.update(ml_metrics)
    
    print("=== ML MODEL EVALUATION METRICS ===")
    print(f"  -> Accuracy (Sign Direction): {metrics.get('ml_accuracy', 0.0):.4f}")
    print(f"  -> F1-Score: {metrics.get('ml_f1_score', 0.0):.4f}")
    print(f"  -> Precision: {metrics.get('ml_precision', 0.0):.4f} | Recall: {metrics.get('ml_recall', 0.0):.4f}")
    print(f"  -> Rank IC: {metrics.get('ml_rank_ic', 0.0):.4f} | IC IR: {metrics.get('ml_ic_ir', 0.0):.4f}")
    print(f"  -> R2 Explanatory Power: {metrics.get('ml_r2_score', 0.0):.4f} | MSE: {metrics.get('ml_mse', 0.0):.6f}")
    
    # Attach model for MLflow registration
    metrics['__bst__'] = bst
    metrics['__selected_features__'] = target_symbols
    
    # Automatically save performance_report.txt and performance_chart.png at project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    save_backtest_report_and_chart(equity_df, metrics, params, project_root)
    
    return portfolio_weights, equity_df, metrics

def run_dual_mode_backtest(
    data_dict: Dict[str, pd.DataFrame],
    params: dict,
    local_data_dir: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executes backtests for both Mode 4 (Walk-Forward OOS) and Mode 5 (Full-Sample)
    and formats a synchronous QuantBT comparative report.
    """
    print("\n====================================================")
    print("   EXECUTING DUAL-MODE BACKTEST (MODE 4 & MODE 5)   ")
    print("====================================================\n")

    # Mode 4: Walk-Forward OOS
    params_mode4 = params.copy()
    params_mode4['split_mode'] = 'train_test_split_2024'
    params_mode4['optimization_mode'] = 'mode_4_is_only_robust'
    print("--- Running Mode 4 (Walk-Forward OOS: 2024 - 2026) ---")
    weights_m4, equity_m4, metrics_m4 = run_strategy_backtest(data_dict, params_mode4, local_data_dir)

    # Mode 5: Full-Sample
    params_mode5 = params.copy()
    params_mode5['split_mode'] = 'full'
    params_mode5['optimization_mode'] = 'mode_5_full_robust'
    print("\n--- Running Mode 5 (Full-Sample: 2020 - 2026) ---")
    weights_m5, equity_m5, metrics_m5 = run_strategy_backtest(data_dict, params_mode5, local_data_dir)

    print("\n====================================================")
    print("   QUANTBT DUAL-MODE COMPARATIVE REPORT")
    print("====================================================")
    print("Metrics (QuantBT)               Mode 4 (OOS 2024-26)   Mode 5 (Full 2020-26)")
    print("-" * 65)
    print(f"Sharpe Ratio                    {metrics_m4.get('sharpe_ratio', 0.0):<22.4f} {metrics_m5.get('sharpe_ratio', 0.0):.4f}")
    print(f"CAGR (%)                        {metrics_m4.get('cagr_pct', 0.0):<22.2f}% {metrics_m5.get('cagr_pct', 0.0):.2f}%")
    print(f"Total Return (%)                {metrics_m4.get('total_return_pct', 0.0):<22.2f}% {metrics_m5.get('total_return_pct', 0.0):.2f}%")
    print(f"Max Drawdown (%)                {metrics_m4.get('max_drawdown_pct', 0.0):<22.2f}% {metrics_m5.get('max_drawdown_pct', 0.0):.2f}%")
    print(f"Profit Factor                   {metrics_m4.get('profit_factor', 0.0):<22.4f} {metrics_m5.get('profit_factor', 0.0):.4f}")
    print(f"Long Hit Rate (%)               {metrics_m4.get('long_hitrate_pct', 0.0):<22.2f}% {metrics_m5.get('long_hitrate_pct', 0.0):.2f}%")
    print(f"Short Hit Rate (%)              {metrics_m4.get('short_hitrate_pct', 0.0):<22.2f}% {metrics_m5.get('short_hitrate_pct', 0.0):.2f}%")
    print(f"ML Sign Accuracy (%)            {metrics_m4.get('ml_accuracy', 0.0)*100:<22.2f}% {metrics_m5.get('ml_accuracy', 0.0)*100:.2f}%")
    print("====================================================\n")

    res_m4 = {"weights": weights_m4, "equity": equity_m4, "metrics": metrics_m4}
    res_m5 = {"weights": weights_m5, "equity": equity_m5, "metrics": metrics_m5}
    return res_m4, res_m5
