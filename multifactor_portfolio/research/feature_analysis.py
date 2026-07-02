import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Add strategy path to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from multifactor_portfolio.training.train import load_ohlcv_data
from multifactor_portfolio.util.factors import CrossSectionalFactorEngine
from multifactor_portfolio.util.data_collector import download_missing_data

def run_feature_analysis():
    print("====================================================")
    # 1. Load parameters and config
    param_path = os.path.join(project_root, "parameters.json")
    with open(param_path) as f:
        pars = json.load(f)
        
    features_args = pars.get('features', {})
    dataset_args = pars.get('dataset', {})
    training_args = pars.get('training', {})
    all_params = {**features_args, **dataset_args, **training_args}
    
    # 2. Load daily OHLCV
    data_path = all_params.get("data_path", "/root/bobby/pool_alpha/alphas_storage/_get_data")
    data_dict = load_ohlcv_data(data_path)
    if not data_dict:
        print("Error: No data loaded.")
        return
        
    # Get all unique dates
    all_dates = pd.Index([])
    for df in data_dict.values():
        all_dates = all_dates.union(df.index)
    all_dates = pd.DatetimeIndex(sorted(all_dates))
    
    # 3. Filter timeline to pre-2024 (Training phase only to prevent leakage)
    train_dates = all_dates[all_dates < pd.Timestamp('2024-01-01')]
    print(f"Feature Analysis Train period: {train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')}")
    
    train_dict = {sym: df.loc[df.index.intersection(train_dates)] for sym, df in data_dict.items()}
    train_dict = {sym: df for sym, df in train_dict.items() if not df.empty}
    
    # 4. Determine dynamic active universe (top 40 liquid symbols) on train period
    temp_engine = CrossSectionalFactorEngine(symbols=[], quantiles=20)
    kline_list = []
    for symbol, df in train_dict.items():
        temp_df = df.copy()
        temp_df['Symbol'] = symbol
        temp_df.index.name = 'time'
        kline_list.append(temp_df.set_index('Symbol', append=True).reorder_levels(['Symbol', 'time']))
        
    klines_all = pd.concat(kline_list).sort_index() if kline_list else pd.DataFrame()
    top_n = all_params.get('top_n_symbols', 40)
    
    # Run cap proxy selection
    _ = temp_engine.calculate_market_cap_proxy(klines_all, top_n=top_n)
    target_symbols = temp_engine.symbols
    print(f"Selected Top {len(target_symbols)} liquid symbols for feature analysis.")
    
    # 5. Load historical funding rates
    start_date = str(klines_all.index.get_level_values('time').min().strftime('%Y-%m-%d'))
    local_data_dir = os.path.join(project_root, "data")
    futures_data = download_missing_data(
        symbols=target_symbols,
        data_dir=local_data_dir,
        start_date=start_date,
        end_date='2023-12-31'
    )
    funding_df = futures_data['funding']
    
    # 6. Generate Panel Dataset
    from multifactor_portfolio.util.macro_collector import download_macro_features
    print("Downloading/Loading macro features...")
    macro_df = download_macro_features(local_data_dir, start_date=start_date, end_date='2023-12-31')
    
    windows = [7, 14, 30, 60, 90]
    print(f"Generating candidate features on windows: {windows}...")
    panel_df = CrossSectionalFactorEngine.prepare_panel_dataset(
        data_dict=train_dict,
        funding_df=funding_df,
        symbols=target_symbols,
        macro_df=macro_df,
        windows=windows,
        lag=all_params.get('lag', 1)
    )
    
    if panel_df.empty:
        print("Error: Generated panel dataset is empty.")
        return
        
    print(f"Panel Dataset ready: {panel_df.shape[0]} rows, {panel_df.shape[1] - 1} features.")
    
    # Separate features and target
    X = panel_df.drop(columns=['target'])
    y = panel_df['target']
    
    # Fill any remaining NaNs in features
    X = X.fillna(0.0)
    
    # 7. Evaluate features
    feature_names = X.columns.tolist()
    
    # A. Mutual Information (MI)
    from sklearn.feature_selection import mutual_info_regression
    print("Calculating Mutual Information (MI)...")
    mi_scores = mutual_info_regression(X, y)
    mi_dict = dict(zip(feature_names, mi_scores))
    
    # B. OLS Single-variable Regression
    import statsmodels.api as sm
    ols_pvalues = {}
    ols_rsquared = {}
    print("Running OLS regressions...")
    for col in feature_names:
        try:
            x_var = sm.add_constant(X[col])
            model = sm.OLS(y, x_var).fit()
            ols_pvalues[col] = model.pvalues[col]
            ols_rsquared[col] = model.rsquared
        except Exception:
            ols_pvalues[col] = 1.0
            ols_rsquared[col] = 0.0
            
    # C. Baseline XGBoost Gain Importance
    import xgboost as xgb
    print("Training baseline XGBoost model...")
    dtrain = xgb.DMatrix(X, label=y)
    xgb_params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.05,
        'max_depth': 4,
        'colsample_bytree': 0.3,
        'verbosity': 0
    }
    bst = xgb.train(xgb_params, dtrain, num_boost_round=100)
    importance_dict = bst.get_score(importance_type='gain')
    total_gain = sum(importance_dict.values()) + 1e-8
    xgb_gain_pct = {name: importance_dict.get(name, 0.0) / total_gain for name in feature_names}
    
    # 8. Filter and Rank
    report_rows = []
    selected_features = []
    
    # Hybrid threshold: MI > 0.002 OR XGBoost Gain Importance > 0.005
    for col in feature_names:
        p_val = ols_pvalues[col]
        r2 = ols_rsquared[col]
        mi = mi_dict[col]
        gain_pct = xgb_gain_pct[col]
        
        # Decide keep/drop
        is_sig_ols = "Sig" if p_val < 0.05 else "Non-sig"
        status = "Keep" if (mi > 0.002 or gain_pct > 0.005) else "Drop"
        
        if status == "Keep":
            selected_features.append(col)
            
        report_rows.append({
            'Feature': col,
            'OLS P-value': f"{p_val:.4f} ({is_sig_ols})",
            'OLS R2': f"{r2:.5f}",
            'MI Score': f"{mi:.4f}",
            'XGBM Gain %': f"{gain_pct * 100:.2f}%",
            'Status': status
        })
        
    report_df = pd.DataFrame(report_rows)
    # Sort by XGBM importance descending
    report_df['xgb_sort'] = [xgb_gain_pct[f] for f in report_df['Feature']]
    report_df = report_df.sort_values(by='xgb_sort', ascending=False).drop(columns=['xgb_sort'])
    
    asset_report = report_df[~report_df['Feature'].str.startswith('macro_')]
    macro_report = report_df[report_df['Feature'].str.startswith('macro_')]
    
    print("\n====================================================================================")
    print("                    1. ASSET ALPHA FEATURES (Used for Model Training)                ")
    print("====================================================================================")
    print(asset_report.to_string(index=False))
    print("====================================================================================")
    
    print("\n====================================================================================")
    print("                    2. GLOBAL MACRO FEATURES (Used for Regime Risk Overlay)          ")
    print("====================================================================================")
    print(macro_report.to_string(index=False))
    print("====================================================================================")
    
    # Save selected features to JSON (Only save asset-specific features for return prediction)
    asset_selected_features = [f for f in selected_features if not f.startswith('macro_')]
    macro_selected_features = [f for f in selected_features if f.startswith('macro_')]
    
    print(f"Total Candidate Features (Mixed): {len(feature_names)}")
    print(f"Total Asset Features Kept (For Training): {len(asset_selected_features)}")
    print(f"Total Macro Features Kept (For Regime Overlay): {len(macro_selected_features)}")
    print("====================================================================================")
    
    research_dir = os.path.join(project_root, "multifactor_portfolio", "research")
    os.makedirs(research_dir, exist_ok=True)
    selected_path = os.path.join(research_dir, "selected_features.json")
    with open(selected_path, "w") as f:
        json.dump(asset_selected_features, f, indent=2)
    print(f"Selected asset feature list saved to: {selected_path} ({len(asset_selected_features)} features)")
    
if __name__ == "__main__":
    run_feature_analysis()
