import os
import joblib
import argparse
import mlflow
import mlflow.pyfunc
from dotenv import load_dotenv
load_dotenv()

class MultifactorPortfolioModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Loads the serialized model/parameter bundle and booster.
        """
        import joblib
        import json
        
        self.model_bundle = joblib.load(context.artifacts["model_bundle"])
        self.params = self.model_bundle["params"]
        self.symbols = self.model_bundle["symbols"]
        
        # Load model booster based on model_type
        model_type = self.params.get('model_type', 'xgboost')
        if model_type == 'lightgbm':
            import lightgbm as lgb
            self.bst = lgb.Booster(model_file=context.artifacts["model_lgb"])
        else:
            import xgboost as xgb
            self.bst = xgb.Booster()
            self.bst.load_model(context.artifacts["model_xgb"])
        
        # Load selected features list
        with open(context.artifacts["selected_features"]) as f:
            self.selected_features = json.load(f)

    def predict(self, context, model_input):
        """
        Receives real-time strategy inputs and generates the target portfolio weights.
        """
        import pandas as pd
        import numpy as np
        import xgboost as xgb
        from multifactor_portfolio.util.factors import CrossSectionalFactorEngine
        from multifactor_portfolio.util.rebalance import calculate_inverse_volatility_weighting, get_underlying_price_df
        
        if not isinstance(model_input, dict):
            return {}

        klines_dict = model_input.get('klines_dict', {})
        funding_df = model_input.get('funding_df', pd.DataFrame())
        
        if not klines_dict:
            return {}
            
        # 1. Determine active universe (top 40 liquid symbols)
        kline_list = []
        for symbol, df in klines_dict.items():
            temp_df = df.copy()
            temp_df['Symbol'] = symbol
            temp_df.index.name = 'time'
            kline_list.append(temp_df.set_index('Symbol', append=True).reorder_levels(['Symbol', 'time']))
            
        klines_all = pd.concat(kline_list).sort_index() if kline_list else pd.DataFrame()
        temp_engine = CrossSectionalFactorEngine(symbols=[], quantiles=self.params.get('quantiles', 20))
        top_n = self.params.get('top_n_symbols', 40)
        _ = temp_engine.calculate_market_cap_proxy(klines_all, top_n=top_n)
        target_symbols = temp_engine.symbols
        
        # 2. Resample funding to daily
        funding_daily = pd.DataFrame()
        if not funding_df.empty:
            funding_daily = funding_df.resample('1D').last()
            
        # Download recent macro features for inference
        import os
        data_dir = os.environ.get("MACRO_DATA_DIR", "/tmp/macro_data")
        
        # Calculate recent date range (180 days history for rolling windows)
        all_dates = pd.Index([])
        for df in klines_dict.values():
            all_dates = all_dates.union(df.index)
        latest_time = pd.DatetimeIndex(sorted(all_dates)).max()
        start_date = (latest_time - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
        
        from multifactor_portfolio.util.macro_collector import download_macro_features
        macro_df = download_macro_features(data_dir, start_date=start_date, end_date=latest_time.strftime('%Y-%m-%d'))
        
        macro_features_df = pd.DataFrame()
        if not macro_df.empty:
            macro_features_df = CrossSectionalFactorEngine.generate_macro_features(macro_df, windows=[7, 14, 30, 60, 90])
            
        # 3. Generate features for active symbols
        symbol_dfs = []
        windows = [7, 14, 30, 60, 90]
        
        for symbol in target_symbols:
            if symbol not in klines_dict:
                continue
            kline_df = klines_dict[symbol]
            if kline_df.empty or len(kline_df) < max(windows):
                continue
                
            if not funding_daily.empty and symbol in funding_daily.columns:
                funding_series = funding_daily[symbol]
            else:
                funding_series = pd.Series(0.0, index=kline_df.index)
                
            # Get the features at the latest timestamp
            features_df = CrossSectionalFactorEngine.generate_features_for_symbol(kline_df, funding_series, windows)
            if not macro_features_df.empty:
                features_df = features_df.join(macro_features_df, how='left')
            latest_features = features_df.iloc[-1:].copy()
            latest_features['Symbol'] = symbol
            symbol_dfs.append(latest_features)
            
        if not symbol_dfs:
            return {}
            
        test_panel = pd.concat(symbol_dfs)
        test_panel.index.name = 'Time'
        test_panel = test_panel.reset_index().set_index(['Time', 'Symbol'])
        
        # Select features
        X_test = test_panel[self.selected_features].fillna(0.0)
        
        # Predict expected returns
        model_type = self.params.get('model_type', 'xgboost')
        if model_type == 'lightgbm':
            preds = self.bst.predict(X_test)
        else:
            dtest = xgb.DMatrix(X_test)
            preds = self.bst.predict(dtest)
        
        # Unstack predictions
        predicted_returns_df = pd.Series(preds, index=test_panel.index).unstack(level='Symbol').fillna(0.0)
        
        # 4. Cross-sectional binning on predictions
        engine_bin = CrossSectionalFactorEngine(symbols=target_symbols, quantiles=self.params.get('quantiles', 20))
        final_weights_df = engine_bin.create_cross_sectional_bins(predicted_returns_df)
        
        # 5. Inverse Volatility scaling
        underlying = get_underlying_price_df(klines_dict, target_symbols)
        if underlying.empty:
            return final_weights_df.iloc[-1].to_dict()
            
        underlying_returns = underlying.pct_change()
        risk_weights = calculate_inverse_volatility_weighting(
            underlying=underlying_returns, 
            weights=final_weights_df, 
            period=120
        )
        
        portfolio_weights = final_weights_df.mul(risk_weights, axis="columns")
        
        # Apply Macro-Regime Risk Overlay during inference
        try:
            if not macro_df.empty:
                macro_df_aligned = macro_df.reindex(portfolio_weights.index).ffill().bfill()
                vix_ma = macro_df_aligned['vix'].rolling(14, min_periods=1).mean()
                fng_ma = macro_df_aligned['fear_greed'].rolling(14, min_periods=1).mean()
                dvol_ma = macro_df_aligned['dvol_btc'].rolling(14, min_periods=1).mean()
                
                stress_flag = (vix_ma > 22.0) | (fng_ma < 30.0) | (dvol_ma > 65.0)
                
                stress_multiplier = self.params.get('stress_multiplier', 0.5)
                regime_multiplier = pd.Series(1.0, index=portfolio_weights.index)
                regime_multiplier[stress_flag.fillna(False)] = stress_multiplier
                
                portfolio_weights = portfolio_weights.mul(regime_multiplier, axis=0)
        except Exception as e:
            print(f"Warning: Failed to apply Macro-Regime Risk Overlay during inference: {e}")
            
        allocation_cap = self.params.get('allocation_cap', 0.2)
        portfolio_weights = portfolio_weights.clip(lower=-allocation_cap, upper=allocation_cap)
        
        return portfolio_weights.iloc[-1].fillna(0.0).to_dict()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='multifactor_portfolio_model')
    parser.add_argument('--run_id', type=str)
    args = parser.parse_args()
    
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    
    run = mlflow.get_run(args.run_id)
    model_uri = f"runs:/{args.run_id}/model"
    
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=args.model_name,
        tags={
            "git_sha": run.data.tags.get("git_sha", "unknown"),
            "run_id": args.run_id,
            "experiment_name": run.info.experiment_id
        }
    )
    print(f"Registered model {args.model_name} version {model_version.version}")

if __name__ == '__main__':
    main()
