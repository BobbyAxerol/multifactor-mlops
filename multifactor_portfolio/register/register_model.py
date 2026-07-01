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
        Loads the serialized model/parameter bundle.
        """
        self.model_bundle = joblib.load(context.artifacts["model_bundle"])
        self.params = self.model_bundle["params"]
        self.symbols = self.model_bundle["symbols"]

    def predict(self, context, model_input):
        """
        Receives strategy inputs and generates the target portfolio weights.
        Supports:
        1. A dictionary of dataframes: {'klines_dict': dict, 'ls_ratio_df': DataFrame, 'oi_df': DataFrame, 'funding_df': DataFrame}
        2. Returns a dictionary of the latest target weights by symbol.
        """
        import pandas as pd
        import numpy as np
        from multifactor_portfolio.util.factors import CrossSectionalFactorEngine
        from multifactor_portfolio.util.rebalance import calculate_inverse_volatility_weighting, get_underlying_price_df
        
        if not isinstance(model_input, dict):
            # Fallback for generic/empty inputs
            return {}

        klines_dict = model_input.get('klines_dict', {})
        ls_ratio_df = model_input.get('ls_ratio_df', pd.DataFrame())
        oi_df = model_input.get('oi_df', pd.DataFrame())
        funding_df = model_input.get('funding_df', pd.DataFrame())
        
        if not klines_dict:
            return {}
            
        # Instantiate factor engine
        engine = CrossSectionalFactorEngine(
            symbols=self.symbols, 
            quantiles=self.params.get('quantiles', 20)
        )
        
        # Calculate primary weights
        final_weights_df = engine.run_factor_engine(
            klines_dict=klines_dict,
            ls_ratio_df=ls_ratio_df,
            oi_df=oi_df,
            funding_df=funding_df,
            momentum_params={
                'ma_length': self.params.get('ma_length', 20),
                'rsi_lower': self.params.get('rsi_lower', 50),
                'rsi_upper': self.params.get('rsi_upper', 60)
            },
            retail_params={
                'volume_period': self.params.get('volume_period', 20)
            },
            carry_window=self.params.get('carry_window', 60),
            risk_window=self.params.get('risk_window', 30),
            top_n_symbols=self.params.get('top_n_symbols', 40)
        )
        
        if final_weights_df.empty:
            return {}
            
        # Refine weights with inverse volatility weighting
        target_symbols = final_weights_df.columns.tolist()
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
        allocation_cap = self.params.get('allocation_cap', 0.2)
        portfolio_weights = portfolio_weights.clip(lower=-allocation_cap, upper=allocation_cap)
        
        # Return the latest target weights as dictionary
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
