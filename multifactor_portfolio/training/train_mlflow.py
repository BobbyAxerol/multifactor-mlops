import os  
import json  
import warnings
warnings.filterwarnings("ignore")
import argparse  
import joblib  
import pandas as pd  
import numpy as np  
import mlflow  
import mlflow.pyfunc  
import datetime  
import xgboost as xgb
from dotenv import load_dotenv
load_dotenv()

from multifactor_portfolio.training.train import load_ohlcv_data, run_strategy_backtest
from multifactor_portfolio.register.register_model import MultifactorPortfolioModelWrapper

def save_performance_chart(perf: pd.DataFrame, filename: str = "performance_chart.png"):
    import matplotlib.pyplot as plt
    if perf.empty:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(perf['time'], perf['equity'], label='Portfolio Equity', color='blue', linewidth=2)
    ax1.set_title('Equity Portfolio Value')
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True)
    ax1.legend()
    
    dd = (perf['equity'] / perf['equity'].cummax()) - 1.0
    ax2.plot(perf['time'], dd, label='Drawdown', color='red', linewidth=1.5)
    ax2.fill_between(perf['time'], dd, 0, color='red', alpha=0.3)
    ax2.set_title('Drawdown Chart')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def _get_git_metadata():
    import subprocess
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
        status = subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip()
        is_dirty = len(status) > 0
        return sha, branch, is_dirty
    except Exception:
        return "unknown", "unknown", True

def main():  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--data_path', type=str, default='/root/bobby/pool_alpha/alphas_storage/_get_data')  
    parser.add_argument('--model_name', type=str, default='multifactor_portfolio_model')  
    args = parser.parse_args()  
  
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))  
    mlflow.set_experiment(os.environ.get('MLFLOW_EXPERIMENT_NAME', 'multifactor_portfolio'))  
  
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    param_path = os.path.join(project_root, "parameters.json")
  
    with open(param_path) as f:  
        pars = json.load(f)  
        
    features_args = pars.get('features', {})
    dataset_args = pars.get('dataset', {})
    training_args = pars.get('training', {})
    
    all_params = {**features_args, **dataset_args, **training_args}
    
    asset_class = all_params.get('asset_class', 'crypto')
    universe_name = all_params.get('universe_name', 'binance_daily')
    top_n = all_params.get('top_n_symbols', 40)
  
    with mlflow.start_run() as run:  
        sha, branch, is_dirty = _get_git_metadata()
        mlflow.set_tag("git_sha", os.environ.get("GITHUB_SHA", sha))
        mlflow.set_tag("git_branch", branch)
        mlflow.set_tag("git_dirty", str(is_dirty))
        
        # Log all strategy parameters
        mlflow.log_params(all_params)
          
        # Load dataset
        raw_dict = load_ohlcv_data(args.data_path)  
        if not raw_dict:  
             raise ValueError(f"No data found at {args.data_path}")  
           
        local_data_dir = os.path.join(project_root, "data")
        portfolio_weights, final_equity_df, metrics = run_strategy_backtest(
            data_dict=raw_dict,
            params=all_params,
            local_data_dir=local_data_dir
        )
        
        # Extract model artifacts
        bst = metrics.pop('__bst__', None)
        active_symbols = metrics.pop('__selected_features__', None)
        
        symbols_list = list(portfolio_weights.columns)
        with open("universe_symbols.json", "w") as sf:
            json.dump(symbols_list, sf, indent=2)
        mlflow.log_artifact("universe_symbols.json")
        
        # Log backtest metrics
        mlflow.log_metrics(metrics)  
        
        if not final_equity_df.empty:
            save_performance_chart(final_equity_df, "performance_chart.png")
            mlflow.log_artifact("performance_chart.png")
          
        # Save model parameters & active symbols list
        model_bundle = {  
            'params': all_params,
            'symbols': symbols_list
        }  
        joblib.dump(model_bundle, "model_bundle.joblib")  
        
        # Save model booster based on model_type
        model_type = all_params.get('model_type', 'xgboost')
        
        model_artifacts = {
            "model_bundle": "model_bundle.joblib",
            "selected_features": selected_path
        }
        
        if model_type == 'lightgbm':
            if bst is not None:
                bst.save_model("model.txt")
            else:
                import lightgbm as lgb
                dummy_bst = lgb.train({'objective': 'regression', 'verbosity': -1}, lgb.Dataset(np.zeros((2,2)), label=np.zeros(2)), num_boost_round=1)
                dummy_bst.save_model("model.txt")
            model_artifacts["model_lgb"] = "model.txt"
        else:
            if bst is not None:
                bst.save_model("model.json")
            else:
                dummy_bst = xgb.train({'objective': 'reg:squarederror'}, xgb.DMatrix(np.zeros((2,2)), label=np.zeros(2)), num_boost_round=1)
                dummy_bst.save_model("model.json")
            model_artifacts["model_xgb"] = "model.json"
            
        # Log Python PyFunc Model
        mlflow.pyfunc.log_model(  
            artifact_path="model",  
            python_model=MultifactorPortfolioModelWrapper(),  
            artifacts=model_artifacts,
            code_paths=[os.path.join(project_root, "multifactor_portfolio")]
        )  
  
        model_uri = f"runs:/{run.info.run_id}/model"  
        try:  
            model_version = mlflow.register_model(  
                model_uri=model_uri,  
                name=args.model_name  
            )  
            mlflow.set_tag("mlflow.model.version", model_version.version)  
            client = mlflow.tracking.MlflowClient()
            desc = (
                f"### XGBoost Model Version Card\n"
                f"- **Trained At**: {datetime.datetime.now().isoformat()}\n"
                f"- **Git Commit**: {sha} (Branch: {branch}, Dirty: {is_dirty})\n"
                f"- **Performance (OOS Backtest)**:\n"
                f"  - Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0)}\n"
                f"  - CAGR: {metrics.get('cagr', 0.0)}\n"
                f"  - Max Drawdown: {metrics.get('max_drawdown', 0.0)}\n"
            )
            client.update_model_version(
                name=args.model_name,
                version=model_version.version,
                description=desc
            )
        except Exception as ex:  
            print(f"Error registering model version: {ex}")
            mlflow.set_tag("mlflow.model.version", "1")  
          
        mlflow.set_tag("run_id", run.info.run_id)  
  
        report = []  
        report.append("====================================================")  
        report.append("          MULTIFACTOR PERFORMANCE REPORT            ")  
        report.append("====================================================")  
        report.append(f"Timestamp: {datetime.datetime.now().isoformat()}")  
        report.append(f"Asset Class: {asset_class}")
        report.append(f"Universe Name: {universe_name}")
        report.append(f"Lag: {all_params.get('lag')}")  
        report.append(f"Transaction Fee: {all_params.get('fee')}")  
        
        if not final_equity_df.empty:
            report.append(f"Data Start Date: {final_equity_df['time'].min().strftime('%Y-%m-%d')}")  
            report.append(f"Data End Date: {final_equity_df['time'].max().strftime('%Y-%m-%d')}")  
        report.append(f"Total Symbols: {len(symbols_list)}")  
        report.append("----------------------------------------------------")  
        report.append("Performance Metrics:")  
        for k, v in metrics.items():  
            report.append(f"  {k}: {v}")  
        report.append("====================================================")  
          
        report_text = "\n".join(report)  
        print(report_text)  
        with open("performance_report.txt", "w") as rf:  
            rf.write(report_text)  

if __name__ == '__main__':  
    main()
