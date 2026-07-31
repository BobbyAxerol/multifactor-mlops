"""
Root Model Registration entry point.
Refits final production model bundle through production_cutoff and registers bundle in MLflow Model Registry.
"""

import sys
import os
import json
import mlflow

from src.multifactor_mlops.config.loader import load_config
from src.multifactor_mlops.data.historical_adapter import HistoricalDataAdapter
from src.multifactor_mlops.pipelines.fit_final import fit_final_production_model

def register_final_model(config_path: str = "parameters.json") -> str:
    """
    Refits final model on full historical dataset and registers production model bundle in MLflow.
    """
    app_config = load_config(config_path)
    
    data_adapter = HistoricalDataAdapter(data_repo_path=app_config.data.data_repo_path)
    data_dict = data_adapter.load_ohlcv_data()
    symbols = list(data_dict.keys())[:app_config.data.top_n_symbols]
    
    print(f"Refitting final production model for {len(symbols)} symbols...")
    bundle = fit_final_production_model(
        data_dict=data_dict,
        symbols=symbols,
        config_path=config_path,
        production_cutoff=app_config.run.production_cutoff
    )

    mlflow.set_experiment(app_config.backtest.engine.upper() + "_Production_Registration")
    with mlflow.start_run(run_name="final_production_refit") as run:
        bundle_dir = os.path.abspath("./artifacts/models/final_bundle")
        bundle.save(bundle_dir)
        
        mlflow.log_artifacts(bundle_dir, artifact_path="model_bundle")
        mlflow.log_params(app_config.model.model_dump())
        mlflow.log_dict(bundle.lineage_manifest, "lineage_manifest.json")
        
        run_id = run.info.run_id
        print(f"Successfully refitted and registered production bundle. Run ID: {run_id}")
        return run_id

if __name__ == "__main__":
    register_final_model()
