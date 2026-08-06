"""
Export immutable production bundle (model + preprocessor + overlay spec).

Requires locked artifacts/model_config.json + artifacts/strategy_config.json.
Fits the final model on rows with label_end_time <= training_cutoff using the
SAME pipeline as backtest folds (PanelDatasetBuilder + preprocessor train-only),
then writes the bundle to artifacts/model_bundle/.

Usage:
    poetry run python src/multifactor_mlops/register/export_bundle.py --training-cutoff 2023-12-31
"""

import os
import json
import argparse

from src.multifactor_mlops.config.loader import load_config
from src.multifactor_mlops.data.loader import load_all_data
from src.multifactor_mlops.pipelines.fit_final import fit_final_production_model

MODEL_CONFIG_PATH = "artifacts/model_config.json"
STRATEGY_CONFIG_PATH = "artifacts/strategy_config.json"
BUNDLE_DIR = "artifacts/model_bundle"


def export_bundle(
    config_path: str = "parameters.json",
    training_cutoff: str = "2023-12-31",
    data_dir: str = "./data",
    output_dir: str = BUNDLE_DIR,
):
    for p in (MODEL_CONFIG_PATH, STRATEGY_CONFIG_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing locked artifact {p}. Fail-fast, no silent defaults.")

    with open(MODEL_CONFIG_PATH, "r") as f:
        model_config = json.load(f)
    with open(STRATEGY_CONFIG_PATH, "r") as f:
        strategy_config = json.load(f)

    app_config = load_config(config_path)
    data_dict, macro_df, funding_dict = load_all_data(
        app_config, data_dir=data_dir, end_date=training_cutoff
    )

    bundle = fit_final_production_model(
        data_dict=data_dict,
        symbols=list(data_dict.keys()),
        macro_df=macro_df,
        funding_df=None,
        config_path=config_path,
        production_cutoff=training_cutoff,
        overlay_params=strategy_config,
    )
    bundle.save(output_dir)
    print(f"[Bundle] Exported -> {output_dir}")
    print(f"[Bundle] features={len(bundle.feature_names)}, rows={bundle.lineage_manifest['total_rows']}")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-cutoff", default="2023-12-31")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()
    export_bundle(training_cutoff=args.training_cutoff, data_dir=args.data_dir)
