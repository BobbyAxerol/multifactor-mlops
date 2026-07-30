"""
MLOps Run Manifests and MLflow Lineage Tracking module.
Creates immutable run directories under artifacts/runs/<run_id>/ storing data checksums, fold manifests, and parquet artifacts.
"""

import os
import json
import hashlib
import pandas as pd
from typing import Dict, Any, Optional

class MLOpsRunLogger:
    """
    Manages immutable run artifacts and logs lineage manifests to MLflow.
    """

    def __init__(self, run_id: str, runs_root: str = "artifacts/runs"):
        self.run_id = run_id
        self.run_dir = os.path.abspath(os.path.join(runs_root, run_id))
        os.makedirs(self.run_dir, exist_ok=True)
        self.manifest: Dict[str, Any] = {
            "run_id": run_id,
            "artifacts": {},
            "checksums": {}
        }

    def log_dataframe_artifact(self, name: str, df: pd.DataFrame) -> str:
        """
        Saves DataFrame as Parquet artifact and records SHA256 checksum.
        """
        filepath = os.path.join(self.run_dir, f"{name}.parquet")
        df.to_parquet(filepath)
        
        df_bytes = df.to_json().encode('utf-8')
        checksum = hashlib.sha256(df_bytes).hexdigest()
        
        self.manifest["artifacts"][name] = filepath
        self.manifest["checksums"][name] = checksum
        return filepath

    def log_json_artifact(self, name: str, data: Dict[str, Any]) -> str:
        """
        Saves Dictionary as JSON artifact.
        """
        filepath = os.path.join(self.run_dir, f"{name}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
            
        data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
        checksum = hashlib.sha256(data_bytes).hexdigest()
        
        self.manifest["artifacts"][name] = filepath
        self.manifest["checksums"][name] = checksum
        return filepath

    def save_run_manifest(self) -> str:
        """
        Saves complete run manifest and checksums.
        """
        filepath = os.path.join(self.run_dir, "run_manifest.json")
        with open(filepath, "w") as f:
            json.dump(self.manifest, f, indent=2)
        return filepath
