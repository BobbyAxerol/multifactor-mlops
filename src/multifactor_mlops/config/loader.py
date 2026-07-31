"""
Configuration loader that validates parameters.json against Pydantic AppConfig schema.
"""

import json
import os
from typing import Union, Dict, Any
from .schema import AppConfig

def load_config(config_input: Union[str, Dict[str, Any]]) -> AppConfig:
    """
    Loads and validates configuration from a JSON filepath or dictionary.
    Raises ValidationError if any parameter or split mode is invalid.
    """
    if isinstance(config_input, str):
        if not os.path.exists(config_input):
            raise FileNotFoundError(f"Configuration file not found: {config_input}")
        with open(config_input, "r") as f:
            raw_data = json.load(f)
    elif isinstance(config_input, dict):
        raw_data = config_input
    else:
        raise TypeError(f"config_input must be a file path or dict, got {type(config_input)}")

    # Handle flat or nested JSON structure
    if "data" in raw_data or "validation" in raw_data:
        return AppConfig.model_validate(raw_data)

    # Flattened JSON mapping fallback
    flat_data = {
        "data": raw_data.get("dataset", raw_data),
        "features": raw_data.get("features", raw_data),
        "validation": raw_data.get("training", raw_data.get("validation", raw_data)),
        "portfolio": raw_data.get("training", raw_data.get("portfolio", raw_data)),
        "backtest": raw_data.get("training", raw_data.get("backtest", raw_data)),
        "model": raw_data.get("model", raw_data)
    }
    return AppConfig.model_validate(flat_data)

def flatten_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens nested parameter sections (features, dataset, training) into top-level key-value pairs.
    Top-level explicit keys override sub-dictionary keys.
    """
    if not isinstance(params, dict):
        return {}

    flat = {}
    for section in ['dataset', 'features', 'training', 'model', 'portfolio', 'backtest', 'validation', 'data']:
        if section in params and isinstance(params[section], dict):
            flat.update(params[section])

    for k, v in params.items():
        if k not in ['dataset', 'features', 'training', 'model', 'portfolio', 'backtest', 'validation', 'data']:
            flat[k] = v

    return flat
