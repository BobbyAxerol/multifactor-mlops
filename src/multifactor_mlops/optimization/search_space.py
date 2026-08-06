"""
Shared Optuna search-space helpers: suggest params from a DECLARED search space
(parameters.json optimization.stageN.search_space). No hardcoded ranges in code.
"""

from typing import Any, Dict

import optuna


def suggest_from_space(trial: optuna.Trial, name: str, spec: Any):
    """
    spec forms:
      {"low": x, "high": y, "step": z}  -> numeric (int if low/high are ints)
      ["a", "b", "c"]                    -> categorical
    """
    if isinstance(spec, dict):
        low = spec["low"]
        high = spec["high"]
        step = spec.get("step")
        if isinstance(low, int) and isinstance(high, int):
            if step is not None:
                return trial.suggest_int(name, low, high, step=step)
            return trial.suggest_int(name, low, high)
        if step is not None:
            return trial.suggest_float(name, float(low), float(high), step=float(step))
        return trial.suggest_float(name, float(low), float(high))
    if isinstance(spec, (list, tuple)):
        return trial.suggest_categorical(name, [str(x) for x in spec])
    raise ValueError(f"Invalid search-space spec for '{name}': {spec!r}")


def suggest_all(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    return {name: suggest_from_space(trial, name, spec) for name, spec in search_space.items()}
