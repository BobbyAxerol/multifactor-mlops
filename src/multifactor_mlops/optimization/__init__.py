"""Optimization subpackage (lazy exports to avoid circular imports)."""

__all__ = [
    "PureMLStage1Optimizer",
    "Stage2StrategyOptimizer",
    "PurgedExpandingFoldBuilder",
    "train_xgb_model",
    "predict_xgb_model",
    "rank_ic"
]


def __getattr__(name):
    if name == "PureMLStage1Optimizer":
        from .stage1_ml_tuning import PureMLStage1Optimizer
        return PureMLStage1Optimizer
    if name == "Stage2StrategyOptimizer":
        from .stage2_strategy_tuning import Stage2StrategyOptimizer
        return Stage2StrategyOptimizer
    if name == "PurgedExpandingFoldBuilder":
        from .folds import PurgedExpandingFoldBuilder
        return PurgedExpandingFoldBuilder
    if name in ("train_xgb_model", "predict_xgb_model", "rank_ic"):
        from . import model_utils
        return getattr(model_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
