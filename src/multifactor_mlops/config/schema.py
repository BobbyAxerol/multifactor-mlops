"""
Pydantic Schema definitions for Multifactor MLOps strategy parameters.
Enforces strict configuration validation and raises immediately on invalid parameters.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict

VALID_SPLIT_MODES = {
    "train_test_split_2024",
    "train_test_split_2023",
    "train_test_split_2022",
    "walk_forward_semi_yearly",
    "walk_forward_quarterly",
    "walk_forward_2024",
    "walk_forward_2023",
    "walk_forward_2022",
    "full"
}

VALID_SPLIT_FREQUENCIES = {"single", "yearly", "semi_yearly", "quarterly", "monthly", "weekly"}
VALID_WINDOW_MODES = {"expanding", "rolling"}

class RunConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    seed: int = 42
    timezone: str = "UTC"
    fail_fast: bool = True
    production_cutoff: Optional[str] = None

class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    asset_class: str = "crypto"
    universe_name: str = "binance_daily"
    top_n_symbols: int = 40
    data_repo_path: str = "/root/bobby/pool_alpha/alphas_storage/_get_data"
    quantbt_repo_path: str = "/root/bobby/pool_alpha/quantbt"
    allow_bfill: bool = False
    # Point-in-time universe: membership by rolling lagged turnover (no
    # full-sample survivorship bias in the traded symbol set).
    use_point_in_time_universe: bool = True
    universe_lookback_days: int = 30
    universe_min_history_days: int = 180

    @field_validator("allow_bfill")
    @classmethod
    def validate_no_bfill(cls, v: bool) -> bool:
        if v:
            raise ValueError("allow_bfill=True is strictly prohibited to prevent look-ahead bias.")
        return v

class FeaturesConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    windows: List[int] = Field(default_factory=lambda: [7, 14, 30, 60, 90])
    quantiles: int = 16
    warmup_policy: str = "drop"
    missing_policy: str = "preserve"
    # Evidence-backed feature selection (research V4.1):
    # keep only factor families that show cross-window tail spread (dev + OOS).
    keep_families: Optional[List[str]] = Field(default=None)
    # Sign-flip these features BEFORE cross-sectional ranking (their tail spread
    # is negative: high values mean FUTURE UNDERPERFORMANCE).
    inverted_features: List[str] = Field(default_factory=list)
    # Macro features are per-day constants -> zero cross-sectional IC by
    # construction; exclude them from the model feature set (still used as overlay).
    use_macro_features: bool = True

class LabelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # CANONICAL: next_close_to_close matches the QuantBT engine realization
    # (decision close D -> fill close D+1 -> exit close D+1+H).
    return_type: str = "next_close_to_close"
    holding_bars: int = 1
    cross_sectional_demean: bool = True

    @field_validator("return_type")
    @classmethod
    def validate_return_type(cls, v: str) -> str:
        if v not in {"next_close_to_close", "next_open_to_open"}:
            raise ValueError(
                f"Invalid return_type '{v}'. Supported: next_close_to_close (canonical), "
                f"next_open_to_open (research only)."
            )
        return v

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model_type: str = "xgboost"
    learning_rate: float = 0.07
    max_depth: int = 5
    num_boost_round: int = 100
    colsample_bytree: float = 0.3
    subsample: float = 0.8
    random_state: int = 42
    train_step_days: int = 4

class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    split_mode: str = "walk_forward_2024"
    split_frequency: str = "quarterly"
    window_mode: str = "expanding"
    train_window: Optional[str] = None
    min_train_bars: int = 120
    min_test_bars: int = 20
    first_oos_date: str = "2024-01-01"
    outer_test_period: str = "quarterly"
    purge_bars: int = 1
    embargo_bars: int = 1
    unsupported_split_policy: str = "raise"

    @field_validator("split_mode")
    @classmethod
    def validate_split_mode(cls, v: str) -> str:
        if not (v in VALID_SPLIT_MODES or v.startswith("train_test_split_") or v.startswith("walk_forward_")):
            raise ValueError(f"Invalid split_mode '{v}'. Supported split modes: {VALID_SPLIT_MODES}")
        return v

    @field_validator("split_frequency")
    @classmethod
    def validate_split_frequency(cls, v: str) -> str:
        if v not in VALID_SPLIT_FREQUENCIES:
            raise ValueError(f"Invalid split_frequency '{v}'. Supported: {VALID_SPLIT_FREQUENCIES}")
        return v

    @field_validator("window_mode")
    @classmethod
    def validate_window_mode(cls, v: str) -> str:
        if v not in VALID_WINDOW_MODES:
            raise ValueError(f"Invalid window_mode '{v}'. Supported: {VALID_WINDOW_MODES}")
        return v

class PortfolioConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    quantiles: int = 16
    allocation_cap: float = 0.25
    inverse_vol_period: int = 42
    volatility_ceiling: float = 0.08
    rebalance_schedule: str = "weekly_friday_exit"
    rebalance_threshold: float = 0.05
    # Signal source: "ml" (XGBoost per-fold) or "composite" (model-free
    # per-timestamp z-score sum of composite_features — evidence-based V4.1).
    signal_mode: str = "ml"
    composite_features: List[str] = Field(default_factory=list)
    stress_vix_threshold: float = 26.0
    stress_fng_threshold: float = 20.0
    stress_dvol_threshold: float = 50.0
    stress_multiplier: float = 0.4
    portfolio_mode: str = "longshort"
    hedge_type: str = "target_weight"

class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    engine: str = "quantbt"
    backend: str = "native_portfolio"
    portfolio_mode: str = "longshort"
    hedge_type: str = "target_weight"
    allow_engine_fallback: bool = False
    initial_capital: float = 100000.0
    leverage: float = 3.0
    fee_rate_per_fill: float = 0.0005
    fee: float = 0.001
    slippage: float = 0.0001
    trading_days_per_year: int = 365
    use_funding: bool = True

    @field_validator("allow_engine_fallback")
    @classmethod
    def validate_no_engine_fallback(cls, v: bool) -> bool:
        if v:
            raise ValueError("allow_engine_fallback=True is strictly prohibited. QuantBT is the sole backtest engine.")
        return v

class Stage1TuningConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    n_trials: int = 30
    random_seed: int = 42
    storage_uri: str = "sqlite:///artifacts/optuna/stage1.db"
    dev_end: str = "2023-12-31"
    inner_start: str = "2022-01-01"
    frequency: str = "quarterly"
    # search_space: {"param": {"low": x, "high": y, "step": z}} (numeric)
    #              | {"param": ["a", "b", "c"]} (categorical)
    search_space: Dict[str, Any] = Field(default_factory=dict)

class Stage2TuningConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    n_trials: int = 15
    random_seed: int = 42
    storage_uri: str = "sqlite:///artifacts/optuna/stage2.db"
    dev_end: str = "2023-12-31"
    inner_start: str = "2022-01-01"
    search_space: Dict[str, Any] = Field(default_factory=dict)

class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    split_mode: str = "walk_forward_2024"
    split_frequency: str = "quarterly"
    window_mode: str = "expanding"
    start_year: int = 2024
    dev_end: str = "2023-12-31"
    n_trials: int = 30
    random_seed: int = 42
    stage1: Stage1TuningConfig = Field(default_factory=Stage1TuningConfig)
    stage2: Stage2TuningConfig = Field(default_factory=Stage2TuningConfig)

class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    run: RunConfig = Field(default_factory=RunConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
