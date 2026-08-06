"""
QuantBT Native Walk-Forward runner (canonical backtest entrypoint).

Runs QuantBTEndpoint.walk_forward with target_mode="portfolio",
optimization_mode="none" (ML hyperparameters are tuned OUTSIDE the engine:
inner purged folds, see stage1_ml_tuning) and FIXED model+strategy params.

Costs: fee_rate is passed as ONE-WAY per-fill rate (QuantBT endpoint semantics:
`fee_rate` has priority over round-trip `fee`). Slippage is bps per fill.
Funding: use_funding + funding_rate (per-symbol Series) charged by the engine.
"""

import sys
from typing import Dict, Any, List, Optional, Union

import pandas as pd

from src.multifactor_mlops.config.schema import AppConfig
from src.multifactor_mlops.backtest.walkforward_strategy import make_strategy_factory


class QuantBTExecutionError(Exception):
    """Raised when QuantBT execution fails. Engine fallback is strictly prohibited."""
    pass


class WalkForwardQuantBTRunner:
    """Single integration boundary for QuantBT walk-forward portfolio backtests."""

    def __init__(self, quantbt_repo_path: str = "/root/bobby/pool_alpha/quantbt"):
        self.quantbt_repo_path = quantbt_repo_path
        self._import_quantbt()

    def _import_quantbt(self):
        if self.quantbt_repo_path not in sys.path:
            sys.path.append(self.quantbt_repo_path)
        if "/root/bobby/pool_alpha" not in sys.path:
            sys.path.append("/root/bobby/pool_alpha")
        try:
            from quantbt.endpoint import QuantBTEndpoint
            self.QuantBTEndpoint = QuantBTEndpoint
        except ImportError as e:
            raise QuantBTExecutionError(f"Failed to import QuantBTEndpoint from {self.quantbt_repo_path}: {e}")

    @staticmethod
    def _common_index(data_dict: Dict[str, pd.DataFrame], symbols: List[str]) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex([])
        for sym in symbols:
            df = data_dict.get(sym)
            if df is not None and not df.empty:
                idx = idx.union(pd.DatetimeIndex(df.index))
        idx = pd.DatetimeIndex(sorted(idx))
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        return idx

    def run(
        self,
        data_dict: Dict[str, pd.DataFrame],
        symbols: List[str],
        app_config: AppConfig,
        params: Dict[str, Any],
        macro_df: Optional[pd.DataFrame] = None,
        funding_rate: Optional[Union[float, Dict[str, pd.Series]]] = None,
        split_mode: Optional[str] = None,
        split_frequency: Optional[str] = None,
        window_mode: Optional[str] = None,
        predictions_cache: Optional[pd.DataFrame] = None,
        funding_wide: Optional[pd.DataFrame] = None,
        universe_membership_df: Optional[pd.DataFrame] = None,
    ):
        """
        Runs the QuantBT native walk-forward backtest and returns the raw result.

        params : merged flat params (model + strategy) passed to the strategy.
        predictions_cache : optional pre-computed OOF predictions (no training).
        funding_wide : daily funding DataFrame (symbol columns) used for the
                       carry FEATURE and (optionally) the engine funding cost.
        universe_membership_df : point-in-time traded-universe mask (Time x Symbol).
        """
        bc = app_config.backtest
        vc = app_config.validation

        # Base strategy params = parameters.json portfolio section (signal_mode,
        # composite_features, quantiles, schedule, overlay...) overridden by locked
        # tuning artifacts passed via `params`.
        merged_params = {**app_config.portfolio.model_dump(), **params}

        # Engine requires a funding entry for EVERY traded symbol: complete gaps with 0.0.
        # Copy all series: the engine aligns/mutates passed series IN PLACE (tz-aware),
        # which would corrupt caller-owned data between trials.
        completed_funding = None
        if isinstance(funding_rate, dict):
            completed_funding = {s: (v.copy() if hasattr(v, "copy") else v) for s, v in funding_rate.items()}
            for sym in symbols:
                if sym not in completed_funding:
                    completed_funding[sym] = pd.Series(0.0, index=self._common_index(data_dict, symbols))
            funding_rate = completed_funding

        strategy = make_strategy_factory(
            data_dict=data_dict,
            symbols=symbols,
            app_config=app_config,
            strategy_params=merged_params,
            macro_df=macro_df,
            predictions_cache=predictions_cache,
            funding_df=funding_wide,
            universe_membership_df=universe_membership_df,
        )

        try:
            bt = self.QuantBTEndpoint.walk_forward(
                strategy_class=strategy,
                split_mode=split_mode or vc.split_mode,
                split_frequency=split_frequency or vc.split_frequency,
                window_mode=window_mode or vc.window_mode,
                train_window=vc.train_window,
                target_mode="portfolio",
                optimization_mode="none",
                portfolio_mode=bc.portfolio_mode,
                backend="native_portfolio",
                hedge_type=bc.hedge_type,
                initial_capital=float(bc.initial_capital),
                leverage=float(bc.leverage),
                fee_rate=float(bc.fee_rate_per_fill),
                slippage=float(bc.slippage),
                use_funding=bool(bc.use_funding),
                funding_rate=funding_rate if funding_rate is not None else 0.0,
                contract_size=1.0,
                report_level="minimal",
            )
            common_index = self._common_index(data_dict, symbols)
            # The engine may align/mutate index tz of the passed frames in place:
            # pass deep copies so caller-owned data stays pristine between trials.
            data_for_engine = {s: df.copy() for s, df in data_dict.items()}
            qbt_res = bt.backtest(
                data=data_for_engine,
                symbols=symbols,
                params=merged_params,
                datetime_index=common_index,
            )
        except Exception as e:
            raise QuantBTExecutionError(f"QuantBT walk-forward backtest execution failed: {e}") from e

        return qbt_res


def extract_metrics(qbt_res, trading_days: int = 365) -> Dict[str, Any]:
    """Extracts native QuantBT metrics (source of truth for PnL statistics)."""
    metrics: Dict[str, Any] = {}
    if hasattr(qbt_res, "show_metrics"):
        try:
            metrics = dict(qbt_res.show_metrics(trading_days=trading_days))
        except Exception as e:
            print(f"Warning: qbt_res.show_metrics() failed: {e}")
    return metrics


def extract_equity(qbt_res, initial_capital: float = 100000.0) -> pd.DataFrame:
    """Extracts daily equity/returns DataFrame with columns time/equity/return."""
    qbt_equity = getattr(qbt_res, "daily_equity", None)
    if qbt_equity is not None and len(qbt_equity) > 0:
        qbt_rets = pd.Series(qbt_equity).pct_change().fillna(0.0)
    else:
        qbt_rets = getattr(qbt_res, "daily_returns", getattr(qbt_res, "portfolio_returns", None))
        if qbt_rets is None or len(qbt_rets) == 0:
            raise QuantBTExecutionError("QuantBT backtest returned empty equity and return series.")
        qbt_equity = (1.0 + qbt_rets).cumprod() * initial_capital

    idx = pd.DatetimeIndex(qbt_equity.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return pd.DataFrame({"time": idx, "equity": pd.Series(qbt_equity).values, "return": pd.Series(qbt_rets).values})
