"""
Single integration boundary for QuantBT Backtest Engine.
Enforces zero-fallback policy: any execution failure raises QuantBTExecutionError immediately.
"""

import sys
import pandas as pd
from typing import Dict, Any, Tuple

class QuantBTExecutionError(Exception):
    """Raised when QuantBT execution fails. Engine fallback is strictly prohibited."""
    pass

class QuantBTRunner:
    """
    Canonical QuantBT runner adapter.
    Interacts with QuantBTEndpoint natively and returns standard metrics and equity curves.
    """

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

    def run_backtest(
        self,
        positions: pd.DataFrame,
        data_dict: Dict[str, pd.DataFrame],
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
        """
        Executes native QuantBT portfolio backtest.

        Parameters
        ----------
        positions : pd.DataFrame
            DataFrame of target portfolio weights (shifted 1 bar for execution lag).
        data_dict : Dict[str, pd.DataFrame]
            Dictionary of asset OHLCV DataFrames.
        params : Dict[str, Any]
            Strategy and backtest configuration parameters.

        Returns
        -------
        equity_df : pd.DataFrame
            DataFrame with columns ['time', 'equity', 'return'].
        metrics_report : Dict[str, Any]
            Native QuantBT metrics dictionary from show_metrics().
        qbt_res : Any
            Raw QuantBT result object.
        """
        # Cost convention: fee_rate is ONE-WAY per fill (QuantBT endpoint honors
        # fee_rate with priority over the legacy round-trip `fee`).
        fee_rate_per_fill = params.get("fee_rate_per_fill")
        if fee_rate_per_fill is None:
            if params.get("fee_rate") is not None:
                fee_rate_per_fill = params["fee_rate"]
            else:
                fee_rate_per_fill = float(params.get("fee", 0.001)) / 2.0  # legacy round-trip -> one-way
        fee_rate = float(fee_rate_per_fill)
        slippage = float(params.get("slippage", 0.0001))
        leverage = float(params.get("leverage", 3.0))
        initial_capital = float(params.get("initial_capital", 100000.0))
        portfolio_mode = str(params.get("portfolio_mode", "longshort"))
        hedge_type = str(params.get("hedge_type", "target_weight"))
        trading_days = int(params.get("trading_days_per_year", 365))
        use_funding = bool(params.get("use_funding", False))
        funding_rate = params.get("funding_rate", 0.0)

        # Causal copy only -- NO full-sample statistics (removed future max clip).
        scaled_weights = positions.copy().fillna(0.0)

        try:
            bt_engine = self.QuantBTEndpoint.portfolio(
                portfolio_mode=portfolio_mode,
                backend="native_portfolio",
                hedge_type=hedge_type,
                initial_capital=initial_capital,
                leverage=leverage,
                asset_type=params.get("asset_class", "crypto"),
                use_funding=use_funding,
                funding_rate=funding_rate,
                fee_rate=fee_rate,
                slippage=slippage,
                contract_size=1.0,
                report_level="minimal"
            )
            qbt_res = bt_engine.backtest(positions=scaled_weights, data=data_dict)
        except Exception as e:
            raise QuantBTExecutionError(f"QuantBT portfolio backtest execution failed: {e}")

        # Extract native metrics report
        metrics_report = {}
        if hasattr(qbt_res, "show_metrics"):
            try:
                metrics_report = qbt_res.show_metrics(trading_days=trading_days)
            except Exception as me:
                print(f"Warning: qbt_res.show_metrics() failed: {me}")

        qbt_equity = getattr(qbt_res, "daily_equity", None)
        if qbt_equity is not None and len(qbt_equity) > 0:
            qbt_rets = qbt_equity.pct_change().fillna(0.0)
        else:
            qbt_rets = getattr(qbt_res, "daily_returns", getattr(qbt_res, "portfolio_returns", None))
            if qbt_rets is None or len(qbt_rets) == 0:
                raise QuantBTExecutionError("QuantBT backtest returned empty equity and return series.")
            qbt_equity = (1.0 + qbt_rets).cumprod() * initial_capital

        eval_dates = qbt_equity.index
        if eval_dates.tz is not None:
            eval_dates = eval_dates.tz_localize(None)

        equity_df = pd.DataFrame({
            "time": eval_dates,
            "equity": qbt_equity.values,
            "return": qbt_rets.values
        }).reset_index(drop=True)

        return equity_df, metrics_report, qbt_res
