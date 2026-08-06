"""
QuantBT Native Walk-Forward ML Strategy Adapter.

Contract (QuantBT WalkForwardEngine, read-only inspection):
  - strategy.build_signal(data, params, train_index, test_index, fold) is called
    MULTIPLE times per fold (2 scoring passes: IS with test_index==train_index and
    OOS; plus 1 final pass). Only final-pass outputs are stitched into the OOS
    equity curve.
  - The returned output MUST be a timestamp-indexed DataFrame of positions
    covering EVERY requested timestamp (engine raises otherwise).
  - The engine does NOT shift positions internally and executes the target row
    at the CLOSE of its index bar -> this adapter applies the canonical
    1-bar execution lag (decision at close D -> executed at close D+1).

Anti-leakage guarantees implemented here:
  - purge: training rows are filtered by label_end_time < fold.test_start.
  - features: only trailing rolling windows through close(D).
  - cross-sectional ranking / label demeaning: per timestamp only.
  - no bfill; no future statistics.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, List, Optional

from src.multifactor_mlops.config.schema import AppConfig
from src.multifactor_mlops.features.panel import PanelDatasetBuilder
from src.multifactor_mlops.features.macro import MacroOverlayTransformer
from src.multifactor_mlops.labels.returns import filter_train_by_label_end
from src.multifactor_mlops.optimization.model_utils import train_xgb_model, predict_xgb_model
from src.multifactor_mlops.portfolio.constructor import PortfolioConstructor
from src.multifactor_mlops.portfolio.overlay import apply_stress_overlay


def _to_naive_index(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


class MultiFactorWalkForwardStrategy:
    """
    Per-fold model training + cross-sectional portfolio weights builder.

    Caches:
      - panel dataset (built once for the whole run; features are causal so the
        same panel serves every fold),
      - one XGBoost model per (fold_id, params-key) because the engine calls
        build_signal 3x per fold.
    """

    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        symbols: List[str],
        app_config: AppConfig,
        strategy_params: Dict[str, Any],
        macro_df: Optional[pd.DataFrame] = None,
        predictions_cache: Optional[pd.DataFrame] = None,
        funding_df: Optional[pd.DataFrame] = None,
        universe_membership_df: Optional[pd.DataFrame] = None,
    ):
        self.data_dict = data_dict
        self.symbols = list(symbols)
        self.app_config = app_config
        self.strategy_params = strategy_params
        self.macro_df = macro_df
        self.funding_df = funding_df
        self.universe_membership_df = universe_membership_df
        # predictions_cache: DataFrame indexed by Time with symbol columns
        # (pre-computed OOF predictions). When provided, NO model is trained.
        self.predictions_cache = predictions_cache

        self._panel: Optional[pd.DataFrame] = None
        self._feature_cols: Optional[List[str]] = None
        self._model_cache: Dict[Any, Any] = {}
        self._macro_features: Optional[pd.DataFrame] = None
        self._closes: Dict[str, pd.Series] = {}
        self._vols: Optional[pd.DataFrame] = None
        self._btc_mom: Optional[pd.Series] = None

    # ------------------------------------------------------------------ setup
    def _build_panel(self) -> pd.DataFrame:
        fc = self.app_config.features
        builder = PanelDatasetBuilder(
            windows=fc.windows,
            cross_sectional_rank=True,
            lag=self.app_config.label.holding_bars,
            keep_families=fc.keep_families,
            inverted_features=fc.inverted_features,
            use_macro_features=fc.use_macro_features,
        )
        panel = builder.build_panel_dataset(
            data_dict=self.data_dict,
            symbols=self.symbols,
            macro_df=self.macro_df,
            funding_df=self.funding_df,
            universe_membership_df=self.universe_membership_df,
        )
        if panel.empty:
            raise ValueError("MultiFactorWalkForwardStrategy: panel dataset is empty.")
        self._feature_cols = PanelDatasetBuilder.feature_columns(panel)
        return panel

    def _get_panel(self) -> pd.DataFrame:
        if self._panel is None:
            self._panel = self._build_panel()
        return self._panel

    def _get_macro_features(self) -> pd.DataFrame:
        if self._macro_features is None:
            if self.macro_df is None or self.macro_df.empty:
                self._macro_features = pd.DataFrame()
            else:
                self._macro_features = MacroOverlayTransformer().transform_macro_df(self.macro_df)
        return self._macro_features

    def _load_closes(self) -> None:
        """Populates per-symbol close series (trailing data only)."""
        if not self._closes:
            for sym in self.symbols:
                df = self.data_dict.get(sym)
                if df is not None and not df.empty and "close" in df.columns:
                    self._closes[sym] = df["close"].sort_index()

    def _get_vols(self) -> pd.DataFrame:
        """Rolling vol per symbol (trailing only). Cached."""
        if self._vols is None:
            self._load_closes()
            period = max(5, int(self.strategy_params.get("inverse_vol_period", 42)))
            close_df = pd.DataFrame(self._closes).sort_index()
            rets = close_df.pct_change()
            self._vols = rets.rolling(period, min_periods=min(10, period)).std()
        return self._vols

    def _get_btc_mom(self) -> pd.Series:
        if self._btc_mom is None:
            self._load_closes()
            anchor = "BTCUSDT" if "BTCUSDT" in self._closes else (self.symbols[0] if self.symbols else None)
            if anchor is None or anchor not in self._closes:
                self._btc_mom = pd.Series(dtype=float)
            else:
                self._btc_mom = self._closes[anchor].pct_change(21)
        return self._btc_mom

    # ------------------------------------------------------------------ model
    def _fit_model(self, train_df: pd.DataFrame, params: Dict[str, Any]):
        return train_xgb_model(train_df, self._feature_cols, params)

    def _get_model(self, train_df: pd.DataFrame, params: Dict[str, Any], fold_id):
        if self.predictions_cache is not None:
            return None
        key = (fold_id, tuple(sorted((k, v) for k, v in params.items() if not isinstance(v, dict))))
        if key not in self._model_cache:
            self._model_cache[key] = self._fit_model(train_df, params)
        return self._model_cache[key]

    def _predict(self, booster, decision_rows: pd.DataFrame) -> pd.Series:
        feature_cols = self._feature_cols
        return pd.Series(
            predict_xgb_model(booster, decision_rows, feature_cols),
            index=decision_rows.index,
        )

    # ------------------------------------------------------------- weights
    def _build_weights(self, preds_wide: pd.DataFrame, decision_times: pd.DatetimeIndex) -> pd.DataFrame:
        """
        preds_wide : decision-time x symbol raw predictions.
        Returns signed target weights (decision-time x symbol), NOT yet shifted.
        """
        sp = self.strategy_params
        quantiles = int(sp.get("quantiles", self.app_config.portfolio.quantiles))
        alloc_cap = float(sp.get("allocation_cap", self.app_config.portfolio.allocation_cap))
        vol_ceiling_raw = sp.get("volatility_ceiling", self.app_config.portfolio.volatility_ceiling)
        vol_ceiling = float(vol_ceiling_raw) if vol_ceiling_raw is not None else None

        preds_clean = preds_wide.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rank_pct = preds_clean.rank(axis=1, pct=True, method="first")

        num_assets = len(preds_wide.columns)
        if num_assets <= 4:
            top_pct = 0.5
        else:
            top_pct = min(0.25, max(0.05, 1.0 / float(max(quantiles, 2))))
        long_mask = rank_pct > (1.0 - top_pct)
        short_mask = rank_pct <= top_pct

        pos_preds = preds_clean.clip(lower=0.0)
        neg_preds = preds_clean.clip(upper=0.0).abs()
        long_weighted = (pos_preds * long_mask.astype(float)).fillna(0.0)
        short_weighted = (neg_preds * short_mask.astype(float)).fillna(0.0)

        # Inverse-volatility weighting (trailing vol only)
        vols = self._get_vols().reindex(preds_wide.index).fillna(np.nan)
        inv_vol = (1.0 / vols.replace(0, np.nan)).fillna(0.0)
        long_weighted = long_weighted * inv_vol
        short_weighted = short_weighted * inv_vol

        long_sums = long_weighted.sum(axis=1).replace(0, 1.0)
        short_sums = short_weighted.sum(axis=1).replace(0, 1.0)
        long_part = long_weighted.div(long_sums, axis=0)
        short_part = -short_weighted.div(short_sums, axis=0)
        weights = (long_part + short_part).fillna(0.0)

        # Volatility ceiling: halve assets whose rolling vol exceeds ceiling
        if vol_ceiling is not None and vol_ceiling > 0.0:
            high_vol = (vols > float(vol_ceiling)).fillna(False)
            weights = weights.mask(high_vol, weights * 0.5)

        # Stress overlay (macro + BTC momentum) -- single shared module
        macro_feats = self._get_macro_features()
        if macro_feats is not None and not macro_feats.empty:
            macro_mult = macro_feats["macro_multiplier"].reindex(weights.index).fillna(1.0)
        else:
            macro_mult = pd.Series(1.0, index=weights.index)
        weights = apply_stress_overlay(
            weights,
            macro_multiplier=macro_mult,
            params=sp,
            btc_mom=self._get_btc_mom(),
        )

        # Per-asset allocation cap
        weights = weights.clip(lower=-alloc_cap, upper=alloc_cap)
        return weights.fillna(0.0)

    # --------------------------------------------------------------- signal
    def build_signal(self, data, params, train_index, test_index, fold):
        """
        QuantBT WalkForwardEngine adapter entrypoint.
        Returns positions DataFrame indexed by `test_index` (already 1-bar lagged).
        """
        panel = self._get_panel()
        test_start = pd.Timestamp(fold.test_start)
        if test_start.tz is not None:
            test_start = test_start.tz_localize(None)

        # Purge: training rows must have label horizon strictly before test_start.
        train_df = filter_train_by_label_end(panel, test_start)
        if train_df.empty:
            raise ValueError(f"build_signal: empty purged train set for fold {fold.fold_id} (test_start={test_start}).")

        req_index = _to_naive_index(test_index)

        if self.predictions_cache is not None:
            # OOF path: predictions pre-computed on the dev window; no training.
            cache_ts = _to_naive_index(self.predictions_cache.index)
            preds = self.predictions_cache.set_axis(cache_ts)
            preds_wide = preds.reindex(req_index).fillna(0.0)
        else:
            booster = self._get_model(train_df, params, fold.fold_id)
            decision_rows = panel[panel.index.get_level_values("Time").isin(req_index)]
            if decision_rows.empty:
                raise ValueError(f"build_signal: no panel rows at requested decision times (fold {fold.fold_id}).")
            preds = self._predict(booster, decision_rows)
            pred_frame = preds.rename("pred").reset_index()
            pred_frame.columns = ["Time", "Symbol", "pred"]
            preds_wide = (
                pred_frame.pivot_table(index="Time", columns="Symbol", values="pred")
                .reindex(req_index)
                .fillna(0.0)
            )

        # build weights for every requested decision day (covers train + test requests)
        if len(req_index) > 0:
            all_times = req_index.union(preds_wide.index).sort_values()
            preds_full = preds_wide.reindex(all_times).fillna(0.0)
            weights = self._build_weights(preds_full, all_times)
            weights = weights.reindex(all_times).fillna(0.0)
        else:
            weights = pd.DataFrame(0.0, index=req_index, columns=self.symbols)

        # Execution schedule (includes the 1-bar lag):
        #  - monday_decide_weekly: Monday close decision -> hold Tue-Fri (low turnover)
        #  - daily: decision at close D -> executed at close D+1
        #  - legacy calendar schedules: shift(1) + calendar hold
        schedule = self.strategy_params.get("rebalance_schedule", "weekly_friday_exit")
        if schedule == "monday_decide_weekly":
            positions = PortfolioConstructor.apply_monday_decide_schedule(weights)
        elif schedule == "daily":
            positions = weights.shift(1).fillna(0.0)
        else:
            positions = weights.shift(1).fillna(0.0)
            positions = PortfolioConstructor.apply_calendar_holding_schedule(positions, schedule=schedule)

        positions = positions.reindex(req_index, columns=self.symbols).fillna(0.0)
        return positions


def make_strategy_factory(
    data_dict: Dict[str, pd.DataFrame],
    symbols: List[str],
    app_config: AppConfig,
    strategy_params: Dict[str, Any],
    macro_df: Optional[pd.DataFrame] = None,
    predictions_cache: Optional[pd.DataFrame] = None,
    funding_df: Optional[pd.DataFrame] = None,
    universe_membership_df: Optional[pd.DataFrame] = None,
):
    """Factory returning a strategy INSTANCE bound to its data/panel context."""
    strategy = MultiFactorWalkForwardStrategy(
        data_dict=data_dict,
        symbols=symbols,
        app_config=app_config,
        strategy_params=strategy_params,
        macro_df=macro_df,
        predictions_cache=predictions_cache,
        funding_df=funding_df,
        universe_membership_df=universe_membership_df,
    )
    return strategy
