# MULTIFACTOR MLOPS — AUDIT & REPAIR PLAN

**Repository audited:** `BobbyAxerol/multifactor-mlops`  
**Branch:** `feat/multifactor-macro-features`  
**Audit date:** 2026-07-30  
**Primary constraints:** preserve the current strategy thesis; use `quantbt` for every backtest; use `trading-historical-data` as the primary market-data source; allow controlled alternative sources only when the primary source is stale or incomplete.

---

## 1. Executive conclusion

The current implementation is **not yet trustworthy for measuring the strategy's edge**. The conceptual strategy can be retained, but the present pipeline contains several P0 defects capable of producing material look-ahead bias, in-sample performance, incorrect alignment, inconsistent live inference, and non-QuantBT backtests.

The most serious issue is the combination of:

1. `parameters.json` requests `train_test_split_2024`, but `training/train.py` does not implement that split name.
2. When no folds are returned, the code silently changes to `full` mode.
3. `full` mode fits and predicts on the same observations.
4. The default backend is `native_portfolio`, so the normal training path can use the repository's custom vectorized backtest rather than QuantBT.
5. Several data paths use `.bfill()`, which can carry future observations backward.

**Consequence:** all currently reported Sharpe, CAGR, drawdown, and Optuna best parameters should be treated as provisional until the P0 repair is complete and the strategy is rerun from a frozen data snapshot.

This plan deliberately **does not add new alpha factors or broaden the strategy**. It retains:

- cross-sectional multifactor prediction;
- XGBoost/LightGBM model family;
- top-liquidity crypto universe;
- momentum, retail-flow, carry, and margin-risk factors;
- macro risk overlay;
- inverse-volatility sizing;
- long-short quantile portfolio construction.

Only the data, timing, validation, tuning, portfolio, backtest, and MLOps contracts are corrected.

---

## 2. P0 findings — must be fixed before trusting any result

| ID | Defect | Current behavior | Required correction |
|---|---|---|---|
| P0-01 | Unsupported split silently becomes full-sample | `train_test_split_2024` is not handled; empty folds fall back to `full`; model fits and predicts on the same panel | Validate split names at config load; unsupported split must raise; remove recursive full-mode fallback |
| P0-02 | Backtest is not QuantBT-only | Default `backend=native_portfolio`; normal path can call the custom `backtest_portfolio` | All scoring, validation, tuning, and final reporting must call a single `QuantBTRunner`; delete/disable performance fallback |
| P0-03 | Future data can be backfilled | Price and macro series use `ffill().bfill()` | Ban `bfill` in the research path; use point-in-time backward joins and keep unavailable values as `NaN` |
| P0-04 | Timing contract is implicit | Feature date, decision time, execution time, label horizon, and effective position time are not represented explicitly | Introduce one canonical timing contract and assert it for every row |
| P0-05 | Training and serving transformations differ | Training cross-sectionally ranks factors; serving passes raw latest features to the model | Build one fitted feature pipeline used by training, validation, and inference |
| P0-06 | Live short weights can flip long | Serving multiplies long/short signs by already signed risk weights | Use the portfolio builder's final signed target weights directly; add sign-preservation tests |
| P0-07 | Data configuration is ignored or hardcoded | `train.py` uses absolute `sys.path`; callers read `data_path` while config contains `data_repo_path` | Replace path injection with a data adapter/package import and validated config/environment resolution |
| P0-08 | Universe selection has look-ahead/survivorship bias | Top symbols can be selected from average turnover over the entire supplied period | Build a point-in-time rolling-liquidity universe, lagged one bar, with listing/history masks |
| P0-09 | Missing predictions can become trades | Missing predictions/features are filled with zero before cross-sectional ranking | Preserve missingness; exclude ineligible symbols before ranking and portfolio construction |
| P0-10 | Costs and position lag are ambiguous | QuantBT path shifts weights; custom path shifts internally; fallback uses a different cost convention and `fee * 2` | Apply position timing once; configure fee per fill once; certify with a deterministic one-trade test |
| P0-11 | Futures funding P&L is omitted by default | Strategy uses funding-related features, while backtest defaults can leave funding cashflows off | Supply point-in-time funding events to QuantBT and explicitly configure funding P&L; never infer missing funding as zero silently |
| P0-12 | Failures are concealed | QuantBT exceptions can fall back to custom backtest; Optuna exceptions become `-999`; missing dates are filled with zero returns | Fail fast with typed errors; no silent fallback; assert date and symbol coverage |

### Immediate decision

Until P0-01 through P0-12 pass automated tests, the pipeline must carry this status:

```text
RESEARCH_STATUS = INVALID_FOR_EDGE_CONCLUSION
```

---

## 3. Detailed audit

### 3.1 Split and out-of-sample validation

The current config and implementation are inconsistent:

- `parameters.json` uses `train_test_split_2024`.
- `split_data()` supports other names but not this value.
- An empty fold list is silently replaced by `full` mode.
- In `full` mode, the model is fitted on `X, y` and predicts on `X` from the same panel.

This is direct in-sample contamination, not a conservative approximation.

#### Required design

Use a strict outer walk-forward evaluation:

```text
outer train window -> optional inner Optuna CV -> fit fold model
outer test window  -> predict once -> construct weights -> QuantBT
repeat folds       -> stitch untouched OOS returns only
```

Rules:

- No test row may participate in feature selection, hyperparameter selection, scaler/ranker fitting, universe estimation, or model fitting.
- No automatic fallback from an OOS scheme to `full` is permitted.
- `full` is allowed only for the final production refit **after** the research configuration is frozen; it must never produce the research backtest score.
- Add purge and embargo based on the label horizon.
- Use explicit fold boundaries from the split object; do not infer a fold cutoff from the first symbol's last timestamp.

Recommended default:

```text
scheme             = expanding walk-forward
first_oos_date     = 2024-01-01
outer_test_period  = quarterly
purge_bars         = decision_to_execution_bars + holding_bars - 1
embargo_bars       = 1
```

### 3.2 Canonical timing and label contract

A single row must carry these concepts separately:

```text
bar_open_time
bar_close_time
feature_available_at
decision_time
execution_time
label_start_time
label_end_time
position_effective_time
```

For daily crypto bars, the recommended default is:

```text
bar D is fully observed at its close
features_D use only information available by bar_close_D
signal_D is decided after bar_close_D
position becomes effective at open_(D+1)
label_D measures open_(D+1) -> open_(D+2)
```

Formally:

$$
y_{i,D} = \frac{O_{i,D+2}}{O_{i,D+1}} - 1
$$

This preserves the idea of predicting the next holding-period return while matching a realizable next-bar execution.

Mandatory assertions:

```python
assert feature_available_at <= decision_time
assert execution_time > decision_time
assert label_start_time == execution_time
assert label_end_time > label_start_time
assert position_effective_time == execution_time
```

Do not combine a close-to-close label with a next-open execution unless the intended holding interval and the resulting overnight exposure are explicitly modeled.

### 3.3 Primary and alternative data sources

#### Primary source

All core OHLCV and available derivatives datasets should be loaded through `trading-historical-data`, preferably its existing loader interface rather than absolute-path imports.

Target interface:

```python
snapshot = HistoricalDataAdapter(config).load(
    symbols=universe_seed,
    start=start,
    end=end,
    fields=["open", "high", "low", "close", "volume", "funding_rate"]
)
```

The adapter must return a canonical schema:

```text
symbol
event_time
available_at
ingested_at
source
field
value
revision_id
```

For OHLCV matrices, keep the matrix internally if it is efficient, but materialize the time/provenance contract in the snapshot manifest.

#### Alternative source policy

Alternative providers may be used only when `trading-historical-data` is stale or missing a requested tail/gap.

Required priority:

```text
1. trading-historical-data frozen snapshot
2. approved alternative provider for an explicit missing interval
3. no data / symbol ineligible
```

Alternative data must never overwrite primary history silently. Each filled interval must record:

```text
provider
requested_at
ingested_at
first_event_time
last_event_time
row_count
checksum
reason_for_fallback
cross_source_validation_result
```

Before combining sources:

- normalize timezone and bar semantics;
- check duplicates, monotonicity, OHLC constraints, and missing bars;
- compare overlap returns/prices against the primary source;
- freeze the retrieved result as a run artifact;
- prohibit network refresh during model inference or historical replay.

#### Current collector defects to remove

- Direct Binance OI/long-short calls use a limited request window without a complete pagination contract.
- Cache validity is based on symbol presence rather than requested date coverage and freshness.
- Existing symbols can prevent the missing date tail from being updated.
- Naive `datetime.now()`/timestamp conversions make timezone semantics ambiguous.
- Local CSV fallback is selected implicitly instead of through source policy.

### 3.4 Point-in-time alignment

Every non-price series must be joined by when it became usable, not only by its calendar date.

Correct pattern:

```python
pd.merge_asof(
    decisions.sort_values("decision_time"),
    observations.sort_values("available_at"),
    left_on="decision_time",
    right_on="available_at",
    by=["symbol", "field"],
    direction="backward",
    tolerance=...
)
```

Forbidden patterns in any alpha, eligibility, label, or P&L path:

```python
.bfill()
.fillna(method="bfill")
reindex(...).bfill()
using a period-end value before its release/availability time
```

Forward fill is allowed only when the economic meaning supports persistence and must include:

```text
value_age
max_age
is_stale
is_observed
```

If `value_age > max_age`, the value becomes unavailable rather than persisting indefinitely.

### 3.5 Macro features

The current macro collector combines DVOL, Fear & Greed, DeFiLlama, and market series by date, then forward/backfills them. It does not preserve a reliable availability timestamp or source vintage.

Required correction:

1. Collect raw observations separately by provider.
2. Preserve provider event timestamp and ingestion timestamp.
3. Derive a conservative `available_at`.
4. Join to the decision grid only backward.
5. Fit rolling transformations using past data only.
6. Use one shared `MacroOverlayTransformer` in training and inference.

Current training and inference use different macro-overlay logic. Training uses one rolling z-score/sigmoid form, while serving uses a short moving-average threshold. This creates a material train-serving skew.

The final model bundle must contain the fitted macro transformation specification; inference must receive a frozen as-of macro snapshot instead of downloading current macro data inside `predict()`.

### 3.6 Asset factors and missingness

`util/factors.py` should become a pure transformation module:

```text
input: canonical point-in-time market snapshot
output: feature frame + availability/missingness metadata
side effects: none
network calls: none
```

Rules:

- rolling windows use observations through time `t` only;
- all cross-sectional ranks are calculated on the eligible universe at the same decision timestamp;
- rolling warm-up rows remain unavailable;
- do not replace rolling warm-up with arbitrary neutral values unless a corresponding missing indicator is explicitly modeled and fitted;
- funding missingness is not automatically equivalent to a true zero funding rate;
- preserve exact ordered feature schema and dtype.

For every generated feature, store metadata:

```text
feature_name
lookback
input_fields
minimum_history
availability_rule
missing_policy
version
```

### 3.7 Point-in-time universe

The present top-symbol logic can use average turnover over the full supplied period. In a full-period run this uses future liquidity and future survivors.

Retain the top-liquidity idea, but calculate it point-in-time:

$$
L_{i,t} = \operatorname{mean}_{s=t-W+1}^{t}(V_{i,s} \times C_{i,s})
$$

The universe used for decision time `t` is based on lagged liquidity:

$$
\mathcal{U}_t = \operatorname{TopN}\left(L_{i,t-1}\right)
$$

Eligibility also requires:

```text
listed by t
not delisted/suspended at execution
minimum history satisfied
required price fields present
tradable at execution timestamp
feature vector complete under the configured policy
```

Store a `universe_membership.parquet` artifact for each run.

### 3.8 Feature selection

Current feature analysis uses a fixed historical cutoff and can overlap an outer test period for some split configurations. Mutual information, OLS, and model gain are fitted on a broad sample rather than nested inside the relevant training window.

Keep the current selection idea, but apply one of two valid modes:

**Recommended:** nested selection per outer fold, fitted only on that fold's training observations.

**Cheaper alternative:** freeze a feature list using a cutoff strictly earlier than the first OOS date, then never update it during the research backtest.

The selected-feature artifact must include:

```text
ordered_features
selection_cutoff
selection_method
training_data_hash
feature_code_hash
config_hash
random_seed
```

OLS p-values should not be treated as decisive evidence in a correlated panel without time/cluster-aware inference. Use them as diagnostics, not as the primary inclusion rule.

### 3.9 Portfolio construction

Construct the portfolio in one module shared by validation and production:

```text
predictions
-> eligibility mask
-> cross-sectional score/rank
-> long and short selection
-> inverse-volatility sizing
-> macro gross-risk multiplier
-> per-asset cap
-> final exposure normalization
-> QuantBT target weights
```

Required invariants:

```python
assert long_weights.sum() >= 0
assert short_weights.sum() <= 0
assert gross_exposure <= gross_limit + tolerance
assert abs(net_exposure) <= net_limit + tolerance
assert abs(weight_i) <= allocation_cap + tolerance
assert weights[~eligible] == 0
```

The serving implementation currently multiplies direction by risk weights that are already signed, which can turn shorts positive. The shared portfolio builder must output final signed weights once; no second sign multiplication is allowed.

Do not fill missing predictions with zero and then rank them. Remove the symbol from that timestamp's eligible cross-section.

With `top_n=40` and `quantiles=20`, each side may contain only roughly two symbols. Keep this configuration if intentional, but require:

```text
minimum_cross_section
minimum_names_per_side
deterministic tie breaker
reported realized gross/net/concentration
```

### 3.10 QuantBT-only backtesting

Create exactly one integration boundary:

```text
backtest/quantbt_runner.py
```

Responsibilities:

- import the pinned `quantbt` package;
- construct the portfolio endpoint;
- validate OHLCV/funding inputs;
- convert final target weights to QuantBT's expected semantics;
- execute the backtest;
- return standardized metrics and audit artifacts;
- fail on missing coverage or endpoint errors.

Forbidden:

- custom equity/P&L as the official score;
- catching a QuantBT error and switching engine;
- filling absent QuantBT returns with zero;
- applying an extra weight shift outside the declared timing adapter;
- doubling fees manually without a documented two-fill event model.

Recommended endpoint intent:

```text
backend        = native_portfolio
portfolio_mode = longshort
hedge_type     = target_weight
```

These are QuantBT settings; `native_portfolio` must not mean a repository-local backtester.

For futures, the runner must explicitly pass and report:

```text
fee rate per fill
slippage convention
leverage
maintenance ratio if applicable
funding cashflows
contract size
position/equity accounting
```

Create a one-trade certification fixture with hand-computed P&L, fee, slippage, funding, and effective timestamp. The test must reconcile QuantBT output exactly within numerical tolerance.

### 3.11 Optuna

The current Optuna objective can optimize an in-sample/full backtest because it calls the same flawed training path. It also contains configuration fields that are not wired into the executed logic, hardcoded early stopping, no durable study storage, and broad exception suppression.

Correct nested process:

```text
FOR each outer fold:
    outer_train
        -> inner purged walk-forward folds
        -> Optuna trial
        -> QuantBT inner-OOS objective only
        -> select parameters
    fit once on all outer_train
    predict untouched outer_test
    QuantBT outer-test backtest
STITCH outer-test results
```

Rules:

- Objective may use only inner-OOS observations.
- Every trial must use QuantBT, not a metric shortcut based on custom P&L.
- Search only parameters that actually change the executed pipeline.
- Validate the search space against the selected model type.
- Set and log all random seeds.
- Use persistent storage and resume by study name.
- Do not mutate the base `parameters.json` with best parameters.
- Save best parameters to an immutable run artifact, then promote explicitly.
- Invalid data/configuration errors fail the study; only mathematically invalid trials may be pruned/failed as trials.

Suggested objective hierarchy:

```text
primary: median inner-fold OOS Sharpe or rank IC
penalty: dispersion across folds, drawdown, turnover, concentration
constraint: minimum observations/trades and both long/short participation
```

Any macro threshold/multiplier included in the search space must be consumed by the same shared `MacroOverlayTransformer`; otherwise remove it as a dead dimension.

### 3.12 Training, validation, final fit, and model registration

Do not combine research evaluation and production fitting in one ambiguous `train.py` path.

Required commands:

```text
train.py --mode validate   # outer WFO, returns OOS artifacts only
train.py --mode final      # refit after research config is frozen
backtest.py                # replay saved OOS weights through QuantBT only
register_model.py          # register the final refit bundle only
```

Important distinction:

- The stitched OOS backtest is generated by multiple fold models.
- The last fold's booster is **not** equivalent to the evaluated strategy.
- After model/config selection, train one final model through a declared `production_cutoff` and register that model.
- Never register a dummy model when training returned `None`; fail the run.

The model bundle must include:

```text
model binary
ordered feature schema
fitted feature transformations
macro overlay specification
universe policy
portfolio policy
label/execution contract
training cutoff
source/data manifest hash
parameters hash
code commit hashes
quantbt version/commit
```

Inference must not fetch live data inside the model wrapper. The wrapper accepts an already validated as-of snapshot and produces target weights using the same feature and portfolio pipeline.

---

## 4. Target repository structure

This is a minimal refactor, not a rewrite of the strategy:

```text
multifactor-mlops/
├── parameters.json
├── src/multifactor_mlops/
│   ├── config/
│   │   ├── schema.py
│   │   └── loader.py
│   ├── data/
│   │   ├── historical_adapter.py
│   │   ├── alternative_adapters.py
│   │   ├── snapshot.py
│   │   ├── point_in_time.py
│   │   └── quality.py
│   ├── universe/
│   │   └── liquidity.py
│   ├── features/
│   │   ├── asset.py
│   │   ├── macro.py
│   │   ├── panel.py
│   │   └── schema.py
│   ├── labels/
│   │   └── returns.py
│   ├── validation/
│   │   ├── splits.py
│   │   └── leakage.py
│   ├── models/
│   │   ├── factory.py
│   │   └── bundle.py
│   ├── portfolio/
│   │   ├── constructor.py
│   │   └── constraints.py
│   ├── backtest/
│   │   └── quantbt_runner.py
│   ├── optimization/
│   │   └── optuna_kernel.py
│   ├── tracking/
│   │   └── mlflow_logger.py
│   └── pipelines/
│       ├── validate.py
│       ├── fit_final.py
│       └── infer.py
├── train.py
├── backtest.py
├── register_model.py
└── tests/
```

### Mapping from current modules

| Current module | Required action |
|---|---|
| `util/data_collector.py` | Retain provider-specific collection code only behind adapters; remove implicit core-data fallback |
| `util/macro_collector.py` | Return raw point-in-time observations with provenance; remove all `bfill`; no feature logic |
| `util/factors.py` | Split into pure asset/macro/panel transformers; no downloads or backtest logic |
| `util/rebalance.py` | Keep portfolio-weight construction only; delete official custom backtest path |
| `training/train.py` | Reduce to orchestration; strict split validation; no silent fallback; no engine fallback |
| `optimization/optuna_kernel.py` | Implement nested inner OOS tuning; use persistent study and QuantBT-only objective |
| `train_mlflow.py` | Log immutable data/fold/prediction/weight/backtest manifests and hashes |
| `register_model.py` | Register only final refit bundle; no network collection; use shared transformers |
| root `model.json` | Remove/rename ambiguous placeholder; model artifacts belong to immutable run directories |

---

## 5. Proposed `parameters.json`

Use one validated configuration file, but separate concerns. Model parameters must not live under `features`; backtest configuration must not determine validation behavior implicitly.

```json
{
  "run": {
    "seed": 42,
    "timezone": "UTC",
    "fail_fast": true,
    "production_cutoff": null
  },
  "data": {
    "primary": {
      "provider": "trading_historical_data",
      "dataset": "binance_daily_matrix",
      "loader": "CryptoDailyMatrix.load_ohlcv",
      "check_validation": true
    },
    "fallback": {
      "enabled": true,
      "providers": ["binance_rest"],
      "scope": "missing_tail_or_explicit_gap_only",
      "max_primary_staleness_days": 1,
      "require_overlap_validation": true,
      "write_back_to_primary": false
    },
    "start": "2020-01-01",
    "end": null,
    "allow_bfill": false,
    "snapshot_root": "artifacts/data"
  },
  "universe": {
    "method": "rolling_turnover_top_n",
    "top_n": 40,
    "lookback_days": 30,
    "selection_lag_bars": 1,
    "min_history_days": 180,
    "minimum_cross_section": 20
  },
  "features": {
    "windows": [7, 14, 30, 60, 90],
    "cross_sectional_rank": true,
    "warmup_policy": "drop",
    "missing_policy": "preserve",
    "feature_set_version": "multifactor_v1"
  },
  "label": {
    "return_type": "next_open_to_open",
    "decision_to_execution_bars": 1,
    "holding_bars": 1,
    "cross_sectional_demean": true
  },
  "model": {
    "type": "xgboost",
    "params": {
      "learning_rate": 0.03,
      "max_depth": 5,
      "n_estimators": 500,
      "colsample_bytree": 0.8,
      "subsample": 0.8,
      "random_state": 42
    }
  },
  "validation": {
    "scheme": "walk_forward",
    "train_window": "expanding",
    "first_oos_date": "2024-01-01",
    "outer_test_period": "quarterly",
    "purge_bars": 1,
    "embargo_bars": 1,
    "unsupported_split_policy": "raise"
  },
  "feature_selection": {
    "mode": "nested_per_outer_fold",
    "methods": ["mutual_information", "model_gain"],
    "minimum_fold_stability": 0.6
  },
  "optimization": {
    "enabled": true,
    "n_trials": 50,
    "seed": 42,
    "study_name": "multifactor_inner_wfo",
    "storage": "sqlite:///artifacts/optuna/multifactor.db",
    "direction": "maximize",
    "objective": "inner_oos_robust_score",
    "early_stopping_trials": 50,
    "search_space": {
      "learning_rate": [0.01, 0.1],
      "max_depth": [3, 8],
      "colsample_bytree": [0.5, 1.0],
      "subsample": [0.5, 1.0]
    }
  },
  "portfolio": {
    "quantiles": 20,
    "minimum_names_per_side": 2,
    "inverse_vol_lookback": 60,
    "gross_target": 1.0,
    "net_target": 0.0,
    "allocation_cap": 0.25,
    "macro_overlay": {
      "enabled": true,
      "method": "rolling_zscore_sigmoid",
      "lookback_days": 120,
      "minimum_multiplier": 0.25,
      "maximum_multiplier": 1.0
    }
  },
  "backtest": {
    "engine": "quantbt",
    "backend": "native_portfolio",
    "portfolio_mode": "longshort",
    "hedge_type": "target_weight",
    "allow_engine_fallback": false,
    "initial_capital": 100000.0,
    "leverage": 3.0,
    "fee_rate_per_fill": 0.0005,
    "slippage": 0.0001,
    "use_funding": true,
    "fill_missing_engine_returns": false
  },
  "mlflow": {
    "experiment_name": "Multifactor_MLOps",
    "tracking_uri": "http://localhost:5000",
    "register_final_model": true
  }
}
```

### Configuration rules

- Validate with Pydantic before any data is loaded.
- Unknown keys and invalid enum values must raise.
- `_available_*` option lists do not belong in runtime parameters.
- Resolve repository locations from installed packages or environment variables, not absolute paths inside source code.
- Base parameters are immutable during a run.
- Runtime overrides and Optuna best parameters are saved under the run artifact directory.

---

## 6. Required MLOps artifacts

Each run must create an immutable directory:

```text
artifacts/runs/<run_id>/
├── resolved_parameters.json
├── source_manifest.json
├── data_manifest.json
├── data_quality.json
├── fold_manifest.json
├── feature_schema.json
├── selected_features.json
├── predictions.parquet
├── target_weights.parquet
├── universe_membership.parquet
├── quantbt_config.json
├── quantbt_result.parquet
├── metrics.json
├── model_bundle/
└── checksums.json
```

Minimum lineage:

```text
Git commit multifactor-mlops
Git commit trading-historical-data
Git commit quantbt
resolved parameters hash
data snapshot hash
feature schema hash
model hash
prediction hash
weight hash
QuantBT result hash
```

MLflow should log these artifacts, not only flattened parameters and summary metrics.

---

## 7. Mandatory automated tests

### Leakage and timing

1. **Future mutation test**: changing any observation after cutoff `T` must not alter features, universe, predictions, or weights at `t <= T`.
2. **No-backfill test**: fail if `bfill` is called in a data/feature/alignment path.
3. **Availability test**: every joined source row satisfies `available_at <= decision_time`.
4. **Label timing test**: label start equals execution time and label end is strictly later.
5. **Fold isolation test**: no outer-test timestamp appears in model/selector fitting indices.
6. **Unsupported split test**: invalid split raises; it never becomes full-sample.

### Data and universe

7. Requested symbol/date coverage must match the frozen snapshot manifest.
8. Cache freshness is checked by dates, not only symbol names.
9. Alternative-source rows are tagged and restricted to approved missing intervals.
10. Universe membership at `t` is unchanged by future turnover/listing data.
11. Warm-up and missing-feature symbols are ineligible rather than neutral-filled.

### Training-serving parity

12. The same as-of snapshot produces byte-equivalent ordered model features in training replay and inference.
13. Macro overlay outputs match between validation and inference.
14. Short target weights remain negative after all sizing and risk overlays.
15. Model bundle rejects missing, extra, reordered, or wrong-dtype features.

### QuantBT accounting

16. Every official backtest reports `engine=quantbt` and a pinned version/commit.
17. A QuantBT exception fails the run; no alternative engine is invoked.
18. One-trade fixture reconciles entry time, exit time, gross P&L, fees, slippage, funding, and equity.
19. Position lag is applied exactly once.
20. Missing QuantBT dates cannot be converted to zero returns silently.
21. Replaying saved target weights reproduces the saved equity curve and metrics.

### Optuna and reproducibility

22. Every trial score contains only inner-OOS returns.
23. Search-space parameters are proven to affect the executed pipeline.
24. Same snapshot, config, code, and seed reproduce predictions/weights within tolerance.
25. Optuna does not edit the base `parameters.json`.

---

## 8. Repair sequence

### PR 1 — Correctness gate

Scope:

- strict config schema;
- implement/rename valid split and remove full fallback;
- canonical timing/label builder;
- QuantBT-only runner;
- remove custom official backtest and silent fallback;
- ban `bfill`;
- preserve missingness;
- fix serving short-sign bug;
- add P0 leakage/accounting tests.

Exit condition:

```text
No current metric is reused.
All P0 tests pass.
A deterministic hand-calculated trade reconciles with QuantBT.
```

### PR 2 — Point-in-time data and parity

Scope:

- `trading-historical-data` adapter;
- controlled alternative-source adapter;
- snapshot and provenance manifests;
- point-in-time macro alignment;
- rolling lagged universe;
- shared feature and macro transformations;
- training-serving parity tests.

Exit condition:

```text
Future mutation test passes.
No historical inference requires a network call.
Training and serving feature hashes match.
```

### PR 3 — Nested tuning and MLOps finalization

Scope:

- inner walk-forward Optuna;
- outer untouched validation;
- immutable best-parameter artifact;
- final refit pipeline;
- complete MLflow lineage;
- deterministic replay of saved weights through QuantBT.

Exit condition:

```text
One command reproduces the full OOS result from a frozen snapshot.
The registered model is the explicit final refit, not the last fold model.
```

---

## 9. Definition of done

The implementation is acceptable only when all statements below are true:

- Every displayed strategy metric comes from QuantBT.
- There is no custom-backtest fallback in the scoring path.
- Unsupported split values raise immediately.
- Research metrics are stitched exclusively from untouched outer-OOS folds.
- Optuna sees only inner-OOS results.
- No `bfill` exists in the decision pipeline.
- Every source is aligned on `available_at <= decision_time`.
- Universe selection uses only lagged historical information.
- Missing/untradable symbols cannot enter a portfolio through neutral filling.
- Label, signal, execution, and position timestamps obey one tested contract.
- Funding, fees, slippage, and position lag are applied exactly once.
- Training, validation, replay, and inference use the same feature and portfolio code.
- A registered model contains its schema, transformations, timing contract, source lineage, and QuantBT version.
- Mutating future data cannot change any earlier decision or P&L.
- Frozen data + code + parameters + seed reproduce the saved weights and QuantBT equity curve.

Only after these conditions pass should the strategy's edge be reevaluated. The first clean result should be treated as a new baseline; it should not be compared mechanically with the contaminated historical metrics.

---

## 10. Audited source locations

Primary implementation areas reviewed:

- Repository branch: <https://github.com/BobbyAxerol/multifactor-mlops/tree/feat/multifactor-macro-features>
- Parameters: <https://github.com/BobbyAxerol/multifactor-mlops/blob/feat/multifactor-macro-features/parameters.json>
- Training/splits/backtest routing: <https://github.com/BobbyAxerol/multifactor-mlops/blob/feat/multifactor-macro-features/training/train.py>
- Factors: <https://github.com/BobbyAxerol/multifactor-mlops/blob/feat/multifactor-macro-features/util/factors.py>
- Rebalance/custom backtest: <https://github.com/BobbyAxerol/multifactor-mlops/blob/feat/multifactor-macro-features/util/rebalance.py>
- Macro collection: <https://github.com/BobbyAxerol/multifactor-mlops/blob/feat/multifactor-macro-features/util/macro_collector.py>
- Market-data collection: <https://github.com/BobbyAxerol/multifactor-mlops/blob/feat/multifactor-macro-features/util/data_collector.py>
- Optuna: <https://github.com/BobbyAxerol/multifactor-mlops/blob/feat/multifactor-macro-features/optimization/optuna_kernel.py>
- Model registration/inference: <https://github.com/BobbyAxerol/multifactor-mlops/blob/feat/multifactor-macro-features/register_model.py>
- QuantBT: <https://github.com/BobbyAxerol/quantbt/tree/dev>
- Primary historical data: <https://github.com/BobbyAxerol/trading-historical-data>
