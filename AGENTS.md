# AGENTS.md

Two-layered ML quant strategy (cross-sectional alpha + macro stress overlay) for crypto futures. The README describes the legacy pipeline; the actively developed code is the v3 package.

## Canonical codebase (single pipeline, legacy removed)

- **Active**: `src/multifactor_mlops/` — config, data, features, labels, optimization, pipelines, portfolio, tracking, universe, backtest modules. All tests import `src.multifactor_mlops.*`.
- The legacy `multifactor_portfolio/` package was **deleted** in V4 cleanup — do not recreate it; the README documents the canonical v4 entrypoints.
- The package is NOT installed; it uses `src.multifactor_mlops` imports. Always run from the repo root.
- `MULTIFACTOR_MLOPS_*.md` at root are historical repair-plan records (V4 resolved them); `RESEARCH_LOG.md` is the living results log.

## Environment (critical)

- This repo has **no pyproject.toml/poetry.lock**. The Poetry env belongs to the parent workspace `/root/bobby/pool_alpha` (project `backtest-env`, Python 3.12.13, venv at `/root/bobby/pool_alpha/.venv`). Plain `python`/`python3` is 3.10 and wrong — always use `poetry run`.
- External deps that are only reachable via `sys.path` insertion (do not pip-install):
  - QuantBT backtest engine at `/root/bobby/pool_alpha/quantbt` — **READ-ONLY, never modify it** (user rule; zero-fallback policy: any failure raises `QuantBTExecutionError`).
  - Raw market data at `/root/bobby/pool_alpha/alphas_storage/_get_data`.
- MLflow: `sqlite:///mlflow.db` from `.env` (`MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `MODEL_NAME`). `mlflow.db`, `mlruns/`, `data/*.csv.gz` are gitignored.

## Commands

```bash
poetry run pytest tests/                     # real verification: phase1/2/3 + integrity + v4 + remaining fixes
poetry run pytest tests/test_phase3.py -q    # focused check
poetry run pre-commit run --all-files        # black + ruff + hooks
```

- CI (`.github/workflows/ci.yml`) runs `tests/` — the same canonical suite as locally.
- `parameters.json` is asserted immutable by `tests/test_phase3.py::test_immutable_base_config` — never let tuning mutate it in place; write tuned params to `artifacts/*.json` instead (model_config.json, strategy_config.json, oof_predictions.csv are the locked tuning artifacts).

## Repo conventions (from `.agents/AGENTS.md`)

- **Never commit, merge, or push to `main`.** Work strictly on the active feature branch (`feat/multifactor-macro-features-v3`); commit there after each completed fix. CI runs on main/dev/feat/*/research/* pushes.
- Before committing: check `git status` (don't stage unrelated work), verify `git config user.name`/`user.email` (identity: BobbyAxerol / vugioan11022002@gmail.com).
- **No large Optuna/training runs without explicit approval** — small runs for syntax debugging only.
- Ask before major refactors or direction changes; after a major change, review the whole integration flow for cross-file synchronization.
