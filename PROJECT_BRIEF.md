# pm-quant — Project Brief

## Objective
Quantitative research framework for binary prediction markets (**Polymarket** and **Kalshi**) built on a **€0 budget** using public APIs and open-source tooling. The project rests on two pillars:
1. **Hierarchical Bayesian Calibration**: model systematic miscalibration across venues, categories, liquidity tiers, and time-to-resolution with full posterior uncertainty.
2. **Taker L2 Execution & Bayesian Kelly**: measure empirical taker execution costs on L2 orderbooks (spread, depth, fees, slippage) and size positions via uncertainty-aware Bayesian Kelly allocation.

## Tech Stack
- **Runtime & Packaging**: Python >= 3.12 managed with `uv`.
- **Core dependencies**: `polars`, `duckdb`, `httpx`, `scipy`, `statsmodels`, `scikit-learn`.
- **Dev group** (`uv sync --dev`): `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`.
- **Isolated `bayes` group**: `pymc`, `numpyro` (excluded from default sync).

## Repository Structure
```text
src/pm_quant/
  ingest/      # Venue API clients and data ingestion pipelines
  evaluation/  # Scoring rules, reliability diagnostics, calibration metrics
  models/      # Frequentist baselines and hierarchical Bayesian models
  execution/   # L2 orderbook reconstruction, slippage, Bayesian Kelly sizing
  backtest/    # Event-driven and walk-forward historical backtesting
  live/        # Monitoring, paper trading, and operational workflows
tests/         # Unit, integration, and opt-in live API tests (-m live)
data/          # Local datasets (gitignored; never committed)
```

## Key 2026 API Specifications
- **Polymarket Data API v2**: filter market trades with `condition=`, enforce `taker_only=true` to avoid maker/taker double-counting, and paginate via `next_cursor`.
- **Kalshi API**: partition historical vs. active data using the `historical/cutoff` boundary; note that `orderbook_fp` responses are **bid-only** (`yes_dollars` and `no_dollars` fixed-point dollar strings, parsed as `Decimal`) and sorted in **ascending** price order (best bid is last; implied asks derived from complementary `1 - bid` levels).

## Working Rules
- **Language rule**: chat communication is in **Italian**; everything inside the repository (code, comments, docstrings, markdown, commit messages) must be in **English**.
- **Scope & hygiene**:
  - Never install the `bayes` dependency group before **Week 9**.
  - Never commit raw/processed datasets (`data/`) or secrets (`.env`, credentials, tokens).
  - Keep live network tests marked with `@pytest.mark.live` (excluded from default `pytest` runs).
