# pm-quant — Prediction Market Calibration & Execution

[![CI](https://github.com/nicoferro17/prediction-market-calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/nicoferro17/prediction-market-calibration/actions/workflows/ci.yml)

`pm-quant` is a quantitative research framework for binary prediction markets (**Polymarket** and **Kalshi**), built end-to-end on public APIs and open-source tooling.
It separates true statistical miscalibration from microstructure friction by pairing hierarchical Bayesian probability estimation with empirical L2 orderbook execution modeling.
Every historical backtest applies counterfactual November 2026 taker fees and strict anti-leakage controls to evaluate realistic, uncertainty-adjusted net alpha.

## Research Architecture (Two Pillars)

1. **Hierarchical Bayesian Calibration (`models/`, `evaluation/`)**
   Models systematic probability miscalibration across venues, market categories, liquidity tiers, and time-to-resolution with full posterior uncertainty, benchmarked against frequentist baselines using proper scoring rules and reliability diagnostics.
2. **Taker L2 Execution & Bayesian Kelly (`ingest/`, `execution/`, `backtest/`)**
   Reconstructs normalized L2 orderbook depth (exact `Decimal` pricing, spread, slippage, and counterfactual taker fees) and sizes positions via posterior-uncertainty-aware Bayesian Kelly allocation under walk-forward historical backtesting.

## Repository Layout

```text
src/pm_quant/
  ingest/      # Async venue API clients (Kalshi v2, Polymarket Gamma/CLOB/Data v2) & schemas
  evaluation/  # Proper scoring rules, reliability curves, calibration diagnostics
  models/      # Frequentist baselines and hierarchical Bayesian calibration models
  execution/   # L2 orderbook slippage, counterfactual taker fees, Bayesian Kelly sizing
  backtest/    # Walk-forward and event-driven historical backtesting
  live/        # Monitoring, paper trading, and operational workflows
docs/          # Live API verification notes & 2026 fee schedules
tests/         # Offline unit tests (default) & opt-in live API integration tests (-m live)
```

## Daily L2 Snapshot Collector

An automated GitHub Actions pipeline ([`daily_pipeline.yml`](.github/workflows/daily_pipeline.yml)) collects normalized dual-side (`yes`/`no`) L2 orderbook snapshots from Polymarket and Kalshi every day at **`17 15 * * *` UTC** (15:17 UTC) and publishes them to the public Hugging Face Dataset: [Nicoferro17/prediction-market-calibration](https://huggingface.co/datasets/Nicoferro17/prediction-market-calibration).

- **Universe selection policy (`ingest/universe.py`)**: deterministic per-venue selection with separate budgets for `target_series` (macro/economics/politics series within a **45-day** resolution horizon) and `volume_core` (highest-24h-volume active markets), explicitly excluding Kalshi MVE/combo tickers (`KXMVE*`) and 5m/15m crypto markets.
- **Versioned Parquet schema (`ingest/storage.py`)**: Zstd-compressed daily files (`l2_snapshots/YYYY-MM-DD.parquet`) written with explicit `L2_SNAPSHOT_SCHEMA_V1` (`schema_version = 'v1'`), exact `Decimal` prices/sizes capped at top-20 depth per side (`bid_prices` desc, `ask_prices` asc), explicit `volume_24h_unit` (`usd` on Polymarket, `contracts` on Kalshi), and DuckDB `read_parquet(..., union_by_name=true)` verification (`scripts/verify_l2_snapshot.py`).

## Quickstart

Requires Python `>= 3.12` and [`uv`](https://docs.astral.sh/uv/).

```bash
# Install core + dev dependencies (ruff, mypy, pytest, pytest-cov)
uv sync --dev

# Run lint, format check, strict mypy, and offline unit test suite
make check

# Optional: run live API integration tests against public Kalshi & Polymarket endpoints
uv run pytest -m live -q
```

## Documentation

- [2026 API Verification & Counterfactual Taker Fee Schedule](docs/api_and_fees_2026.md) — Live endpoint schema checks, Polymarket CLOB vs. Data API v2 comparison, and November 2026 taker fee formulas for Polymarket International and Kalshi.
