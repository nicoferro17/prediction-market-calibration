# Project Status

## Done
- Week 1 (week-1-done): Initialized `pm-quant` repo and CI (`ruff`, `mypy --strict`, `pytest`), verified live Polymarket/Kalshi APIs and documented Nov 2026 counterfactual taker fees (`docs/api_and_fees_2026.md`), built immutable schemas (`schemas.py`), `AsyncRateLimitedClient` (`http.py`), `KalshiClient` (`kalshi.py`), `PolymarketClient` (`polymarket.py`), offline unit suites (68 tests, 92% coverage on `pm_quant.ingest`), opt-in live integration tests (`tests/ingest/test_live_integration.py`, `@pytest.mark.live`), and portfolio `README.md`.
- Week 2 (week-2-done): Built deterministic L2 universe selector (`universe.py`, separate target-series/core budgets, 45d horizon, MVE & 5m/15m crypto exclusions), versioned Zstd Parquet storage (`storage.py`, `L2_SNAPSHOT_SCHEMA_V1`, top-20 depth cap, explicit `snapshot_date`, DuckDB `union_by_name=true`), async daily collector (`collector.py`, exact Kalshi weather series expansion, `scripts/collect_daily.py`), idempotent HF Datasets sync (`hf_sync.py`), scheduled GitHub Actions workflow (`daily_pipeline.yml`, cron `17 15 * * *` UTC, DuckDB verify before HF upload), hardened 6-invariant verifier (`scripts/verify_l2_snapshot.py`, PASS on first live HF snapshot `2026-11-23.parquet`), and README/brief docs (192 tests, 94% coverage on `pm_quant.ingest`).

## In Progress
- None.

## Next Atomic Task
- Week 3 Block 1: Define historical backfill perimeter & point-in-time trade/candlestick schema.

## Known Issues
- None.
