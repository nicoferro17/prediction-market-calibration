# Project Status

## Done
- Week 1 (week-1-done): Initialized `pm-quant` repo and CI (`ruff`, `mypy --strict`, `pytest`), verified live Polymarket/Kalshi APIs and documented Nov 2026 counterfactual taker fees (`docs/api_and_fees_2026.md`), built immutable schemas (`schemas.py`), `AsyncRateLimitedClient` (`http.py`), `KalshiClient` (`kalshi.py`), `PolymarketClient` (`polymarket.py`), offline unit suites (68 tests, 92% coverage on `pm_quant.ingest`), opt-in live integration tests (`tests/ingest/test_live_integration.py`, `@pytest.mark.live`), and portfolio `README.md`.
- Week 2 (week-2-done): Built deterministic L2 universe selector (`universe.py`, separate target-series/core budgets, 45d horizon, MVE & 5m/15m crypto exclusions), versioned Zstd Parquet storage (`storage.py`, `L2_SNAPSHOT_SCHEMA_V1`, top-20 depth cap, explicit `snapshot_date`, DuckDB `union_by_name=true`), async daily collector (`collector.py`, exact Kalshi weather series expansion, `scripts/collect_daily.py`), idempotent HF Datasets sync (`hf_sync.py`), scheduled GitHub Actions workflow (`daily_pipeline.yml`, cron `17 15 * * *` UTC, DuckDB verify before HF upload), hardened 6-invariant verifier (`scripts/verify_l2_snapshot.py`, PASS on first live HF snapshot `2026-11-23.parquet`), and README/brief docs (192 tests, 94% coverage on `pm_quant.ingest`).
- Week 3 Block 1: Added `historical_schema.py` (`BackfillPerimeter`, `ResolvedMarket`, `HistoricalSnapshotRecord`, `HISTORICAL_CALIBRATION_SCHEMA_V1`, pure `parse_gamma_resolved_market`/`parse_kalshi_resolved_market` leaving `parse_gamma_market()` untouched, deterministic SHA-256 variable-tau sampler covering `tau < 1d`, atomic Zstd Parquet + DuckDB `union_by_name=true` I/O) and offline suite `test_historical_schema.py` (226 tests, 93% coverage on `pm_quant.ingest`).
- Week 3 Block 2: Extended `polymarket.py` with ISO-8601 UTC `end_date_min`/`end_date_max` on `_fetch_gamma_markets_raw()`/`get_markets()`, `get_resolved_markets()`, and pagination-truncation tracking (`get_trades_with_truncation()`); implemented `backfill_polymarket.py` (monthly chronological Gamma slicing, SHA-256 balanced `(category, YYYY-MM)` stratification, strict `<= t_i` as-of price with Data v2 `[t_i-7d, t_i]` -> CLOB `[t_i-14d, t_i]` fallback before `/v2/trades`, mini-batch `taker_only=true` `<= t_i` volume with `trades_truncated`, resumable append-only JSONL checkpoint) and offline `test_backfill_polymarket.py` (237 tests, 92% coverage on `pm_quant.ingest`).

## In Progress
- None.

## Next Atomic Task
- Week 3 Block 3: Implement point-in-time Kalshi historical backfill (`backfill_kalshi.py` with `/historical/cutoff` routing, candlestick price extraction `<= t_i`, `vol_usd_le_t = sum(vol_contracts * p_candle)` alongside `vol_contracts_le_t` with `vol_unit="contracts_notional"`, resumable JSONL checkpoint, and offline `test_backfill_kalshi.py`).

## Known Issues
- None.
