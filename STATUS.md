# Project Status

## Done
- Week 1 (week-1-done): Initialized `pm-quant` repo and CI (`ruff`, `mypy --strict`, `pytest`), verified live Polymarket/Kalshi APIs and documented Nov 2026 counterfactual taker fees (`docs/api_and_fees_2026.md`), built immutable schemas (`schemas.py`), `AsyncRateLimitedClient` (`http.py`), `KalshiClient` (`kalshi.py`), `PolymarketClient` (`polymarket.py`), offline unit suites (68 tests, 92% coverage on `pm_quant.ingest`), opt-in live integration tests (`tests/ingest/test_live_integration.py`, `@pytest.mark.live`), and portfolio `README.md`.
- Week 2 (week-2-done): Built deterministic L2 universe selector (`universe.py`, separate target-series/core budgets, 45d horizon, MVE & 5m/15m crypto exclusions), versioned Zstd Parquet storage (`storage.py`, `L2_SNAPSHOT_SCHEMA_V1`, top-20 depth cap, explicit `snapshot_date`, DuckDB `union_by_name=true`), async daily collector (`collector.py`, exact Kalshi weather series expansion, `scripts/collect_daily.py`), idempotent HF Datasets sync (`hf_sync.py`), scheduled GitHub Actions workflow (`daily_pipeline.yml`, cron `17 15 * * *` UTC, DuckDB verify before HF upload), hardened 6-invariant verifier (`scripts/verify_l2_snapshot.py`, PASS on first live HF snapshot `2026-11-23.parquet`), and README/brief docs (192 tests, 94% coverage on `pm_quant.ingest`).
- Week 3 Block 1: Added `historical_schema.py` (`BackfillPerimeter`, `ResolvedMarket`, `HistoricalSnapshotRecord`, `HISTORICAL_CALIBRATION_SCHEMA_V1`, pure `parse_gamma_resolved_market`/`parse_kalshi_resolved_market` leaving `parse_gamma_market()` untouched, deterministic SHA-256 variable-tau sampler covering `tau < 1d`, atomic Zstd Parquet + DuckDB `union_by_name=true` I/O) and offline suite `test_historical_schema.py` (226 tests, 93% coverage on `pm_quant.ingest`).

## In Progress
- None.

## Next Atomic Task
- Week 3 Block 2: Implement point-in-time Polymarket historical backfill (`polymarket.py` `end_date_min/max` + `get_resolved_markets()`, `backfill_polymarket.py`, offline `test_backfill_polymarket.py`).
  - Locked Block 2 decisions: (1) Gamma monthly chronological slicing via `end_date_min`/`end_date_max` (NEVER `order=volume*`), initial 1.5-2x oversampling before guards; (2) fetch Data v2 prices (`[t_i-7d, t_i]`, `exclude_settlement=True`) -> CLOB fallback (`[t_i-14d, t_i]`) BEFORE `/v2/trades` and skip trades if price missing/outside `[min_price, max_price]`; (3) `/v2/trades` mini-batches of 1-5 `condition_ids` (`taker_only=true`, `cursor=`, `limit=1000`, `max_pages=15`, per-market `trades_truncated`, never Gamma `volume`); (4) resumable append-only JSONL checkpoint `data/backfill_cache/{venue}_{perimeter_hash}.jsonl` (`checkpoint_version`, `perimeter_hash`, `market_id`, `status: ok|skipped`, `skip_reason`, `fetched_at_utc`, `record` with `Decimal` as `str`; ignore truncated tail line, never cache transient HTTP errors); (5) emit `outcome_side="outcome_0"` when `is_yes_no=False`, populate `fee_type` from `metadata.extra`, and keep Kalshi Block 3 default `vol_usd_le_t = sum(vol_contracts * p_candle)` alongside `vol_contracts_le_t` (`vol_unit="contracts_notional"`).

## Known Issues
- None.
