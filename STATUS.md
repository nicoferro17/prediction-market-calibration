# Project Status

## Done
- Week 1 (week-1-done): Initialized `pm-quant` repo and CI (`ruff`, `mypy --strict`, `pytest`), verified live Polymarket/Kalshi APIs and documented Nov 2026 counterfactual taker fees (`docs/api_and_fees_2026.md`), built immutable schemas (`schemas.py`), `AsyncRateLimitedClient` (`http.py`), `KalshiClient` (`kalshi.py`), `PolymarketClient` (`polymarket.py`), offline unit suites (68 tests, 92% coverage on `pm_quant.ingest`), opt-in live integration tests (`tests/ingest/test_live_integration.py`, `@pytest.mark.live`), and portfolio `README.md`.
- Week 2 Block 1: Implemented deterministic L2 universe selection policy (`src/pm_quant/ingest/universe.py`) with separate target-series/core per-venue budgets, 45d target horizon filter, MVE/combo and 5m/15m crypto exclusions, live screening helpers (`parse_gamma_live_candidate`, `parse_kalshi_live_candidate`, `get_live_market_candidates`) preserving `parse_gamma_market` anti-leakage invariants, and offline unit tests (`tests/ingest/test_universe.py`; 143 total tests passing, 94% coverage on `pm_quant.ingest`).
- Week 2 Block 2: Implemented Parquet L2 snapshot storage (`src/pm_quant/ingest/storage.py`) with explicit `L2_SNAPSHOT_SCHEMA_V1` (`schema_version='v1'`), defensive L2 top-15..20 depth cap and re-sorting (`orderbooks_to_frame`), atomic Zstd daily Parquet writer (`write_daily_parquet`), DuckDB `union_by_name=true` reader (`read_snapshots_duckdb`), and offline unit tests including v1+v2 schema evolution (`tests/ingest/test_storage.py`; 156 total tests passing, 93% coverage on `pm_quant.ingest`).
- Week 2 Block 3: Implemented async daily L2 collector orchestrator (`src/pm_quant/ingest/collector.py`, `CollectionSummary`, `collect_daily_l2`) with Kalshi `mve_filter=exclude` + explicit target series, Polymarket `volume24hr` + target tag scans, dual-side L2 batching with depth cap and per-venue error isolation, CLI (`scripts/collect_daily.py`, `Makefile` `collect` target), and offline `MockTransport` tests (`tests/ingest/test_collector.py`; 166 total tests passing, 94% coverage on `pm_quant.ingest`).
- Week 2 Block 4: Added `huggingface-hub` (keeping `bayes` group isolated), implemented idempotent Hugging Face Datasets daily Parquet upload (`src/pm_quant/ingest/hf_sync.py`, `validate_daily_parquet_for_upload`, `upload_daily_parquet`, `HfUploadApi`) with DuckDB pre-upload validation and token redaction, integrated `--upload-hf`/`--hf-repo` in `scripts/collect_daily.py`, and added 100% offline tests (`tests/ingest/test_hf_sync.py`; 180 total tests passing, 94% coverage on `pm_quant.ingest`).

## In Progress
- None.

## Next Atomic Task
- Week 2 Block 5: Configure scheduled daily L2 collection workflow (GitHub Actions + HF upload) and close Week 2

## Known Issues
- None.
