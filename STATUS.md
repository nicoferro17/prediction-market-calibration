# Project Status

## Done
- Week 1 (week-1-done): Initialized `pm-quant` repo and CI (`ruff`, `mypy --strict`, `pytest`), verified live Polymarket/Kalshi APIs and documented Nov 2026 counterfactual taker fees (`docs/api_and_fees_2026.md`), built immutable schemas (`schemas.py`), `AsyncRateLimitedClient` (`http.py`), `KalshiClient` (`kalshi.py`), `PolymarketClient` (`polymarket.py`), offline unit suites (68 tests, 92% coverage on `pm_quant.ingest`), opt-in live integration tests (`tests/ingest/test_live_integration.py`, `@pytest.mark.live`), and portfolio `README.md`.
- Week 2 Block 1: Implemented deterministic L2 universe selection policy (`src/pm_quant/ingest/universe.py`) with separate target-series/core per-venue budgets, 45d target horizon filter, MVE/combo and 5m/15m crypto exclusions, live screening helpers (`parse_gamma_live_candidate`, `parse_kalshi_live_candidate`, `get_live_market_candidates`) preserving `parse_gamma_market` anti-leakage invariants, and offline unit tests (`tests/ingest/test_universe.py`; 143 total tests passing, 94% coverage on `pm_quant.ingest`).

## In Progress
- None.

## Next Atomic Task
- Week 2 Block 2: Implement Parquet `schema_version` writer & daily L2 snapshot collector

## Known Issues
- None.
