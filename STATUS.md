# Project Status

## Done
- Week 1, Day 1: Initialized `pm-quant` repository (`pyproject.toml`, `uv` dependency groups, `src/pm_quant/` package layout, `py.typed`, `Makefile`, `PROJECT_BRIEF.md`, `STATUS.md`, smoke test, GitHub Actions CI, `.gitignore`).
- Week 1, Day 2: Added read-only Polymarket and Kalshi API smoke test script (`scripts/smoke_test_apis.py`), manual GitHub Actions workflow (`.github/workflows/smoke_test_apis.yml`), and extended `Makefile` checks to `scripts/`.
- Week 1, Day 3: Documented Nov 2026 GitHub Actions API smoke-test results (9/9 HTTP 200, no WAF blocks) and counterfactual taker fee schedules (`docs/api_and_fees_2026.md`), implemented immutable normalized ingestion schemas (`src/pm_quant/ingest/schemas.py`) and `AsyncRateLimitedClient` (`src/pm_quant/ingest/http.py`), and added unit tests (`tests/ingest/test_http_and_schemas.py`).
- Week 1, Day 4: Implemented pure Kalshi bid-only orderbook and market parsers (`parse_kalshi_orderbook`, `parse_kalshi_market`, `parse_kalshi_timestamp`) and async `KalshiClient` (`src/pm_quant/ingest/kalshi.py`, re-exported via `src/pm_quant/ingest/__init__.py`) with cached `/historical/cutoff` routing, cursor pagination, <=100 batch chunking for orderbooks and live candlesticks, and archived candlestick fetching.
- Week 1, Day 5: Added comprehensive offline unit test suite (`tests/ingest/test_kalshi.py`, 24 tests, 93% coverage on `pm_quant.ingest.kalshi`) covering ascending bid-only parsing, YES/NO symmetry identities, `max_depth=20` truncation, empty/null/one-sided/dirty-level edge cases, `/historical/cutoff` caching and routing, cursor pagination, and 235-ticker batch chunking; fixed `/markets/orderbooks` to pass `depth` in query parameters.
- Week 1, Day 6: Implemented Polymarket pure parsers (`parse_gamma_market`, `parse_clob_orderbook`, `parse_v2_trade`) and async `PolymarketClient` (`src/pm_quant/ingest/polymarket.py`, re-exported via `src/pm_quant/ingest/__init__.py`) covering Gamma JSON-string/list parsing with strict `is_yes_no` and no lifetime-volume leakage, CLOB explicit re-sorting and top-20 depth cap (`GET /book`, `POST /books`, `GET /prices-history`), Data API v2 `/trades` (<=20 `condition=` chunking, forced `taker_only=true`, `next_cursor` pagination, no `market=`/`offset=`), and `/v2/prices-history` (15-day window splitting, ordered deduplication, empty-block resolution-decay warning), with offline unit tests (`tests/ingest/test_polymarket.py`, 25 tests, 90% coverage on `pm_quant.ingest.polymarket`).

## In Progress
- None.

## Next Atomic Task
- Week 1, Day 7: Add opt-in live integration tests (`tests/ingest/test_live_integration.py`, `@pytest.mark.live`), verify >85% test coverage on `pm_quant.ingest`, update `README.md`, and close Week 1 (`week-1-done`).

## Known Issues
- None.
