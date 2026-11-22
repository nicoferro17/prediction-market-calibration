# Project Status

## Done
- Week 1, Day 1: Initialized `pm-quant` repository (`pyproject.toml`, `uv` dependency groups, `src/pm_quant/` package layout, `py.typed`, `Makefile`, `PROJECT_BRIEF.md`, `STATUS.md`, smoke test, GitHub Actions CI, `.gitignore`).
- Week 1, Day 2: Added read-only Polymarket and Kalshi API smoke test script (`scripts/smoke_test_apis.py`), manual GitHub Actions workflow (`.github/workflows/smoke_test_apis.yml`), and extended `Makefile` checks to `scripts/`.
- Week 1, Day 3: Documented Nov 2026 GitHub Actions API smoke-test results (9/9 HTTP 200, no WAF blocks) and counterfactual taker fee schedules (`docs/api_and_fees_2026.md`), implemented immutable normalized ingestion schemas (`src/pm_quant/ingest/schemas.py`) and `AsyncRateLimitedClient` (`src/pm_quant/ingest/http.py`), and added unit tests (`tests/ingest/test_http_and_schemas.py`).

## In Progress
- None.

## Next Atomic Task
- Week 1, Day 4: Implement Polymarket and Kalshi venue ingestion clients on top of `AsyncRateLimitedClient` and normalized schemas.

## Known Issues
- None.
