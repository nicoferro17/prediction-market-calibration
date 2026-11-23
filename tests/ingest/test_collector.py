"""Offline unit and end-to-end tests for the daily L2 collector orchestrator and CLI."""

from __future__ import annotations

import importlib.util
import json
import logging
from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from pm_quant.ingest import (
    L2_SNAPSHOT_SCHEMA_V1,
    SCHEMA_VERSION,
    AsyncRateLimitedClient,
    CollectionSummary,
    KalshiClient,
    PolymarketClient,
    UniversePolicy,
    collect_daily_l2,
    read_snapshots_duckdb,
)

FIXED_TS = datetime(2026, 11, 23, 14, 30, 0, tzinfo=UTC)
FIXED_DATE = date(2026, 11, 23)


async def _noop_sleep(_delay: float) -> None:
    """Instant async sleep for fast retry/503 offline tests."""


def _make_kalshi_client(
    handler: Any,
    *,
    max_retries: int = 2,
) -> KalshiClient:
    http = AsyncRateLimitedClient(
        max_retries=max_retries,
        sleep=_noop_sleep,
        transport=httpx.MockTransport(handler),
        now_fn=lambda: FIXED_TS,
    )
    return KalshiClient(http_client=http, now_fn=lambda: FIXED_TS)


def _make_polymarket_client(
    handler: Any,
    *,
    max_retries: int = 2,
) -> PolymarketClient:
    http = AsyncRateLimitedClient(
        max_retries=max_retries,
        sleep=_noop_sleep,
        transport=httpx.MockTransport(handler),
        now_fn=lambda: FIXED_TS,
    )
    return PolymarketClient(http_client=http, now_fn=lambda: FIXED_TS)


def _load_cli_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "collect_daily.py"
    spec = importlib.util.spec_from_file_location("collect_daily_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ascending_kalshi_levels(start_cents: int, count: int) -> list[list[str]]:
    """Generate ``count`` ascending [price, size] string pairs starting at ``start_cents``."""
    return [
        [f"{Decimal(start_cents + i) / Decimal(100):.4f}", f"{10 + i}.00"] for i in range(count)
    ]


def _scrambled_clob_levels(start_cents: int, count: int) -> list[dict[str, str]]:
    """Generate ``count`` deterministically scrambled CLOB ``{price, size}`` dicts."""
    ordered = [
        {"price": f"{Decimal(start_cents + i) / Decimal(100):.4f}", "size": f"{25 + i}.50"}
        for i in range(count)
    ]
    return [ordered[(i * 7) % count] for i in range(count)]


# ---------------------------------------------------------------------------
# 1. End-to-end offline collection across Polymarket & Kalshi
# ---------------------------------------------------------------------------


async def test_collect_daily_l2_end_to_end_offline(tmp_path: Path) -> None:
    """Full pipeline: HTTP params, exclusions, target series, dual sides, Parquet & DuckDB."""
    gamma_requests: list[dict[str, str]] = []
    clob_post_bodies: list[list[dict[str, Any]]] = []
    kalshi_market_requests: list[dict[str, str]] = []
    kalshi_book_requests: list[httpx.QueryParams] = []

    def polymarket_handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "gamma-api.polymarket.com" and request.url.path == "/markets":
            params = dict(request.url.params)
            gamma_requests.append(params)
            if params.get("tag_id") == "84":
                # Low-volume Weather market below core threshold ($5,000), saved via target tag.
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": "pm-weather-1",
                            "conditionId": "0xpm_weather_lowvol",
                            "question": "Will NYC high temperature exceed 55F tomorrow?",
                            "slug": "nyc-high-temp-55f",
                            "category": "Weather",
                            "outcomes": '["Yes", "No"]',
                            "clobTokenIds": '["tok_pm_w_yes", "tok_pm_w_no"]',
                            "endDate": "2026-11-26T23:59:00Z",
                            "active": True,
                            "closed": False,
                            "enableOrderBook": True,
                            "acceptingOrders": True,
                            "negRisk": False,
                            "feeType": "weather",
                            "orderPriceMinTickSize": 0.01,
                            "volume24hr": "12.50",
                            "openInterest": "250.00",
                        }
                    ],
                    request=request,
                )

            # General high-volume scan: includes a valid non-Yes/No core market + crypto 5m noise.
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "pm-core-1",
                        "conditionId": "0xpm_nba_core",
                        "question": "Lakers vs Celtics winner",
                        "slug": "lakers-vs-celtics",
                        "category": "Sports",
                        "outcomes": '["Lakers", "Celtics"]',
                        "clobTokenIds": '["tok_pm_lakers", "tok_pm_celtics"]',
                        "endDate": "2026-12-05T03:00:00Z",
                        "active": True,
                        "closed": False,
                        "enableOrderBook": True,
                        "acceptingOrders": True,
                        "negRisk": False,
                        "feeType": "sports",
                        "orderPriceMinTickSize": 0.01,
                        "volume24hr": 85000.0,
                        "openInterest": 150000.0,
                    },
                    {
                        "id": "pm-crypto-5m",
                        "conditionId": "0xpm_btc_5m_excluded",
                        "question": "Will BTC go up in the next 5m?",
                        "slug": "btc-updown-5m-1795444200",
                        "category": "Crypto",
                        "outcomes": '["Up", "Down"]',
                        "clobTokenIds": '["tok_5m_up", "tok_5m_down"]',
                        "endDate": "2026-11-23T14:35:00Z",
                        "active": True,
                        "closed": False,
                        "enableOrderBook": True,
                        "acceptingOrders": True,
                        "volume24hr": 999999.0,
                    },
                ],
                request=request,
            )

        if request.url.host == "clob.polymarket.com" and request.url.path == "/books":
            assert request.method == "POST"
            body = json.loads(request.content.decode("utf-8"))
            assert isinstance(body, list)
            clob_post_bodies.append(body)

            books_by_id: dict[str, dict[str, Any]] = {
                # 25 scrambled levels per side -> must be re-sorted and capped to <= 20.
                "tok_pm_lakers": {
                    "asset_id": "tok_pm_lakers",
                    "market": "0xpm_nba_core",
                    "bids": _scrambled_clob_levels(20, 25),
                    "asks": _scrambled_clob_levels(50, 25),
                },
                "tok_pm_celtics": {
                    "asset_id": "tok_pm_celtics",
                    "market": "0xpm_nba_core",
                    "bids": [{"price": "0.49", "size": "120.0"}],
                    "asks": [{"price": "0.51", "size": "90.0"}],
                },
                # One-sided book (bids only) for weather YES.
                "tok_pm_w_yes": {
                    "asset_id": "tok_pm_w_yes",
                    "market": "0xpm_weather_lowvol",
                    "bids": [{"price": "0.35", "size": "40.0"}],
                    "asks": [],
                },
                # Empty book for weather NO.
                "tok_pm_w_no": {
                    "asset_id": "tok_pm_w_no",
                    "market": "0xpm_weather_lowvol",
                    "bids": [],
                    "asks": [],
                },
            }
            returned = [
                books_by_id[item["token_id"]] for item in body if item["token_id"] in books_by_id
            ]
            return httpx.Response(200, json=returned, request=request)

        raise AssertionError(f"Unexpected Polymarket request: {request.method} {request.url}")

    def kalshi_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/trade-api/v2/markets":
            params = dict(request.url.params)
            kalshi_market_requests.append(params)
            if params.get("series_ticker") == "KXHIGHNY":
                # Low-volume Weather series below Kalshi core threshold (1,000 contracts).
                return httpx.Response(
                    200,
                    json={
                        "markets": [
                            {
                                "ticker": "KXHIGHNY-26NOV24-T55",
                                "event_ticker": "KXHIGHNY-26NOV24",
                                "series_ticker": "KXHIGHNY",
                                "title": "NYC high temp > 55F on Nov 24?",
                                "status": "active",
                                "market_type": "binary",
                                "close_time": "2026-11-24T23:59:00Z",
                                "volume_24h_fp": "5.00",
                                "open_interest_fp": "80.00",
                            }
                        ],
                        "cursor": "",
                    },
                    request=request,
                )

            # General open scan: valid core market + an MVE combo that must still be filtered out.
            return httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXPOL-26DEC-BILL",
                            "event_ticker": "KXPOL-26DEC",
                            "series_ticker": "KXPOL",
                            "title": "Will bill X pass Congress?",
                            "category": "Politics",
                            "status": "open",
                            "market_type": "binary",
                            "close_time": "2026-12-20T23:59:00Z",
                            "volume_24h_fp": "3500.00",
                            "open_interest_fp": "9000.00",
                        },
                        {
                            "ticker": "KXMVECROSSCATEGORY-S2026-COMBO",
                            "event_ticker": "KXMVECROSSCATEGORY-S2026",
                            "title": "Cross-category multi-leg parlay combo",
                            "status": "open",
                            "market_type": "binary",
                            "mve_collection_ticker": "KXMVECROSSCATEGORY",
                            "close_time": "2026-12-01T23:59:00Z",
                            "volume_24h_fp": "95000.00",
                        },
                    ],
                    "cursor": "",
                },
                request=request,
            )

        if request.url.path == "/trade-api/v2/markets/orderbooks":
            kalshi_book_requests.append(request.url.params)
            requested = request.url.params.get_list("tickers")
            raw_books: dict[str, dict[str, Any]] = {
                # 28 ascending bid levels per side -> must be capped to top 20.
                "KXPOL-26DEC-BILL": {
                    "ticker": "KXPOL-26DEC-BILL",
                    "orderbook_fp": {
                        "yes_dollars": _ascending_kalshi_levels(15, 28),
                        "no_dollars": _ascending_kalshi_levels(25, 28),
                    },
                },
                # One-sided YES-only bid -> yes_book has bids only, no_book has derived asks only.
                "KXHIGHNY-26NOV24-T55": {
                    "ticker": "KXHIGHNY-26NOV24-T55",
                    "orderbook_fp": {
                        "yes_dollars": [["0.4000", "25.00"]],
                        "no_dollars": [],
                    },
                },
            }
            return httpx.Response(
                200,
                json={"orderbooks": [raw_books[t] for t in requested if t in raw_books]},
                request=request,
            )

        raise AssertionError(f"Unexpected Kalshi request: {request.method} {request.url}")

    policy = UniversePolicy(
        target_series_prefixes=("KXHIGHNY",),
        polymarket_target_tag_ids=("84",),
    )
    kalshi_client = _make_kalshi_client(kalshi_handler)
    pm_client = _make_polymarket_client(polymarket_handler)

    try:
        summary = await collect_daily_l2(
            output_dir=tmp_path,
            policy=policy,
            max_depth=20,
            kalshi_client=kalshi_client,
            polymarket_client=pm_client,
            now_fn=lambda: FIXED_TS,
        )
    finally:
        await kalshi_client._http.aclose()
        await pm_client._http.aclose()

    # --- Verify HTTP request parameters ---
    assert len(kalshi_market_requests) == 2
    assert all(req.get("status") == "open" for req in kalshi_market_requests)
    assert all(req.get("mve_filter") == "exclude" for req in kalshi_market_requests)
    assert any(req.get("series_ticker") == "KXHIGHNY" for req in kalshi_market_requests)

    assert len(kalshi_book_requests) == 1
    assert kalshi_book_requests[0]["depth"] == "20"
    assert set(kalshi_book_requests[0].get_list("tickers")) == {
        "KXHIGHNY-26NOV24-T55",
        "KXPOL-26DEC-BILL",
    }

    assert len(gamma_requests) == 2
    general_gamma = [r for r in gamma_requests if "tag_id" not in r]
    tag_gamma = [r for r in gamma_requests if r.get("tag_id") == "84"]
    assert len(general_gamma) == 1
    assert general_gamma[0]["active"] == "true"
    assert general_gamma[0]["closed"] == "false"
    assert general_gamma[0]["order"] == "volume24hr"
    assert general_gamma[0]["ascending"] == "false"
    assert len(tag_gamma) == 1

    assert len(clob_post_bodies) == 1
    requested_tokens = {item["token_id"] for item in clob_post_bodies[0]}
    assert requested_tokens == {
        "tok_pm_lakers",
        "tok_pm_celtics",
        "tok_pm_w_yes",
        "tok_pm_w_no",
    }

    # --- Verify CollectionSummary ---
    assert isinstance(summary, CollectionSummary)
    assert summary.snapshot_date == FIXED_DATE
    assert summary.ts_utc == FIXED_TS
    assert summary.parquet_path == tmp_path / "2026-11-23.parquet"
    assert summary.parquet_path.exists()
    assert summary.file_size_bytes == summary.parquet_path.stat().st_size
    assert summary.file_size_bytes > 0
    assert summary.total_rows == 8
    assert summary.rows_by_venue == {"polymarket": 4, "kalshi": 4}
    assert summary.selected_by_reason == {"target_series": 2, "volume_core": 2}
    # Empty: tok_pm_w_no (1). One-sided: tok_pm_w_yes + KXHIGHNY yes & no (3).
    assert summary.empty_books_count == 1
    assert summary.one_sided_books_count == 3
    assert summary.failed_venues == ()

    with pytest.raises(FrozenInstanceError):
        summary.total_rows = 0  # type: ignore[misc]
    with pytest.raises(TypeError):
        summary.rows_by_venue["kalshi"] = 999  # type: ignore[index]

    # --- Re-read Parquet via DuckDB and verify schema, exclusions, labels, and <= 20 cap ---
    df = read_snapshots_duckdb(summary.parquet_path)
    assert df.schema == L2_SNAPSHOT_SCHEMA_V1
    assert df.height == 8
    assert set(df["schema_version"].to_list()) == {SCHEMA_VERSION}

    market_ids = set(df["market_id"].to_list())
    assert "0xpm_btc_5m_excluded" not in market_ids
    assert "KXMVECROSSCATEGORY-S2026-COMBO" not in market_ids
    assert market_ids == {
        "0xpm_nba_core",
        "0xpm_weather_lowvol",
        "KXHIGHNY-26NOV24-T55",
        "KXPOL-26DEC-BILL",
    }

    # Verify depth cap <= 20 across every row.
    assert max(df["bid_depth"].to_list()) == 20
    assert max(df["ask_depth"].to_list()) == 20
    assert all(len(prices) <= 20 for prices in df["bid_prices"].to_list())
    assert all(len(prices) <= 20 for prices in df["ask_prices"].to_list())

    # Verify low-volume weather series included with selection_reason == "target_series".
    weather_rows = df.filter(df["market_id"] == "KXHIGHNY-26NOV24-T55")
    assert weather_rows.height == 2
    assert set(weather_rows["outcome_side"].to_list()) == {"yes", "no"}
    assert set(weather_rows["selection_reason"].to_list()) == {"target_series"}
    assert weather_rows["outcome_label"].to_list() == ["No", "Yes"]

    pm_weather_rows = df.filter(df["market_id"] == "0xpm_weather_lowvol")
    assert pm_weather_rows.height == 2
    assert set(pm_weather_rows["selection_reason"].to_list()) == {"target_series"}

    # Verify Polymarket non-Yes/No outcome labels populated from metadata.outcomes.
    nba_rows = df.filter(df["market_id"] == "0xpm_nba_core").sort("outcome_side")
    assert nba_rows["outcome_side"].to_list() == ["no", "yes"]
    assert nba_rows["outcome_label"].to_list() == ["Celtics", "Lakers"]
    assert nba_rows["is_yes_no"].to_list() == [False, False]
    assert nba_rows["bid_depth"].to_list() == [1, 20]
    assert nba_rows["ask_depth"].to_list() == [1, 20]

    # Verify Kalshi core market emitted both sides with reciprocal top-of-book prices.
    kalshi_core = df.filter(df["market_id"] == "KXPOL-26DEC-BILL").sort("outcome_side")
    assert kalshi_core["outcome_side"].to_list() == ["no", "yes"]
    assert kalshi_core["bid_depth"].to_list() == [20, 20]
    assert kalshi_core["ask_depth"].to_list() == [20, 20]
    no_row = kalshi_core.row(0, named=True)
    yes_row = kalshi_core.row(1, named=True)
    assert yes_row["best_bid"] == Decimal("0.420000")
    assert no_row["best_ask"] == Decimal("0.580000")
    assert yes_row["best_bid"] + no_row["best_ask"] == Decimal("1.000000")


# ---------------------------------------------------------------------------
# 2. Custom max_depth=15, reversed ("No", "Yes") outcomes & validation guards
# ---------------------------------------------------------------------------


async def test_collect_daily_l2_custom_max_depth_reversed_outcomes_and_validation(
    tmp_path: Path,
) -> None:
    """max_depth=15 propagates, reversed ('No','Yes') maps sides cleanly, bad inputs fail fast."""
    seen_kalshi_depth: list[str] = []

    def pm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/markets":
            return httpx.Response(
                200,
                json=[
                    {
                        "conditionId": "0xpm_reversed",
                        "question": "Reversed outcomes test",
                        "outcomes": '["No", "Yes"]',
                        "clobTokenIds": '["tok_no_first", "tok_yes_second"]',
                        "endDate": "2026-12-01T00:00:00Z",
                        "active": True,
                        "closed": False,
                        "enableOrderBook": True,
                        "acceptingOrders": True,
                        "volume24hr": 12000,
                    }
                ],
                request=request,
            )
        if request.url.path == "/books":
            return httpx.Response(
                200,
                json=[
                    {
                        "asset_id": "tok_no_first",
                        "bids": _scrambled_clob_levels(10, 22),
                        "asks": _scrambled_clob_levels(55, 22),
                    },
                    {
                        "asset_id": "tok_yes_second",
                        "bids": _scrambled_clob_levels(15, 22),
                        "asks": _scrambled_clob_levels(60, 22),
                    },
                ],
                request=request,
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    def kalshi_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/trade-api/v2/markets":
            return httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXCPI-26DEC-T30",
                            "series_ticker": "KXCPI",
                            "title": "CPI > 3.0%?",
                            "status": "open",
                            "market_type": "binary",
                            "close_time": "2026-12-15T13:30:00Z",
                            "volume_24h_fp": "2000.00",
                        }
                    ],
                    "cursor": "",
                },
                request=request,
            )
        if request.url.path == "/trade-api/v2/markets/orderbooks":
            seen_kalshi_depth.append(request.url.params["depth"])
            return httpx.Response(
                200,
                json={
                    "orderbooks": [
                        {
                            "ticker": "KXCPI-26DEC-T30",
                            "orderbook_fp": {
                                "yes_dollars": _ascending_kalshi_levels(10, 25),
                                "no_dollars": _ascending_kalshi_levels(15, 25),
                            },
                        }
                    ]
                },
                request=request,
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    kalshi_client = _make_kalshi_client(kalshi_handler)
    pm_client = _make_polymarket_client(pm_handler)

    try:
        # Fail-fast validation guards before any network calls.
        with pytest.raises(ValueError, match="max_depth"):
            await collect_daily_l2(
                output_dir=tmp_path,
                max_depth=14,
                kalshi_client=kalshi_client,
                polymarket_client=pm_client,
            )
        with pytest.raises(ValueError, match="max_depth"):
            await collect_daily_l2(
                output_dir=tmp_path,
                max_depth=21,
                kalshi_client=kalshi_client,
                polymarket_client=pm_client,
            )
        with pytest.raises(TypeError, match="datetime.date"):
            await collect_daily_l2(
                output_dir=tmp_path,
                snapshot_date=FIXED_TS,
                kalshi_client=kalshi_client,
                polymarket_client=pm_client,
            )

        summary = await collect_daily_l2(
            output_dir=tmp_path,
            policy=UniversePolicy(target_series_prefixes=()),
            max_depth=15,
            snapshot_date=date(2026, 11, 24),
            kalshi_client=kalshi_client,
            polymarket_client=pm_client,
            now_fn=lambda: FIXED_TS,
        )
    finally:
        await kalshi_client._http.aclose()
        await pm_client._http.aclose()

    assert seen_kalshi_depth == ["15"]
    assert summary.snapshot_date == date(2026, 11, 24)
    assert summary.parquet_path == tmp_path / "2026-11-24.parquet"

    df = read_snapshots_duckdb(summary.parquet_path)
    assert max(df["bid_depth"].to_list()) == 15
    assert max(df["ask_depth"].to_list()) == 15

    pm_rows = df.filter(df["venue"] == "polymarket").sort("outcome_side")
    assert pm_rows["outcome_side"].to_list() == ["no", "yes"]
    assert pm_rows["outcome_label"].to_list() == ["No", "Yes"]
    assert pm_rows["token_id"].to_list() == ["tok_no_first", "tok_yes_second"]


# ---------------------------------------------------------------------------
# 3. Resilience: single-venue persistent HTTP 503 isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("failing_venue", ["kalshi", "polymarket"])
async def test_collect_daily_l2_single_venue_persistent_503_isolation(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    failing_venue: str,
) -> None:
    """Persistent 503 on one venue is isolated into failed_venues while the other completes."""

    def healthy_pm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/markets":
            return httpx.Response(
                200,
                json=[
                    {
                        "conditionId": "0xpm_healthy",
                        "question": "Healthy Polymarket binary?",
                        "outcomes": '["Yes", "No"]',
                        "clobTokenIds": '["tok_h_y", "tok_h_n"]',
                        "endDate": "2026-12-10T00:00:00Z",
                        "active": True,
                        "closed": False,
                        "enableOrderBook": True,
                        "acceptingOrders": True,
                        "volume24hr": 20000,
                    }
                ],
                request=request,
            )
        if request.url.path == "/books":
            return httpx.Response(
                200,
                json=[
                    {
                        "asset_id": "tok_h_y",
                        "bids": [{"price": "0.52", "size": "100"}],
                        "asks": [{"price": "0.54", "size": "100"}],
                    },
                    {
                        "asset_id": "tok_h_n",
                        "bids": [{"price": "0.46", "size": "100"}],
                        "asks": [{"price": "0.48", "size": "100"}],
                    },
                ],
                request=request,
            )
        raise AssertionError(f"Unexpected PM request: {request.url}")

    def healthy_kalshi_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/trade-api/v2/markets":
            return httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXHEALTHY-26DEC",
                            "title": "Healthy Kalshi binary?",
                            "status": "open",
                            "market_type": "binary",
                            "close_time": "2026-12-10T00:00:00Z",
                            "volume_24h_fp": "4000.00",
                        }
                    ],
                    "cursor": "",
                },
                request=request,
            )
        if request.url.path == "/trade-api/v2/markets/orderbooks":
            return httpx.Response(
                200,
                json={
                    "orderbooks": [
                        {
                            "ticker": "KXHEALTHY-26DEC",
                            "orderbook_fp": {
                                "yes_dollars": [["0.4800", "50"]],
                                "no_dollars": [["0.5000", "60"]],
                            },
                        }
                    ]
                },
                request=request,
            )
        raise AssertionError(f"Unexpected Kalshi request: {request.url}")

    def always_503_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable", request=request)

    kalshi_client = _make_kalshi_client(
        always_503_handler if failing_venue == "kalshi" else healthy_kalshi_handler
    )
    pm_client = _make_polymarket_client(
        always_503_handler if failing_venue == "polymarket" else healthy_pm_handler
    )

    try:
        with caplog.at_level(logging.WARNING, logger="pm_quant.ingest.collector"):
            summary = await collect_daily_l2(
                output_dir=tmp_path,
                policy=UniversePolicy(target_series_prefixes=()),
                kalshi_client=kalshi_client,
                polymarket_client=pm_client,
                now_fn=lambda: FIXED_TS,
            )
    finally:
        await kalshi_client._http.aclose()
        await pm_client._http.aclose()

    healthy_venue = "polymarket" if failing_venue == "kalshi" else "kalshi"
    assert summary.failed_venues == (failing_venue,)
    assert summary.total_rows == 2
    assert summary.rows_by_venue[failing_venue] == 0
    assert summary.rows_by_venue[healthy_venue] == 2
    assert summary.selected_by_reason == {"target_series": 0, "volume_core": 1}
    assert any(failing_venue in rec.message for rec in caplog.records)

    df = read_snapshots_duckdb(summary.parquet_path)
    assert df.height == 2
    assert set(df["venue"].to_list()) == {healthy_venue}


async def test_collect_daily_l2_orderbooks_503_and_target_series_404_isolation(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Orderbook-stage 503 isolates Kalshi; single target-tag 404 warns without dropping venue."""

    def pm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/markets":
            if request.url.params.get("tag_id") == "999":
                return httpx.Response(404, text="Not Found", request=request)
            return httpx.Response(
                200,
                json=[
                    {
                        "conditionId": "0xpm_ok",
                        "question": "Polymarket ok?",
                        "outcomes": '["Yes", "No"]',
                        "clobTokenIds": '["tok_ok_y", "tok_ok_n"]',
                        "endDate": "2026-12-10T00:00:00Z",
                        "active": True,
                        "closed": False,
                        "enableOrderBook": True,
                        "acceptingOrders": True,
                        "volume24hr": 15000,
                    }
                ],
                request=request,
            )
        if request.url.path == "/books":
            return httpx.Response(
                200,
                json=[
                    {
                        "asset_id": "tok_ok_y",
                        "bids": [{"price": "0.50", "size": "10"}],
                        "asks": [{"price": "0.52", "size": "10"}],
                    },
                    {
                        "asset_id": "tok_ok_n",
                        "bids": [{"price": "0.48", "size": "10"}],
                        "asks": [{"price": "0.50", "size": "10"}],
                    },
                ],
                request=request,
            )
        raise AssertionError(f"Unexpected PM request: {request.url}")

    def kalshi_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/trade-api/v2/markets":
            return httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXBOOKFAIL-1",
                            "title": "Kalshi book fails with 503",
                            "status": "open",
                            "market_type": "binary",
                            "close_time": "2026-12-10T00:00:00Z",
                            "volume_24h_fp": "3000.00",
                        }
                    ],
                    "cursor": "",
                },
                request=request,
            )
        if request.url.path == "/trade-api/v2/markets/orderbooks":
            return httpx.Response(503, text="Orderbook endpoint down", request=request)
        raise AssertionError(f"Unexpected Kalshi request: {request.url}")

    kalshi_client = _make_kalshi_client(kalshi_handler)
    pm_client = _make_polymarket_client(pm_handler)

    try:
        with caplog.at_level(logging.WARNING, logger="pm_quant.ingest.collector"):
            summary = await collect_daily_l2(
                output_dir=tmp_path,
                policy=UniversePolicy(
                    target_series_prefixes=(),
                    polymarket_target_tag_ids=("999",),
                ),
                kalshi_client=kalshi_client,
                polymarket_client=pm_client,
                now_fn=lambda: FIXED_TS,
            )
    finally:
        await kalshi_client._http.aclose()
        await pm_client._http.aclose()

    assert summary.failed_venues == ("kalshi",)
    assert summary.rows_by_venue == {"polymarket": 2, "kalshi": 0}
    assert summary.selected_by_reason == {"target_series": 0, "volume_core": 1}
    assert any("tag_id=999" in rec.message for rec in caplog.records)
    assert any("orderbook batch fetch failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# 4. Resilience: both venues empty or failing -> RuntimeError (fail-fast CI)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["both_503", "both_empty", "mixed_503_and_empty"])
async def test_collect_daily_l2_zero_rows_raises_runtime_error(
    tmp_path: Path,
    mode: str,
) -> None:
    """When both venues fail or return zero usable snapshots, raise RuntimeError without writing."""

    def always_503(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Down", request=request)

    def empty_pm(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[], request=request)

    def empty_kalshi(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"markets": [], "cursor": ""}, request=request)

    if mode == "both_503":
        k_handler, p_handler = always_503, always_503
    elif mode == "both_empty":
        k_handler, p_handler = empty_kalshi, empty_pm
    else:
        k_handler, p_handler = always_503, empty_pm

    kalshi_client = _make_kalshi_client(k_handler)
    pm_client = _make_polymarket_client(p_handler)

    try:
        with pytest.raises(RuntimeError, match="0 rows"):
            await collect_daily_l2(
                output_dir=tmp_path,
                policy=UniversePolicy(target_series_prefixes=()),
                kalshi_client=kalshi_client,
                polymarket_client=pm_client,
                now_fn=lambda: FIXED_TS,
            )
    finally:
        await kalshi_client._http.aclose()
        await pm_client._http.aclose()

    assert list(tmp_path.glob("*.parquet")) == []


# ---------------------------------------------------------------------------
# 5. CLI scripts/collect_daily.py (args, summary table, dry-run, fail-on-partial)
# ---------------------------------------------------------------------------


def test_cli_collect_daily_flags_summary_table_and_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI parses flags, overrides policy budgets, prints compact summary, and sets exit codes."""
    cli = _load_cli_module()
    captured_calls: list[dict[str, Any]] = []

    parquet_file = tmp_path / "2026-11-23.parquet"
    parquet_file.write_bytes(b"PAR1" + b"0" * 2044)

    def _make_summary(*, failed_venues: tuple[str, ...] = ()) -> CollectionSummary:
        return CollectionSummary(
            snapshot_date=FIXED_DATE,
            ts_utc=FIXED_TS,
            parquet_path=parquet_file,
            file_size_bytes=parquet_file.stat().st_size,
            total_rows=10,
            rows_by_venue={"polymarket": 6, "kalshi": 4 if not failed_venues else 0},
            selected_by_reason={"target_series": 2, "volume_core": 3},
            empty_books_count=1,
            one_sided_books_count=2,
            failed_venues=failed_venues,
        )

    async def fake_collect_ok(**kwargs: Any) -> CollectionSummary:
        captured_calls.append(kwargs)
        return _make_summary()

    monkeypatch.setattr(cli, "collect_daily_l2", fake_collect_ok)
    exit_code = cli.main(
        [
            "--output-dir",
            str(tmp_path),
            "--date",
            "2026-11-23",
            "--max-depth",
            "18",
            "--max-core-per-venue",
            "50",
            "--max-target-series-per-venue",
            "25",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert len(captured_calls) == 1
    assert captured_calls[0]["output_dir"] == tmp_path
    assert captured_calls[0]["snapshot_date"] == FIXED_DATE
    assert captured_calls[0]["max_depth"] == 18
    assert captured_calls[0]["policy"].max_core_per_venue == 50
    assert captured_calls[0]["policy"].max_target_series_per_venue == 25

    out = capsys.readouterr().out
    assert "Rows (polymarket)" in out
    assert "Rows (kalshi)" in out
    assert "Selected (target_series)" in out
    assert "Selected (volume_core)" in out
    assert "Empty books" in out
    assert "2.00 KB" in out
    assert str(parquet_file) in out
    assert "dry-run" in out

    # Partial venue failure: exit 0 without --fail-on-partial, exit != 0 with --fail-on-partial.
    async def fake_collect_partial(**kwargs: Any) -> CollectionSummary:
        return _make_summary(failed_venues=("kalshi",))

    monkeypatch.setattr(cli, "collect_daily_l2", fake_collect_partial)
    assert cli.main(["--output-dir", str(tmp_path), "--dry-run"]) == 0
    capsys.readouterr()

    partial_code = cli.main(["--output-dir", str(tmp_path), "--dry-run", "--fail-on-partial"])
    partial_io = capsys.readouterr()
    assert partial_code != 0
    assert "Failed venues" in partial_io.out
    assert "kalshi" in partial_io.out
    assert "--fail-on-partial" in partial_io.err

    # Total collection RuntimeError -> non-zero exit code with stderr message.
    async def fake_collect_fail(**kwargs: Any) -> CollectionSummary:
        raise RuntimeError("Daily L2 collection produced 0 rows across all venues")

    monkeypatch.setattr(cli, "collect_daily_l2", fake_collect_fail)
    fail_code = cli.main(["--output-dir", str(tmp_path)])
    fail_io = capsys.readouterr()
    assert fail_code != 0
    assert "ERROR:" in fail_io.err

    # Invalid CLI arguments are rejected by argparse.
    with pytest.raises(SystemExit):
        cli.main(["--max-depth", "14"])
    with pytest.raises(SystemExit):
        cli.main(["--max-depth", "21"])
    with pytest.raises(SystemExit):
        cli.main(["--date", "2026/11/23"])
    with pytest.raises(SystemExit):
        cli.main(["--max-core-per-venue", "-1"])


async def test_collect_daily_l2_target_series_override_and_omitted_books(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit kalshi_target_series, partial 404 series, and omitted batch books warn cleanly."""
    seen_series: list[str] = []

    def pm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/markets":
            return httpx.Response(
                200,
                json=[
                    {
                        "conditionId": "0xpm_partial",
                        "question": "Partial CLOB return?",
                        "outcomes": '["Yes", "No"]',
                        "clobTokenIds": '["tok_present", "tok_missing"]',
                        "endDate": "2026-12-10T00:00:00Z",
                        "active": True,
                        "closed": False,
                        "enableOrderBook": True,
                        "acceptingOrders": True,
                        "volume24hr": 10000,
                    }
                ],
                request=request,
            )
        if request.url.path == "/books":
            # Deliberately omit tok_missing from CLOB response.
            return httpx.Response(
                200,
                json=[
                    {
                        "asset_id": "tok_present",
                        "bids": [{"price": "0.55", "size": "20"}],
                        "asks": [{"price": "0.57", "size": "20"}],
                    }
                ],
                request=request,
            )
        raise AssertionError(f"Unexpected PM request: {request.url}")

    def kalshi_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/trade-api/v2/markets":
            series = request.url.params.get("series_ticker")
            if series is not None:
                seen_series.append(series)
                if series == "KXBAD":
                    return httpx.Response(404, text="Unknown series", request=request)
                if series == "KXHIGHNY":
                    return httpx.Response(
                        200,
                        json={
                            "markets": [
                                {
                                    "ticker": "KXHIGHNY-26NOV25-T50",
                                    "series_ticker": "KXHIGHNY",
                                    "title": "NYC high > 50F?",
                                    "status": "open",
                                    "market_type": "binary",
                                    "close_time": "2026-11-25T23:59:00Z",
                                    "volume_24h_fp": "1.00",
                                },
                                {
                                    "ticker": "KXHIGHNY-26NOV25-OMITTED",
                                    "series_ticker": "KXHIGHNY",
                                    "title": "NYC high > 60F?",
                                    "status": "open",
                                    "market_type": "binary",
                                    "close_time": "2026-11-25T23:59:00Z",
                                    "volume_24h_fp": "1.00",
                                },
                            ],
                            "cursor": "",
                        },
                        request=request,
                    )
            return httpx.Response(200, json={"markets": [], "cursor": ""}, request=request)
        if request.url.path == "/trade-api/v2/markets/orderbooks":
            # Deliberately omit KXHIGHNY-26NOV25-OMITTED.
            return httpx.Response(
                200,
                json={
                    "orderbooks": [
                        {
                            "ticker": "KXHIGHNY-26NOV25-T50",
                            "orderbook_fp": {
                                "yes_dollars": [["0.4500", "15"]],
                                "no_dollars": [["0.5300", "15"]],
                            },
                        }
                    ]
                },
                request=request,
            )
        raise AssertionError(f"Unexpected Kalshi request: {request.url}")

    kalshi_client = _make_kalshi_client(kalshi_handler)
    pm_client = _make_polymarket_client(pm_handler)

    try:
        with caplog.at_level(logging.WARNING, logger="pm_quant.ingest.collector"):
            summary = await collect_daily_l2(
                output_dir=tmp_path,
                policy=UniversePolicy(target_series_prefixes=("KXHIGHNY",)),
                kalshi_target_series=(" KXHIGHNY ", "KXHIGHNY", "KXBAD", ""),
                kalshi_client=kalshi_client,
                polymarket_client=pm_client,
                now_fn=lambda: FIXED_TS,
            )
    finally:
        await kalshi_client._http.aclose()
        await pm_client._http.aclose()

    assert seen_series == ["KXHIGHNY", "KXBAD"]
    assert summary.failed_venues == ()
    assert summary.rows_by_venue == {"polymarket": 1, "kalshi": 2}
    assert any("series_ticker=KXBAD" in rec.message for rec in caplog.records)
    assert any("omitted 1/2 selected tickers" in rec.message for rec in caplog.records)
    assert any("omitted 1/2 selected token_ids" in rec.message for rec in caplog.records)
