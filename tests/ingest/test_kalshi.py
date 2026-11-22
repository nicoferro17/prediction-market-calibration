"""Offline unit tests for Kalshi pure parsers and asynchronous KalshiClient."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import httpx
import pytest

from pm_quant.ingest import (
    AsyncRateLimitedClient,
    KalshiClient,
    PriceLevel,
    parse_kalshi_market,
    parse_kalshi_orderbook,
    parse_kalshi_timestamp,
)

FIXED_TS = datetime(2026, 11, 22, 18, 0, 0, tzinfo=UTC)
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


# ---------------------------------------------------------------------------
# 1. Bid-only ascending orderbook parsing & 2. YES/NO symmetry identities
# ---------------------------------------------------------------------------


def test_parse_orderbook_ascending_bid_only_and_reciprocal_asks() -> None:
    """Best bid is the last element of ascending arrays; asks equal 1 - opposite bid."""
    raw_payload: dict[str, Any] = {
        "ticker": "KXFED-26DEC-T425",
        "orderbook_fp": {
            "yes_dollars": [
                ["0.3000", "50"],
                ["0.4200", "120"],
                ["0.4500", "80"],
            ],
            "no_dollars": [
                ["0.4000", "60"],
                ["0.5100", "90"],
                ["0.5300", "40"],
            ],
        },
    }

    yes_book, no_book = parse_kalshi_orderbook(
        raw_payload,
        ticker="KXFED-26DEC-T425",
        ts_utc=FIXED_TS,
    )

    assert yes_book.venue == "kalshi"
    assert yes_book.market_id == "KXFED-26DEC-T425"
    assert yes_book.outcome_side == "yes"
    assert no_book.outcome_side == "no"

    # Best YES bid is the LAST element of ascending yes_dollars, not the first.
    assert yes_book.best_bid == Decimal("0.4500")
    # Best YES ask is exact Decimal("1") - Decimal("0.5300") = Decimal("0.4700").
    assert yes_book.best_ask == Decimal("0.4700")
    assert yes_book.mid == Decimal("0.4600")
    assert yes_book.spread == Decimal("0.0200")
    assert not yes_book.is_crossed

    # YES bids strictly descending, YES asks strictly ascending.
    assert yes_book.bids == (
        PriceLevel(Decimal("0.4500"), Decimal("80")),
        PriceLevel(Decimal("0.4200"), Decimal("120")),
        PriceLevel(Decimal("0.3000"), Decimal("50")),
    )
    assert yes_book.asks == (
        PriceLevel(Decimal("0.4700"), Decimal("40")),
        PriceLevel(Decimal("0.4900"), Decimal("90")),
        PriceLevel(Decimal("0.6000"), Decimal("60")),
    )
    assert all(
        yes_book.bids[i].price > yes_book.bids[i + 1].price for i in range(len(yes_book.bids) - 1)
    )
    assert all(
        yes_book.asks[i].price < yes_book.asks[i + 1].price for i in range(len(yes_book.asks) - 1)
    )


def test_parse_orderbook_yes_no_symmetry_identities() -> None:
    """YES and NO books satisfy exact Decimal reciprocal identities."""
    raw_book: dict[str, Any] = {
        "yes_dollars": [["0.3000", "50"], ["0.4200", "120"], ["0.4500", "80"]],
        "no_dollars": [["0.4000", "60"], ["0.5100", "90"], ["0.5300", "40"]],
    }

    yes_book, no_book = parse_kalshi_orderbook(
        raw_book,
        ticker="KXINFL-26NOV-T30",
        ts_utc=FIXED_TS,
    )

    assert yes_book.best_bid is not None
    assert yes_book.best_ask is not None
    assert no_book.best_bid is not None
    assert no_book.best_ask is not None
    assert yes_book.mid is not None
    assert no_book.mid is not None

    assert yes_book.best_bid + no_book.best_ask == Decimal("1")
    assert yes_book.best_ask + no_book.best_bid == Decimal("1")
    assert yes_book.spread == no_book.spread
    assert yes_book.mid + no_book.mid == Decimal("1")

    # Level-by-level mirror check: every NO ask mirrors the corresponding YES bid.
    assert len(no_book.asks) == len(yes_book.bids)
    for y_bid, n_ask in zip(yes_book.bids, no_book.asks, strict=True):
        assert y_bid.price + n_ask.price == Decimal("1")
        assert y_bid.size == n_ask.size

    assert len(yes_book.asks) == len(no_book.bids)
    for n_bid, y_ask in zip(no_book.bids, yes_book.asks, strict=True):
        assert n_bid.price + y_ask.price == Decimal("1")
        assert n_bid.size == y_ask.size


# ---------------------------------------------------------------------------
# 3. Depth capping (max_depth=20 retains the TOP 20 / last 20 ascending)
# ---------------------------------------------------------------------------


def test_parse_orderbook_max_depth_keeps_top_20_best_levels() -> None:
    """With 35 ascending levels per side, max_depth=20 retains the last 20 (best) levels."""
    # Ascending prices: 0.1000, 0.1100, ..., 0.4400 (35 levels)
    yes_levels = [
        [f"{Decimal('0.1000') + Decimal('0.0100') * i:.4f}", str(i + 1)] for i in range(35)
    ]
    # Ascending prices: 0.2000, 0.2100, ..., 0.5400 (35 levels)
    no_levels = [
        [f"{Decimal('0.2000') + Decimal('0.0100') * i:.4f}", str(i + 101)] for i in range(35)
    ]

    yes_book, no_book = parse_kalshi_orderbook(
        {"orderbook_fp": {"yes_dollars": yes_levels, "no_dollars": no_levels}},
        ticker="KXDEPTH-TEST",
        ts_utc=FIXED_TS,
        max_depth=20,
    )

    assert yes_book.depth_levels == (20, 20)
    assert no_book.depth_levels == (20, 20)

    # Top 20 YES bids are indices 34 down to 15: 0.4400 .. 0.2500 (NOT 0.1000 .. 0.2900).
    assert yes_book.bids[0] == PriceLevel(Decimal("0.4400"), Decimal("35"))
    assert yes_book.bids[-1] == PriceLevel(Decimal("0.2500"), Decimal("16"))

    # Top 20 NO bids are indices 34 down to 15: 0.5400 .. 0.3500.
    assert no_book.bids[0] == PriceLevel(Decimal("0.5400"), Decimal("135"))
    assert no_book.bids[-1] == PriceLevel(Decimal("0.3500"), Decimal("116"))

    # Top 20 YES asks derived from top 20 NO bids: 0.4600 .. 0.6500 (ascending).
    assert yes_book.asks[0] == PriceLevel(Decimal("0.4600"), Decimal("135"))
    assert yes_book.asks[-1] == PriceLevel(Decimal("0.6500"), Decimal("116"))


# ---------------------------------------------------------------------------
# 4. Robust edge cases (empty, None/missing keys, one-sided, dirty levels)
# ---------------------------------------------------------------------------


def test_parse_orderbook_empty_book() -> None:
    """Empty yes_dollars and no_dollars yield None for best_bid, best_ask, mid, and spread."""
    yes_book, no_book = parse_kalshi_orderbook(
        {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}},
        ticker="KXEMPTY-1",
        ts_utc=FIXED_TS,
    )

    for book in (yes_book, no_book):
        assert book.bids == ()
        assert book.asks == ()
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid is None
        assert book.spread is None
        assert not book.is_crossed


@pytest.mark.parametrize(
    "raw_book",
    [
        None,
        {},
        {"orderbook_fp": None},
        {"orderbook_fp": {}},
        {"yes_dollars": None, "no_dollars": None},
        {"orderbook_fp": {"yes_dollars": None}},
        {"yes_dollars": "not-a-list", "no_dollars": 123},
    ],
)
def test_parse_orderbook_none_or_missing_keys_no_crash(raw_book: dict[str, Any] | None) -> None:
    """Missing keys, None payloads, or malformed containers safely produce empty books."""
    yes_book, no_book = parse_kalshi_orderbook(
        raw_book,
        ticker="KXNULL-1",
        ts_utc=FIXED_TS,
    )

    assert yes_book.bids == ()
    assert yes_book.asks == ()
    assert no_book.bids == ()
    assert no_book.asks == ()
    assert yes_book.best_bid is None
    assert yes_book.best_ask is None


def test_parse_orderbook_one_sided_yes_only() -> None:
    """One-sided book populates yes_book.best_bid and leaves ask/spread/mid as None."""
    yes_book, no_book = parse_kalshi_orderbook(
        {"yes_dollars": [["0.3500", "25"], ["0.4000", "10"]], "no_dollars": []},
        ticker="KXONESIDE-1",
        ts_utc=FIXED_TS,
    )

    assert yes_book.best_bid == Decimal("0.4000")
    assert yes_book.best_ask is None
    assert yes_book.spread is None
    assert yes_book.mid is None

    # Symmetrically, NO book has no bids, but has derived asks from YES bids.
    assert no_book.best_bid is None
    assert no_book.best_ask == Decimal("0.6000")
    assert no_book.spread is None
    assert no_book.mid is None


def test_parse_orderbook_filters_dirty_levels() -> None:
    """Zero/negative sizes, boundary prices (0.0000, 1.0000), and invalid entries are discarded."""
    raw_book: dict[str, Any] = {
        "yes_dollars": [
            ["0.0000", "100"],  # price == 0 -> filtered
            ["-0.1000", "50"],  # price < 0 -> filtered
            ["0.2500", "0.00"],  # size == 0 -> filtered
            ["0.3000", "-5"],  # size < 0 -> filtered
            ["0.4000", "15.5"],  # valid
            ["1.0000", "200"],  # price == 1 -> filtered
            ["1.0500", "10"],  # price > 1 -> filtered
            ["NaN", "10"],  # non-finite -> filtered
            ["0.4200", ""],  # empty size -> filtered
            ["bad", "10"],  # invalid decimal -> filtered
            ["only-one-element"],  # malformed entry -> filtered
            [True, 10],  # boolean -> filtered
            ["0.4500", 20],  # numeric int size -> valid
        ],
        "no_dollars": [
            ["0.0000", "10"],
            ["0.5000", "0.00"],
            ["0.5200", "30"],
            ["1.0000", "10"],
        ],
    }

    yes_book, no_book = parse_kalshi_orderbook(
        raw_book,
        ticker="KXDIRTY-1",
        ts_utc=FIXED_TS,
    )

    assert yes_book.bids == (
        PriceLevel(Decimal("0.4500"), Decimal("20")),
        PriceLevel(Decimal("0.4000"), Decimal("15.5")),
    )
    assert yes_book.asks == (PriceLevel(Decimal("0.4800"), Decimal("30")),)
    assert no_book.bids == (PriceLevel(Decimal("0.5200"), Decimal("30")),)
    assert no_book.asks == (
        PriceLevel(Decimal("0.5500"), Decimal("20")),
        PriceLevel(Decimal("0.6000"), Decimal("15.5")),
    )


def test_parse_orderbook_input_validation() -> None:
    """Invalid ticker, max_depth < 1, or naive ts_utc raise ValueError."""
    with pytest.raises(ValueError, match="ticker"):
        parse_kalshi_orderbook({}, ticker="   ", ts_utc=FIXED_TS)
    with pytest.raises(ValueError, match="max_depth"):
        parse_kalshi_orderbook({}, ticker="KX-1", ts_utc=FIXED_TS, max_depth=0)
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_kalshi_orderbook({}, ticker="KX-1", ts_utc=datetime(2026, 11, 22, 18, 0, 0))


# ---------------------------------------------------------------------------
# Timestamp & Market metadata pure parsers
# ---------------------------------------------------------------------------


def test_parse_kalshi_timestamp_variants_and_errors() -> None:
    """ISO-8601 Z, offset-aware, and naive strings normalize to UTC; invalid inputs raise."""
    assert parse_kalshi_timestamp("2026-09-22T00:00:00Z") == datetime(2026, 9, 22, 0, 0, tzinfo=UTC)
    assert parse_kalshi_timestamp("2026-09-22T02:00:00+02:00") == datetime(
        2026, 9, 22, 0, 0, tzinfo=UTC
    )
    assert parse_kalshi_timestamp("2026-09-22T00:00:00") == datetime(2026, 9, 22, 0, 0, tzinfo=UTC)

    with pytest.raises(ValueError, match="ISO-8601 string"):
        parse_kalshi_timestamp(12345)
    with pytest.raises(ValueError, match="non-empty"):
        parse_kalshi_timestamp("   ")
    with pytest.raises(ValueError, match="invalid ISO-8601"):
        parse_kalshi_timestamp("not-a-timestamp")


def test_parse_kalshi_market_fields_and_end_time_no_settlement_fallback() -> None:
    """end_time_utc never falls back to settlement_ts and ignores Go zero-value timestamps."""
    settled_only_raw: dict[str, Any] = {
        "ticker": "KXHIST-25JAN-A",
        "event_ticker": "KXHIST-25JAN",
        "title": "Historical settled contract",
        "status": "FINALIZED",
        "market_type": "binary",
        "close_time": "0001-01-01T00:00:00Z",  # Go zero-value -> ignored
        "expected_expiration_time": "",
        "settlement_ts": "2025-01-15T12:00:00Z",
        "result": "yes",
    }

    meta = parse_kalshi_market(settled_only_raw)
    assert meta.venue == "kalshi"
    assert meta.market_id == "KXHIST-25JAN-A"
    assert meta.event_id == "KXHIST-25JAN"
    assert meta.category == "KXHIST"
    assert meta.status == "finalized"
    assert meta.is_yes_no is True
    assert meta.outcomes == ("Yes", "No")
    assert meta.end_time_utc is None
    assert meta.extra["settlement_ts"] == "2025-01-15T12:00:00Z"
    assert meta.extra["result"] == "yes"

    # Scalar market with close_time fallback to expected_expiration_time
    scalar_meta = parse_kalshi_market(
        {
            "ticker": "KXSCALAR-1",
            "series_ticker": "KXSCALAR",
            "subtitle": "Fallback subtitle",
            "market_type": "scalar",
            "expected_expiration_time": "2026-12-31T23:59:00Z",
        }
    )
    assert scalar_meta.title == "Fallback subtitle"
    assert scalar_meta.category == "KXSCALAR"
    assert scalar_meta.is_yes_no is False
    assert scalar_meta.outcomes == ()
    assert scalar_meta.status == "unknown"
    assert scalar_meta.end_time_utc == datetime(2026, 12, 31, 23, 59, tzinfo=UTC)

    with pytest.raises(ValueError, match="ticker"):
        parse_kalshi_market({"title": "Missing ticker"})


# ---------------------------------------------------------------------------
# 5. Historical cutoff caching, is_archived, and /markets vs /historical/markets
# ---------------------------------------------------------------------------


async def test_historical_cutoff_cached_and_is_archived_boundaries() -> None:
    """/historical/cutoff is fetched once across repeated/concurrent calls; boundary is strict <."""
    cutoff_calls = 0
    cutoff_str = "2026-09-22T00:00:00Z"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal cutoff_calls
        assert request.url.path == "/trade-api/v2/historical/cutoff"
        cutoff_calls += 1
        return httpx.Response(
            200,
            json={
                "market_settled_ts": cutoff_str,
                "trades_created_ts": cutoff_str,
                "orders_updated_ts": cutoff_str,
            },
            request=request,
        )

    transport = httpx.MockTransport(handler)
    async with KalshiClient(transport=transport) as client:
        assert client.cached_historical_cutoff is None

        # Repeated and concurrent calls must hit the endpoint only once.
        first, second, third = await asyncio.gather(
            client.get_historical_cutoff(),
            client.get_historical_cutoff(),
            client.get_historical_cutoff(),
        )
        fourth = await client.get_historical_cutoff()

        expected_cutoff = datetime(2026, 9, 22, 0, 0, 0, tzinfo=UTC)
        assert first == second == third == fourth == expected_cutoff
        assert client.cached_historical_cutoff == expected_cutoff
        assert cutoff_calls == 1

        # Boundary checks: strictly before -> True; equal -> False; after -> False.
        before = expected_cutoff - timedelta(seconds=1)
        equal = expected_cutoff
        after = expected_cutoff + timedelta(seconds=1)
        # Also test non-UTC timezone-aware equivalent of `before` (UTC+2).
        before_tz = before.astimezone(timezone(timedelta(hours=2)))

        assert await client.is_archived(before) is True
        assert await client.is_archived(before_tz) is True
        assert await client.is_archived(equal) is False
        assert await client.is_archived(after) is False
        assert cutoff_calls == 1

        # Naive datetime must raise ValueError.
        with pytest.raises(ValueError, match="timezone-aware"):
            await client.is_archived(datetime(2026, 9, 21, 23, 59, 59))

        # force_refresh=True triggers a second fetch.
        cutoff_str = "2026-10-01T00:00:00Z"
        refreshed = await client.get_historical_cutoff(force_refresh=True)
        assert refreshed == datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC)
        assert cutoff_calls == 2


async def test_get_markets_routing_and_cursor_pagination() -> None:
    """archived=False hits /markets with status; archived=True hits /historical/markets."""
    seen_requests: list[tuple[str, dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        seen_requests.append((request.url.path, params))
        cursor = params.get("cursor")

        if request.url.path == "/trade-api/v2/markets":
            if cursor is None:
                return httpx.Response(
                    200,
                    json={
                        "markets": [
                            {"ticker": "KXLIVE-1", "title": "Live 1", "status": "open"},
                            {"ticker": "KXLIVE-2", "title": "Live 2", "status": "open"},
                        ],
                        "cursor": "live-page-2",
                    },
                    request=request,
                )
            if cursor == "live-page-2":
                return httpx.Response(
                    200,
                    json={
                        "markets": [
                            # Duplicate ticker across pages must be deduplicated.
                            {"ticker": "KXLIVE-2", "title": "Live 2 dup", "status": "open"},
                            {"ticker": "KXLIVE-3", "title": "Live 3", "status": "open"},
                        ],
                        "cursor": "",
                    },
                    request=request,
                )

        if request.url.path == "/trade-api/v2/historical/markets":
            if cursor is None:
                return httpx.Response(
                    200,
                    json={
                        "markets": [
                            {"ticker": "KXARCH-1", "title": "Archived 1", "status": "finalized"},
                        ],
                        "cursor": "arch-page-2",
                    },
                    request=request,
                )
            if cursor == "arch-page-2":
                return httpx.Response(
                    200,
                    json={
                        "markets": [
                            {"ticker": "KXARCH-2", "title": "Archived 2", "status": "finalized"},
                        ],
                        "cursor": None,
                    },
                    request=request,
                )

        raise AssertionError(f"Unexpected request: {request.url}")

    transport = httpx.MockTransport(handler)
    async with KalshiClient(transport=transport) as client:
        live = await client.get_markets(
            status="open",
            archived=False,
            limit=2,
            series_ticker="KXLIVE",
            event_ticker="KXLIVE-EVT",
            mve_filter="exclude",
        )
        archived = await client.get_markets(status="settled", archived=True, limit=1)

    assert [m.market_id for m in live] == ["KXLIVE-1", "KXLIVE-2", "KXLIVE-3"]
    assert [m.market_id for m in archived] == ["KXARCH-1", "KXARCH-2"]

    assert seen_requests[0][0] == "/trade-api/v2/markets"
    assert seen_requests[0][1]["status"] == "open"
    assert seen_requests[0][1]["limit"] == "2"
    assert seen_requests[0][1]["series_ticker"] == "KXLIVE"
    assert seen_requests[0][1]["event_ticker"] == "KXLIVE-EVT"
    assert seen_requests[0][1]["mve_filter"] == "exclude"
    assert "cursor" not in seen_requests[0][1]

    assert seen_requests[1][0] == "/trade-api/v2/markets"
    assert seen_requests[1][1]["cursor"] == "live-page-2"

    assert seen_requests[2][0] == "/trade-api/v2/historical/markets"
    assert "status" not in seen_requests[2][1]
    assert seen_requests[2][1]["limit"] == "1"

    assert seen_requests[3][0] == "/trade-api/v2/historical/markets"
    assert "status" not in seen_requests[3][1]
    assert seen_requests[3][1]["cursor"] == "arch-page-2"


async def test_get_markets_max_pages_and_validation() -> None:
    """max_pages caps pagination; max_pages=0 makes zero calls; invalid bounds raise."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"markets": [{"ticker": f"KX-{calls}"}], "cursor": f"next-{calls}"},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    async with KalshiClient(transport=transport) as client:
        assert await client.get_markets(max_pages=0) == []
        assert calls == 0

        limited = await client.get_markets(max_pages=1)
        assert [m.market_id for m in limited] == ["KX-1"]
        assert calls == 1

        with pytest.raises(ValueError, match="limit"):
            await client.get_markets(limit=0)
        with pytest.raises(ValueError, match="limit"):
            await client.get_markets(limit=1001)
        with pytest.raises(ValueError, match="max_pages"):
            await client.get_markets(max_pages=-1)
        with pytest.raises(ValueError, match="mutually exclusive"):
            await client.get_markets(
                archived=True,
                series_ticker="KXFED",
                event_ticker="KXFED-26DEC",
            )
        with pytest.raises(ValueError, match="mutually exclusive"):
            await client.get_markets(
                archived=True,
                series_ticker="KXFED",
                mve_filter="exclude",
            )
        with pytest.raises(ValueError, match="mve_filter='exclude'"):
            await client.get_markets(archived=True, mve_filter="only")

        # Single valid filter on /historical/markets succeeds.
        filtered_archived = await client.get_markets(
            archived=True,
            mve_filter="exclude",
            max_pages=1,
        )
        assert [m.market_id for m in filtered_archived] == ["KX-2"]


# ---------------------------------------------------------------------------
# 6. Batch chunking for get_orderbooks_batch (235 tickers -> 100, 100, 35)
# ---------------------------------------------------------------------------


async def test_get_orderbooks_batch_chunks_235_tickers_with_depth_20() -> None:
    """235 tickers split into exactly 3 requests of sizes 100, 100, and 35 with depth=20."""
    chunk_sizes: list[int] = []
    seen_depths: list[str | None] = []
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        tickers = request.url.params.get_list("tickers")
        chunk_sizes.append(len(tickers))
        seen_depths.append(request.url.params.get("depth"))

        orderbooks = [
            {
                "ticker": ticker,
                "orderbook_fp": {
                    "yes_dollars": [["0.4000", "10"], ["0.4500", "25"]],
                    "no_dollars": [["0.5000", "15"], ["0.5300", "30"]],
                },
            }
            for ticker in tickers
        ]
        return httpx.Response(200, json={"orderbooks": orderbooks}, request=request)

    tickers_235 = [f"KXBATCH-{i:03d}" for i in range(235)]
    transport = httpx.MockTransport(handler)
    async with KalshiClient(transport=transport, now_fn=lambda: FIXED_TS) as client:
        books = await client.get_orderbooks_batch(tickers_235, depth=20)

    assert len(chunk_sizes) == 3
    assert sorted(chunk_sizes, reverse=True) == [100, 100, 35]
    assert seen_depths == ["20", "20", "20"]
    assert seen_paths == ["/trade-api/v2/markets/orderbooks"] * 3

    assert len(books) == 235
    sample_yes, sample_no = books["KXBATCH-000"]
    assert sample_yes.best_bid == Decimal("0.4500")
    assert sample_yes.best_ask == Decimal("0.4700")
    assert sample_no.best_bid == Decimal("0.5300")
    assert sample_no.best_ask == Decimal("0.5500")
    assert sample_yes.ts_utc == FIXED_TS


async def test_get_orderbooks_batch_empty_dedup_positional_fallback_and_validation() -> None:
    """Empty tickers make zero calls; duplicates deduplicate; missing ticker falls back."""
    requests_made = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests_made
        requests_made += 1
        tickers = request.url.params.get_list("tickers")
        assert tickers == ["KX-A", "KX-B"]
        assert request.url.params.get("depth") == "5"
        # Omit explicit "ticker" field to exercise positional alignment fallback.
        return httpx.Response(
            200,
            json={
                "orderbooks": [
                    {"orderbook_fp": {"yes_dollars": [["0.4000", "10"]], "no_dollars": []}},
                    {"orderbook_fp": {"yes_dollars": [], "no_dollars": [["0.6000", "5"]]}},
                ]
            },
            request=request,
        )

    transport = httpx.MockTransport(handler)
    async with KalshiClient(transport=transport) as client:
        assert await client.get_orderbooks_batch(["", "   "]) == {}
        assert requests_made == 0

        res = await client.get_orderbooks_batch(
            ["KX-A", "  KX-A ", "KX-B", ""],
            depth=5,
            ts_utc=FIXED_TS,
        )
        assert requests_made == 1
        assert res["KX-A"][0].best_bid == Decimal("0.4000")
        assert res["KX-B"][1].best_bid == Decimal("0.6000")

        with pytest.raises(ValueError, match="depth"):
            await client.get_orderbooks_batch(["KX-A"], depth=0)
        with pytest.raises(ValueError, match="timezone-aware"):
            await client.get_orderbooks_batch(["KX-A"], ts_utc=datetime(2026, 11, 22, 18, 0))


# ---------------------------------------------------------------------------
# Candlesticks (live batch + historical) & client lifecycle / error handling
# ---------------------------------------------------------------------------


async def test_get_candlesticks_batch_chunks_and_params() -> None:
    """Live candlesticks batch sends comma-separated market_tickers in chunks of <= 100."""
    seen_chunks: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/trade-api/v2/markets/candlesticks"
        raw_tickers = request.url.params.get("market_tickers", "")
        chunk = raw_tickers.split(",") if raw_tickers else []
        seen_chunks.append(chunk)
        assert request.url.params.get("start_ts") == "1700000000"
        assert request.url.params.get("end_ts") == "1700003600"
        assert request.url.params.get("period_interval") == "60"
        assert request.url.params.get("include_latest_before_start") == "true"
        return httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "market_ticker": t,
                        "candlesticks": [{"end_period_ts": 1700003600, "volume": 12}],
                    }
                    for t in chunk
                ]
            },
            request=request,
        )

    tickers = [f"KXCAN-{i:03d}" for i in range(105)]
    transport = httpx.MockTransport(handler)
    async with KalshiClient(transport=transport) as client:
        assert await client.get_candlesticks_batch([], 1700000000, 1700003600) == {}
        candles = await client.get_candlesticks_batch(
            tickers,
            start_ts=1700000000,
            end_ts=1700003600,
            period_interval=60,
            include_latest_before_start=True,
        )

        with pytest.raises(ValueError, match="start_ts"):
            await client.get_candlesticks_batch(["KX-1"], start_ts=200, end_ts=100)
        with pytest.raises(ValueError, match="period_interval"):
            await client.get_candlesticks_batch(["KX-1"], 100, 200, period_interval=0)
        with pytest.raises(ValueError, match="max_batch_size"):
            await client.get_candlesticks_batch(["KX-1"], 100, 200, max_batch_size=101)

    assert sorted(len(c) for c in seen_chunks) == [5, 100]
    assert len(candles) == 105
    assert candles["KXCAN-000"] == [{"end_period_ts": 1700003600, "volume": 12}]


async def test_get_historical_candlesticks_and_validation() -> None:
    """Archived candlesticks hit /historical/markets/{ticker}/candlesticks with valid intervals."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.raw_path.startswith(
            b"/trade-api/v2/historical/markets/KXHIST%2F1/candlesticks"
        )
        assert request.url.params.get("period_interval") == "1440"
        return httpx.Response(
            200,
            json={"candlesticks": [{"end_period_ts": 1700086400, "volume": 50}]},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    async with KalshiClient(transport=transport) as client:
        rows = await client.get_historical_candlesticks(
            "KXHIST/1",
            start_ts=1700000000,
            end_ts=1700086400,
            period_interval=1440,
        )
        assert rows == [{"end_period_ts": 1700086400, "volume": 50}]

        with pytest.raises(ValueError, match="ticker"):
            await client.get_historical_candlesticks("  ", 100, 200)
        with pytest.raises(ValueError, match="start_ts"):
            await client.get_historical_candlesticks("KX-1", 300, 200)
        with pytest.raises(ValueError, match="period_interval"):
            await client.get_historical_candlesticks("KX-1", 100, 200, period_interval=15)


async def test_client_external_http_client_and_malformed_payloads() -> None:
    """Externally supplied AsyncRateLimitedClient is reused; malformed API payloads raise."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/historical/cutoff"):
            return httpx.Response(200, json=["unexpected-list"], request=request)
        if request.url.path.endswith("/markets"):
            return httpx.Response(200, json={"markets": "not-a-list"}, request=request)
        if request.url.path.endswith("/markets/orderbooks"):
            return httpx.Response(200, json={"orderbooks": "not-a-list"}, request=request)
        return httpx.Response(200, json={"candlesticks": None}, request=request)

    with pytest.raises(ValueError, match="base_url"):
        KalshiClient(base_url="   ")

    async with AsyncRateLimitedClient(transport=httpx.MockTransport(handler)) as shared_http:
        async with KalshiClient(base_url=BASE_URL, http_client=shared_http) as client:
            assert client.base_url == BASE_URL
            with pytest.raises(ValueError, match="expected JSON object"):
                await client.get_historical_cutoff()
            with pytest.raises(ValueError, match="expected 'markets' list"):
                await client.get_markets()
            with pytest.raises(ValueError, match="expected 'orderbooks' list"):
                await client.get_orderbooks_batch(["KX-1"])
            assert await client.get_historical_candlesticks("KX-1", 100, 200, 60) == []
        # External client must still be open after KalshiClient exits.
        await client.aclose()
        assert await shared_http.get_json(f"{BASE_URL}/historical/markets/KX-1/candlesticks") == {
            "candlesticks": None
        }
