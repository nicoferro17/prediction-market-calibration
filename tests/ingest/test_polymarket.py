"""Offline unit tests for Polymarket pure parsers and asynchronous PolymarketClient."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import httpx
import pytest

from pm_quant.ingest import (
    DEFAULT_CLOB_BASE_URL,
    DEFAULT_DATA_API_V2_BASE_URL,
    DEFAULT_GAMMA_BASE_URL,
    POLYMARKET_V2_HISTORY_MAX_WINDOW_SECONDS,
    AsyncRateLimitedClient,
    PolymarketClient,
    PriceLevel,
    parse_clob_orderbook,
    parse_gamma_market,
    parse_v2_trade,
)

FIXED_TS = datetime(2026, 11, 22, 18, 0, 0, tzinfo=UTC)
DAY_SECONDS = 86_400


# ---------------------------------------------------------------------------
# 1. Gamma market parsing (JSON strings vs lists, is_yes_no, anti-leakage)
# ---------------------------------------------------------------------------


def test_parse_gamma_market_json_string_and_list_formats() -> None:
    """clobTokenIds and outcomes parse identically from JSON strings or Python lists."""
    raw_serialized: dict[str, Any] = {
        "id": "551234",
        "conditionId": "0xabc123def456",
        "question": "Will the Fed cut rates in December 2026?",
        "slug": "fed-rate-cut-dec-2026",
        "category": "Economics",
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": '["1111111111", "2222222222"]',
        "endDate": "2026-12-18T19:00:00Z",
        "active": True,
        "closed": False,
        "negRisk": True,
        "negRiskMarketID": "0xnegriskgroup",
        "enableOrderBook": True,
        "acceptingOrders": True,
        "feesEnabled": True,
        "feeType": "economics",
        "orderPriceMinTickSize": 0.001,
        "orderMinSize": 5,
        "events": [{"id": "evt-900", "slug": "fed-dec-2026"}],
        # Final lifetime volume & quote fields must NEVER leak into MarketMetadata.
        "volume": "4250000.50",
        "volumeNum": 4250000.50,
        "volumeClob": 4200000.0,
        "volume24hr": 120000.0,
        "liquidity": "350000.00",
        "outcomePrices": '["0.62", "0.38"]',
        "bestBid": 0.61,
        "bestAsk": 0.63,
        "lastTradePrice": 0.62,
    }
    raw_list: dict[str, Any] = {
        **raw_serialized,
        "outcomes": ["Yes", "No"],
        "clobTokenIds": ["1111111111", "2222222222"],
        "endDate": "2026-12-18T21:00:00+02:00",
    }

    meta_str = parse_gamma_market(raw_serialized)
    meta_list = parse_gamma_market(raw_list)

    for meta in (meta_str, meta_list):
        assert meta.venue == "polymarket"
        assert meta.market_id == "0xabc123def456"
        assert meta.event_id == "evt-900"
        assert meta.title == "Will the Fed cut rates in December 2026?"
        assert meta.category == "Economics"
        assert meta.outcomes == ("Yes", "No")
        assert meta.is_yes_no is True
        assert meta.end_time_utc == datetime(2026, 12, 18, 19, 0, 0, tzinfo=UTC)
        assert meta.status == "active"
        assert meta.extra["clob_token_ids"] == ("1111111111", "2222222222")
        assert meta.extra["neg_risk"] is True
        assert meta.extra["neg_risk_market_id"] == "0xnegriskgroup"
        assert meta.extra["enable_order_book"] is True
        assert meta.extra["tick_size"] == "0.001"
        assert meta.extra["min_order_size"] == "5"

        # Anti-leakage: Gamma lifetime volumes and snapshot prices must never be exposed.
        forbidden_keys = {
            "volume",
            "volumeNum",
            "volume_num",
            "volumeClob",
            "volume_clob",
            "volume24hr",
            "liquidity",
            "outcomePrices",
            "outcome_prices",
            "bestBid",
            "bestAsk",
            "lastTradePrice",
        }
        assert forbidden_keys.isdisjoint(meta.extra.keys())


@pytest.mark.parametrize(
    ("raw_outcomes", "expected_is_yes_no"),
    [
        (["Yes", "No"], True),
        ('["Yes", "No"]', True),
        ([" NO ", " yes "], True),
        (["YES", "NO"], True),
        (["Lakers", "Celtics"], False),
        ('["Lakers", "Celtics"]', False),
        (["Up", "Down"], False),
        ('["Up", "Down"]', False),
        (["Yes", "No", "Maybe"], False),
        (["Yes"], False),
        ([], False),
        ("not-valid-json", False),
        (None, False),
    ],
)
def test_parse_gamma_market_is_yes_no_indicator(
    raw_outcomes: Any,
    expected_is_yes_no: bool,
) -> None:
    """is_yes_no is True strictly for exact binary {'yes', 'no'} outcomes."""
    meta = parse_gamma_market(
        {
            "conditionId": "0xindicator",
            "question": "Indicator check",
            "outcomes": raw_outcomes,
            "clobTokenIds": ["10", "20"],
        }
    )
    assert meta.is_yes_no is expected_is_yes_no


def test_parse_gamma_market_edge_cases_and_errors() -> None:
    """Fallback title/event/category, status derivation, endDateIso, and missing conditionId."""
    archived_meta = parse_gamma_market(
        {
            "conditionId": "0xarchived",
            "slug": "archived-slug",
            "outcomes": '["Up", "Down"]',
            "clobTokenIds": "",
            "endDate": "0001-01-01T00:00:00Z",
            "endDateIso": "2026-10-01",
            "archived": True,
            "closed": True,
            "negRisk": "false",
            "events": [{"slug": "evt-slug", "category": "Crypto"}],
        }
    )
    assert archived_meta.title == "archived-slug"
    assert archived_meta.event_id == "evt-slug"
    assert archived_meta.category == "Crypto"
    assert archived_meta.status == "archived"
    assert archived_meta.end_time_utc == datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC)
    assert archived_meta.extra["clob_token_ids"] == ()
    assert archived_meta.extra["neg_risk"] is False

    closed_meta = parse_gamma_market({"conditionId": "0xclosed", "closed": True})
    assert closed_meta.status == "closed"
    assert closed_meta.end_time_utc is None

    unknown_meta = parse_gamma_market({"conditionId": "0xunknown", "endDate": "bad-date"})
    assert unknown_meta.status == "unknown"
    assert unknown_meta.end_time_utc is None

    explicit_meta = parse_gamma_market({"conditionId": "0xexplicit", "status": "RESOLVED"})
    assert explicit_meta.status == "resolved"

    with pytest.raises(ValueError, match="conditionId"):
        parse_gamma_market({"question": "Missing conditionId"})


# ---------------------------------------------------------------------------
# 2. CLOB orderbook parsing (unsorted inputs, Decimal precision, top-20 cap)
# ---------------------------------------------------------------------------


def test_parse_clob_orderbook_resorts_disordered_levels_and_caps_top_20() -> None:
    """Deliberately disordered bids/asks are explicitly re-sorted and capped to top 20."""
    # Construct 30 bid prices from 0.10 to 0.39 and shuffle them deterministically.
    bid_prices = [Decimal("0.10") + Decimal("0.01") * i for i in range(30)]
    bid_order = [5, 29, 0, 17, 24, 12, 28, 3, 21, 9, 25, 1, 19, 14, 27] + [
        i for i in range(30) if i not in {5, 29, 0, 17, 24, 12, 28, 3, 21, 9, 25, 1, 19, 14, 27}
    ]
    raw_bids = [{"price": f"{bid_prices[i]:.2f}", "size": str((i + 1) * 10)} for i in bid_order]

    # Construct 30 ask prices from 0.41 to 0.70 and reverse/shuffle them (worst first, like CLOB).
    ask_prices = [Decimal("0.41") + Decimal("0.01") * i for i in range(30)]
    ask_order = [29, 4, 18, 0, 11, 25, 2, 20, 1, 15, 28, 7, 13, 3, 22] + [
        i
        for i in range(29, -1, -1)
        if i not in {29, 4, 18, 0, 11, 25, 2, 20, 1, 15, 28, 7, 13, 3, 22}
    ]
    raw_asks = [{"price": f"{ask_prices[i]:.2f}", "size": str((i + 1) * 5)} for i in ask_order]

    book = parse_clob_orderbook(
        {"bids": raw_bids, "asks": raw_asks},
        token_id="99887766",
        ts_utc=FIXED_TS,
        max_depth=20,
    )

    assert book.venue == "polymarket"
    assert book.market_id == "99887766"
    assert book.outcome_side == "yes"
    assert book.ts_utc == FIXED_TS
    assert book.depth_levels == (20, 20)

    # Top 20 bids: descending from 0.39 down to 0.20.
    assert book.bids[0] == PriceLevel(Decimal("0.39"), Decimal("300"))
    assert book.bids[-1] == PriceLevel(Decimal("0.20"), Decimal("110"))
    assert all(book.bids[i].price > book.bids[i + 1].price for i in range(len(book.bids) - 1))

    # Top 20 asks: ascending from 0.41 up to 0.60.
    assert book.asks[0] == PriceLevel(Decimal("0.41"), Decimal("5"))
    assert book.asks[-1] == PriceLevel(Decimal("0.60"), Decimal("100"))
    assert all(book.asks[i].price < book.asks[i + 1].price for i in range(len(book.asks) - 1))

    # Exact Decimal derived properties.
    assert book.best_bid == Decimal("0.39")
    assert book.best_ask == Decimal("0.41")
    assert book.spread == Decimal("0.02")
    assert book.mid == Decimal("0.40")
    assert not book.is_crossed


def test_parse_clob_orderbook_filters_dirty_levels_and_edge_cases() -> None:
    """Zero/negative sizes, out-of-range prices, and malformed levels are filtered cleanly."""
    raw_book: dict[str, Any] = {
        "bids": [
            {"price": "0.45", "size": "0"},  # size == 0 -> filtered
            {"price": "0.44", "size": "-12.5"},  # size < 0 -> filtered
            {"price": "0.00", "size": "100"},  # price == 0 -> filtered
            {"price": "1.00", "size": "50"},  # price == 1 -> filtered
            {"price": "1.20", "size": "10"},  # price > 1 -> filtered
            {"price": "NaN", "size": "10"},  # non-finite -> filtered
            {"price": "0.42", "size": "25.5"},  # valid
            ["0.46", "15"],  # valid 2-tuple/list format
            {"price": True, "size": 10},  # bool -> filtered
            "malformed-entry",  # invalid -> filtered
        ],
        "asks": [
            {"price": "0.55", "size": "0.0"},  # filtered
            {"price": "0.52", "size": "30"},  # valid
            {"price": "0.49", "size": "12.25"},  # valid (better ask, out of order)
            {"price": "bad", "size": "10"},  # filtered
        ],
    }

    book = parse_clob_orderbook(raw_book, token_id="tok-dirty", ts_utc=FIXED_TS, outcome_side="no")
    assert book.outcome_side == "no"
    assert book.bids == (
        PriceLevel(Decimal("0.46"), Decimal("15")),
        PriceLevel(Decimal("0.42"), Decimal("25.5")),
    )
    assert book.asks == (
        PriceLevel(Decimal("0.49"), Decimal("12.25")),
        PriceLevel(Decimal("0.52"), Decimal("30")),
    )

    empty = parse_clob_orderbook(None, token_id="tok-empty", ts_utc=FIXED_TS)
    assert empty.bids == ()
    assert empty.asks == ()
    assert empty.best_bid is None
    assert empty.best_ask is None

    with pytest.raises(ValueError, match="token_id"):
        parse_clob_orderbook({}, token_id="  ", ts_utc=FIXED_TS)
    with pytest.raises(ValueError, match="max_depth"):
        parse_clob_orderbook({}, token_id="tok-1", ts_utc=FIXED_TS, max_depth=0)
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_clob_orderbook({}, token_id="tok-1", ts_utc=datetime(2026, 11, 22, 18, 0))


# ---------------------------------------------------------------------------
# 3. Data API v2 trade parser
# ---------------------------------------------------------------------------


def test_parse_v2_trade_variants_and_validation() -> None:
    """Snake_case v2 trade payloads parse into exact Decimal TradeRecords."""
    trade = parse_v2_trade(
        {
            "proxy_wallet": "0x53805554",
            "side": "SELL",
            "token_id": "5448579069",
            "condition_id": "0x2093b156",
            "size": 10.5,
            "price": 0.62,
            "timestamp": 1763834400,
            "transaction_hash": "0xa39fe4d9",
        }
    )
    assert trade.venue == "polymarket"
    assert trade.market_id == "0x2093b156"
    assert trade.token_id == "5448579069"
    assert trade.price == Decimal("0.62")
    assert trade.size == Decimal("10.5")
    assert trade.notional_usd == Decimal("6.510")
    assert trade.ts_utc == datetime.fromtimestamp(1763834400, tz=UTC)
    assert trade.side == "sell"
    assert trade.tx_hash == "0xa39fe4d9"

    # Explicit usdc_size, millisecond epoch timestamp, BUY side, asset fallback.
    trade_ms = parse_v2_trade(
        {
            "condition_id": "0xms",
            "asset": 123456,
            "price": "0.50",
            "size": "20",
            "usdc_size": "10.00",
            "timestamp": 1763834400000,
            "side": "buy",
            "tx_hash": "0xtx2",
        }
    )
    assert trade_ms.token_id == "123456"
    assert trade_ms.notional_usd == Decimal("10.00")
    assert trade_ms.ts_utc == datetime.fromtimestamp(1763834400, tz=UTC)
    assert trade_ms.side == "buy"
    assert trade_ms.tx_hash == "0xtx2"

    # ISO-8601 string timestamp and unknown side -> side=None.
    trade_iso = parse_v2_trade(
        {
            "condition_id": "0xiso",
            "price": "0.25",
            "size": "4",
            "timestamp": "2026-11-22T18:00:00Z",
            "side": "UNKNOWN",
        }
    )
    assert trade_iso.ts_utc == FIXED_TS
    assert trade_iso.side is None

    with pytest.raises(ValueError, match="condition_id"):
        parse_v2_trade({"price": 0.5, "size": 1, "timestamp": 1700000000})
    with pytest.raises(ValueError, match="price"):
        parse_v2_trade({"condition_id": "0x1", "price": 1.5, "size": 1, "timestamp": 1700000000})
    with pytest.raises(ValueError, match="size"):
        parse_v2_trade({"condition_id": "0x1", "price": 0.5, "size": -2, "timestamp": 1700000000})
    with pytest.raises(ValueError, match="timestamp"):
        parse_v2_trade({"condition_id": "0x1", "price": 0.5, "size": 1, "timestamp": "bad-ts"})


# ---------------------------------------------------------------------------
# 4. Gamma API client pagination
# ---------------------------------------------------------------------------


async def test_gamma_get_markets_pagination_dedup_and_filters() -> None:
    """Gamma /markets paginates with limit/offset until short page and deduplicates."""
    seen_params: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith(f"{DEFAULT_GAMMA_BASE_URL}/markets")
        params = dict(request.url.params)
        seen_params.append(params)
        offset = int(params["offset"])

        if offset == 0:
            return httpx.Response(
                200,
                json=[
                    {
                        "conditionId": "0xm1",
                        "question": "Market 1",
                        "outcomes": '["Yes", "No"]',
                        "clobTokenIds": '["101", "102"]',
                        "active": True,
                        "closed": False,
                    },
                    {
                        "conditionId": "0xm2",
                        "question": "Market 2",
                        "outcomes": ["Lakers", "Celtics"],
                        "clobTokenIds": ["201", "202"],
                        "active": True,
                        "closed": False,
                    },
                ],
                request=request,
            )
        if offset == 2:
            # Short page (1 < limit=2) with a duplicate conditionId must terminate cleanly.
            return httpx.Response(
                200,
                json=[
                    {
                        "conditionId": "0xm2",
                        "question": "Market 2 duplicate",
                        "outcomes": ["Lakers", "Celtics"],
                        "clobTokenIds": ["201", "202"],
                    },
                ],
                request=request,
            )
        raise AssertionError(f"Unexpected offset: {offset}")

    async with PolymarketClient(transport=httpx.MockTransport(handler)) as client:
        markets = await client.get_markets(
            active=True,
            closed=False,
            archived=False,
            limit=2,
            order="endDate",
            ascending=True,
            tag_id=42,
            slug="sports",
        )
        assert await client.get_markets(max_pages=0) == []

    assert [m.market_id for m in markets] == ["0xm1", "0xm2"]
    assert markets[0].is_yes_no is True
    assert markets[1].is_yes_no is False
    assert len(seen_params) == 2
    assert seen_params[0]["offset"] == "0"
    assert seen_params[0]["limit"] == "2"
    assert seen_params[0]["active"] == "true"
    assert seen_params[0]["closed"] == "false"
    assert seen_params[0]["archived"] == "false"
    assert seen_params[0]["order"] == "endDate"
    assert seen_params[0]["ascending"] == "true"
    assert seen_params[0]["tag_id"] == "42"
    assert seen_params[0]["slug"] == "sports"
    assert seen_params[1]["offset"] == "2"


# ---------------------------------------------------------------------------
# 5. CLOB API client methods (GET /book, POST /books, GET /prices-history)
# ---------------------------------------------------------------------------


async def test_clob_get_orderbook_batch_and_price_history() -> None:
    """CLOB /book, POST /books batching, and /prices-history work end-to-end offline."""
    post_bodies: list[Any] = []
    seen_history_params: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/book":
            assert request.method == "GET"
            assert request.url.params.get("token_id") == "tok-single"
            return httpx.Response(
                200,
                json={
                    "asset_id": "tok-single",
                    "bids": [{"price": "0.30", "size": "10"}, {"price": "0.45", "size": "25"}],
                    "asks": [{"price": "0.60", "size": "15"}, {"price": "0.48", "size": "40"}],
                },
                request=request,
            )

        if request.url.path == "/books":
            assert request.method == "POST"
            body = json.loads(request.content.decode())
            post_bodies.append(body)
            # Omit asset_id on the second item of a 2-item batch to test positional fallback.
            items: list[dict[str, Any]] = []
            for idx, entry in enumerate(body):
                item: dict[str, Any] = {
                    "bids": [{"price": "0.40", "size": "50"}],
                    "asks": [{"price": "0.42", "size": "30"}],
                }
                if idx == 0:
                    item["asset_id"] = entry["token_id"]
                items.append(item)
            return httpx.Response(200, json=items, request=request)

        if request.url.path == "/prices-history":
            assert request.method == "GET"
            seen_history_params.append(dict(request.url.params))
            return httpx.Response(
                200,
                json={
                    "history": [
                        {"t": 1700003600, "p": 0.55},
                        {"t": 1700000000, "p": 0.50},
                        {"t": 1700003600, "p": 0.55},  # duplicate
                    ]
                },
                request=request,
            )

        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async with PolymarketClient(
        transport=httpx.MockTransport(handler),
        now_fn=lambda: FIXED_TS,
    ) as client:
        single = await client.get_orderbook("tok-single")
        assert single.best_bid == Decimal("0.45")
        assert single.best_ask == Decimal("0.48")
        assert single.ts_utc == FIXED_TS

        assert await client.get_orderbooks_batch(["", "   "]) == {}
        batch = await client.get_orderbooks_batch(
            ["tok-1", "tok-1", "tok-2", "tok-3"],
            max_depth=10,
            max_batch_size=2,
        )
        assert sorted(batch.keys()) == ["tok-1", "tok-2", "tok-3"]
        assert len(post_bodies) == 2
        assert sorted(len(b) for b in post_bodies) == [1, 2]
        assert post_bodies[0] == [{"token_id": "tok-1"}, {"token_id": "tok-2"}]
        assert post_bodies[1] == [{"token_id": "tok-3"}]

        start_dt = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
        end_dt = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)
        hist = await client.get_clob_price_history(
            "tok-hist",
            start_ts=start_dt,
            end_ts=end_dt,
            fidelity=60,
        )
        assert hist == [
            {"t": 1700000000, "p": 0.50},
            {"t": 1700003600, "p": 0.55},
        ]
        assert seen_history_params[0] == {
            "market": "tok-hist",
            "startTs": str(int(start_dt.timestamp())),
            "endTs": str(int(end_dt.timestamp())),
            "fidelity": "60",
        }

        # Interval preset mode (without start_ts / end_ts).
        await client.get_clob_price_history("tok-hist", interval="1d")
        assert seen_history_params[1] == {"market": "tok-hist", "interval": "1d"}

        with pytest.raises(ValueError, match="interval"):
            await client.get_clob_price_history("tok-hist", start_ts=100, end_ts=200, interval="1d")
        with pytest.raises(ValueError, match="start_ts"):
            await client.get_clob_price_history("tok-hist", start_ts=300, end_ts=200)
        with pytest.raises(ValueError, match="fidelity"):
            await client.get_clob_price_history("tok-hist", fidelity=0)


# ---------------------------------------------------------------------------
# 6. Data API v2 /v2/trades (45 IDs -> 20+20+5, condition=, taker_only=true,
#    NO market=/offset=, 2-page next_cursor pagination)
# ---------------------------------------------------------------------------


async def test_data_v2_trades_chunks_45_conditions_and_paginates_next_cursor() -> None:
    """45 condition_ids split into 3 chunks (20+20+5), enforce v2 params, and page via cursor."""
    chunk_requests: list[tuple[list[str], dict[str, str], str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith(f"{DEFAULT_DATA_API_V2_BASE_URL}/trades")
        params = dict(request.url.params)
        raw_query = request.url.query.decode()

        # Mandatory Data API v2 invariants in every single request.
        assert "condition" in params
        assert params.get("taker_only") == "true"
        assert "market" not in params
        assert "offset" not in params
        assert "market=" not in raw_query
        assert "offset=" not in raw_query

        conditions = [c for c in params["condition"].split(",") if c]
        assert 1 <= len(conditions) <= 20
        chunk_requests.append((conditions, params, raw_query))

        cursor = params.get("cursor")
        first_cond = conditions[0]

        if cursor is None:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "condition_id": first_cond,
                            "token_id": f"tok-{first_cond}-p1",
                            "price": 0.60,
                            "size": 100,
                            "timestamp": 1763830000,
                            "side": "BUY",
                            "transaction_hash": f"0xtx-{first_cond}-1",
                        }
                    ],
                    "pagination": {
                        "has_more": True,
                        "next_cursor": f"cur-{first_cond}-2",
                    },
                },
                request=request,
            )

        assert cursor == f"cur-{first_cond}-2"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "condition_id": first_cond,
                        "token_id": f"tok-{first_cond}-p2",
                        "price": 0.40,
                        "size": 50,
                        "timestamp": 1763829000,
                        "side": "SELL",
                        "transaction_hash": f"0xtx-{first_cond}-2",
                    }
                ],
                "pagination": {
                    "has_more": False,
                    "next_cursor": None,
                },
            },
            request=request,
        )

    condition_ids_45 = [f"0xcond{i:02d}" for i in range(45)]
    async with PolymarketClient(transport=httpx.MockTransport(handler)) as client:
        trades = await client.get_trades(condition_ids_45)

    # 3 chunks (20 + 20 + 5) * 2 pages per chunk = 6 requests total.
    assert len(chunk_requests) == 6
    page1_chunk_sizes = [
        len(conds) for conds, params, _ in chunk_requests if "cursor" not in params
    ]
    page2_chunk_sizes = [len(conds) for conds, params, _ in chunk_requests if "cursor" in params]
    assert sorted(page1_chunk_sizes, reverse=True) == [20, 20, 5]
    assert sorted(page2_chunk_sizes, reverse=True) == [20, 20, 5]

    # Every chunk returned 2 trades (1 per page) -> 6 TradeRecords total.
    assert len(trades) == 6
    assert [t.tx_hash for t in trades] == [
        "0xtx-0xcond00-1",
        "0xtx-0xcond00-2",
        "0xtx-0xcond20-1",
        "0xtx-0xcond20-2",
        "0xtx-0xcond40-1",
        "0xtx-0xcond40-2",
    ]
    assert trades[0].notional_usd == Decimal("60.00")
    assert trades[1].notional_usd == Decimal("20.00")


async def test_data_v2_trades_max_pages_null_data_and_validation() -> None:
    """max_pages caps pagination, data=null is safe, and empty/invalid inputs behave cleanly."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "data": None,
                "pagination": {"has_more": True, "next_cursor": f"page-{calls + 1}"},
            },
            request=request,
        )

    async with PolymarketClient(transport=httpx.MockTransport(handler)) as client:
        assert await client.get_trades(["", "  "]) == []
        assert await client.get_trades(["0x1"], max_pages=0) == []
        assert calls == 0

        res = await client.get_trades(["0x1"], max_pages=1)
        assert res == []
        assert calls == 1

        with pytest.raises(ValueError, match="max_pages"):
            await client.get_trades(["0x1"], max_pages=-1)
        with pytest.raises(ValueError, match="limit"):
            await client.get_trades(["0x1"], limit=0)
        with pytest.raises(ValueError, match="limit"):
            await client.get_trades(["0x1"], limit=1001)


# ---------------------------------------------------------------------------
# 7. Data API v2 /v2/prices-history (35-day window -> 3 blocks <= 15d,
#    ordered deduplication, empty-block resolution decay warning)
# ---------------------------------------------------------------------------


async def test_data_v2_price_history_35_day_window_splits_dedups_and_warns_on_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """35-day window issues exact 3 requests (<=15d each), deduplicates, and warns if empty."""
    t0 = 1_760_000_000
    t_end = t0 + 35 * DAY_SECONDS
    seen_windows: list[tuple[int, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith(f"{DEFAULT_DATA_API_V2_BASE_URL}/prices-history")
        params = dict(request.url.params)
        assert params["token_id"] == "tok-35d"
        assert params.get("bucket_seconds") == "3600"
        win_start = int(params["start"])
        win_end = int(params["end"])
        assert 0 < win_end - win_start <= POLYMARKET_V2_HISTORY_MAX_WINDOW_SECONDS
        seen_windows.append((win_start, win_end))

        # Block 1 (days 0..15): empty to simulate resolution decay and trigger a warning.
        if win_start == t0:
            return httpx.Response(200, json={"data": []}, request=request)

        # Block 2 (days 15..30): unsorted points + boundary timestamp shared with block 3.
        if win_start == t0 + 15 * DAY_SECONDS:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "timestamp": t0 + 20 * DAY_SECONDS,
                            "price": 0.54,
                            "resolution_seconds": 3600,
                        },
                        {
                            "timestamp": t0 + 16 * DAY_SECONDS,
                            "price": 0.51,
                            "resolution_seconds": 3600,
                        },
                        {
                            "timestamp": t0 + 30 * DAY_SECONDS,
                            "price": 0.60,
                            "resolution_seconds": 3600,
                        },
                    ]
                },
                request=request,
            )

        # Block 3 (days 30..35): duplicate boundary timestamp + newer point.
        if win_start == t0 + 30 * DAY_SECONDS:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "timestamp": t0 + 30 * DAY_SECONDS,
                            "price": 0.60,
                            "resolution_seconds": 3600,
                        },
                        {
                            "timestamp": t0 + 34 * DAY_SECONDS,
                            "price": 0.67,
                            "resolution_seconds": 3600,
                        },
                    ]
                },
                request=request,
            )

        raise AssertionError(f"Unexpected window: {win_start}..{win_end}")

    with caplog.at_level(logging.WARNING, logger="pm_quant.ingest.polymarket"):
        async with PolymarketClient(transport=httpx.MockTransport(handler)) as client:
            points = await client.get_v2_price_history(
                "tok-35d",
                start_ts=t0,
                end_ts=t_end,
                bucket_seconds=3600,
            )

    # Exact 3 requests covering 15d + 15d + 5d, each <= 15 days.
    assert len(seen_windows) == 3
    sorted_windows = sorted(seen_windows)
    assert sorted_windows == [
        (t0, t0 + 15 * DAY_SECONDS),
        (t0 + 15 * DAY_SECONDS, t0 + 30 * DAY_SECONDS),
        (t0 + 30 * DAY_SECONDS, t0 + 35 * DAY_SECONDS),
    ]
    assert all(
        end - start <= POLYMARKET_V2_HISTORY_MAX_WINDOW_SECONDS for start, end in seen_windows
    )

    # Deduplicated and strictly sorted ascending by timestamp.
    timestamps = [p["timestamp"] for p in points]
    assert timestamps == [
        t0 + 16 * DAY_SECONDS,
        t0 + 20 * DAY_SECONDS,
        t0 + 30 * DAY_SECONDS,
        t0 + 34 * DAY_SECONDS,
    ]
    assert [p["price"] for p in points] == [0.51, 0.54, 0.60, 0.67]

    # Warning emitted for the empty block (resolution decay).
    warnings = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "empty block" in warnings[0]
    assert "tok-35d" in warnings[0]


async def test_data_v2_price_history_interval_mode_cursor_and_validation() -> None:
    """Relative interval mode, cursor pagination, datetime bounds, and input validation."""
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        if params.get("cursor") is None and params.get("interval") == "1d":
            return httpx.Response(
                200,
                json={
                    "data": [{"timestamp": 200, "price": 0.52, "resolution_seconds": 60}],
                    "pagination": {"has_more": True, "next_cursor": "hist-p2"},
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "data": [{"timestamp": 100, "price": 0.48, "resolution_seconds": 60}],
                "pagination": {"has_more": False, "next_cursor": None},
            },
            request=request,
        )

    async with PolymarketClient(transport=httpx.MockTransport(handler)) as client:
        pts = await client.get_v2_price_history("tok-int", interval="1d")
        assert [p["timestamp"] for p in pts] == [100, 200]
        assert calls[0] == {"token_id": "tok-int", "interval": "1d"}
        assert calls[1] == {"token_id": "tok-int", "interval": "1d", "cursor": "hist-p2"}

        # Equal start_ts == end_ts with timezone-aware datetimes makes 1 request.
        tz_dt = FIXED_TS.astimezone(timezone(timedelta(hours=2)))
        eq_pts = await client.get_v2_price_history("tok-eq", start_ts=tz_dt, end_ts=FIXED_TS)
        assert len(eq_pts) == 1

        with pytest.raises(ValueError, match="token_id"):
            await client.get_v2_price_history("   ", interval="1d")
        with pytest.raises(ValueError, match="both be provided"):
            await client.get_v2_price_history("tok-int", start_ts=100)
        with pytest.raises(ValueError, match="start_ts"):
            await client.get_v2_price_history("tok-int", start_ts=200, end_ts=100)
        with pytest.raises(ValueError, match="bucket_seconds"):
            await client.get_v2_price_history("tok-int", interval="1d", bucket_seconds=0)


# ---------------------------------------------------------------------------
# 8. Client lifecycle, external http_client reuse, and malformed payloads
# ---------------------------------------------------------------------------


async def test_client_external_http_client_and_malformed_payloads() -> None:
    """Externally supplied AsyncRateLimitedClient is preserved; malformed responses raise."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/markets":
            return httpx.Response(200, json={"unexpected": "object"}, request=request)
        if request.url.path == "/book":
            return httpx.Response(200, json=["unexpected-list"], request=request)
        if request.url.path == "/books":
            return httpx.Response(200, json={"unexpected": "object"}, request=request)
        if request.url.path == "/prices-history":
            return httpx.Response(200, json={"history": "not-a-list"}, request=request)
        if request.url.path == "/v2/trades":
            return httpx.Response(200, json={"data": "not-a-list"}, request=request)
        if request.url.path == "/v2/prices-history":
            return httpx.Response(200, json=["not-an-object"], request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    with pytest.raises(ValueError, match="gamma_base_url"):
        PolymarketClient(gamma_base_url="   ")
    with pytest.raises(ValueError, match="clob_base_url"):
        PolymarketClient(clob_base_url="")
    with pytest.raises(ValueError, match="data_api_base_url"):
        PolymarketClient(data_api_base_url="  ")

    async with AsyncRateLimitedClient(transport=httpx.MockTransport(handler)) as shared_http:
        async with PolymarketClient(http_client=shared_http) as client:
            assert client.gamma_base_url == DEFAULT_GAMMA_BASE_URL
            assert client.clob_base_url == DEFAULT_CLOB_BASE_URL
            assert client.data_api_base_url == DEFAULT_DATA_API_V2_BASE_URL

            with pytest.raises(ValueError, match="expected JSON list from Gamma"):
                await client.get_markets()
            with pytest.raises(ValueError, match="expected JSON object from CLOB /book"):
                await client.get_orderbook("tok-1")
            with pytest.raises(ValueError, match="expected JSON list from CLOB POST /books"):
                await client.get_orderbooks_batch(["tok-1"])
            with pytest.raises(ValueError, match="expected 'history' list"):
                await client.get_clob_price_history("tok-1")
            with pytest.raises(ValueError, match="expected 'data' list"):
                await client.get_trades(["0x1"])
            with pytest.raises(ValueError, match="expected JSON object from Data API v2"):
                await client.get_v2_price_history("tok-1", interval="1d")

        # External HTTP client must remain usable after PolymarketClient closes.
        await client.aclose()
        assert await shared_http.get_json("https://example.test/health") == {"ok": True}
