"""Unit tests for normalized ingestion schemas and AsyncRateLimitedClient."""

from __future__ import annotations

import asyncio
import random
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import httpx
import pytest

from pm_quant.ingest import (
    SCHEMA_VERSION,
    AsyncRateLimitedClient,
    MarketMetadata,
    NormalizedOrderBook,
    PriceLevel,
    TradeRecord,
    parse_retry_after,
)

# ---------------------------------------------------------------------------
# AsyncRateLimitedClient tests
# ---------------------------------------------------------------------------


async def test_retry_429_respects_numeric_retry_after() -> None:
    """HTTP 429 with numeric Retry-After sleeps for that exact delay and recovers."""
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "2.5"}, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    transport = httpx.MockTransport(handler)
    async with AsyncRateLimitedClient(
        transport=transport,
        max_retries=3,
        sleep=fake_sleep,
        rng=random.Random(42),
    ) as client:
        payload = await client.get_json("https://example.test/markets")

    assert payload == {"ok": True}
    assert attempts == 2
    assert sleeps == [pytest.approx(2.5)]


async def test_retry_429_respects_http_date_retry_after() -> None:
    """HTTP 429 with HTTP-date Retry-After computes delta seconds against current UTC time."""
    attempts = 0
    sleeps: list[float] = []
    fixed_now = datetime(2026, 11, 22, 18, 0, 0, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "Sun, 22 Nov 2026 18:00:07 GMT"},
                request=request,
            )
        return httpx.Response(200, json={"status": "recovered"}, request=request)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    transport = httpx.MockTransport(handler)
    async with AsyncRateLimitedClient(
        transport=transport,
        max_retries=2,
        sleep=fake_sleep,
        now_fn=lambda: fixed_now,
    ) as client:
        data = await client.get_json("https://example.test/trades")

    assert data == {"status": "recovered"}
    assert attempts == 2
    assert sleeps == [pytest.approx(7.0)]


async def test_retry_503_honors_retry_after_and_max_retry_after_cap() -> None:
    """HTTP 503 honors Retry-After, honoring max_retry_after cap or uncapped when None."""
    attempts = 0
    sleeps_capped: list[float] = []
    sleeps_uncapped: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts in (1, 3):
            return httpx.Response(503, headers={"Retry-After": "120"}, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    async def sleep_capped(delay: float) -> None:
        sleeps_capped.append(delay)

    async def sleep_uncapped(delay: float) -> None:
        sleeps_uncapped.append(delay)

    async with AsyncRateLimitedClient(
        transport=httpx.MockTransport(handler),
        max_retries=2,
        max_retry_after=15.0,
        sleep=sleep_capped,
    ) as client:
        assert await client.get_json("https://example.test/capped") == {"ok": True}

    async with AsyncRateLimitedClient(
        transport=httpx.MockTransport(handler),
        max_retries=2,
        max_retry_after=None,
        sleep=sleep_uncapped,
    ) as client:
        assert await client.get_json("https://example.test/uncapped") == {"ok": True}

    assert sleeps_capped == [pytest.approx(15.0)]
    assert sleeps_uncapped == [pytest.approx(120.0)]


async def test_retry_503_recovers_with_exponential_backoff_and_jitter() -> None:
    """Transient HTTP 503 responses retry with bounded exponential backoff and recover."""
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            return httpx.Response(503, text="service unavailable", request=request)
        return httpx.Response(200, json={"attempt": attempts}, request=request)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    transport = httpx.MockTransport(handler)
    async with AsyncRateLimitedClient(
        transport=transport,
        max_retries=3,
        backoff_base=1.0,
        backoff_max=10.0,
        sleep=fake_sleep,
        rng=random.Random(0),
    ) as client:
        data = await client.get_json("https://example.test/book")

    assert data == {"attempt": 3}
    assert attempts == 3
    assert len(sleeps) == 2
    # Equal jitter bounds: attempt 0 -> [0.5, 1.0], attempt 1 -> [1.0, 2.0]
    assert 0.5 <= sleeps[0] <= 1.0
    assert 1.0 <= sleeps[1] <= 2.0


async def test_retry_transport_error_recovers() -> None:
    """Transient httpx.TransportError (e.g. ConnectTimeout) retries and recovers."""
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectTimeout("timed out", request=request)
        return httpx.Response(200, json={"recovered": True}, request=request)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    transport = httpx.MockTransport(handler)
    async with AsyncRateLimitedClient(
        transport=transport,
        max_retries=2,
        backoff_base=0.5,
        sleep=fake_sleep,
        rng=random.Random(7),
    ) as client:
        data = await client.get_json("https://example.test/cutoff")

    assert data == {"recovered": True}
    assert attempts == 2
    assert len(sleeps) == 1
    assert 0.25 <= sleeps[0] <= 0.5


async def test_exhausted_transport_error_reraises() -> None:
    """Persistent httpx.TransportError re-raises once max_retries is exhausted."""
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectTimeout("still down", request=request)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async with AsyncRateLimitedClient(
        transport=httpx.MockTransport(handler),
        max_retries=2,
        backoff_base=0.5,
        sleep=fake_sleep,
        rng=random.Random(3),
    ) as client:
        with pytest.raises(httpx.ConnectTimeout, match="still down"):
            await client.get("https://example.test/offline")

    assert attempts == 3
    assert len(sleeps) == 2


@pytest.mark.parametrize("status_code", [400, 404, 410])
async def test_fail_fast_on_4xx_without_retry(status_code: int) -> None:
    """Non-429 4xx responses fail fast immediately without sleeping or retrying."""
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(status_code, json={"error": "client error"}, request=request)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    transport = httpx.MockTransport(handler)
    async with AsyncRateLimitedClient(
        transport=transport,
        max_retries=3,
        sleep=fake_sleep,
    ) as client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.get("https://example.test/bad-request")

    assert exc_info.value.response.status_code == status_code
    assert attempts == 1
    assert sleeps == []


async def test_exhausted_retries_raises_http_status_error() -> None:
    """Persistent HTTP 503 raises HTTPStatusError once max_retries is exhausted."""
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, request=request)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    transport = httpx.MockTransport(handler)
    async with AsyncRateLimitedClient(
        transport=transport,
        max_retries=2,
        sleep=fake_sleep,
        rng=random.Random(1),
    ) as client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.get("https://example.test/down")

    assert exc_info.value.response.status_code == 503
    assert attempts == 3
    assert len(sleeps) == 2


async def test_semaphore_limits_concurrency() -> None:
    """AsyncRateLimitedClient never exceeds max_concurrency in-flight requests."""
    in_flight = 0
    peak_in_flight = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, json={"url": str(request.url)}, request=request)

    transport = httpx.MockTransport(handler)
    async with AsyncRateLimitedClient(transport=transport, max_concurrency=3) as client:
        responses = await asyncio.gather(
            *(client.get(f"https://example.test/item/{idx}") for idx in range(10))
        )

    assert len(responses) == 10
    assert peak_in_flight == 3


def test_parse_retry_after_invalid_and_edge_cases() -> None:
    """parse_retry_after returns None for missing/invalid headers and clamps negatives to 0."""
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("not-a-date") is None
    assert parse_retry_after("-5") == 0.0
    assert parse_retry_after("0") == 0.0
    assert parse_retry_after("3.25") == pytest.approx(3.25)


# ---------------------------------------------------------------------------
# Normalized schema tests
# ---------------------------------------------------------------------------


def test_orderbook_normal_and_exact_decimal_spread() -> None:
    """Normal two-sided book exposes exact Decimal best_bid, best_ask, mid, and spread."""
    ts = datetime(2026, 11, 22, 18, 0, 0, tzinfo=UTC)
    book = NormalizedOrderBook(
        venue="polymarket",
        market_id="0xabc123",
        outcome_side="yes",
        ts_utc=ts,
        bids=(
            PriceLevel(price=Decimal("0.30"), size=Decimal("150.5")),
            PriceLevel(price=Decimal("0.29"), size=Decimal("500")),
        ),
        asks=(
            PriceLevel(price=Decimal("0.31"), size=Decimal("200")),
            PriceLevel(price=Decimal("0.33"), size=Decimal("100")),
            PriceLevel(price=Decimal("0.35"), size=Decimal("50")),
        ),
        token_id="998877",
    )

    assert book.schema_version == SCHEMA_VERSION == "v1"
    assert book.token_id == "998877"
    assert book.best_bid == Decimal("0.30")
    assert book.best_ask == Decimal("0.31")
    assert book.mid == Decimal("0.305")
    # Exact Decimal subtraction avoids binary float drift (0.31 - 0.30 != 0.010000000000000009)
    assert book.spread == Decimal("0.01")
    assert isinstance(book.spread, Decimal)
    assert book.is_crossed is False
    assert book.depth_levels == (2, 3)


def test_orderbook_empty_book_returns_none_safely() -> None:
    """Empty orderbooks return None for price-derived properties and False for is_crossed."""
    ts = datetime(2026, 11, 22, 18, 0, 0, tzinfo=UTC)
    book = NormalizedOrderBook(
        venue="kalshi",
        market_id="KXMVECROSSCATEGORY-EXAMPLE",
        outcome_side="yes",
        ts_utc=ts,
        bids=(),
        asks=(),
    )

    assert book.token_id is None
    assert book.best_bid is None
    assert book.best_ask is None
    assert book.mid is None
    assert book.spread is None
    assert book.is_crossed is False
    assert book.depth_levels == (0, 0)


def test_orderbook_one_sided_books_return_none_for_mid_and_spread() -> None:
    """One-sided books expose the available side and return None for missing/derived metrics."""
    ts = datetime(2026, 11, 22, 18, 0, 0, tzinfo=UTC)

    bid_only = NormalizedOrderBook(
        venue="kalshi",
        market_id="KX-BID-ONLY",
        outcome_side="yes",
        ts_utc=ts,
        bids=(
            PriceLevel(price=Decimal("0.42"), size=Decimal("10")),
            PriceLevel(price=Decimal("0.45"), size=Decimal("25")),
        ),
        asks=(),
    )
    assert bid_only.best_bid == Decimal("0.45")
    assert bid_only.best_ask is None
    assert bid_only.mid is None
    assert bid_only.spread is None
    assert bid_only.is_crossed is False
    assert bid_only.depth_levels == (2, 0)

    ask_only = NormalizedOrderBook(
        venue="polymarket",
        market_id="0xaskonly",
        outcome_side="no",
        ts_utc=ts,
        bids=(),
        asks=(
            PriceLevel(price=Decimal("0.60"), size=Decimal("100")),
            PriceLevel(price=Decimal("0.58"), size=Decimal("40")),
        ),
    )
    assert ask_only.best_bid is None
    assert ask_only.best_ask == Decimal("0.58")
    assert ask_only.mid is None
    assert ask_only.spread is None
    assert ask_only.is_crossed is False
    assert ask_only.depth_levels == (0, 2)


def test_orderbook_crossed_book_detection() -> None:
    """A book with best_bid > best_ask reports is_crossed=True and negative Decimal spread."""
    ts = datetime(2026, 11, 22, 18, 0, 0, tzinfo=UTC)
    book = NormalizedOrderBook(
        venue="polymarket",
        market_id="0xcrossed",
        outcome_side="yes",
        ts_utc=ts,
        bids=(PriceLevel(price=Decimal("0.55"), size=Decimal("50")),),
        asks=(PriceLevel(price=Decimal("0.52"), size=Decimal("50")),),
    )

    assert book.is_crossed is True
    assert book.spread == Decimal("-0.03")
    assert book.mid == Decimal("0.535")


def test_schemas_immutability_utc_normalization_and_validation() -> None:
    """Dataclasses are frozen/slotted, normalize aware timestamps to UTC, and validate bounds."""
    ts_utc = datetime(2026, 11, 22, 18, 0, 0, tzinfo=UTC)
    ts_plus2 = datetime(2026, 11, 22, 20, 0, 0, tzinfo=timezone(timedelta(hours=2)))

    level = PriceLevel(price=Decimal("0.50"), size=Decimal("10"))
    with pytest.raises(FrozenInstanceError):
        level.price = Decimal("0.51")  # type: ignore[misc]

    book = NormalizedOrderBook(
        venue="kalshi",
        market_id="KX-TZ",
        outcome_side="yes",
        ts_utc=ts_plus2,
        bids=(),
        asks=(),
    )
    assert book.ts_utc == ts_utc
    assert book.ts_utc.tzinfo is UTC
    assert book.ts_utc.utcoffset() == timedelta(0)

    metadata = MarketMetadata(
        venue="polymarket",
        market_id="0xcond",
        event_id="evt-1",
        title="Will X happen?",
        category="Economics",
        outcomes=("Yes", "No"),
        is_yes_no=True,
        end_time_utc=ts_plus2,
        status="open",
        extra={"enable_order_book": True},
    )
    assert metadata.end_time_utc == ts_utc
    assert metadata.end_time_utc is not None
    assert metadata.end_time_utc.tzinfo is UTC
    assert metadata.extra["enable_order_book"] is True
    with pytest.raises(TypeError):
        metadata.extra["mutated"] = True  # type: ignore[index]

    trade = TradeRecord(
        venue="polymarket",
        market_id="0xcond",
        token_id="12345",
        price=Decimal("0.62"),
        size=Decimal("100"),
        notional_usd=Decimal("62.00"),
        ts_utc=ts_plus2,
        side="buy",
        tx_hash="0xtxhash",
    )
    assert trade.schema_version == SCHEMA_VERSION
    assert trade.notional_usd == Decimal("62.00")
    assert trade.ts_utc == ts_utc
    assert trade.ts_utc.tzinfo is UTC

    with pytest.raises(ValueError, match="price"):
        PriceLevel(price=Decimal("1.01"), size=Decimal("1"))

    with pytest.raises(ValueError, match="price"):
        PriceLevel(price=Decimal("NaN"), size=Decimal("1"))

    with pytest.raises(ValueError, match="size"):
        PriceLevel(price=Decimal("0.50"), size=Decimal("0"))

    with pytest.raises(ValueError, match="size"):
        PriceLevel(price=Decimal("0.50"), size=Decimal("-1"))

    with pytest.raises(ValueError, match="size"):
        TradeRecord(
            venue="polymarket",
            market_id="0xcond",
            token_id=None,
            price=Decimal("0.50"),
            size=Decimal("-1"),
            notional_usd=Decimal("0"),
            ts_utc=ts_utc,
            side=None,
        )

    with pytest.raises(ValueError, match="timezone-aware"):
        NormalizedOrderBook(
            venue="kalshi",
            market_id="KX-NAIVE",
            outcome_side="yes",
            ts_utc=datetime(2026, 11, 22, 18, 0, 0),
            bids=(),
            asks=(),
        )
