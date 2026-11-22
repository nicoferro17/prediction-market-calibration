"""Kalshi Trade API v2 pure parsers and asynchronous ingestion client.

This module intentionally separates **pure, deterministic parsing functions**
(no I/O, no mutable state) from the **asynchronous HTTP client** built on top
of :class:`pm_quant.ingest.http.AsyncRateLimitedClient`.

Key Kalshi 2026 API invariants enforced here:
- ``orderbook_fp`` arrays (``yes_dollars``, ``no_dollars``) are **bid-only**
  and sorted in **ascending** price order (best bid is the last element).
- Complementary asks are derived via ``Decimal("1") - opposite_bid_price``
  with identical contract sizes.
- ``GET /markets/orderbooks`` requires **repeated** ``tickers`` query
  parameters (up to 100 per call), whereas ``GET /markets/candlesticks``
  requires a **comma-separated** ``market_tickers`` parameter (up to 100 per
  call).
- Markets and candlesticks settled strictly before ``market_settled_ts`` from
  ``GET /historical/cutoff`` are served from ``/historical/*`` endpoints.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from types import TracebackType
from typing import Any, Final
from urllib.parse import quote

import httpx

from pm_quant.ingest.http import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    AsyncRateLimitedClient,
    NowFn,
)
from pm_quant.ingest.schemas import MarketMetadata, NormalizedOrderBook, PriceLevel

DEFAULT_KALSHI_BASE_URL: Final[str] = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_EXTERNAL_BASE_URL: Final[str] = "https://external-api.kalshi.com/trade-api/v2"
KALSHI_MAX_BATCH_TICKERS: Final[int] = 100
KALSHI_MAX_PAGE_LIMIT: Final[int] = 1000
DEFAULT_ORDERBOOK_DEPTH: Final[int] = 20
DEFAULT_MARKETS_LIMIT: Final[int] = 200
DEFAULT_CANDLESTICK_INTERVAL: Final[int] = 60
HISTORICAL_CANDLESTICK_INTERVALS: Final[frozenset[int]] = frozenset({1, 60, 1440})

_ZERO: Final[Decimal] = Decimal("0")
_ONE: Final[Decimal] = Decimal("1")

_MARKET_EXTRA_FIELDS: Final[tuple[str, ...]] = (
    "market_type",
    "series_ticker",
    "subtitle",
    "yes_sub_title",
    "no_sub_title",
    "result",
    "can_close_early",
    "created_time",
    "open_time",
    "close_time",
    "expected_expiration_time",
    "latest_expiration_time",
    "settlement_ts",
    "settlement_value_dollars",
    "mve_collection_ticker",
)


# ---------------------------------------------------------------------------
# Pure parsing helpers (no I/O, no mutable state)
# ---------------------------------------------------------------------------


def _ensure_utc(dt: datetime, field_name: str) -> datetime:
    """Validate that ``dt`` is timezone-aware and normalize it to UTC."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware (UTC), got naive datetime {dt!r}")
    return dt.astimezone(UTC)


def parse_kalshi_timestamp(raw: Any, *, field_name: str = "timestamp") -> datetime:
    """Parse an ISO-8601 Kalshi timestamp string into a UTC-aware ``datetime``."""
    if not isinstance(raw, str):
        raise ValueError(f"{field_name} must be an ISO-8601 string, got {type(raw).__name__}")
    text = raw.strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 {field_name}: {raw!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_optional_market_timestamp(raw: Any) -> datetime | None:
    """Parse an optional market ISO timestamp into UTC, treating blank/zero values as ``None``."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    # Treat Go zero-value timestamps (0001-01-01T00:00:00Z) as unset.
    if parsed.year <= 1:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _non_empty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _to_decimal(value: Any) -> Decimal | None:
    """Convert a fixed-point string or numeric scalar to a finite ``Decimal``, or ``None``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            dec = Decimal(text)
        except InvalidOperation:
            return None
        return dec if dec.is_finite() else None
    if isinstance(value, int | float):
        try:
            dec = Decimal(str(value))
        except InvalidOperation:
            return None
        return dec if dec.is_finite() else None
    return None


def _parse_kalshi_bid_levels(raw_levels: Any, max_depth: int) -> tuple[PriceLevel, ...]:
    """Parse, filter, sort descending (best bid first), and truncate Kalshi bid levels.

    Kalshi returns ``[price_str, count_str]`` pairs sorted ascending by price
    (best bid last). Levels with ``size <= 0`` or ``price`` outside the open
    interval ``(0, 1)`` are discarded before sorting and truncation.
    """
    if not isinstance(raw_levels, list | tuple):
        return ()

    valid_levels: list[PriceLevel] = []
    for entry in raw_levels:
        if not isinstance(entry, list | tuple) or len(entry) < 2:
            continue
        price = _to_decimal(entry[0])
        size = _to_decimal(entry[1])
        if price is None or size is None:
            continue
        if size <= _ZERO or price <= _ZERO or price >= _ONE:
            continue
        valid_levels.append(PriceLevel(price=price, size=size))

    valid_levels.sort(key=lambda level: level.price, reverse=True)
    return tuple(valid_levels[:max_depth])


def _derive_complementary_asks(
    opposite_bids_desc: tuple[PriceLevel, ...],
) -> tuple[PriceLevel, ...]:
    """Derive ascending asks (best/lowest ask first) from descending opposite-side bids."""
    return tuple(
        PriceLevel(price=_ONE - level.price, size=level.size) for level in opposite_bids_desc
    )


def parse_kalshi_orderbook(
    raw_book: Mapping[str, Any] | None,
    ticker: str,
    ts_utc: datetime,
    max_depth: int = DEFAULT_ORDERBOOK_DEPTH,
) -> tuple[NormalizedOrderBook, NormalizedOrderBook]:
    """Parse a bid-only Kalshi orderbook payload into coherent ``(yes_book, no_book)`` snapshots.

    Accepts either an outer envelope containing ``orderbook_fp`` (such as an item
    from ``GET /markets/orderbooks`` or the response of
    ``GET /markets/{ticker}/orderbook``) or a mapping containing ``yes_dollars``
    and ``no_dollars`` directly.

    Ordering & reciprocal invariants:
    - ``yes_book.bids``: parsed from ``yes_dollars``, sorted descending (best bid at index 0),
      truncated to the top ``max_depth`` valid levels.
    - ``yes_book.asks``: derived from ``no_dollars`` as ``Decimal("1") - no_bid_price``
      with identical sizes, sorted ascending (best/lowest ask at index 0), truncated to
      ``max_depth``.
    - ``no_book``: symmetric mirror (``no_bids`` descending from ``no_dollars``;
      ``no_asks`` ascending from ``Decimal("1") - yes_bid_price``).
    - Levels with ``size <= 0`` or ``price`` outside ``(0, 1)`` are filtered out.
    - Empty (``[]``, ``None``, missing keys) and one-sided books are handled safely.
    """
    clean_ticker = _non_empty_str(ticker)
    if clean_ticker is None:
        raise ValueError("ticker must be a non-empty string")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")
    ts_norm = _ensure_utc(ts_utc, "ts_utc")

    book_payload: Mapping[str, Any]
    if raw_book is None:
        book_payload = {}
    elif "orderbook_fp" in raw_book:
        nested = raw_book.get("orderbook_fp")
        book_payload = nested if isinstance(nested, Mapping) else {}
    else:
        book_payload = raw_book

    yes_bids = _parse_kalshi_bid_levels(book_payload.get("yes_dollars"), max_depth)
    no_bids = _parse_kalshi_bid_levels(book_payload.get("no_dollars"), max_depth)

    yes_asks = _derive_complementary_asks(no_bids)
    no_asks = _derive_complementary_asks(yes_bids)

    yes_book = NormalizedOrderBook(
        venue="kalshi",
        market_id=clean_ticker,
        outcome_side="yes",
        ts_utc=ts_norm,
        bids=yes_bids,
        asks=yes_asks,
    )
    no_book = NormalizedOrderBook(
        venue="kalshi",
        market_id=clean_ticker,
        outcome_side="no",
        ts_utc=ts_norm,
        bids=no_bids,
        asks=no_asks,
    )
    return yes_book, no_book


def _extract_category_or_series(raw: Mapping[str, Any], event_id: str | None) -> str | None:
    """Resolve market category from explicit category/series fields or event ticker prefix."""
    for key in ("category", "series_ticker", "series"):
        value = _non_empty_str(raw.get(key))
        if value is not None:
            return value
    if event_id is not None and "-" in event_id:
        prefix = event_id.split("-", 1)[0].strip()
        if prefix:
            return prefix
    return None


def parse_kalshi_market(raw: Mapping[str, Any]) -> MarketMetadata:
    """Parse a raw Kalshi market object (live or historical) into ``MarketMetadata``.

    ``end_time_utc`` is resolved in UTC from trading-close / expiration fields
    only (``close_time`` -> ``expected_expiration_time`` ->
    ``latest_expiration_time``) and **never** falls back to ``settlement_ts`` so
    time-to-resolution calculations are not biased by post-close settlement lag.
    ``settlement_ts`` (if present) is preserved inside ``extra`` for archive
    routing checks.
    """
    ticker = _non_empty_str(raw.get("ticker"))
    if ticker is None:
        raise ValueError("Kalshi market payload missing required non-empty 'ticker'")

    event_id = _non_empty_str(raw.get("event_ticker"))
    title = (
        _non_empty_str(raw.get("title"))
        or _non_empty_str(raw.get("yes_sub_title"))
        or _non_empty_str(raw.get("subtitle"))
        or ticker
    )
    category = _extract_category_or_series(raw, event_id)

    # Never fall back to settlement_ts for market expiration / trading close.
    end_time_utc = (
        _parse_optional_market_timestamp(raw.get("close_time"))
        or _parse_optional_market_timestamp(raw.get("expected_expiration_time"))
        or _parse_optional_market_timestamp(raw.get("latest_expiration_time"))
    )

    raw_status = _non_empty_str(raw.get("status"))
    status = raw_status.lower() if raw_status is not None else "unknown"

    raw_market_type = _non_empty_str(raw.get("market_type"))
    is_yes_no = raw_market_type is None or raw_market_type.lower() == "binary"
    outcomes: tuple[str, ...] = ("Yes", "No") if is_yes_no else ()

    extra: dict[str, Any] = {}
    for field_name in _MARKET_EXTRA_FIELDS:
        value = raw.get(field_name)
        if value is not None and value != "":
            extra[field_name] = value

    return MarketMetadata(
        venue="kalshi",
        market_id=ticker,
        event_id=event_id,
        title=title,
        category=category,
        outcomes=outcomes,
        is_yes_no=is_yes_no,
        end_time_utc=end_time_utc,
        status=status,
        extra=extra,
    )


def _chunk_tickers(tickers: Sequence[str], chunk_size: int) -> list[list[str]]:
    """Deduplicate non-empty tickers in order and split into fixed-size chunks."""
    unique = list(dict.fromkeys(clean for raw in tickers if (clean := raw.strip())))
    return [unique[idx : idx + chunk_size] for idx in range(0, len(unique), chunk_size)]


# ---------------------------------------------------------------------------
# Asynchronous Kalshi HTTP client
# ---------------------------------------------------------------------------


class KalshiClient:
    """Asynchronous read-only client for Kalshi Trade API v2 public endpoints."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_KALSHI_BASE_URL,
        http_client: AsyncRateLimitedClient | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
        now_fn: NowFn | None = None,
    ) -> None:
        cleaned_base_url = base_url.strip().rstrip("/")
        if not cleaned_base_url:
            raise ValueError("base_url must be a non-empty URL")

        self._base_url = cleaned_base_url
        self._now_fn: NowFn = now_fn if now_fn is not None else (lambda: datetime.now(UTC))
        self._historical_cutoff: datetime | None = None
        self._cutoff_lock = asyncio.Lock()

        if http_client is not None:
            self._http = http_client
            self._owns_http = False
        else:
            self._http = AsyncRateLimitedClient(
                base_url=cleaned_base_url,
                max_concurrency=max_concurrency,
                max_retries=max_retries,
                timeout=timeout,
                transport=transport,
                now_fn=self._now_fn,
            )
            self._owns_http = True

    @property
    def base_url(self) -> str:
        """Configured Kalshi Trade API v2 base URL."""
        return self._base_url

    @property
    def cached_historical_cutoff(self) -> datetime | None:
        """Return the in-memory cached ``market_settled_ts`` cutoff, or ``None`` if unfetched."""
        return self._historical_cutoff

    async def __aenter__(self) -> KalshiClient:
        if self._owns_http:
            await self._http.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._owns_http:
            await self._http.__aexit__(exc_type, exc_val, exc_tb)

    async def aclose(self) -> None:
        """Close the owned underlying HTTP client (no-op if externally supplied)."""
        if self._owns_http:
            await self._http.aclose()

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    async def get_historical_cutoff(self, *, force_refresh: bool = False) -> datetime:
        """Fetch and cache the UTC ``market_settled_ts`` boundary from ``GET /historical/cutoff``.

        Subsequent calls reuse the instance-level in-memory cache unless
        ``force_refresh=True`` is specified.
        """
        if self._historical_cutoff is not None and not force_refresh:
            return self._historical_cutoff

        async with self._cutoff_lock:
            if self._historical_cutoff is not None and not force_refresh:
                return self._historical_cutoff

            payload = await self._http.get_json(self._url("/historical/cutoff"))
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"expected JSON object from /historical/cutoff, got {type(payload).__name__}"
                )
            cutoff = parse_kalshi_timestamp(
                payload.get("market_settled_ts"),
                field_name="market_settled_ts",
            )
            self._historical_cutoff = cutoff
            return cutoff

    async def is_archived(self, settled_ts: datetime) -> bool:
        """Return ``True`` if ``settled_ts`` is strictly before the cached historical cutoff.

        Validates that ``settled_ts`` is timezone-aware, lazily populates the
        in-memory cutoff cache via :meth:`get_historical_cutoff` if needed, and
        compares in UTC.
        """
        settled_utc = _ensure_utc(settled_ts, "settled_ts")
        cutoff = await self.get_historical_cutoff()
        return settled_utc < cutoff

    async def get_markets(
        self,
        status: str | None = "open",
        archived: bool = False,
        limit: int = DEFAULT_MARKETS_LIMIT,
        max_pages: int | None = None,
        *,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        mve_filter: str | None = None,
    ) -> list[MarketMetadata]:
        """Paginate through live (``/markets``) or archived (``/historical/markets``) markets.

        Note that ``GET /historical/markets`` only holds settled markets and does
        not accept a ``status`` query filter, so ``status`` is omitted when
        ``archived=True``. Results are deduplicated by market ticker across pages.
        """
        if limit < 1 or limit > KALSHI_MAX_PAGE_LIMIT:
            raise ValueError(f"limit must be in [1, {KALSHI_MAX_PAGE_LIMIT}], got {limit}")
        if max_pages is not None and max_pages < 0:
            raise ValueError(f"max_pages must be >= 0 or None, got {max_pages}")
        if max_pages == 0:
            return []

        endpoint = self._url("/historical/markets" if archived else "/markets")
        results: list[MarketMetadata] = []
        seen_tickers: set[str] = set()
        cursor: str | None = None
        pages_fetched = 0

        while True:
            params: dict[str, Any] = {"limit": limit}
            if not archived and status is not None and status.strip():
                params["status"] = status.strip()
            if series_ticker is not None and series_ticker.strip():
                params["series_ticker"] = series_ticker.strip()
            if event_ticker is not None and event_ticker.strip():
                params["event_ticker"] = event_ticker.strip()
            if mve_filter is not None and mve_filter.strip():
                params["mve_filter"] = mve_filter.strip()
            if cursor is not None:
                params["cursor"] = cursor

            payload = await self._http.get_json(endpoint, params=params)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"expected JSON object from {endpoint}, got {type(payload).__name__}"
                )

            raw_markets = payload.get("markets")
            if raw_markets is None:
                raw_markets = []
            if not isinstance(raw_markets, list):
                raise ValueError(
                    f"expected 'markets' list from {endpoint}, got {type(raw_markets).__name__}"
                )

            for item in raw_markets:
                if not isinstance(item, Mapping):
                    continue
                metadata = parse_kalshi_market(item)
                if metadata.market_id not in seen_tickers:
                    seen_tickers.add(metadata.market_id)
                    results.append(metadata)

            pages_fetched += 1
            if max_pages is not None and pages_fetched >= max_pages:
                break

            next_cursor = _non_empty_str(payload.get("cursor"))
            if next_cursor is None or next_cursor == cursor:
                break
            cursor = next_cursor

        return results

    async def _fetch_orderbooks_chunk(
        self,
        chunk: list[str],
        depth: int,
        explicit_ts_utc: datetime | None,
    ) -> dict[str, tuple[NormalizedOrderBook, NormalizedOrderBook]]:
        endpoint = self._url("/markets/orderbooks")
        # Kalshi /markets/orderbooks requires repeated `tickers` query parameters
        # (?tickers=A&tickers=B); comma-joining treats "A,B" as one ticker name.
        payload = await self._http.get_json(endpoint, params={"tickers": chunk, "depth": depth})
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"expected JSON object from /markets/orderbooks, got {type(payload).__name__}"
            )

        raw_books = payload.get("orderbooks")
        if raw_books is None:
            return {}
        if not isinstance(raw_books, list):
            raise ValueError(
                "expected 'orderbooks' list from /markets/orderbooks, "
                f"got {type(raw_books).__name__}"
            )

        snapshot_ts = (
            explicit_ts_utc
            if explicit_ts_utc is not None
            else _ensure_utc(self._now_fn(), "ts_utc")
        )
        parsed: dict[str, tuple[NormalizedOrderBook, NormalizedOrderBook]] = {}
        for idx, entry in enumerate(raw_books):
            if not isinstance(entry, Mapping):
                continue
            entry_ticker = _non_empty_str(entry.get("ticker"))
            if entry_ticker is None and len(raw_books) == len(chunk):
                entry_ticker = chunk[idx]
            if entry_ticker is None:
                continue
            parsed[entry_ticker] = parse_kalshi_orderbook(
                entry,
                ticker=entry_ticker,
                ts_utc=snapshot_ts,
                max_depth=depth,
            )
        return parsed

    async def get_orderbooks_batch(
        self,
        tickers: Sequence[str],
        depth: int = DEFAULT_ORDERBOOK_DEPTH,
        *,
        ts_utc: datetime | None = None,
    ) -> dict[str, tuple[NormalizedOrderBook, NormalizedOrderBook]]:
        """Fetch normalized ``(yes_book, no_book)`` snapshots via ``GET /markets/orderbooks``.

        Automatically deduplicates and chunks ``tickers`` into batches of at most
        100 elements (Kalshi's API limit) and truncates each side to ``depth``
        best price levels.
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        explicit_ts = _ensure_utc(ts_utc, "ts_utc") if ts_utc is not None else None

        chunks = _chunk_tickers(tickers, KALSHI_MAX_BATCH_TICKERS)
        if not chunks:
            return {}

        chunk_results = await asyncio.gather(
            *(self._fetch_orderbooks_chunk(chunk, depth, explicit_ts) for chunk in chunks)
        )
        merged: dict[str, tuple[NormalizedOrderBook, NormalizedOrderBook]] = {}
        for chunk_map in chunk_results:
            merged.update(chunk_map)
        return merged

    async def _fetch_candlesticks_chunk(
        self,
        chunk: list[str],
        start_ts: int,
        end_ts: int,
        period_interval: int,
        include_latest_before_start: bool,
    ) -> dict[str, list[dict[str, Any]]]:
        endpoint = self._url("/markets/candlesticks")
        # Unlike /markets/orderbooks, /markets/candlesticks expects a comma-separated
        # `market_tickers` string (up to 100 tickers per call).
        params: dict[str, Any] = {
            "market_tickers": ",".join(chunk),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        if include_latest_before_start:
            params["include_latest_before_start"] = "true"

        payload = await self._http.get_json(endpoint, params=params)
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"expected JSON object from /markets/candlesticks, got {type(payload).__name__}"
            )

        raw_markets = payload.get("markets")
        if raw_markets is None:
            return {}
        if not isinstance(raw_markets, list):
            raise ValueError(
                "expected 'markets' list from /markets/candlesticks, "
                f"got {type(raw_markets).__name__}"
            )

        result: dict[str, list[dict[str, Any]]] = {}
        for entry in raw_markets:
            if not isinstance(entry, Mapping):
                continue
            ticker = _non_empty_str(entry.get("market_ticker")) or _non_empty_str(
                entry.get("ticker")
            )
            if ticker is None:
                continue
            raw_candles = entry.get("candlesticks")
            if isinstance(raw_candles, list):
                result[ticker] = [dict(c) for c in raw_candles if isinstance(c, Mapping)]
            else:
                result[ticker] = []
        return result

    async def get_candlesticks_batch(
        self,
        tickers: Sequence[str],
        start_ts: int,
        end_ts: int,
        period_interval: int = DEFAULT_CANDLESTICK_INTERVAL,
        *,
        include_latest_before_start: bool = False,
        max_batch_size: int = KALSHI_MAX_BATCH_TICKERS,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch live candlesticks via ``GET /markets/candlesticks`` in chunks of <= 100 tickers.

        Note: Kalshi caps each batch call at 10,000 total candlestick periods
        across all tickers in the batch; callers requesting very long fine-grained
        windows can lower ``max_batch_size`` accordingly.
        """
        if start_ts > end_ts:
            raise ValueError(f"start_ts ({start_ts}) must be <= end_ts ({end_ts})")
        if period_interval < 1:
            raise ValueError(f"period_interval must be >= 1, got {period_interval}")
        if max_batch_size < 1 or max_batch_size > KALSHI_MAX_BATCH_TICKERS:
            raise ValueError(
                f"max_batch_size must be in [1, {KALSHI_MAX_BATCH_TICKERS}], got {max_batch_size}"
            )

        chunks = _chunk_tickers(tickers, max_batch_size)
        if not chunks:
            return {}

        chunk_results = await asyncio.gather(
            *(
                self._fetch_candlesticks_chunk(
                    chunk,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    period_interval=period_interval,
                    include_latest_before_start=include_latest_before_start,
                )
                for chunk in chunks
            )
        )
        merged: dict[str, list[dict[str, Any]]] = {}
        for chunk_map in chunk_results:
            merged.update(chunk_map)
        return merged

    async def get_historical_candlesticks(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = DEFAULT_CANDLESTICK_INTERVAL,
    ) -> list[dict[str, Any]]:
        """Fetch archived candlesticks from ``GET /historical/markets/{ticker}/candlesticks``.

        Kalshi accepts ``period_interval`` values of ``1`` (1m), ``60`` (1h), or
        ``1440`` (1d), with at most 5,000 periods per request.
        """
        clean_ticker = _non_empty_str(ticker)
        if clean_ticker is None:
            raise ValueError("ticker must be a non-empty string")
        if start_ts > end_ts:
            raise ValueError(f"start_ts ({start_ts}) must be <= end_ts ({end_ts})")
        if period_interval not in HISTORICAL_CANDLESTICK_INTERVALS:
            allowed = sorted(HISTORICAL_CANDLESTICK_INTERVALS)
            raise ValueError(
                f"period_interval must be one of {allowed} for historical candlesticks, "
                f"got {period_interval}"
            )

        encoded_ticker = quote(clean_ticker, safe="")
        endpoint = self._url(f"/historical/markets/{encoded_ticker}/candlesticks")
        payload = await self._http.get_json(
            endpoint,
            params={
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
        )
        if not isinstance(payload, Mapping):
            raise ValueError(f"expected JSON object from {endpoint}, got {type(payload).__name__}")

        raw_candles = payload.get("candlesticks")
        if raw_candles is None:
            return []
        if not isinstance(raw_candles, list):
            raise ValueError(
                f"expected 'candlesticks' list from {endpoint}, got {type(raw_candles).__name__}"
            )
        return [dict(candle) for candle in raw_candles if isinstance(candle, Mapping)]
