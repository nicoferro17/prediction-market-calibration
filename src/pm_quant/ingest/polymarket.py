"""Polymarket Gamma, CLOB, and Data API v2 pure parsers and asynchronous client.

This module separates **pure, deterministic parsing functions** (no I/O, no
mutable state) from the **asynchronous HTTP client** built on top of
:class:`pm_quant.ingest.http.AsyncRateLimitedClient`.

Key Polymarket November 2026 API invariants enforced here:
- **Gamma API** (``https://gamma-api.polymarket.com``): ``clobTokenIds`` and
  ``outcomes`` may arrive either as Python lists or as JSON-serialized strings.
  ``is_yes_no`` is ``True`` strictly when normalized outcomes equal
  ``{"yes", "no"}`` (length 2), preserving the ``I_YES/NO`` contract indicator.
  Final Gamma volume/liquidity snapshots are never exposed as point-in-time
  features to prevent lookahead leakage.
- **CLOB API** (``https://clob.polymarket.com``): L2 orderbook ``bids`` and
  ``asks`` are converted to ``Decimal``, filtered for ``size > 0`` and valid
  prices in ``(0, 1)``, **explicitly re-sorted** (bids descending, asks
  ascending — never trusting server order), and capped to ``max_depth`` levels.
- **Data API v2** (``https://data-api.polymarket.com/v2``):
  - ``GET /v2/trades`` chunks ``condition_ids`` to at most 20 per call, always
    sends ``condition=`` (never legacy ``market=``) and ``taker_only=true``
    (preventing maker/taker volume double-counting), and paginates via
    ``pagination.next_cursor`` sent back as ``cursor=`` (never ``offset=``).
  - ``GET /v2/prices-history`` automatically splits windows exceeding 15 days
    into contiguous blocks of at most 15 days, deduplicates and sorts points by
    timestamp, and logs a warning whenever a block returns empty (resolution
    decay on historical buckets).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from types import TracebackType
from typing import Any, Final

import httpx

from pm_quant.ingest.historical_schema import ResolvedMarket, parse_gamma_resolved_market
from pm_quant.ingest.http import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    AsyncRateLimitedClient,
    NowFn,
)
from pm_quant.ingest.schemas import (
    MarketMetadata,
    NormalizedOrderBook,
    OutcomeSide,
    PriceLevel,
    TradeRecord,
    TradeSide,
)
from pm_quant.ingest.universe import LiveMarketCandidate

logger = logging.getLogger(__name__)

DEFAULT_GAMMA_BASE_URL: Final[str] = "https://gamma-api.polymarket.com"
DEFAULT_CLOB_BASE_URL: Final[str] = "https://clob.polymarket.com"
DEFAULT_DATA_API_V2_BASE_URL: Final[str] = "https://data-api.polymarket.com/v2"

DEFAULT_ORDERBOOK_DEPTH: Final[int] = 20
DEFAULT_GAMMA_LIMIT: Final[int] = 100
POLYMARKET_GAMMA_MAX_LIMIT: Final[int] = 500
POLYMARKET_MAX_BOOKS_BATCH: Final[int] = 500
POLYMARKET_MAX_CONDITION_BATCH: Final[int] = 20
DEFAULT_TRADES_LIMIT: Final[int] = 500
POLYMARKET_MAX_TRADES_LIMIT: Final[int] = 1000
POLYMARKET_V2_HISTORY_MAX_WINDOW_DAYS: Final[int] = 15
POLYMARKET_V2_HISTORY_MAX_WINDOW_SECONDS: Final[int] = (
    POLYMARKET_V2_HISTORY_MAX_WINDOW_DAYS * 86_400
)
POLYMARKET_V2_BUCKET_MIN_SECONDS: Final[int] = 60
POLYMARKET_V2_BUCKET_MAX_SECONDS: Final[int] = 86_400

_ZERO: Final[Decimal] = Decimal("0")
_ONE: Final[Decimal] = Decimal("1")
_YES_NO_SET: Final[frozenset[str]] = frozenset({"yes", "no"})


# ---------------------------------------------------------------------------
# Pure parsing helpers (no I/O, no mutable state)
# ---------------------------------------------------------------------------


def _ensure_utc(dt: datetime, field_name: str) -> datetime:
    """Validate that ``dt`` is timezone-aware and normalize it to UTC."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware (UTC), got naive datetime {dt!r}")
    return dt.astimezone(UTC)


def _format_gamma_iso_utc(value: str | datetime, field_name: str) -> tuple[str, datetime]:
    """Validate and serialize a Gamma date bound into an ISO-8601 UTC string and ``datetime``."""
    if isinstance(value, datetime):
        dt_utc = _ensure_utc(value, field_name)
        return dt_utc.isoformat().replace("+00:00", "Z"), dt_utc
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} must be a non-empty ISO-8601 string or datetime")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"invalid ISO-8601 {field_name}: {value!r}") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            dt_utc = parsed.replace(tzinfo=UTC)
        else:
            dt_utc = parsed.astimezone(UTC)
        return dt_utc.isoformat().replace("+00:00", "Z"), dt_utc
    raise ValueError(
        f"{field_name} must be an ISO-8601 str or timezone-aware datetime, "
        f"got {type(value).__name__}"
    )


def _to_epoch_seconds(value: int | datetime, field_name: str) -> int:
    """Convert an integer epoch timestamp or timezone-aware ``datetime`` to UTC epoch seconds."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an int or timezone-aware datetime, got bool")
    if isinstance(value, datetime):
        return int(_ensure_utc(value, field_name).timestamp())
    if isinstance(value, int):
        return value
    raise ValueError(
        f"{field_name} must be an int or timezone-aware datetime, got {type(value).__name__}"
    )


def _non_empty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _id_to_str(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, int):
        return str(value)
    return None


def _to_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return default


def _to_decimal(value: Any) -> Decimal | None:
    """Convert a string or numeric scalar to a finite ``Decimal``, or ``None``."""
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
        if isinstance(value, float) and not math.isfinite(value):
            return None
        try:
            dec = Decimal(str(value))
        except InvalidOperation:
            return None
        return dec if dec.is_finite() else None
    return None


def _parse_optional_iso_timestamp(raw: Any) -> datetime | None:
    """Parse an optional ISO-8601 timestamp into UTC, returning ``None`` if unset or invalid."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.year <= 1:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_polymarket_timestamp(raw: Any, *, field_name: str = "timestamp") -> datetime:
    """Parse a Unix epoch (seconds or ms) or ISO-8601 timestamp into a UTC-aware ``datetime``."""
    if isinstance(raw, bool) or raw is None:
        raise ValueError(f"{field_name} is required, got {raw!r}")
    if isinstance(raw, datetime):
        return _ensure_utc(raw, field_name)
    if isinstance(raw, int | float):
        if not math.isfinite(raw):
            raise ValueError(f"{field_name} must be finite, got {raw!r}")
        seconds = raw / 1000.0 if abs(raw) >= 1e12 else float(raw)
        return datetime.fromtimestamp(seconds, tz=UTC)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError(f"{field_name} must be a non-empty timestamp")
        try:
            numeric = float(text)
        except ValueError:
            numeric = math.nan
        if math.isfinite(numeric):
            seconds = numeric / 1000.0 if abs(numeric) >= 1e12 else numeric
            return datetime.fromtimestamp(seconds, tz=UTC)
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"invalid {field_name}: {raw!r}") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    raise ValueError(f"unsupported {field_name} type: {type(raw).__name__}")


def _parse_string_sequence(raw: Any) -> tuple[str, ...]:
    """Parse a Gamma field that may be either a Python list or a JSON-encoded list string."""
    candidate: Any = raw
    if isinstance(candidate, str):
        text = candidate.strip()
        if not text:
            return ()
        try:
            candidate = json.loads(text)
        except ValueError:
            return ()
    if not isinstance(candidate, list | tuple):
        return ()

    items: list[str] = []
    for entry in candidate:
        converted = _id_to_str(entry)
        if converted is not None:
            items.append(converted)
    return tuple(items)


def _extract_gamma_event_and_category(raw: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Resolve ``event_id`` and ``category`` from top-level fields or nested Gamma ``events``."""
    event_id = _id_to_str(raw.get("eventId")) or _id_to_str(raw.get("event_id"))
    category = _non_empty_str(raw.get("category"))

    events = raw.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, Mapping):
                continue
            if event_id is None:
                event_id = _id_to_str(event.get("id")) or _non_empty_str(event.get("slug"))
            if category is None:
                category = _non_empty_str(event.get("category"))
            if event_id is not None and category is not None:
                break

    return event_id, category


def _derive_gamma_status(raw: Mapping[str, Any]) -> str:
    explicit = _non_empty_str(raw.get("status"))
    if explicit is not None:
        return explicit.lower()
    if _to_bool(raw.get("archived"), default=False):
        return "archived"
    if _to_bool(raw.get("closed"), default=False):
        return "closed"
    if _to_bool(raw.get("active"), default=False):
        return "active"
    return "unknown"


def parse_gamma_market(raw: Mapping[str, Any]) -> MarketMetadata:
    """Parse a raw Polymarket Gamma market payload into normalized ``MarketMetadata``.

    Handles ``clobTokenIds`` and ``outcomes`` seamlessly whether they arrive as
    Python lists or as JSON-serialized strings. Sets ``is_yes_no = True`` **only**
    when there are exactly two outcomes whose lowercased, trimmed labels equal
    ``{"yes", "no"}`` (e.g. ``False`` for ``["Lakers", "Celtics"]`` or
    ``["Up", "Down"]``).

    Anti-leakage policy:
    - ``end_time_utc`` is parsed strictly from ``endDate`` (falling back only to
      ``endDateIso`` when ``endDate`` is absent) and never from post-resolution
      oracle settlement timestamps.
    - Final lifetime Gamma ``volume`` / ``liquidity`` / snapshot price fields are
      intentionally **excluded** from ``MarketMetadata`` and ``extra`` so they
      cannot leak into point-in-time statistical features.
    """
    condition_id = _non_empty_str(raw.get("conditionId")) or _non_empty_str(raw.get("condition_id"))
    if condition_id is None:
        raise ValueError("Gamma market payload missing required non-empty 'conditionId'")

    outcomes = _parse_string_sequence(raw.get("outcomes"))
    clob_token_ids = _parse_string_sequence(raw.get("clobTokenIds"))

    normalized_outcomes = {label.strip().lower() for label in outcomes}
    is_yes_no = len(outcomes) == 2 and normalized_outcomes == _YES_NO_SET

    event_id, category = _extract_gamma_event_and_category(raw)
    title = (
        _non_empty_str(raw.get("question"))
        or _non_empty_str(raw.get("title"))
        or _non_empty_str(raw.get("slug"))
        or condition_id
    )
    end_time_utc = _parse_optional_iso_timestamp(
        raw.get("endDate")
    ) or _parse_optional_iso_timestamp(raw.get("endDateIso"))
    status = _derive_gamma_status(raw)

    neg_risk_raw = raw.get("negRisk") if "negRisk" in raw else raw.get("neg_risk")
    extra: dict[str, Any] = {
        "clob_token_ids": clob_token_ids,
        "neg_risk": _to_bool(neg_risk_raw, default=False),
    }

    optional_str_fields: tuple[tuple[str, str], ...] = (
        ("id", "gamma_id"),
        ("slug", "slug"),
        ("questionID", "question_id"),
        ("negRiskMarketID", "neg_risk_market_id"),
        ("feeType", "fee_type"),
    )
    for raw_key, extra_key in optional_str_fields:
        value = _id_to_str(raw.get(raw_key))
        if value is not None:
            extra[extra_key] = value

    optional_bool_fields: tuple[tuple[str, str], ...] = (
        ("enableOrderBook", "enable_order_book"),
        ("acceptingOrders", "accepting_orders"),
        ("feesEnabled", "fees_enabled"),
        ("active", "active"),
        ("closed", "closed"),
        ("archived", "archived"),
    )
    for raw_key, extra_key in optional_bool_fields:
        if raw_key in raw and raw.get(raw_key) is not None:
            extra[extra_key] = _to_bool(raw.get(raw_key))

    tick_size = _to_decimal(raw.get("orderPriceMinTickSize"))
    if tick_size is not None:
        extra["tick_size"] = str(tick_size)
    min_order_size = _to_decimal(raw.get("orderMinSize"))
    if min_order_size is not None:
        extra["min_order_size"] = str(min_order_size)

    return MarketMetadata(
        venue="polymarket",
        market_id=condition_id,
        event_id=event_id,
        title=title,
        category=category,
        outcomes=outcomes,
        is_yes_no=is_yes_no,
        end_time_utc=end_time_utc,
        status=status,
        extra=extra,
    )


def _parse_non_negative_decimal(value: Any) -> Decimal | None:
    """Convert a scalar to a finite non-negative ``Decimal``, or ``None``."""
    dec = _to_decimal(value)
    if dec is None or dec < _ZERO:
        return None
    return dec


def _extract_gamma_tag_ids(
    raw: Mapping[str, Any],
    extra_tag_ids: Sequence[str | int] | None = None,
) -> tuple[str, ...]:
    """Extract normalized Gamma tag IDs from top-level fields, nested events, and caller context."""
    found: list[str] = []

    def _collect_tags_field(raw_tags: Any) -> None:
        if isinstance(raw_tags, list | tuple):
            for entry in raw_tags:
                if isinstance(entry, Mapping):
                    tid = _id_to_str(entry.get("id"))
                else:
                    tid = _id_to_str(entry)
                if tid is not None:
                    found.append(tid)

    _collect_tags_field(raw.get("tags"))
    for key in ("tagIds", "tag_ids"):
        found.extend(_parse_string_sequence(raw.get(key)))

    events = raw.get("events")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, Mapping):
                _collect_tags_field(event.get("tags"))

    if extra_tag_ids is not None:
        for extra_tag in extra_tag_ids:
            tid = _id_to_str(extra_tag)
            if tid is not None:
                found.append(tid)

    return tuple(dict.fromkeys(found))


def parse_gamma_live_candidate(
    raw: Mapping[str, Any],
    *,
    tag_ids: Sequence[str | int] | None = None,
) -> LiveMarketCandidate:
    """Parse a raw Gamma market payload into a live screening ``LiveMarketCandidate``.

    Delegates metadata extraction strictly to :func:`parse_gamma_market` (keeping
    ``MarketMetadata.extra`` free of volume/liquidity fields) and extracts
    ``volume24hr`` (USD notional) for live ``t_now`` universe screening only.
    """
    metadata = parse_gamma_market(raw)

    volume_24h: Decimal | None = None
    for vol_key in ("volume24hr", "volume24hrClob", "volume_24hr"):
        parsed_vol = _parse_non_negative_decimal(raw.get(vol_key))
        if parsed_vol is not None:
            volume_24h = parsed_vol
            break

    open_interest: Decimal | None = None
    for oi_key in ("openInterest", "open_interest"):
        parsed_oi = _parse_non_negative_decimal(raw.get(oi_key))
        if parsed_oi is not None:
            open_interest = parsed_oi
            break

    extracted_tags = _extract_gamma_tag_ids(raw, tag_ids)
    return LiveMarketCandidate(
        metadata=metadata,
        volume_24h=volume_24h,
        open_interest=open_interest,
        volume_24h_unit="usd",
        tag_ids=extracted_tags,
    )


def _parse_clob_side_levels(
    raw_levels: Any,
    *,
    descending: bool,
    max_depth: int,
) -> tuple[PriceLevel, ...]:
    """Parse, filter, explicitly sort, and depth-cap Polymarket CLOB book levels.

    Accepts both standard CLOB objects ``{"price": "...", "size": "..."}`` and
    two-element ``[price, size]`` sequences. Discards levels with ``size <= 0``
    or ``price`` outside ``(0, 1)``, sorts explicitly regardless of server order,
    and keeps the top ``max_depth`` levels.
    """
    if not isinstance(raw_levels, list | tuple):
        return ()

    valid_levels: list[PriceLevel] = []
    for entry in raw_levels:
        raw_price: Any
        raw_size: Any
        if isinstance(entry, Mapping):
            raw_price = entry.get("price")
            raw_size = entry.get("size")
        elif isinstance(entry, list | tuple) and len(entry) >= 2:
            raw_price = entry[0]
            raw_size = entry[1]
        else:
            continue

        price = _to_decimal(raw_price)
        size = _to_decimal(raw_size)
        if price is None or size is None:
            continue
        if size <= _ZERO or price <= _ZERO or price >= _ONE:
            continue
        valid_levels.append(PriceLevel(price=price, size=size))

    valid_levels.sort(key=lambda level: level.price, reverse=descending)
    return tuple(valid_levels[:max_depth])


def parse_clob_orderbook(
    raw: Mapping[str, Any] | None,
    token_id: str,
    ts_utc: datetime,
    max_depth: int = DEFAULT_ORDERBOOK_DEPTH,
    *,
    outcome_side: OutcomeSide = "yes",
    market_id: str | None = None,
) -> NormalizedOrderBook:
    """Parse a Polymarket CLOB ``/book`` or ``/books`` item into ``NormalizedOrderBook``.

    Enforces exact ``Decimal`` conversion, explicit re-sorting (``bids``
    descending so best bid is at index 0; ``asks`` ascending so best ask is at
    index 0 — never relying on server order), filtering of ``size <= 0`` and
    out-of-range prices, and truncation to the top ``max_depth`` levels per side.

    ``market_id`` is resolved from the explicit ``market_id`` argument or the
    CLOB payload's ``market`` (``condition_id``) field so snapshots join cleanly
    with ``MarketMetadata`` and ``TradeRecord``, falling back to ``token_id``
    only when neither is provided. ``token_id`` is always preserved in
    ``NormalizedOrderBook.token_id``.
    """
    clean_token_id = _non_empty_str(token_id)
    if clean_token_id is None:
        raise ValueError("token_id must be a non-empty string")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")
    ts_norm = _ensure_utc(ts_utc, "ts_utc")

    payload: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    resolved_market_id = (
        _non_empty_str(market_id)
        or _non_empty_str(payload.get("market"))
        or _non_empty_str(payload.get("condition_id"))
        or clean_token_id
    )
    bids = _parse_clob_side_levels(payload.get("bids"), descending=True, max_depth=max_depth)
    asks = _parse_clob_side_levels(payload.get("asks"), descending=False, max_depth=max_depth)

    return NormalizedOrderBook(
        venue="polymarket",
        market_id=resolved_market_id,
        outcome_side=outcome_side,
        ts_utc=ts_norm,
        bids=bids,
        asks=asks,
        token_id=clean_token_id,
    )


def parse_v2_trade(raw: Mapping[str, Any]) -> TradeRecord:
    """Parse a snake_case Polymarket Data API v2 ``/v2/trades`` item into ``TradeRecord``."""
    condition_id = _non_empty_str(raw.get("condition_id")) or _non_empty_str(raw.get("condition"))
    if condition_id is None:
        raise ValueError("Data API v2 trade payload missing required non-empty 'condition_id'")

    price = _to_decimal(raw.get("price"))
    if price is None or price < _ZERO or price > _ONE:
        raise ValueError(f"Data API v2 trade has invalid price: {raw.get('price')!r}")

    size = _to_decimal(raw.get("size"))
    if size is None or size < _ZERO:
        raise ValueError(f"Data API v2 trade has invalid size: {raw.get('size')!r}")

    explicit_notional = _to_decimal(raw.get("usdc_size"))
    notional_usd = (
        explicit_notional
        if explicit_notional is not None and explicit_notional >= _ZERO
        else price * size
    )

    ts_utc = _parse_polymarket_timestamp(raw.get("timestamp"), field_name="timestamp")

    token_id = (
        _id_to_str(raw.get("token_id"))
        or _id_to_str(raw.get("asset"))
        or _id_to_str(raw.get("asset_id"))
    )

    raw_side = _non_empty_str(raw.get("side"))
    side: TradeSide | None = None
    if raw_side is not None:
        lowered = raw_side.lower()
        if lowered == "buy":
            side = "buy"
        elif lowered == "sell":
            side = "sell"

    tx_hash = _non_empty_str(raw.get("transaction_hash")) or _non_empty_str(raw.get("tx_hash"))

    return TradeRecord(
        venue="polymarket",
        market_id=condition_id,
        token_id=token_id,
        price=price,
        size=size,
        notional_usd=notional_usd,
        ts_utc=ts_utc,
        side=side,
        tx_hash=tx_hash,
    )


def _chunk_ids(ids: Sequence[str], chunk_size: int) -> list[list[str]]:
    """Deduplicate non-empty identifiers in order and split into fixed-size chunks."""
    unique = list(dict.fromkeys(clean for raw in ids if (clean := raw.strip())))
    return [unique[idx : idx + chunk_size] for idx in range(0, len(unique), chunk_size)]


def _split_time_windows(
    start_ts: int,
    end_ts: int,
    max_span_seconds: int = POLYMARKET_V2_HISTORY_MAX_WINDOW_SECONDS,
) -> list[tuple[int, int]]:
    """Split ``[start_ts, end_ts]`` into contiguous windows of at most ``max_span_seconds``."""
    if start_ts == end_ts:
        return [(start_ts, end_ts)]
    windows: list[tuple[int, int]] = []
    cursor = start_ts
    while cursor < end_ts:
        window_end = min(cursor + max_span_seconds, end_ts)
        windows.append((cursor, window_end))
        cursor = window_end
    return windows


def _extract_point_timestamp(point: Mapping[str, Any], primary_key: str) -> int | None:
    """Extract an integer epoch timestamp from a price-history point mapping."""
    raw = point.get(primary_key)
    if raw is None and primary_key != "t":
        raw = point.get("t")
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and math.isfinite(raw):
        return int(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            parsed = _parse_optional_iso_timestamp(text)
            return int(parsed.timestamp()) if parsed is not None else None
    return None


def _is_settlement_point(point: Mapping[str, Any]) -> bool:
    """Return ``True`` if a v2 history point marks an on-chain settlement (resolution_seconds=0)."""
    res = point.get("resolution_seconds")
    return not isinstance(res, bool) and isinstance(res, int | float) and res == 0


def _deduplicate_and_sort_history(
    points: Sequence[Mapping[str, Any]],
    *,
    timestamp_key: str,
    price_key: str,
    exclude_settlement: bool = False,
) -> list[dict[str, Any]]:
    """Normalize prices to ``Decimal``, deduplicate by timestamp, and sort ascending."""
    by_timestamp: dict[int, dict[str, Any]] = {}
    for point in points:
        if exclude_settlement and _is_settlement_point(point):
            continue
        ts = _extract_point_timestamp(point, timestamp_key)
        if ts is None:
            continue
        price = _to_decimal(point.get(price_key))
        if price is None or price < _ZERO or price > _ONE:
            continue
        if ts not in by_timestamp:
            normalized = dict(point)
            normalized[price_key] = price
            by_timestamp[ts] = normalized
    return [by_timestamp[ts] for ts in sorted(by_timestamp)]


# ---------------------------------------------------------------------------
# Asynchronous Polymarket HTTP client
# ---------------------------------------------------------------------------


class PolymarketClient:
    """Asynchronous read-only client for Polymarket Gamma, CLOB, and Data API v2."""

    def __init__(
        self,
        *,
        gamma_base_url: str = DEFAULT_GAMMA_BASE_URL,
        clob_base_url: str = DEFAULT_CLOB_BASE_URL,
        data_api_base_url: str = DEFAULT_DATA_API_V2_BASE_URL,
        http_client: AsyncRateLimitedClient | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
        now_fn: NowFn | None = None,
    ) -> None:
        self._gamma_base_url = self._validate_base_url(gamma_base_url, "gamma_base_url")
        self._clob_base_url = self._validate_base_url(clob_base_url, "clob_base_url")
        self._data_api_base_url = self._validate_base_url(data_api_base_url, "data_api_base_url")
        self._now_fn: NowFn = now_fn if now_fn is not None else (lambda: datetime.now(UTC))

        if http_client is not None:
            self._http = http_client
            self._owns_http = False
        else:
            self._http = AsyncRateLimitedClient(
                max_concurrency=max_concurrency,
                max_retries=max_retries,
                timeout=timeout,
                transport=transport,
                now_fn=self._now_fn,
            )
            self._owns_http = True

    @staticmethod
    def _validate_base_url(url: str, name: str) -> str:
        cleaned = url.strip().rstrip("/")
        if not cleaned:
            raise ValueError(f"{name} must be a non-empty URL")
        return cleaned

    @property
    def gamma_base_url(self) -> str:
        """Configured Polymarket Gamma API base URL."""
        return self._gamma_base_url

    @property
    def clob_base_url(self) -> str:
        """Configured Polymarket CLOB API base URL."""
        return self._clob_base_url

    @property
    def data_api_base_url(self) -> str:
        """Configured Polymarket Data API v2 base URL."""
        return self._data_api_base_url

    async def __aenter__(self) -> PolymarketClient:
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

    def _gamma_url(self, path: str) -> str:
        return f"{self._gamma_base_url}/{path.lstrip('/')}"

    def _clob_url(self, path: str) -> str:
        return f"{self._clob_base_url}/{path.lstrip('/')}"

    def _data_url(self, path: str) -> str:
        return f"{self._data_api_base_url}/{path.lstrip('/')}"

    # ------------------------------------------------------------------
    # Gamma API
    # ------------------------------------------------------------------

    async def _fetch_gamma_markets_raw(
        self,
        *,
        active: bool | None = True,
        closed: bool | None = False,
        archived: bool | None = None,
        limit: int = DEFAULT_GAMMA_LIMIT,
        offset: int = 0,
        max_pages: int | None = None,
        order: str | None = None,
        ascending: bool | None = None,
        tag_id: int | str | None = None,
        slug: str | None = None,
        end_date_min: str | datetime | None = None,
        end_date_max: str | datetime | None = None,
    ) -> list[Mapping[str, Any]]:
        if limit < 1 or limit > POLYMARKET_GAMMA_MAX_LIMIT:
            raise ValueError(f"limit must be in [1, {POLYMARKET_GAMMA_MAX_LIMIT}], got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        if max_pages is not None and max_pages < 0:
            raise ValueError(f"max_pages must be >= 0 or None, got {max_pages}")

        end_date_min_iso: str | None = None
        end_date_min_dt: datetime | None = None
        if end_date_min is not None:
            end_date_min_iso, end_date_min_dt = _format_gamma_iso_utc(
                end_date_min,
                "end_date_min",
            )

        end_date_max_iso: str | None = None
        end_date_max_dt: datetime | None = None
        if end_date_max is not None:
            end_date_max_iso, end_date_max_dt = _format_gamma_iso_utc(
                end_date_max,
                "end_date_max",
            )

        if (
            end_date_min_dt is not None
            and end_date_max_dt is not None
            and end_date_min_dt > end_date_max_dt
        ):
            raise ValueError(
                f"end_date_min ({end_date_min_iso}) must be <= end_date_max ({end_date_max_iso})"
            )

        if max_pages == 0:
            return []

        endpoint = self._gamma_url("/markets")
        raw_items: list[Mapping[str, Any]] = []
        current_offset = offset
        pages_fetched = 0

        while True:
            params: dict[str, Any] = {"limit": limit, "offset": current_offset}
            if active is not None:
                params["active"] = "true" if active else "false"
            if closed is not None:
                params["closed"] = "true" if closed else "false"
            if archived is not None:
                params["archived"] = "true" if archived else "false"
            if order is not None and order.strip():
                params["order"] = order.strip()
            if ascending is not None:
                params["ascending"] = "true" if ascending else "false"
            if tag_id is not None:
                tag_str = _id_to_str(tag_id)
                if tag_str is not None:
                    params["tag_id"] = tag_str
            if slug is not None and slug.strip():
                params["slug"] = slug.strip()
            if end_date_min_iso is not None:
                params["end_date_min"] = end_date_min_iso
            if end_date_max_iso is not None:
                params["end_date_max"] = end_date_max_iso

            payload = await self._http.get_json(endpoint, params=params)
            if not isinstance(payload, list):
                raise ValueError(
                    f"expected JSON list from Gamma /markets, got {type(payload).__name__}"
                )

            if not payload:
                break

            for item in payload:
                if isinstance(item, Mapping):
                    raw_items.append(item)

            pages_fetched += 1
            if max_pages is not None and pages_fetched >= max_pages:
                break
            if len(payload) < limit:
                break
            current_offset += len(payload)

        return raw_items

    async def get_markets(
        self,
        *,
        active: bool | None = True,
        closed: bool | None = False,
        archived: bool | None = None,
        limit: int = DEFAULT_GAMMA_LIMIT,
        offset: int = 0,
        max_pages: int | None = None,
        order: str | None = None,
        ascending: bool | None = None,
        tag_id: int | str | None = None,
        slug: str | None = None,
        end_date_min: str | datetime | None = None,
        end_date_max: str | datetime | None = None,
    ) -> list[MarketMetadata]:
        """Paginate through Polymarket Gamma ``GET /markets`` with limit/offset pagination.

        Stops when an empty or short page (``len(page) < limit``) is returned, or
        once ``max_pages`` pages have been fetched. Deduplicates markets by
        ``conditionId`` across pages. Optional ``end_date_min`` and
        ``end_date_max`` bounds are serialized as ISO-8601 UTC timestamps.
        """
        raw_items = await self._fetch_gamma_markets_raw(
            active=active,
            closed=closed,
            archived=archived,
            limit=limit,
            offset=offset,
            max_pages=max_pages,
            order=order,
            ascending=ascending,
            tag_id=tag_id,
            slug=slug,
            end_date_min=end_date_min,
            end_date_max=end_date_max,
        )
        results: list[MarketMetadata] = []
        seen_conditions: set[str] = set()
        for item in raw_items:
            metadata = parse_gamma_market(item)
            if metadata.market_id not in seen_conditions:
                seen_conditions.add(metadata.market_id)
                results.append(metadata)
        return results

    async def get_resolved_markets(
        self,
        *,
        active: bool | None = None,
        closed: bool | None = True,
        archived: bool | None = None,
        limit: int = DEFAULT_GAMMA_LIMIT,
        offset: int = 0,
        max_pages: int | None = None,
        order: str | None = None,
        ascending: bool | None = None,
        tag_id: int | str | None = None,
        slug: str | None = None,
        end_date_min: str | datetime | None = None,
        end_date_max: str | datetime | None = None,
    ) -> list[ResolvedMarket]:
        """Fetch closed Gamma ``/markets`` and parse resolved binary ``ResolvedMarket`` objects.

        Delegates payload conversion strictly to :func:`parse_gamma_resolved_market`
        (leaving :func:`parse_gamma_market` untouched and anti-leakage clean),
        discarding open, non-binary, void, or ambiguously settled markets and
        deduplicating by ``conditionId``.
        """
        raw_items = await self._fetch_gamma_markets_raw(
            active=active,
            closed=closed,
            archived=archived,
            limit=limit,
            offset=offset,
            max_pages=max_pages,
            order=order,
            ascending=ascending,
            tag_id=tag_id,
            slug=slug,
            end_date_min=end_date_min,
            end_date_max=end_date_max,
        )
        results: list[ResolvedMarket] = []
        seen_conditions: set[str] = set()
        for item in raw_items:
            try:
                resolved = parse_gamma_resolved_market(item)
            except ValueError:
                logger.debug("Skipping malformed Gamma market payload during resolved discovery")
                continue
            if resolved is None:
                continue
            if resolved.metadata.market_id not in seen_conditions:
                seen_conditions.add(resolved.metadata.market_id)
                results.append(resolved)
        return results

    async def get_live_market_candidates(
        self,
        *,
        active: bool | None = True,
        closed: bool | None = False,
        archived: bool | None = None,
        limit: int = DEFAULT_GAMMA_LIMIT,
        offset: int = 0,
        max_pages: int | None = None,
        order: str | None = None,
        ascending: bool | None = None,
        tag_id: int | str | None = None,
        slug: str | None = None,
    ) -> list[LiveMarketCandidate]:
        """Fetch active Gamma ``/markets`` as live screening ``LiveMarketCandidate`` objects.

        Defaults to ``active=True, closed=False`` and supports an optional
        ``tag_id`` for targeted Weather/Economics series screening. Extracts
        ``volume24hr`` (USD notional) via :func:`parse_gamma_live_candidate`
        without exposing volume/liquidity on ``MarketMetadata.extra``.
        """
        raw_items = await self._fetch_gamma_markets_raw(
            active=active,
            closed=closed,
            archived=archived,
            limit=limit,
            offset=offset,
            max_pages=max_pages,
            order=order,
            ascending=ascending,
            tag_id=tag_id,
            slug=slug,
        )
        tag_str = _id_to_str(tag_id) if tag_id is not None else None
        extra_tags = (tag_str,) if tag_str is not None else None

        results: list[LiveMarketCandidate] = []
        seen_conditions: set[str] = set()
        for item in raw_items:
            candidate = parse_gamma_live_candidate(item, tag_ids=extra_tags)
            if candidate.metadata.market_id not in seen_conditions:
                seen_conditions.add(candidate.metadata.market_id)
                results.append(candidate)
        return results

    # ------------------------------------------------------------------
    # CLOB API
    # ------------------------------------------------------------------

    async def get_orderbook(
        self,
        token_id: str,
        *,
        max_depth: int = DEFAULT_ORDERBOOK_DEPTH,
        outcome_side: OutcomeSide = "yes",
        market_id: str | None = None,
        ts_utc: datetime | None = None,
    ) -> NormalizedOrderBook:
        """Fetch and normalize a single L2 orderbook via ``GET /book?token_id=...``."""
        clean_token_id = _non_empty_str(token_id)
        if clean_token_id is None:
            raise ValueError("token_id must be a non-empty string")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        explicit_ts = _ensure_utc(ts_utc, "ts_utc") if ts_utc is not None else None

        endpoint = self._clob_url("/book")
        payload = await self._http.get_json(endpoint, params={"token_id": clean_token_id})
        if not isinstance(payload, Mapping):
            raise ValueError(f"expected JSON object from CLOB /book, got {type(payload).__name__}")
        snapshot_ts = (
            explicit_ts if explicit_ts is not None else _ensure_utc(self._now_fn(), "ts_utc")
        )
        return parse_clob_orderbook(
            payload,
            token_id=clean_token_id,
            ts_utc=snapshot_ts,
            max_depth=max_depth,
            outcome_side=outcome_side,
            market_id=market_id,
        )

    async def _fetch_orderbooks_chunk(
        self,
        chunk: list[str],
        *,
        max_depth: int,
        outcome_side: OutcomeSide,
        outcome_sides: Mapping[str, OutcomeSide] | None,
        market_ids: Mapping[str, str] | None,
        explicit_ts_utc: datetime | None,
    ) -> dict[str, NormalizedOrderBook]:
        endpoint = self._clob_url("/books")
        body = [{"token_id": tid} for tid in chunk]
        response = await self._http.request("POST", endpoint, json=body)
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(
                f"expected JSON list from CLOB POST /books, got {type(payload).__name__}"
            )

        snapshot_ts = (
            explicit_ts_utc
            if explicit_ts_utc is not None
            else _ensure_utc(self._now_fn(), "ts_utc")
        )
        parsed: dict[str, NormalizedOrderBook] = {}
        for idx, entry in enumerate(payload):
            if not isinstance(entry, Mapping):
                continue
            entry_token_id = _id_to_str(entry.get("asset_id")) or _id_to_str(entry.get("token_id"))
            if entry_token_id is None and len(payload) == len(chunk):
                entry_token_id = chunk[idx]
            if entry_token_id is None:
                continue
            resolved_side = (
                outcome_sides[entry_token_id]
                if outcome_sides is not None and entry_token_id in outcome_sides
                else outcome_side
            )
            resolved_market_id = market_ids.get(entry_token_id) if market_ids is not None else None
            parsed[entry_token_id] = parse_clob_orderbook(
                entry,
                token_id=entry_token_id,
                ts_utc=snapshot_ts,
                max_depth=max_depth,
                outcome_side=resolved_side,
                market_id=resolved_market_id,
            )
        return parsed

    async def get_orderbooks_batch(
        self,
        token_ids: Sequence[str],
        *,
        max_depth: int = DEFAULT_ORDERBOOK_DEPTH,
        outcome_side: OutcomeSide = "yes",
        outcome_sides: Mapping[str, OutcomeSide] | None = None,
        market_ids: Mapping[str, str] | None = None,
        ts_utc: datetime | None = None,
        max_batch_size: int = POLYMARKET_MAX_BOOKS_BATCH,
    ) -> dict[str, NormalizedOrderBook]:
        """Fetch multiple L2 orderbooks via ``POST /books`` in deduplicated chunks.

        When fetching mixed YES and NO outcome tokens in a single batch, pass
        ``outcome_sides={token_id: "yes" | "no", ...}`` so each book preserves
        its true outcome side. Optionally pass ``market_ids={token_id: condition_id}``
        to override or supply ``NormalizedOrderBook.market_id`` when the CLOB
        response omits ``market``.
        """
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if max_batch_size < 1 or max_batch_size > POLYMARKET_MAX_BOOKS_BATCH:
            raise ValueError(
                f"max_batch_size must be in [1, {POLYMARKET_MAX_BOOKS_BATCH}], got {max_batch_size}"
            )
        explicit_ts = _ensure_utc(ts_utc, "ts_utc") if ts_utc is not None else None

        chunks = _chunk_ids(token_ids, max_batch_size)
        if not chunks:
            return {}

        if outcome_sides is not None:
            missing_sides = [tid for chunk in chunks for tid in chunk if tid not in outcome_sides]
            if missing_sides:
                raise ValueError(
                    f"outcome_sides missing mapping for token_id(s): {missing_sides[:5]}"
                )

        chunk_results = await asyncio.gather(
            *(
                self._fetch_orderbooks_chunk(
                    chunk,
                    max_depth=max_depth,
                    outcome_side=outcome_side,
                    outcome_sides=outcome_sides,
                    market_ids=market_ids,
                    explicit_ts_utc=explicit_ts,
                )
                for chunk in chunks
            )
        )
        merged: dict[str, NormalizedOrderBook] = {}
        for chunk_map in chunk_results:
            merged.update(chunk_map)
        return merged

    async def get_clob_price_history(
        self,
        token_id: str,
        start_ts: int | datetime | None = None,
        end_ts: int | datetime | None = None,
        fidelity: int | None = None,
        *,
        interval: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch historical prices from CLOB ``GET /prices-history``.

        CLOB expects ``market=<token_id>`` with either ``startTs``/``endTs`` or
        a relative ``interval`` preset, plus an optional ``fidelity`` resolution
        in minutes. Returned points have exact ``Decimal`` prices (``p``),
        deduplicated and sorted ascending by ``t``.
        """
        clean_token_id = _non_empty_str(token_id)
        if clean_token_id is None:
            raise ValueError("token_id must be a non-empty string")
        if fidelity is not None and fidelity < 1:
            raise ValueError(f"fidelity must be >= 1, got {fidelity}")

        clean_interval = _non_empty_str(interval)
        if clean_interval is not None and (start_ts is not None or end_ts is not None):
            raise ValueError(
                "CLOB /prices-history 'interval' cannot be combined with start_ts/end_ts"
            )

        start_epoch = _to_epoch_seconds(start_ts, "start_ts") if start_ts is not None else None
        end_epoch = _to_epoch_seconds(end_ts, "end_ts") if end_ts is not None else None
        if start_epoch is not None and end_epoch is not None and start_epoch > end_epoch:
            raise ValueError(f"start_ts ({start_epoch}) must be <= end_ts ({end_epoch})")

        params: dict[str, Any] = {"market": clean_token_id}
        if start_epoch is not None:
            params["startTs"] = start_epoch
        if end_epoch is not None:
            params["endTs"] = end_epoch
        if fidelity is not None:
            params["fidelity"] = fidelity
        if clean_interval is not None:
            params["interval"] = clean_interval

        endpoint = self._clob_url("/prices-history")
        payload = await self._http.get_json(endpoint, params=params)
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"expected JSON object from CLOB /prices-history, got {type(payload).__name__}"
            )
        raw_history = payload.get("history")
        if raw_history is None:
            return []
        if not isinstance(raw_history, list):
            raise ValueError(
                "expected 'history' list from CLOB /prices-history, "
                f"got {type(raw_history).__name__}"
            )

        mappings = [item for item in raw_history if isinstance(item, Mapping)]
        return _deduplicate_and_sort_history(mappings, timestamp_key="t", price_key="p")

    # ------------------------------------------------------------------
    # Data API v2
    # ------------------------------------------------------------------

    async def _fetch_trades_chunk(
        self,
        chunk: list[str],
        *,
        limit: int,
        max_pages: int | None,
    ) -> tuple[list[TradeRecord], bool]:
        endpoint = self._data_url("/trades")
        condition_param = ",".join(chunk)
        trades: list[TradeRecord] = []
        cursor: str | None = None
        pages_fetched = 0
        truncated = False

        while True:
            # Data API v2 invariants: ALWAYS use `condition=` (never `market=`),
            # ALWAYS force `taker_only=true`, and ALWAYS paginate via `cursor=`
            # from `pagination.next_cursor` (never `offset=`).
            params: dict[str, Any] = {
                "condition": condition_param,
                "taker_only": "true",
                "limit": limit,
            }
            if cursor is not None:
                params["cursor"] = cursor

            payload = await self._http.get_json(endpoint, params=params)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"expected JSON object from Data API v2 /trades, got {type(payload).__name__}"
                )

            raw_data = payload.get("data")
            if raw_data is None:
                raw_data = []
            if not isinstance(raw_data, list):
                raise ValueError(
                    f"expected 'data' list from Data API v2 /trades, got {type(raw_data).__name__}"
                )

            for item in raw_data:
                if not isinstance(item, Mapping):
                    continue
                trades.append(parse_v2_trade(item))

            pages_fetched += 1

            pagination = payload.get("pagination")
            has_next_page = False
            next_cursor: str | None = None
            if isinstance(pagination, Mapping) and pagination.get("has_more") is not False:
                candidate_cursor = _non_empty_str(pagination.get("next_cursor"))
                if candidate_cursor is not None and candidate_cursor != cursor:
                    has_next_page = True
                    next_cursor = candidate_cursor

            if not has_next_page:
                break
            if max_pages is not None and pages_fetched >= max_pages:
                truncated = True
                break
            cursor = next_cursor

        return trades, truncated

    async def get_trades_with_truncation(
        self,
        condition_ids: Sequence[str],
        max_pages: int | None = None,
        *,
        limit: int = DEFAULT_TRADES_LIMIT,
    ) -> tuple[list[TradeRecord], bool]:
        """Fetch taker-only trades from ``GET /v2/trades`` and report pagination truncation.

        Returns ``(trades, truncated)`` where ``truncated`` is ``True`` if at
        least one condition chunk stopped at ``max_pages`` while ``pagination``
        still indicated another page (``has_more`` not ``False`` with a new
        ``next_cursor``).
        """
        if max_pages is not None and max_pages < 0:
            raise ValueError(f"max_pages must be >= 0 or None, got {max_pages}")
        if limit < 1 or limit > POLYMARKET_MAX_TRADES_LIMIT:
            raise ValueError(f"limit must be in [1, {POLYMARKET_MAX_TRADES_LIMIT}], got {limit}")
        if max_pages == 0:
            return [], False

        chunks = _chunk_ids(condition_ids, POLYMARKET_MAX_CONDITION_BATCH)
        if not chunks:
            return [], False

        chunk_results = await asyncio.gather(
            *(self._fetch_trades_chunk(chunk, limit=limit, max_pages=max_pages) for chunk in chunks)
        )
        merged: list[TradeRecord] = []
        any_truncated = False
        for chunk_trades, chunk_truncated in chunk_results:
            merged.extend(chunk_trades)
            any_truncated = any_truncated or chunk_truncated
        return merged, any_truncated

    async def get_trades(
        self,
        condition_ids: Sequence[str],
        max_pages: int | None = None,
        *,
        limit: int = DEFAULT_TRADES_LIMIT,
    ) -> list[TradeRecord]:
        """Fetch taker-only historical trades from Data API v2 ``GET /v2/trades``.

        Enforces all Data API v2 invariants:
        - Deduplicates and splits ``condition_ids`` into chunks of at most 20 IDs.
        - Always sends ``condition=<csv>`` (never legacy ``market=``).
        - Always forces ``taker_only=true`` to prevent maker/taker double-counting.
        - Paginates via ``pagination.next_cursor`` sent back as ``cursor=``
          (never legacy ``offset=``), capped at ``max_pages`` pages per chunk.
        """
        trades, _ = await self.get_trades_with_truncation(
            condition_ids,
            max_pages=max_pages,
            limit=limit,
        )
        return trades

    async def _fetch_v2_history_block(
        self,
        token_id: str,
        *,
        start_epoch: int | None,
        end_epoch: int | None,
        interval: str | None,
        bucket_seconds: int | None,
    ) -> list[Mapping[str, Any]]:
        endpoint = self._data_url("/prices-history")
        block_points: list[Mapping[str, Any]] = []
        cursor: str | None = None

        while True:
            params: dict[str, Any] = {"token_id": token_id}
            if start_epoch is not None and end_epoch is not None:
                params["start"] = start_epoch
                params["end"] = end_epoch
            if interval is not None:
                params["interval"] = interval
            if bucket_seconds is not None:
                params["bucket_seconds"] = bucket_seconds
            if cursor is not None:
                params["cursor"] = cursor

            payload = await self._http.get_json(endpoint, params=params)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    "expected JSON object from Data API v2 /prices-history, "
                    f"got {type(payload).__name__}"
                )

            raw_data = payload.get("data")
            if raw_data is None:
                raw_data = []
            if not isinstance(raw_data, list):
                raise ValueError(
                    "expected 'data' list from Data API v2 /prices-history, "
                    f"got {type(raw_data).__name__}"
                )

            for item in raw_data:
                if isinstance(item, Mapping):
                    block_points.append(item)

            pagination = payload.get("pagination")
            if not isinstance(pagination, Mapping):
                break
            if pagination.get("has_more") is False:
                break
            next_cursor = _non_empty_str(pagination.get("next_cursor"))
            if next_cursor is None or next_cursor == cursor:
                break
            cursor = next_cursor

        if not block_points:
            logger.warning(
                "Polymarket Data API v2 /prices-history returned an empty block "
                "(possible resolution decay): token_id=%s start=%s end=%s interval=%s "
                "bucket_seconds=%s",
                token_id,
                start_epoch,
                end_epoch,
                interval,
                bucket_seconds,
            )

        return block_points

    async def get_v2_price_history(
        self,
        token_id: str,
        start_ts: int | datetime | None = None,
        end_ts: int | datetime | None = None,
        interval: str | None = None,
        *,
        bucket_seconds: int | None = None,
        exclude_settlement: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch historical prices from Data API v2 ``GET /v2/prices-history``.

        Enforces Data API v2 window exclusivity (either ``start_ts``/``end_ts`` or
        ``interval``, never both or neither) and ``bucket_seconds`` in
        ``[60, 86400]``. Automatically splits explicit ``[start_ts, end_ts]``
        windows spanning more than 15 days into contiguous blocks of at most 15
        days, logs a warning if any block returns empty (historical resolution
        decay), converts ``price`` to exact ``Decimal``, optionally drops
        non-tradable on-chain settlement points (``resolution_seconds == 0``),
        and returns points deduplicated and sorted ascending by ``timestamp``.
        """
        clean_token_id = _non_empty_str(token_id)
        if clean_token_id is None:
            raise ValueError("token_id must be a non-empty string")
        if bucket_seconds is not None and (
            isinstance(bucket_seconds, bool)
            or bucket_seconds < POLYMARKET_V2_BUCKET_MIN_SECONDS
            or bucket_seconds > POLYMARKET_V2_BUCKET_MAX_SECONDS
        ):
            raise ValueError(
                f"bucket_seconds must be in [{POLYMARKET_V2_BUCKET_MIN_SECONDS}, "
                f"{POLYMARKET_V2_BUCKET_MAX_SECONDS}], got {bucket_seconds}"
            )

        clean_interval = _non_empty_str(interval)
        has_explicit_window = start_ts is not None or end_ts is not None

        if clean_interval is not None and has_explicit_window:
            raise ValueError(
                "Data API v2 /prices-history 'interval' cannot be combined with start_ts/end_ts"
            )
        if (start_ts is None) != (end_ts is None):
            raise ValueError("start_ts and end_ts must either both be provided or both be None")
        if clean_interval is None and not has_explicit_window:
            raise ValueError(
                "Data API v2 /prices-history requires either 'interval' or both start_ts and end_ts"
            )

        if start_ts is not None and end_ts is not None:
            start_epoch = _to_epoch_seconds(start_ts, "start_ts")
            end_epoch = _to_epoch_seconds(end_ts, "end_ts")
            if start_epoch > end_epoch:
                raise ValueError(f"start_ts ({start_epoch}) must be <= end_ts ({end_epoch})")

            windows = _split_time_windows(
                start_epoch,
                end_epoch,
                max_span_seconds=POLYMARKET_V2_HISTORY_MAX_WINDOW_SECONDS,
            )
            block_lists = await asyncio.gather(
                *(
                    self._fetch_v2_history_block(
                        clean_token_id,
                        start_epoch=win_start,
                        end_epoch=win_end,
                        interval=None,
                        bucket_seconds=bucket_seconds,
                    )
                    for win_start, win_end in windows
                )
            )
            all_points: list[Mapping[str, Any]] = []
            for block in block_lists:
                all_points.extend(block)
            return _deduplicate_and_sort_history(
                all_points,
                timestamp_key="timestamp",
                price_key="price",
                exclude_settlement=exclude_settlement,
            )

        points = await self._fetch_v2_history_block(
            clean_token_id,
            start_epoch=None,
            end_epoch=None,
            interval=clean_interval,
            bucket_seconds=bucket_seconds,
        )
        return _deduplicate_and_sort_history(
            points,
            timestamp_key="timestamp",
            price_key="price",
            exclude_settlement=exclude_settlement,
        )
