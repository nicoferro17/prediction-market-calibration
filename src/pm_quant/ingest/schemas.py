"""Normalized venue-agnostic market data schemas for Polymarket and Kalshi."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Final, Literal

SCHEMA_VERSION: Final[str] = "v1"

Venue = Literal["polymarket", "kalshi"]
OutcomeSide = Literal["yes", "no"]
TradeSide = Literal["buy", "sell"]

_ZERO: Final[Decimal] = Decimal("0")
_ONE: Final[Decimal] = Decimal("1")
_TWO: Final[Decimal] = Decimal("2")
_EMPTY_EXTRA: Final[Mapping[str, Any]] = MappingProxyType({})


def _validate_probability_price(value: Decimal, field_name: str = "price") -> None:
    if not value.is_finite() or value < _ZERO or value > _ONE:
        raise ValueError(f"{field_name} must be a finite Decimal in [0, 1], got {value!r}")


def _validate_non_negative(value: Decimal, field_name: str) -> None:
    if not value.is_finite() or value < _ZERO:
        raise ValueError(f"{field_name} must be a finite non-negative Decimal, got {value!r}")


def _validate_tz_aware(dt: datetime, field_name: str) -> None:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware (UTC), got naive datetime {dt!r}")


@dataclass(frozen=True, slots=True)
class PriceLevel:
    """Single aggregated L2 price level with exact Decimal price and size."""

    price: Decimal
    size: Decimal

    def __post_init__(self) -> None:
        _validate_probability_price(self.price, "price")
        _validate_non_negative(self.size, "size")


@dataclass(frozen=True, slots=True)
class NormalizedOrderBook:
    """Venue-agnostic L2 orderbook snapshot for a single binary outcome side.

    By convention, ``bids`` are ordered descending by price (best bid first) and
    ``asks`` are ordered ascending by price (best ask first). Derived properties
    compute extrema directly so they remain safe even on empty, one-sided, or
    unsorted inputs.
    """

    venue: Venue
    market_id: str
    outcome_side: OutcomeSide
    ts_utc: datetime
    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_tz_aware(self.ts_utc, "ts_utc")

    @property
    def best_bid(self) -> Decimal | None:
        """Return the highest bid price, or ``None`` if the bid side is empty."""
        if not self.bids:
            return None
        return max(level.price for level in self.bids)

    @property
    def best_ask(self) -> Decimal | None:
        """Return the lowest ask price, or ``None`` if the ask side is empty."""
        if not self.asks:
            return None
        return min(level.price for level in self.asks)

    @property
    def mid(self) -> Decimal | None:
        """Return the midpoint price ``(best_bid + best_ask) / 2``, or ``None`` if one-sided."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / _TWO

    @property
    def spread(self) -> Decimal | None:
        """Return the inside spread ``best_ask - best_bid``, or ``None`` if one-sided."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid

    @property
    def is_crossed(self) -> bool:
        """Return ``True`` if ``best_bid > best_ask``; ``False`` for empty or one-sided books."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return False
        return bid > ask

    @property
    def depth_levels(self) -> tuple[int, int]:
        """Return ``(bid_level_count, ask_level_count)`` for the snapshot."""
        return (len(self.bids), len(self.asks))


@dataclass(frozen=True, slots=True)
class MarketMetadata:
    """Normalized market descriptor shared across venues."""

    venue: Venue
    market_id: str
    event_id: str | None
    title: str
    category: str | None
    outcomes: tuple[str, ...]
    is_yes_no: bool
    end_time_utc: datetime | None
    status: str
    extra: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_EXTRA)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.end_time_utc is not None:
            _validate_tz_aware(self.end_time_utc, "end_time_utc")
        if not isinstance(self.extra, MappingProxyType):
            object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """Normalized executed trade record."""

    venue: Venue
    market_id: str
    token_id: str | None
    price: Decimal
    size: Decimal
    notional_usd: Decimal
    ts_utc: datetime
    side: TradeSide | None
    tx_hash: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_probability_price(self.price, "price")
        _validate_non_negative(self.size, "size")
        _validate_non_negative(self.notional_usd, "notional_usd")
        _validate_tz_aware(self.ts_utc, "ts_utc")
