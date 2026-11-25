"""Historical point-in-time calibration schema, resolved-market parsers, and tau sampler.

This module defines the venue-agnostic data contract shared across Polymarket and
Kalshi historical backfills:

- **Anti-leakage separation**: :class:`ResolvedMarket` wraps
  :class:`pm_quant.ingest.schemas.MarketMetadata` without ever modifying
  :func:`pm_quant.ingest.polymarket.parse_gamma_market` or exposing Gamma final
  lifetime ``volume``/``liquidity``/quote fields. ``settlement_ts_utc`` is stored
  exclusively on ``ResolvedMarket`` for Kalshi ``/historical/cutoff`` archive
  routing and is never used to compute time-to-resolution (``tau_days``).
- **Deterministic variable-tau sampling**: :func:`sample_variable_tau_snapshot_ts`
  assigns exactly one point-in-time timestamp ``t_i`` per ``(venue, market_id)``
  via SHA-256 hashing over ``(seed, venue, market_id)``, enforcing a pre-resolution
  anti-oracle-delay buffer and covering ``tau < 1d`` alongside multi-day horizons.
- **Versioned Parquet storage**: ``HISTORICAL_CALIBRATION_SCHEMA_V1`` serializes
  records atomically with Zstandard compression and reads them back across schema
  versions via DuckDB ``read_parquet(..., union_by_name=true)``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final, Literal

import duckdb
import polars as pl

from pm_quant.ingest.schemas import SCHEMA_VERSION, MarketMetadata, Venue

BinaryOutcome = Literal[0, 1]
HistoricalOutcomeSide = Literal["yes", "outcome_0"]
HistoricalVolumeUnit = Literal["usd", "contracts_notional"]

DEFAULT_MAX_MARKETS_PER_VENUE: Final[int] = 500
DEFAULT_MIN_VOL_USD: Final[Decimal] = Decimal("1000")
DEFAULT_MIN_PRICE: Final[Decimal] = Decimal("0.01")
DEFAULT_MAX_PRICE: Final[Decimal] = Decimal("0.99")
DEFAULT_PRE_RESOLUTION_BUFFER_HOURS: Final[int] = 2
DEFAULT_BACKFILL_SEED: Final[int] = 42
DEFAULT_SHORT_TAU_TARGET_SHARE: Final[float] = 0.25
DEFAULT_FALLBACK_LOOKBACK_DAYS: Final[int] = 30
MAX_SAMPLER_TAU_DAYS: Final[int] = 180

_SECONDS_PER_DAY: Final[int] = 86_400
_MICROSECONDS_PER_DAY: Final[Decimal] = Decimal("86400000000")
_ZERO: Final[Decimal] = Decimal("0")
_ONE: Final[Decimal] = Decimal("1")
_PRICE_TAU_QUANT: Final[Decimal] = Decimal("0.000001")
_VOL_QUANT: Final[Decimal] = Decimal("0.01")
_UINT64_MODULUS: Final[float] = float(1 << 64)

_VALID_VENUES: Final[frozenset[str]] = frozenset({"polymarket", "kalshi"})
_VALID_OUTCOME_SIDES: Final[frozenset[str]] = frozenset({"yes", "outcome_0"})
_VALID_VOL_UNITS: Final[frozenset[str]] = frozenset({"usd", "contracts_notional"})
_CLOSED_GAMMA_STATUSES: Final[frozenset[str]] = frozenset({"closed", "archived", "resolved"})
_MULTI_DAY_TAU_BUCKETS: Final[tuple[tuple[float, float], ...]] = (
    (1.0, 3.0),
    (3.0, 7.0),
    (7.0, 30.0),
    (30.0, float(MAX_SAMPLER_TAU_DAYS)),
)

HISTORICAL_CALIBRATION_SCHEMA_V1: Final[pl.Schema] = pl.Schema(
    {
        "schema_version": pl.String(),
        "venue": pl.String(),
        "market_id": pl.String(),
        "token_id": pl.String(),
        "event_id": pl.String(),
        "title": pl.String(),
        "category": pl.String(),
        "outcome_side": pl.String(),
        "outcome_label": pl.String(),
        "is_yes_no": pl.Boolean(),
        "neg_risk": pl.Boolean(),
        "fee_type": pl.String(),
        "open_time_utc": pl.Datetime("us", "UTC"),
        "snapshot_ts_utc": pl.Datetime("us", "UTC"),
        "end_time_utc": pl.Datetime("us", "UTC"),
        "tau_days": pl.Decimal(12, 6),
        "log1p_tau": pl.Float64(),
        "p_mkt": pl.Decimal(12, 6),
        "vol_usd_le_t": pl.Decimal(18, 2),
        "vol_contracts_le_t": pl.Decimal(18, 2),
        "vol_unit": pl.String(),
        "trade_count_le_t": pl.UInt32(),
        "trades_truncated": pl.Boolean(),
        "price_source": pl.String(),
        "outcome_y": pl.UInt8(),
        "hist_yes_bid": pl.Decimal(12, 6),
        "hist_yes_ask": pl.Decimal(12, 6),
    }
)


# ---------------------------------------------------------------------------
# Internal pure validation & parsing helpers
# ---------------------------------------------------------------------------


def _normalize_utc(dt: datetime, field_name: str) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware (UTC), got naive datetime {dt!r}")
    return dt.astimezone(UTC)


def _validate_probability(value: Decimal, field_name: str) -> None:
    if not value.is_finite() or value < _ZERO or value > _ONE:
        raise ValueError(f"{field_name} must be a finite Decimal in [0, 1], got {value!r}")


def _validate_positive_decimal(value: Decimal, field_name: str) -> None:
    if not value.is_finite() or value <= _ZERO:
        raise ValueError(f"{field_name} must be a finite positive Decimal (> 0), got {value!r}")


def _validate_non_negative_decimal(value: Decimal, field_name: str) -> None:
    if not value.is_finite() or value < _ZERO:
        raise ValueError(f"{field_name} must be a finite non-negative Decimal, got {value!r}")


def _validate_binary_outcome(value: Any, field_name: str = "outcome_y") -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError(f"{field_name} must be strictly 0 or 1, got {value!r}")


def _non_empty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _to_decimal(value: Any) -> Decimal | None:
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
    """Parse an optional ISO-8601 timestamp into UTC, treating blank/zero values as ``None``."""
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


def _parse_optional_decimal_sequence(raw: Any) -> tuple[Decimal, ...] | None:
    """Parse a Python sequence or JSON-encoded list into finite ``Decimal`` values."""
    candidate: Any = raw
    if isinstance(candidate, str):
        text = candidate.strip()
        if not text:
            return None
        try:
            candidate = json.loads(text)
        except ValueError:
            return ()
    if candidate is None:
        return None
    if not isinstance(candidate, list | tuple):
        return ()

    decimals: list[Decimal] = []
    for item in candidate:
        parsed = _to_decimal(item)
        if parsed is None:
            return ()
        decimals.append(parsed)
    return tuple(decimals)


def _quantize_if_needed(value: Decimal | None, quant: Decimal, max_scale: int) -> Decimal | None:
    """Quantize a ``Decimal`` only if its fractional precision exceeds ``max_scale``."""
    if value is None:
        return None
    exp = value.as_tuple().exponent
    if isinstance(exp, int) and exp < -max_scale:
        return value.quantize(quant, rounding=ROUND_HALF_EVEN)
    return value


def compute_tau_metrics(snapshot_ts_utc: datetime, end_time_utc: datetime) -> tuple[Decimal, float]:
    """Compute exact ``(tau_days, log1p_tau)`` from UTC snapshot and market-end timestamps.

    Raises ``ValueError`` if either timestamp is naive or ``snapshot_ts_utc >= end_time_utc``.
    """
    snap = _normalize_utc(snapshot_ts_utc, "snapshot_ts_utc")
    end = _normalize_utc(end_time_utc, "end_time_utc")
    delta = end - snap
    delta_us = (delta.days * _SECONDS_PER_DAY + delta.seconds) * 1_000_000 + delta.microseconds
    if delta_us <= 0:
        raise ValueError(
            f"snapshot_ts_utc ({snap.isoformat()}) must be strictly before "
            f"end_time_utc ({end.isoformat()})"
        )
    tau_days = (Decimal(delta_us) / _MICROSECONDS_PER_DAY).quantize(
        _PRICE_TAU_QUANT,
        rounding=ROUND_HALF_EVEN,
    )
    if tau_days <= _ZERO:
        tau_days = _PRICE_TAU_QUANT
    log1p_tau = math.log1p(float(tau_days))
    return tau_days, log1p_tau


# ---------------------------------------------------------------------------
# Immutable dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BackfillPerimeter:
    """Finite, reproducible historical backfill perimeter and guard thresholds."""

    start_date_utc: datetime
    end_date_utc: datetime
    max_markets_per_venue: int = DEFAULT_MAX_MARKETS_PER_VENUE
    min_vol_usd: Decimal = DEFAULT_MIN_VOL_USD
    min_price: Decimal = DEFAULT_MIN_PRICE
    max_price: Decimal = DEFAULT_MAX_PRICE
    pre_resolution_buffer_hours: int = DEFAULT_PRE_RESOLUTION_BUFFER_HOURS
    seed: int = DEFAULT_BACKFILL_SEED
    short_tau_target_share: float = DEFAULT_SHORT_TAU_TARGET_SHARE

    def __post_init__(self) -> None:
        start_norm = _normalize_utc(self.start_date_utc, "start_date_utc")
        end_norm = _normalize_utc(self.end_date_utc, "end_date_utc")
        if start_norm >= end_norm:
            raise ValueError(
                f"start_date_utc ({start_norm.isoformat()}) must be strictly before "
                f"end_date_utc ({end_norm.isoformat()})"
            )
        object.__setattr__(self, "start_date_utc", start_norm)
        object.__setattr__(self, "end_date_utc", end_norm)

        if isinstance(self.max_markets_per_venue, bool) or self.max_markets_per_venue < 1:
            raise ValueError(
                f"max_markets_per_venue must be an integer >= 1, got {self.max_markets_per_venue!r}"
            )
        _validate_non_negative_decimal(self.min_vol_usd, "min_vol_usd")
        _validate_probability(self.min_price, "min_price")
        _validate_probability(self.max_price, "max_price")
        if self.min_price >= self.max_price:
            raise ValueError(
                f"min_price ({self.min_price}) must be strictly less than "
                f"max_price ({self.max_price})"
            )
        if (
            isinstance(self.pre_resolution_buffer_hours, bool)
            or self.pre_resolution_buffer_hours < 1
        ):
            raise ValueError(
                "pre_resolution_buffer_hours must be an integer >= 1, "
                f"got {self.pre_resolution_buffer_hours!r}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        if (
            isinstance(self.short_tau_target_share, bool)
            or not math.isfinite(self.short_tau_target_share)
            or self.short_tau_target_share < 0.0
            or self.short_tau_target_share > 1.0
        ):
            raise ValueError(
                "short_tau_target_share must be a finite float in [0.0, 1.0], "
                f"got {self.short_tau_target_share!r}"
            )


@dataclass(frozen=True, slots=True)
class ResolvedMarket:
    """Resolved binary market pairing clean ``MarketMetadata`` with its settled outcome.

    ``settlement_ts_utc`` is retained **only** for Kalshi ``/historical/cutoff``
    archive routing and must **never** be used to compute ``tau_days``.
    """

    metadata: MarketMetadata
    outcome_y: BinaryOutcome
    open_time_utc: datetime | None = None
    settlement_ts_utc: datetime | None = None

    def __post_init__(self) -> None:
        _validate_binary_outcome(self.outcome_y, "outcome_y")
        if self.open_time_utc is not None:
            object.__setattr__(
                self,
                "open_time_utc",
                _normalize_utc(self.open_time_utc, "open_time_utc"),
            )
        if self.settlement_ts_utc is not None:
            object.__setattr__(
                self,
                "settlement_ts_utc",
                _normalize_utc(self.settlement_ts_utc, "settlement_ts_utc"),
            )


@dataclass(frozen=True, slots=True)
class HistoricalSnapshotRecord:
    """Single point-in-time historical calibration observation (1 row per market).

    ``hist_yes_bid`` and ``hist_yes_ask`` are populated strictly for Pillar 2
    Kalshi historical execution cost modeling and must never be used as Pillar 1
    calibration features.
    """

    venue: Venue
    market_id: str
    token_id: str | None
    event_id: str | None
    title: str
    category: str | None
    outcome_side: HistoricalOutcomeSide
    outcome_label: str | None
    is_yes_no: bool
    neg_risk: bool | None
    open_time_utc: datetime | None
    snapshot_ts_utc: datetime
    end_time_utc: datetime
    tau_days: Decimal
    log1p_tau: float
    p_mkt: Decimal
    vol_usd_le_t: Decimal
    vol_contracts_le_t: Decimal | None
    vol_unit: HistoricalVolumeUnit
    trade_count_le_t: int | None
    price_source: str
    outcome_y: BinaryOutcome
    fee_type: str | None = None
    trades_truncated: bool = False
    hist_yes_bid: Decimal | None = None
    hist_yes_ask: Decimal | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if _non_empty_str(self.schema_version) is None:
            raise ValueError("schema_version must be a non-empty string")
        if self.venue not in _VALID_VENUES:
            raise ValueError(f"venue must be one of {sorted(_VALID_VENUES)}, got {self.venue!r}")
        if _non_empty_str(self.market_id) is None:
            raise ValueError("market_id must be a non-empty string")
        if self.outcome_side not in _VALID_OUTCOME_SIDES:
            raise ValueError(
                f"outcome_side must be one of {sorted(_VALID_OUTCOME_SIDES)}, "
                f"got {self.outcome_side!r}"
            )
        if self.open_time_utc is not None:
            object.__setattr__(
                self,
                "open_time_utc",
                _normalize_utc(self.open_time_utc, "open_time_utc"),
            )
        snap_norm = _normalize_utc(self.snapshot_ts_utc, "snapshot_ts_utc")
        end_norm = _normalize_utc(self.end_time_utc, "end_time_utc")
        if snap_norm >= end_norm:
            raise ValueError(
                f"snapshot_ts_utc ({snap_norm.isoformat()}) must be strictly before "
                f"end_time_utc ({end_norm.isoformat()})"
            )
        object.__setattr__(self, "snapshot_ts_utc", snap_norm)
        object.__setattr__(self, "end_time_utc", end_norm)

        _validate_positive_decimal(self.tau_days, "tau_days")
        if (
            isinstance(self.log1p_tau, bool)
            or not math.isfinite(self.log1p_tau)
            or self.log1p_tau <= 0.0
        ):
            raise ValueError(f"log1p_tau must be a finite positive float, got {self.log1p_tau!r}")
        _validate_probability(self.p_mkt, "p_mkt")
        _validate_non_negative_decimal(self.vol_usd_le_t, "vol_usd_le_t")
        if self.vol_contracts_le_t is not None:
            _validate_non_negative_decimal(self.vol_contracts_le_t, "vol_contracts_le_t")
        if self.vol_unit not in _VALID_VOL_UNITS:
            raise ValueError(
                f"vol_unit must be one of {sorted(_VALID_VOL_UNITS)}, got {self.vol_unit!r}"
            )
        if self.trade_count_le_t is not None and (
            isinstance(self.trade_count_le_t, bool) or self.trade_count_le_t < 0
        ):
            raise ValueError(
                "trade_count_le_t must be None or a non-negative int, "
                f"got {self.trade_count_le_t!r}"
            )
        if _non_empty_str(self.price_source) is None:
            raise ValueError("price_source must be a non-empty string")
        _validate_binary_outcome(self.outcome_y, "outcome_y")
        if self.hist_yes_bid is not None:
            _validate_probability(self.hist_yes_bid, "hist_yes_bid")
        if self.hist_yes_ask is not None:
            _validate_probability(self.hist_yes_ask, "hist_yes_ask")


# ---------------------------------------------------------------------------
# Pure resolved-market parsers (zero I/O, anti-leakage)
# ---------------------------------------------------------------------------


def _canonical_outcome_index(metadata: MarketMetadata) -> int:
    """Return the index of the canonical tracked outcome (``'yes'`` for YES/NO, else ``0``)."""
    if metadata.is_yes_no:
        for idx, label in enumerate(metadata.outcomes):
            if label.strip().lower() == "yes":
                return idx
    return 0


def _extract_gamma_outcome_y_from_tokens(
    raw_tokens: Any,
    outcomes: tuple[str, ...],
    canonical_idx: int,
) -> BinaryOutcome | None:
    """Extract unambiguous binary outcome from Gamma/CLOB ``tokens[*].winner`` entries."""
    if not isinstance(raw_tokens, list | tuple) or len(raw_tokens) != 2:
        return None
    if not all(isinstance(entry, Mapping) for entry in raw_tokens):
        return None

    winners = [entry.get("winner") for entry in raw_tokens]
    if not all(isinstance(flag, bool) for flag in winners):
        return None
    if sum(1 for flag in winners if flag is True) != 1:
        return None

    token_labels = [_non_empty_str(entry.get("outcome")) for entry in raw_tokens]
    if all(label is not None for label in token_labels):
        lowered_tokens = [label.lower() for label in token_labels if label is not None]
        lowered_outcomes = [label.strip().lower() for label in outcomes]
        if set(lowered_tokens) == set(lowered_outcomes) and len(set(lowered_tokens)) == 2:
            canonical_label = lowered_outcomes[canonical_idx]
            for label_lower, is_winner in zip(lowered_tokens, winners, strict=True):
                if label_lower == canonical_label:
                    return 1 if is_winner else 0
        return None

    if any(label is not None for label in token_labels):
        return None

    return 1 if winners[canonical_idx] is True else 0


def parse_gamma_resolved_market(raw: Mapping[str, Any]) -> ResolvedMarket | None:
    """Parse a closed, unambiguously resolved Polymarket Gamma market into ``ResolvedMarket``.

    Delegates metadata normalization strictly to
    :func:`pm_quant.ingest.polymarket.parse_gamma_market` so ``MarketMetadata``
    never exposes Gamma lifetime ``volume``/``liquidity``/quote fields. Returns
    ``None`` for open, non-binary, 50-50, void, or ambiguously resolved markets.
    """
    from pm_quant.ingest.polymarket import parse_gamma_market

    metadata = parse_gamma_market(raw)

    if metadata.extra.get("closed") is False:
        return None
    is_closed = metadata.extra.get("closed") is True or metadata.status in _CLOSED_GAMMA_STATUSES
    if not is_closed:
        return None

    if len(metadata.outcomes) != 2 or any(not label.strip() for label in metadata.outcomes):
        return None

    canonical_idx = _canonical_outcome_index(metadata)
    outcome_y: BinaryOutcome | None = None

    raw_prices = raw.get("outcomePrices") if "outcomePrices" in raw else raw.get("outcome_prices")
    parsed_prices = _parse_optional_decimal_sequence(raw_prices)
    if parsed_prices is not None:
        if len(parsed_prices) != 2:
            return None
        p0, p1 = parsed_prices
        if p0 == _ONE and p1 == _ZERO:
            winner_idx = 0
        elif p0 == _ZERO and p1 == _ONE:
            winner_idx = 1
        else:
            # Ambiguous, 0.5/0.5, void (0/0), or unsettled split -> discard.
            return None
        outcome_y = 1 if winner_idx == canonical_idx else 0
    else:
        outcome_y = _extract_gamma_outcome_y_from_tokens(
            raw.get("tokens"),
            metadata.outcomes,
            canonical_idx,
        )

    if outcome_y is None:
        return None

    open_time_utc = (
        _parse_optional_iso_timestamp(raw.get("startDate"))
        or _parse_optional_iso_timestamp(raw.get("startDateIso"))
        or _parse_optional_iso_timestamp(raw.get("createdAt"))
        or _parse_optional_iso_timestamp(raw.get("acceptingOrdersTimestamp"))
    )

    return ResolvedMarket(
        metadata=metadata,
        outcome_y=outcome_y,
        open_time_utc=open_time_utc,
        settlement_ts_utc=None,
    )


def parse_kalshi_resolved_market(raw: Mapping[str, Any]) -> ResolvedMarket | None:
    """Parse a settled binary Kalshi market payload into ``ResolvedMarket``.

    Delegates metadata normalization strictly to
    :func:`pm_quant.ingest.kalshi.parse_kalshi_market` (where ``end_time_utc``
    comes exclusively from ``close_time`` / expiration fields, never
    ``settlement_ts``). Accepts strictly ``result in {'yes', 'no'}`` and stores
    ``settlement_ts`` only on ``ResolvedMarket.settlement_ts_utc`` for
    ``/historical/cutoff`` archive routing.
    """
    from pm_quant.ingest.kalshi import parse_kalshi_market

    metadata = parse_kalshi_market(raw)
    if not metadata.is_yes_no or len(metadata.outcomes) != 2:
        return None

    raw_result = _non_empty_str(raw.get("result"))
    if raw_result is None:
        return None
    lowered_result = raw_result.lower()
    if lowered_result == "yes":
        outcome_y: BinaryOutcome = 1
    elif lowered_result == "no":
        outcome_y = 0
    else:
        # Discard void, scalar, all_no, or any non-binary settlement result.
        return None

    open_time_utc = _parse_optional_iso_timestamp(
        raw.get("open_time")
    ) or _parse_optional_iso_timestamp(raw.get("created_time"))
    settlement_ts_utc = _parse_optional_iso_timestamp(raw.get("settlement_ts"))

    return ResolvedMarket(
        metadata=metadata,
        outcome_y=outcome_y,
        open_time_utc=open_time_utc,
        settlement_ts_utc=settlement_ts_utc,
    )


# ---------------------------------------------------------------------------
# Deterministic SHA-256 variable-tau snapshot sampler
# ---------------------------------------------------------------------------


def sample_variable_tau_snapshot_ts(
    market: ResolvedMarket,
    perimeter: BackfillPerimeter,
) -> datetime | None:
    """Deterministically sample 1 variable-tau snapshot timestamp for ``market``.

    Uses SHA-256 over ``f"{perimeter.seed}:{venue}:{market_id}"`` (independent of
    call order and ``PYTHONHASHSEED``) to draw a single UTC timestamp
    ``t_i <= end_time_utc - pre_resolution_buffer_hours`` across log-stratified
    ``tau`` buckets:

    - Short-lived markets (``lifetime <= 24h``) sample directly in ``tau < 1d``
      between ``open_time_utc`` and ``t_max``.
    - Multi-day markets allocate ``perimeter.short_tau_target_share`` to the
      ``[buffer, 1d)`` bucket and distribute the remaining weight uniformly across
      feasible multi-day buckets (``1-3d``, ``3-7d``, ``7-30d``, ``>30d``),
      sampling log-uniformly in ``log(1 + tau)`` within the chosen bucket.
    - When ``open_time_utc`` is absent, falls back to a conservative
      ``DEFAULT_FALLBACK_LOOKBACK_DAYS`` (30-day) window prior to ``t_max``.
    """
    end_time = market.metadata.end_time_utc
    if end_time is None:
        return None

    buffer = timedelta(hours=perimeter.pre_resolution_buffer_hours)
    t_max = end_time - buffer

    if market.open_time_utc is not None:
        t_min = market.open_time_utc
        if t_min >= t_max:
            return None
    else:
        t_min = t_max - timedelta(days=DEFAULT_FALLBACK_LOOKBACK_DAYS)

    tau_min_s = float(buffer.total_seconds())
    tau_max_s = float((end_time - t_min).total_seconds())
    if tau_max_s <= tau_min_s:
        return None

    tau_min_days = tau_min_s / _SECONDS_PER_DAY
    capped_max_tau_days = min(tau_max_s / _SECONDS_PER_DAY, float(MAX_SAMPLER_TAU_DAYS))
    if capped_max_tau_days <= tau_min_days:
        return t_max

    key = f"{perimeter.seed}:{market.metadata.venue}:{market.metadata.market_id}".encode()
    digest = hashlib.sha256(key).digest()
    u_bucket = int.from_bytes(digest[0:8], "big") / _UINT64_MODULUS
    u_within = int.from_bytes(digest[8:16], "big") / _UINT64_MODULUS

    short_hi = min(1.0, capped_max_tau_days)
    short_bucket: tuple[float, float] | None = (
        (tau_min_days, short_hi) if tau_min_days < short_hi else None
    )

    multi_buckets: list[tuple[float, float]] = []
    for lo, hi in _MULTI_DAY_TAU_BUCKETS:
        eff_lo = max(lo, tau_min_days)
        eff_hi = min(hi, capped_max_tau_days)
        if eff_lo < eff_hi:
            multi_buckets.append((eff_lo, eff_hi))

    is_short_choice = False
    if short_bucket is not None and not multi_buckets:
        chosen_lo, chosen_hi = short_bucket
        is_short_choice = True
    elif short_bucket is None and multi_buckets:
        idx = min(len(multi_buckets) - 1, int(u_bucket * len(multi_buckets)))
        chosen_lo, chosen_hi = multi_buckets[idx]
    elif short_bucket is not None and multi_buckets:
        w_short = perimeter.short_tau_target_share
        if w_short >= 1.0 or u_bucket < w_short:
            chosen_lo, chosen_hi = short_bucket
            is_short_choice = True
        else:
            rem_u = (u_bucket - w_short) / (1.0 - w_short)
            idx = min(len(multi_buckets) - 1, int(rem_u * len(multi_buckets)))
            chosen_lo, chosen_hi = multi_buckets[idx]
    else:
        return t_max

    log_lo = math.log1p(chosen_lo)
    log_hi = math.log1p(chosen_hi)
    sampled_tau_days = math.expm1(log_lo + u_within * (log_hi - log_lo))

    min_sec = int(math.ceil(tau_min_s))
    max_sec = int(math.floor(min(tau_max_s, float(MAX_SAMPLER_TAU_DAYS * _SECONDS_PER_DAY))))
    upper_sec = max_sec
    if is_short_choice and min_sec < _SECONDS_PER_DAY:
        # Preserve strictly tau_days < 1.0 when rounding near the 24h boundary.
        upper_sec = min(max_sec, _SECONDS_PER_DAY - 1)

    if upper_sec < min_sec:
        tau_seconds = min_sec
    else:
        raw_sec = int(round(sampled_tau_days * _SECONDS_PER_DAY))
        tau_seconds = max(min_sec, min(raw_sec, upper_sec))

    candidate_ts = end_time - timedelta(seconds=tau_seconds)
    if candidate_ts < t_min:
        candidate_ts = t_min
    if candidate_ts > t_max:
        candidate_ts = t_max
    return candidate_ts


# ---------------------------------------------------------------------------
# Polars serialization, atomic Zstd Parquet write, and DuckDB union_by_name read
# ---------------------------------------------------------------------------


def historical_records_to_frame(
    records: Iterable[HistoricalSnapshotRecord],
) -> pl.DataFrame:
    """Convert historical snapshot records into a strictly typed Polars DataFrame.

    Enforces deterministic ordering by ``(venue, end_time_utc, market_id)`` and
    raises ``ValueError`` immediately if duplicate ``(venue, market_id)`` keys
    are present (hard guard against duplicating ``y_i`` in the likelihood).
    """
    sorted_records = sorted(
        records,
        key=lambda rec: (rec.venue, rec.end_time_utc, rec.market_id),
    )

    if not sorted_records:
        return pl.DataFrame(schema=HISTORICAL_CALIBRATION_SCHEMA_V1)

    seen_keys: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []

    for rec in sorted_records:
        row_key = (rec.venue, rec.market_id)
        if row_key in seen_keys:
            raise ValueError(f"Duplicate historical snapshot key (venue, market_id): {row_key!r}")
        seen_keys.add(row_key)

        rows.append(
            {
                "schema_version": rec.schema_version,
                "venue": rec.venue,
                "market_id": rec.market_id,
                "token_id": rec.token_id,
                "event_id": rec.event_id,
                "title": rec.title,
                "category": rec.category,
                "outcome_side": rec.outcome_side,
                "outcome_label": rec.outcome_label,
                "is_yes_no": rec.is_yes_no,
                "neg_risk": rec.neg_risk,
                "fee_type": rec.fee_type,
                "open_time_utc": rec.open_time_utc,
                "snapshot_ts_utc": rec.snapshot_ts_utc,
                "end_time_utc": rec.end_time_utc,
                "tau_days": _quantize_if_needed(rec.tau_days, _PRICE_TAU_QUANT, 6),
                "log1p_tau": rec.log1p_tau,
                "p_mkt": _quantize_if_needed(rec.p_mkt, _PRICE_TAU_QUANT, 6),
                "vol_usd_le_t": _quantize_if_needed(rec.vol_usd_le_t, _VOL_QUANT, 2),
                "vol_contracts_le_t": _quantize_if_needed(
                    rec.vol_contracts_le_t,
                    _VOL_QUANT,
                    2,
                ),
                "vol_unit": rec.vol_unit,
                "trade_count_le_t": rec.trade_count_le_t,
                "trades_truncated": rec.trades_truncated,
                "price_source": rec.price_source,
                "outcome_y": rec.outcome_y,
                "hist_yes_bid": _quantize_if_needed(rec.hist_yes_bid, _PRICE_TAU_QUANT, 6),
                "hist_yes_ask": _quantize_if_needed(rec.hist_yes_ask, _PRICE_TAU_QUANT, 6),
            }
        )

    return pl.DataFrame(rows, schema=HISTORICAL_CALIBRATION_SCHEMA_V1)


def write_historical_parquet(
    records: Iterable[HistoricalSnapshotRecord],
    output_path: str | Path,
) -> Path:
    """Atomically write historical snapshots to ``output_path`` with Zstandard compression."""
    frame = historical_records_to_frame(records)
    if frame["schema_version"].null_count() > 0:
        raise ValueError("schema_version column must not contain null values")

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f".{target_path.stem}.",
        suffix=".parquet.tmp",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        frame.write_parquet(tmp_path, compression="zstd")
        os.replace(tmp_path, target_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return target_path


def read_historical_duckdb(parquet_glob: str | Path) -> pl.DataFrame:
    """Read one or more historical Parquet files via DuckDB ``union_by_name=true``."""
    with duckdb.connect(":memory:") as con:
        con.execute("SET TimeZone = 'UTC'")
        rel = con.sql(
            "SELECT * FROM read_parquet(?, union_by_name=true)",
            params=[str(parquet_glob)],
        )
        return pl.DataFrame(rel)
