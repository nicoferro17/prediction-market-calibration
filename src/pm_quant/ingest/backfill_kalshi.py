"""Point-in-time historical backfill pipeline for settled Kalshi binary markets.

This module implements the Week 3 Block 3 Kalshi historical ingestion workflow
with strict anti-leakage guarantees and ``/historical/cutoff`` archive routing:

- **Dual discovery with server-side MVE exclusion**: queries both
  ``GET /markets?status=settled&mve_filter=exclude`` (bounded by
  ``min_close_ts``/``max_close_ts``) and
  ``GET /historical/markets?mve_filter=exclude`` (respecting Kalshi's mutual
  exclusivity on archive filters), plus targeted Weather/Economics series scans
  when within ``max_live_pages`` budget.
- **Deterministic balanced stratification**: deduplicates by ticker and reuses
  :func:`pm_quant.ingest.backfill_polymarket.stratify_resolved_markets` with
  :func:`is_eligible_resolved_kalshi` (excluding combo/MVE and 5m/15m crypto).
- **Cutoff-guided candlestick routing**: calls ``GET /historical/cutoff`` once
  and routes post-cutoff markets (``settlement_ts >= cutoff`` or sourced from
  ``/markets``) to ``GET /markets/candlesticks`` batch (``<= 100`` tickers,
  calibrated ``max_batch_size`` under the 10,000-period cap) and pre-cutoff
  markets (``settlement_ts < cutoff`` or sourced from ``/historical/markets``)
  to ``GET /historical/markets/{ticker}/candlesticks`` (``period_interval`` in
  ``{60, 1440}``, automatically stepping up to ``1440`` above 5,000 hours).
- **Strict point-in-time extraction (zero lookahead)**:
  :func:`extract_kalshi_point_in_time` discards every candle with
  ``end_period_ts > snapshot_ts_utc``, extracts ``p_mkt`` from the latest valid
  bivariate ``(yes_bid + yes_ask) / 2`` (``price_source='kalshi_mid'``, storing
  ``hist_yes_bid``/``hist_yes_ask`` strictly for Pillar 2) with fallback to
  ``price_close`` (``price_source='kalshi_close'``), and computes
  ``vol_contracts_le_t`` and ``vol_usd_le_t`` exclusively from candles
  ``<= snapshot_ts_utc`` — never from the market object's final lifetime volume.
- **Resumable append-only JSONL checkpoint**: shares the identical checkpoint
  format and crash-safe resume semantics with
  :func:`pm_quant.ingest.backfill_polymarket.backfill_polymarket`.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final, Literal

import httpx

from pm_quant.ingest.backfill_polymarket import (
    DEFAULT_BACKFILL_MAX_CONCURRENCY,
    DEFAULT_OVERSAMPLE_FACTOR,
    _append_checkpoint_entry,
    _load_checkpoint,
    compute_perimeter_hash,
    stratify_resolved_markets,
)
from pm_quant.ingest.collector import DEFAULT_KALSHI_WEATHER_SERIES_BY_PREFIX
from pm_quant.ingest.historical_schema import (
    DEFAULT_FALLBACK_LOOKBACK_DAYS,
    BackfillPerimeter,
    HistoricalSnapshotRecord,
    ResolvedMarket,
    compute_tau_metrics,
    sample_variable_tau_snapshot_ts,
)
from pm_quant.ingest.kalshi import (
    KALSHI_MAX_BATCH_TICKERS,
    KALSHI_MAX_PAGE_LIMIT,
    KalshiClient,
)
from pm_quant.ingest.schemas import MarketMetadata
from pm_quant.ingest.universe import (
    DEFAULT_MACRO_SERIES_PREFIXES,
    DEFAULT_WEATHER_SERIES_PREFIXES,
    is_short_term_crypto,
)

logger = logging.getLogger(__name__)

DEFAULT_KALSHI_MAX_LIVE_PAGES: Final[int] = 20
DEFAULT_KALSHI_MAX_ARCHIVED_PAGES: Final[int] = 20
DEFAULT_KALSHI_BACKFILL_INTERVAL: Final[int] = 60
KALSHI_BACKFILL_ALLOWED_INTERVALS: Final[frozenset[int]] = frozenset({60, 1440})
KALSHI_BATCH_CANDLESTICKS_MAX_PERIODS: Final[int] = 10_000
KALSHI_HISTORICAL_CANDLESTICKS_MAX_PERIODS: Final[int] = 5_000

_ZERO: Final[Decimal] = Decimal("0")
_ONE: Final[Decimal] = Decimal("1")
_TWO: Final[Decimal] = Decimal("2")
_HUNDRED: Final[Decimal] = Decimal("100")

_COMBO_TICKER_PREFIXES: Final[tuple[str, ...]] = ("KXMVE", "KXMULTI")
_COMBO_TEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[^a-z0-9])"
    r"(?:combo|combos|parlay|parlays|same[-_ ]?game[-_ ]?parlay|multi[-_ ]?leg|multileg)"
    r"(?:$|[^a-z0-9])",
    re.IGNORECASE,
)

KalshiPriceSource = Literal["kalshi_mid", "kalshi_close"]
_MarketSource = Literal["live", "archived"]


# ---------------------------------------------------------------------------
# Pure helpers & immutable candlestick models (zero I/O)
# ---------------------------------------------------------------------------


def _ensure_utc(dt: datetime, field_name: str) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware (UTC), got naive datetime {dt!r}")
    return dt.astimezone(UTC)


def _non_empty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _extra_str(meta: MarketMetadata, key: str) -> str | None:
    return _non_empty_str(meta.extra.get(key))


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


def _parse_candle_timestamp_utc(raw: Any) -> datetime | None:
    """Parse a candlestick ``end_period_ts`` (epoch s/ms, ISO-8601, or aware datetime) into UTC."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, datetime):
        if raw.tzinfo is None or raw.utcoffset() is None:
            return None
        return raw.astimezone(UTC)
    if isinstance(raw, int | float):
        if not math.isfinite(raw):
            return None
        seconds = raw / 1000.0 if abs(raw) >= 1e12 else float(raw)
        try:
            return datetime.fromtimestamp(seconds, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            numeric = math.nan
        if math.isfinite(numeric):
            seconds = numeric / 1000.0 if abs(numeric) >= 1e12 else numeric
            try:
                return datetime.fromtimestamp(seconds, tz=UTC)
            except (OverflowError, OSError, ValueError):
                return None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _parse_dollar_price(raw: Any) -> Decimal | None:
    """Parse a fixed-point dollar price string/number strictly in ``[0, 1]``."""
    dec = _to_decimal(raw)
    if dec is None or dec < _ZERO or dec > _ONE:
        return None
    return dec


def _parse_cents_or_dollar_price(raw: Any) -> Decimal | None:
    """Parse a Kalshi integer-cents (``0..100``) or dollar (``[0, 1]``) price into ``[0, 1]``."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        if raw < 0 or raw > 100:
            return None
        return Decimal(raw) / _HUNDRED
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        dec = _to_decimal(text)
        if dec is None or dec < _ZERO or dec > _HUNDRED:
            return None
        if "." not in text and "e" not in text.lower():
            return dec / _HUNDRED
        if dec <= _ONE:
            return dec
        return dec / _HUNDRED
    dec = _to_decimal(raw)
    if dec is None or dec < _ZERO or dec > _HUNDRED:
        return None
    if dec <= _ONE:
        return dec
    return dec / _HUNDRED


def _extract_candle_field_price(
    raw: Mapping[str, Any],
    *,
    nested_key: str,
    flat_dollars_key: str,
    flat_cents_key: str,
) -> Decimal | None:
    """Extract a normalized ``[0, 1]`` close price from nested or flat Kalshi candlestick fields."""
    nested = raw.get(nested_key)
    if isinstance(nested, Mapping):
        if nested.get("close_dollars") is not None:
            return _parse_dollar_price(nested.get("close_dollars"))
        if nested.get("close") is not None:
            return _parse_cents_or_dollar_price(nested.get("close"))

    if raw.get(flat_dollars_key) is not None:
        return _parse_dollar_price(raw.get(flat_dollars_key))
    if raw.get(flat_cents_key) is not None:
        return _parse_cents_or_dollar_price(raw.get(flat_cents_key))
    return None


@dataclass(frozen=True, slots=True)
class KalshiCandle:
    """Normalized Kalshi OHLCV candlestick with UTC timestamp and ``[0, 1]`` Decimal prices."""

    end_period_ts: datetime
    yes_bid_close: Decimal | None
    yes_ask_close: Decimal | None
    price_close: Decimal | None
    volume_contracts: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "end_period_ts",
            _ensure_utc(self.end_period_ts, "end_period_ts"),
        )
        for field_name in ("yes_bid_close", "yes_ask_close", "price_close"):
            val = getattr(self, field_name)
            if val is not None and (not val.is_finite() or val < _ZERO or val > _ONE):
                raise ValueError(f"{field_name} must be in [0, 1], got {val!r}")
        if not self.volume_contracts.is_finite() or self.volume_contracts < _ZERO:
            raise ValueError(
                f"volume_contracts must be a finite Decimal >= 0, got {self.volume_contracts!r}"
            )

    @property
    def has_valid_bivariate_quote(self) -> bool:
        """Return ``True`` when both YES bid and ask are present and non-crossed."""
        return (
            self.yes_bid_close is not None
            and self.yes_ask_close is not None
            and self.yes_bid_close <= self.yes_ask_close
        )

    @property
    def mid_price(self) -> Decimal | None:
        """Return ``(yes_bid_close + yes_ask_close) / 2`` for valid bivariate candles."""
        if not self.has_valid_bivariate_quote:
            return None
        assert self.yes_bid_close is not None and self.yes_ask_close is not None
        return (self.yes_bid_close + self.yes_ask_close) / _TWO

    @property
    def effective_price(self) -> Decimal | None:
        """Best point-in-time contract price on this candle for notional volume weighting."""
        mid = self.mid_price
        if mid is not None:
            return mid
        if self.price_close is not None:
            return self.price_close
        if self.yes_bid_close is not None:
            return self.yes_bid_close
        if self.yes_ask_close is not None:
            return self.yes_ask_close
        return None


@dataclass(frozen=True, slots=True)
class KalshiPointInTime:
    """Point-in-time price and cumulative volume extracted strictly from candles ``<= t_i``."""

    p_mkt: Decimal
    price_source: KalshiPriceSource
    hist_yes_bid: Decimal | None
    hist_yes_ask: Decimal | None
    vol_contracts_le_t: Decimal
    vol_usd_le_t: Decimal
    candle_count_le_t: int


def parse_kalshi_candlestick(raw: Mapping[str, Any]) -> KalshiCandle | None:
    """Parse a raw Kalshi candlestick object into a normalized :class:`KalshiCandle`.

    Supports both modern fixed-point ``*_dollars`` strings (e.g. ``"0.5200"``)
    and legacy integer cents ``0..100`` (divided by 100) across nested
    (``yes_bid.close_dollars``/``yes_bid.close``) and flat field layouts, as
    well as ``volume_fp`` / ``volume`` contract counts. Returns ``None`` if
    ``end_period_ts`` is missing/invalid or volume is negative/malformed.
    """
    end_period_ts = _parse_candle_timestamp_utc(raw.get("end_period_ts"))
    if end_period_ts is None:
        return None

    yes_bid_close = _extract_candle_field_price(
        raw,
        nested_key="yes_bid",
        flat_dollars_key="yes_bid_close_dollars",
        flat_cents_key="yes_bid_close",
    )
    yes_ask_close = _extract_candle_field_price(
        raw,
        nested_key="yes_ask",
        flat_dollars_key="yes_ask_close_dollars",
        flat_cents_key="yes_ask_close",
    )
    price_close = _extract_candle_field_price(
        raw,
        nested_key="price",
        flat_dollars_key="price_close_dollars",
        flat_cents_key="price_close",
    )

    raw_vol: Any = None
    if raw.get("volume_fp") is not None:
        raw_vol = raw.get("volume_fp")
    elif raw.get("volume") is not None:
        raw_vol = raw.get("volume")

    if raw_vol is None:
        volume_contracts = _ZERO
    else:
        parsed_vol = _to_decimal(raw_vol)
        if parsed_vol is None or parsed_vol < _ZERO:
            return None
        volume_contracts = parsed_vol

    return KalshiCandle(
        end_period_ts=end_period_ts,
        yes_bid_close=yes_bid_close,
        yes_ask_close=yes_ask_close,
        price_close=price_close,
        volume_contracts=volume_contracts,
    )


def extract_kalshi_point_in_time(
    candles: Sequence[KalshiCandle | Mapping[str, Any]],
    snapshot_ts_utc: datetime,
) -> KalshiPointInTime | None:
    """Extract strict point-in-time price and cumulative volume for ``end_period_ts <= t_i``.

    Anti-leakage guarantees:
    - Strictly discards every candle with ``end_period_ts > snapshot_ts_utc``.
    - Computes ``p_mkt`` from ``(yes_bid_close + yes_ask_close) / 2`` of the
      latest valid bivariate candle (``price_source='kalshi_mid'``, populating
      ``hist_yes_bid`` and ``hist_yes_ask`` for Pillar 2 only), falling back to
      ``price_close`` (``price_source='kalshi_close'``) when no valid bivariate
      candle exists ``<= snapshot_ts_utc``.
    - Aggregates ``vol_contracts_le_t = sum(c.volume_contracts)`` and
      ``vol_usd_le_t = sum(c.volume_contracts * p_candle)`` (falling back to
      ``Decimal("1")`` only when a candle carries no price at all) strictly
      over candles ``<= snapshot_ts_utc``.
    """
    snap_utc = _ensure_utc(snapshot_ts_utc, "snapshot_ts_utc")

    valid_candles: list[KalshiCandle] = []
    for item in candles:
        candle = item if isinstance(item, KalshiCandle) else parse_kalshi_candlestick(item)
        if candle is None:
            continue
        if candle.end_period_ts <= snap_utc:
            valid_candles.append(candle)

    if not valid_candles:
        return None

    valid_candles.sort(key=lambda c: c.end_period_ts)

    vol_contracts_le_t = _ZERO
    vol_usd_le_t = _ZERO
    latest_bivariate: KalshiCandle | None = None
    latest_close: KalshiCandle | None = None

    for candle in valid_candles:
        vol_contracts_le_t += candle.volume_contracts
        candle_price = candle.effective_price if candle.effective_price is not None else _ONE
        vol_usd_le_t += candle.volume_contracts * candle_price

        if candle.has_valid_bivariate_quote:
            latest_bivariate = candle
        if candle.price_close is not None:
            latest_close = candle

    if latest_bivariate is not None:
        mid = latest_bivariate.mid_price
        assert mid is not None
        return KalshiPointInTime(
            p_mkt=mid,
            price_source="kalshi_mid",
            hist_yes_bid=latest_bivariate.yes_bid_close,
            hist_yes_ask=latest_bivariate.yes_ask_close,
            vol_contracts_le_t=vol_contracts_le_t,
            vol_usd_le_t=vol_usd_le_t,
            candle_count_le_t=len(valid_candles),
        )

    if latest_close is not None and latest_close.price_close is not None:
        return KalshiPointInTime(
            p_mkt=latest_close.price_close,
            price_source="kalshi_close",
            hist_yes_bid=None,
            hist_yes_ask=None,
            vol_contracts_le_t=vol_contracts_le_t,
            vol_usd_le_t=vol_usd_le_t,
            candle_count_le_t=len(valid_candles),
        )

    return None


def is_eligible_resolved_kalshi(market: ResolvedMarket) -> bool:
    """Return ``True`` if ``market`` is a settled binary non-combo Kalshi market."""
    meta = market.metadata
    if meta.venue != "kalshi":
        return False

    if not meta.is_yes_no or len(meta.outcomes) != 2:
        return False
    if any(not outcome.strip() for outcome in meta.outcomes):
        return False

    market_type = _extra_str(meta, "market_type")
    if market_type is not None and market_type.lower() != "binary":
        return False

    mve_collection = meta.extra.get("mve_collection_ticker")
    if isinstance(mve_collection, str) and mve_collection.strip():
        return False
    if mve_collection is not None and not isinstance(mve_collection, str) and bool(mve_collection):
        return False

    mve_legs = meta.extra.get("mve_selected_legs")
    if isinstance(mve_legs, tuple | list) and len(mve_legs) > 0:
        return False

    series_ticker = _extra_str(meta, "series_ticker")
    for ident in (meta.market_id, meta.event_id, series_ticker):
        if ident is not None and ident.upper().startswith(_COMBO_TICKER_PREFIXES):
            return False

    combo_fields = (
        meta.market_id,
        meta.event_id,
        series_ticker,
        meta.category,
        meta.title,
    )
    if any(field is not None and _COMBO_TEXT_RE.search(field) for field in combo_fields):
        return False

    if is_short_term_crypto(meta):
        return False

    return True


def _resolve_default_kalshi_target_series() -> tuple[str, ...]:
    """Expand default Weather and Economics/Macro prefixes into exact Kalshi series tickers."""
    expanded: list[str] = []
    seen: set[str] = set()
    for prefix in (*DEFAULT_WEATHER_SERIES_PREFIXES, *DEFAULT_MACRO_SERIES_PREFIXES):
        upper = prefix.strip().upper()
        if not upper:
            continue
        series_list = DEFAULT_KALSHI_WEATHER_SERIES_BY_PREFIX.get(upper, (upper,))
        for series in series_list:
            clean = series.strip().upper()
            if clean and clean not in seen:
                seen.add(clean)
                expanded.append(clean)
    return tuple(expanded)


# ---------------------------------------------------------------------------
# Internal orchestration helpers (cutoff routing & period-cap calibration)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _KalshiCandidate:
    market: ResolvedMarket
    snapshot_ts_utc: datetime
    start_ts: int
    end_ts: int
    pre_cutoff: bool


@dataclass(frozen=True, slots=True)
class _LiveCandleBatch:
    candidates: tuple[_KalshiCandidate, ...]
    start_ts: int
    end_ts: int
    period_interval: int
    max_batch_size: int


def _is_pre_cutoff_market(
    market: ResolvedMarket,
    source: _MarketSource,
    cutoff_utc: datetime,
) -> bool:
    """Route a settled market to ``/historical/*`` vs live ``/markets/*`` using ``cutoff_utc``."""
    if market.settlement_ts_utc is not None:
        return market.settlement_ts_utc < cutoff_utc
    return source == "archived"


def _compute_candle_window(market: ResolvedMarket, snapshot_ts_utc: datetime) -> tuple[int, int]:
    """Compute inclusive Unix-second ``(start_ts, end_ts)`` up to ``snapshot_ts_utc``."""
    end_ts = int(snapshot_ts_utc.timestamp())
    if market.open_time_utc is not None and market.open_time_utc < snapshot_ts_utc:
        start_dt = market.open_time_utc
    else:
        start_dt = snapshot_ts_utc - timedelta(days=DEFAULT_FALLBACK_LOOKBACK_DAYS)
    start_ts = min(int(start_dt.timestamp()), end_ts)
    return start_ts, end_ts


def _estimate_periods(start_ts: int, end_ts: int, period_interval: int) -> int:
    span_seconds = max(0, end_ts - start_ts)
    step_seconds = period_interval * 60
    return max(1, math.ceil(span_seconds / step_seconds) + 1)


def _choose_historical_interval_and_start(
    start_ts: int,
    end_ts: int,
    requested_interval: int,
) -> tuple[int, int]:
    """Select ``period_interval`` in ``{60, 1440}`` respecting Kalshi's 5,000-period archive cap."""
    interval = requested_interval
    if (
        interval == 60
        and _estimate_periods(start_ts, end_ts, 60) > KALSHI_HISTORICAL_CANDLESTICKS_MAX_PERIODS
    ):
        interval = 1440

    max_span_seconds = (KALSHI_HISTORICAL_CANDLESTICKS_MAX_PERIODS - 1) * interval * 60
    clamped_start = max(start_ts, end_ts - max_span_seconds)
    return interval, min(clamped_start, end_ts)


def _plan_live_candlestick_batches(
    candidates: Sequence[_KalshiCandidate],
    requested_interval: int,
) -> list[_LiveCandleBatch]:
    """Pack post-cutoff markets into batches under <= 100 tickers and <= 10,000 periods/call."""
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: (c.start_ts, c.end_ts, c.market.metadata.market_id))
    batches: list[_LiveCandleBatch] = []

    current: list[_KalshiCandidate] = []
    cur_start = 0
    cur_end = 0
    cur_interval = requested_interval

    def _flush() -> None:
        if not current:
            return
        periods_per_ticker = _estimate_periods(cur_start, cur_end, cur_interval)
        calibrated_batch_size = max(
            1,
            min(
                KALSHI_MAX_BATCH_TICKERS,
                KALSHI_BATCH_CANDLESTICKS_MAX_PERIODS // periods_per_ticker,
            ),
        )
        batches.append(
            _LiveCandleBatch(
                candidates=tuple(current),
                start_ts=cur_start,
                end_ts=cur_end,
                period_interval=cur_interval,
                max_batch_size=calibrated_batch_size,
            )
        )

    for cand in ordered:
        cand_interval = requested_interval
        if (
            cand_interval == 60
            and _estimate_periods(cand.start_ts, cand.end_ts, 60)
            > KALSHI_BATCH_CANDLESTICKS_MAX_PERIODS
        ):
            cand_interval = 1440

        if not current:
            current = [cand]
            cur_start = cand.start_ts
            cur_end = cand.end_ts
            cur_interval = cand_interval
            continue

        if cand_interval != cur_interval or len(current) >= KALSHI_MAX_BATCH_TICKERS:
            _flush()
            current = [cand]
            cur_start = cand.start_ts
            cur_end = cand.end_ts
            cur_interval = cand_interval
            continue

        next_start = min(cur_start, cand.start_ts)
        next_end = max(cur_end, cand.end_ts)
        next_periods = _estimate_periods(next_start, next_end, cur_interval)
        if (len(current) + 1) * next_periods <= KALSHI_BATCH_CANDLESTICKS_MAX_PERIODS:
            current.append(cand)
            cur_start = next_start
            cur_end = next_end
        else:
            _flush()
            current = [cand]
            cur_start = cand.start_ts
            cur_end = cand.end_ts
            cur_interval = cand_interval

    _flush()
    return batches


async def _discover_kalshi_resolved_markets(
    client: KalshiClient,
    perimeter: BackfillPerimeter,
    *,
    max_live_pages: int,
    max_archived_pages: int,
    target_series: tuple[str, ...],
    semaphore: asyncio.Semaphore,
) -> tuple[list[ResolvedMarket], dict[str, _MarketSource]]:
    """Run dual live + archived Kalshi discovery with ``mve_filter='exclude'``."""
    min_close_ts = int(perimeter.start_date_utc.timestamp())
    max_close_ts = int(perimeter.end_date_utc.timestamp())

    live_core_pages = 0
    series_to_scan: tuple[str, ...] = ()
    if max_live_pages > 0:
        if target_series and max_live_pages > 1:
            live_core_pages = max(1, max_live_pages // 2)
            remaining_budget = max_live_pages - live_core_pages
            series_to_scan = target_series[:remaining_budget]
        else:
            live_core_pages = max_live_pages

    async def _fetch_live_core() -> list[ResolvedMarket]:
        if live_core_pages <= 0:
            return []
        async with semaphore:
            return await client.get_resolved_markets(
                archived=False,
                limit=KALSHI_MAX_PAGE_LIMIT,
                max_pages=live_core_pages,
                mve_filter="exclude",
                min_close_ts=min_close_ts,
                max_close_ts=max_close_ts,
            )

    async def _fetch_live_series(series_ticker: str) -> list[ResolvedMarket]:
        async with semaphore:
            try:
                return await client.get_resolved_markets(
                    archived=False,
                    limit=KALSHI_MAX_PAGE_LIMIT,
                    max_pages=1,
                    series_ticker=series_ticker,
                    mve_filter="exclude",
                    min_close_ts=min_close_ts,
                    max_close_ts=max_close_ts,
                )
            except httpx.HTTPError as exc:
                logger.warning(
                    "Kalshi targeted series discovery failed for series_ticker=%s: %s",
                    series_ticker,
                    exc,
                )
                return []

    async def _fetch_archived() -> list[ResolvedMarket]:
        if max_archived_pages <= 0:
            return []
        async with semaphore:
            return await client.get_resolved_markets(
                archived=True,
                limit=KALSHI_MAX_PAGE_LIMIT,
                max_pages=max_archived_pages,
                mve_filter="exclude",
            )

    live_core_task = _fetch_live_core()
    archived_task = _fetch_archived()
    series_tasks = [_fetch_live_series(series) for series in series_to_scan]

    all_results = await asyncio.gather(live_core_task, archived_task, *series_tasks)
    live_core_markets = all_results[0]
    archived_markets = all_results[1]
    series_market_lists = all_results[2:]

    deduped: list[ResolvedMarket] = []
    source_by_ticker: dict[str, _MarketSource] = {}

    for market in live_core_markets:
        ticker = market.metadata.market_id
        if ticker not in source_by_ticker:
            source_by_ticker[ticker] = "live"
            deduped.append(market)

    for series_list in series_market_lists:
        for market in series_list:
            ticker = market.metadata.market_id
            if ticker not in source_by_ticker:
                source_by_ticker[ticker] = "live"
                deduped.append(market)

    for market in archived_markets:
        ticker = market.metadata.market_id
        if ticker not in source_by_ticker:
            source_by_ticker[ticker] = "archived"
            deduped.append(market)

    return deduped, source_by_ticker


async def _fetch_live_batch_candles(
    client: KalshiClient,
    batch: _LiveCandleBatch,
    semaphore: asyncio.Semaphore,
) -> dict[str, list[dict[str, Any]]] | None:
    tickers = [cand.market.metadata.market_id for cand in batch.candidates]
    async with semaphore:
        try:
            return await client.get_candlesticks_batch(
                tickers,
                start_ts=batch.start_ts,
                end_ts=batch.end_ts,
                period_interval=batch.period_interval,
                include_latest_before_start=True,
                max_batch_size=batch.max_batch_size,
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "Transient HTTP error fetching live Kalshi candlesticks batch (%d tickers): %s",
                len(tickers),
                exc,
            )
            return None


async def _fetch_historical_market_candles(
    client: KalshiClient,
    cand: _KalshiCandidate,
    requested_interval: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, list[dict[str, Any]] | None]:
    ticker = cand.market.metadata.market_id
    eff_interval, eff_start_ts = _choose_historical_interval_and_start(
        cand.start_ts,
        cand.end_ts,
        requested_interval,
    )
    async with semaphore:
        try:
            candles = await client.get_historical_candlesticks(
                ticker,
                start_ts=eff_start_ts,
                end_ts=cand.end_ts,
                period_interval=eff_interval,
            )
            return ticker, candles
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return ticker, []
            logger.warning(
                "Transient HTTP %d fetching historical Kalshi candlesticks for %s: %s",
                exc.response.status_code,
                ticker,
                exc,
            )
            return ticker, None
        except httpx.HTTPError as exc:
            logger.warning(
                "Transient HTTP error fetching historical Kalshi candlesticks for %s: %s",
                ticker,
                exc,
            )
            return ticker, None


def _build_kalshi_snapshot_record(
    cand: _KalshiCandidate,
    pit: KalshiPointInTime,
) -> HistoricalSnapshotRecord:
    market = cand.market
    meta = market.metadata
    end_time_utc = meta.end_time_utc
    if end_time_utc is None:
        raise ValueError(f"Kalshi market {meta.market_id} missing required end_time_utc")

    tau_days, log1p_tau = compute_tau_metrics(cand.snapshot_ts_utc, end_time_utc)

    return HistoricalSnapshotRecord(
        venue="kalshi",
        market_id=meta.market_id,
        token_id=None,
        event_id=meta.event_id,
        title=meta.title,
        category=meta.category,
        outcome_side="yes",
        outcome_label="Yes",
        is_yes_no=True,
        neg_risk=None,
        fee_type=_extra_str(meta, "fee_type"),
        open_time_utc=market.open_time_utc,
        snapshot_ts_utc=cand.snapshot_ts_utc,
        end_time_utc=end_time_utc,
        tau_days=tau_days,
        log1p_tau=log1p_tau,
        p_mkt=pit.p_mkt,
        vol_usd_le_t=pit.vol_usd_le_t,
        vol_contracts_le_t=pit.vol_contracts_le_t,
        vol_unit="contracts_notional",
        trade_count_le_t=None,
        trades_truncated=False,
        price_source=pit.price_source,
        outcome_y=market.outcome_y,
        hist_yes_bid=pit.hist_yes_bid,
        hist_yes_ask=pit.hist_yes_ask,
    )


# ---------------------------------------------------------------------------
# Public async orchestrator
# ---------------------------------------------------------------------------


async def backfill_kalshi(
    perimeter: BackfillPerimeter,
    *,
    client: KalshiClient | None = None,
    checkpoint_path: Path | str | None = None,
    max_live_pages: int = DEFAULT_KALSHI_MAX_LIVE_PAGES,
    max_archived_pages: int = DEFAULT_KALSHI_MAX_ARCHIVED_PAGES,
    period_interval: int = DEFAULT_KALSHI_BACKFILL_INTERVAL,
    max_concurrency: int = DEFAULT_BACKFILL_MAX_CONCURRENCY,
    oversample_factor: float = DEFAULT_OVERSAMPLE_FACTOR,
    target_series: Sequence[str] | None = None,
) -> list[HistoricalSnapshotRecord]:
    """Run point-in-time historical backfill for settled Kalshi binary markets.

    Calls ``GET /historical/cutoff`` once, performs dual live + archived
    discovery with ``mve_filter='exclude'``, stratifies deterministically with
    :func:`pm_quant.ingest.backfill_polymarket.stratify_resolved_markets`,
    routes candlesticks across ``/markets/candlesticks`` (batch) and
    ``/historical/markets/{ticker}/candlesticks`` (archive), enforces strict
    ``end_period_ts <= snapshot_ts_utc`` anti-leakage, and persists progress in
    an append-only resumable JSONL checkpoint.
    """
    if isinstance(max_live_pages, bool) or max_live_pages < 0:
        raise ValueError(f"max_live_pages must be an integer >= 0, got {max_live_pages!r}")
    if isinstance(max_archived_pages, bool) or max_archived_pages < 0:
        raise ValueError(f"max_archived_pages must be an integer >= 0, got {max_archived_pages!r}")
    if (
        isinstance(period_interval, bool)
        or period_interval not in KALSHI_BACKFILL_ALLOWED_INTERVALS
    ):
        allowed = sorted(KALSHI_BACKFILL_ALLOWED_INTERVALS)
        raise ValueError(f"period_interval must be one of {allowed}, got {period_interval!r}")
    if isinstance(max_concurrency, bool) or max_concurrency < 1:
        raise ValueError(f"max_concurrency must be an integer >= 1, got {max_concurrency!r}")
    if (
        isinstance(oversample_factor, bool)
        or not math.isfinite(oversample_factor)
        or oversample_factor < 1.0
    ):
        raise ValueError(
            f"oversample_factor must be a finite float >= 1.0, got {oversample_factor!r}"
        )

    resolved_ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else None
    perimeter_hash = compute_perimeter_hash(perimeter)

    ok_records: dict[str, HistoricalSnapshotRecord] = {}
    skipped_ids: set[str] = set()
    if resolved_ckpt_path is not None:
        ok_records, skipped_ids = _load_checkpoint(resolved_ckpt_path, perimeter_hash)

    def _sorted_final() -> list[HistoricalSnapshotRecord]:
        return sorted(
            ok_records.values(),
            key=lambda rec: (rec.venue, rec.end_time_utc, rec.market_id),
        )[: perimeter.max_markets_per_venue]

    if len(ok_records) >= perimeter.max_markets_per_venue:
        return _sorted_final()

    resolved_target_series = (
        tuple(dict.fromkeys(clean for s in target_series if (clean := s.strip().upper())))
        if target_series is not None
        else _resolve_default_kalshi_target_series()
    )

    owns_client = client is None
    active_client = KalshiClient(max_concurrency=max_concurrency) if client is None else client
    semaphore = asyncio.Semaphore(max_concurrency)

    try:
        if owns_client:
            await active_client.__aenter__()

        cutoff_utc = await active_client.get_historical_cutoff()

        discovered, source_by_ticker = await _discover_kalshi_resolved_markets(
            active_client,
            perimeter,
            max_live_pages=max_live_pages,
            max_archived_pages=max_archived_pages,
            target_series=resolved_target_series,
            semaphore=semaphore,
        )

        candidate_budget = max(
            perimeter.max_markets_per_venue,
            int(math.ceil(perimeter.max_markets_per_venue * oversample_factor)),
        )
        stratified = stratify_resolved_markets(
            discovered,
            perimeter,
            max_markets=candidate_budget,
            eligibility_fn=is_eligible_resolved_kalshi,
        )
        pending = [
            market
            for market in stratified
            if market.metadata.market_id not in ok_records
            and market.metadata.market_id not in skipped_ids
        ]

        cursor = 0
        while cursor < len(pending) and len(ok_records) < perimeter.max_markets_per_venue:
            needed = perimeter.max_markets_per_venue - len(ok_records)
            wave = pending[cursor : cursor + needed]
            cursor += len(wave)

            live_candidates: list[_KalshiCandidate] = []
            hist_candidates: list[_KalshiCandidate] = []

            for market in wave:
                ticker = market.metadata.market_id
                snapshot_ts = sample_variable_tau_snapshot_ts(market, perimeter)
                if snapshot_ts is None:
                    skipped_ids.add(ticker)
                    _append_checkpoint_entry(
                        resolved_ckpt_path,
                        perimeter_hash=perimeter_hash,
                        market_id=ticker,
                        status="skipped",
                        skip_reason="no_valid_snapshot_ts",
                    )
                    continue

                start_ts, end_ts = _compute_candle_window(market, snapshot_ts)
                source = source_by_ticker.get(ticker, "live")
                pre_cutoff = _is_pre_cutoff_market(market, source, cutoff_utc)
                cand = _KalshiCandidate(
                    market=market,
                    snapshot_ts_utc=snapshot_ts,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    pre_cutoff=pre_cutoff,
                )
                if pre_cutoff:
                    hist_candidates.append(cand)
                else:
                    live_candidates.append(cand)

            candles_by_ticker: dict[str, list[dict[str, Any]]] = {}
            transient_failed_tickers: set[str] = set()

            if live_candidates:
                live_batches = _plan_live_candlestick_batches(live_candidates, period_interval)
                batch_results = await asyncio.gather(
                    *(
                        _fetch_live_batch_candles(active_client, batch, semaphore)
                        for batch in live_batches
                    )
                )
                for batch, batch_map in zip(live_batches, batch_results, strict=True):
                    if batch_map is None:
                        for cand in batch.candidates:
                            transient_failed_tickers.add(cand.market.metadata.market_id)
                        continue
                    for cand in batch.candidates:
                        ticker = cand.market.metadata.market_id
                        candles_by_ticker[ticker] = batch_map.get(ticker, [])

            if hist_candidates:
                hist_results = await asyncio.gather(
                    *(
                        _fetch_historical_market_candles(
                            active_client,
                            cand,
                            period_interval,
                            semaphore,
                        )
                        for cand in hist_candidates
                    )
                )
                for ticker, raw_candles in hist_results:
                    if raw_candles is None:
                        transient_failed_tickers.add(ticker)
                    else:
                        candles_by_ticker[ticker] = raw_candles

            wave_candidates = live_candidates + hist_candidates
            for cand in wave_candidates:
                if len(ok_records) >= perimeter.max_markets_per_venue:
                    break
                ticker = cand.market.metadata.market_id
                if ticker in transient_failed_tickers:
                    # Never cache transient HTTP errors so they are retried on resume.
                    continue

                pit = extract_kalshi_point_in_time(
                    candles_by_ticker.get(ticker, ()),
                    cand.snapshot_ts_utc,
                )
                if pit is None:
                    skipped_ids.add(ticker)
                    _append_checkpoint_entry(
                        resolved_ckpt_path,
                        perimeter_hash=perimeter_hash,
                        market_id=ticker,
                        status="skipped",
                        skip_reason="no_candlestick_history",
                    )
                    continue

                if pit.p_mkt < perimeter.min_price or pit.p_mkt > perimeter.max_price:
                    skipped_ids.add(ticker)
                    _append_checkpoint_entry(
                        resolved_ckpt_path,
                        perimeter_hash=perimeter_hash,
                        market_id=ticker,
                        status="skipped",
                        skip_reason="price_out_of_bounds",
                    )
                    continue

                if pit.vol_usd_le_t < perimeter.min_vol_usd:
                    skipped_ids.add(ticker)
                    _append_checkpoint_entry(
                        resolved_ckpt_path,
                        perimeter_hash=perimeter_hash,
                        market_id=ticker,
                        status="skipped",
                        skip_reason="vol_below_min",
                    )
                    continue

                record = _build_kalshi_snapshot_record(cand, pit)
                ok_records[ticker] = record
                _append_checkpoint_entry(
                    resolved_ckpt_path,
                    perimeter_hash=perimeter_hash,
                    market_id=ticker,
                    status="ok",
                    record=record,
                )

        return _sorted_final()
    finally:
        if owns_client:
            await active_client.__aexit__(None, None, None)
