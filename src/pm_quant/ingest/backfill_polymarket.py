"""Point-in-time historical backfill pipeline for resolved Polymarket binary markets.

This module implements the Week 3 Block 2 Polymarket historical ingestion
workflow with strict anti-leakage guarantees:

- **Chronological Gamma discovery**: slices ``[start_date_utc, end_date_utc]``
  into monthly windows using ``end_date_min``/``end_date_max`` (never
  ``order=volume*``) to avoid volume-selection bias and unbounded pagination.
- **Deterministic balanced stratification**: :func:`stratify_resolved_markets`
  filters closed binary non-combo markets (reusing
  :func:`pm_quant.ingest.universe.is_short_term_crypto` and requiring 2 valid
  CLOB token IDs) and samples balanced ``(category_norm, YYYY-MM)`` strata via
  SHA-256 hashing seeded by ``perimeter.seed``.
- **Strict point-in-time prices & trades (zero lookahead)**:
  - Samples exactly one variable-``tau`` timestamp ``t_i`` per market via
    :func:`pm_quant.ingest.historical_schema.sample_variable_tau_snapshot_ts`.
  - Fetches Data API v2 prices on ``[t_i - 7d, t_i]`` with
    ``exclude_settlement=True`` and automatically falls back to CLOB
    ``/prices-history`` on ``[t_i - 14d, t_i]`` when v2 returns empty
    (resolution decay), always enforcing ``point_ts <= t_i`` (never
    nearest-neighbor) and skipping ``/v2/trades`` whenever the price is missing
    or outside ``[min_price, max_price]``.
  - Fetches taker-only trades in mini-batches (``<= 20`` condition IDs, default
    5) via ``GET /v2/trades`` (``condition=``, ``taker_only=true``, ``cursor=``),
    summing ``notional_usd`` strictly for ``ts_utc <= t_i`` (never Gamma
    lifetime ``volume``) and setting ``trades_truncated=True`` whenever
    pagination hits ``max_trade_pages_per_batch`` with more pages remaining.
- **Resumable append-only JSONL checkpoint**: skips already-completed markets
  on resume, ignores truncated trailing lines from crashes, and never caches
  transient HTTP errors.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final, Literal, cast

import httpx

from pm_quant.ingest.historical_schema import (
    BackfillPerimeter,
    BinaryOutcome,
    HistoricalOutcomeSide,
    HistoricalSnapshotRecord,
    HistoricalVolumeUnit,
    ResolvedMarket,
    compute_tau_metrics,
    sample_variable_tau_snapshot_ts,
)
from pm_quant.ingest.polymarket import (
    POLYMARKET_GAMMA_MAX_LIMIT,
    POLYMARKET_MAX_CONDITION_BATCH,
    POLYMARKET_MAX_TRADES_LIMIT,
    PolymarketClient,
)
from pm_quant.ingest.schemas import MarketMetadata, TradeRecord, Venue
from pm_quant.ingest.universe import is_short_term_crypto

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION: Final[int] = 1
DEFAULT_BACKFILL_CACHE_DIR: Final[Path] = Path("data/backfill_cache")
DEFAULT_MAX_GAMMA_PAGES: Final[int] = 30
DEFAULT_MAX_TRADE_PAGES_PER_BATCH: Final[int] = 20
DEFAULT_BACKFILL_MAX_CONCURRENCY: Final[int] = 4
DEFAULT_TRADE_BATCH_SIZE: Final[int] = 5
DEFAULT_OVERSAMPLE_FACTOR: Final[float] = 2.0
DEFAULT_V2_PRICE_LOOKBACK_DAYS: Final[int] = 7
DEFAULT_CLOB_PRICE_LOOKBACK_DAYS: Final[int] = 14

_ZERO: Final[Decimal] = Decimal("0")
_ONE: Final[Decimal] = Decimal("1")
_COMBO_TICKER_PREFIXES: Final[tuple[str, ...]] = ("KXMVE", "KXMULTI")
_COMBO_TEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[^a-z0-9])"
    r"(?:combo|combos|parlay|parlays|same[-_ ]?game[-_ ]?parlay|multi[-_ ]?leg|multileg)"
    r"(?:$|[^a-z0-9])",
    re.IGNORECASE,
)

CheckpointStatus = Literal["ok", "skipped"]


# ---------------------------------------------------------------------------
# Pure helpers & validation (zero I/O)
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


def _parse_point_timestamp_utc(raw: Any) -> datetime | None:
    """Parse a price-history timestamp (epoch seconds/ms, ISO-8601, or aware datetime) into UTC."""
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


def _is_settlement_point(point: Mapping[str, Any]) -> bool:
    """Return ``True`` if a Data API v2 point marks on-chain settlement (resolution_seconds=0)."""
    res = point.get("resolution_seconds")
    return not isinstance(res, bool) and isinstance(res, int | float) and res == 0


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_clob_token_ids(meta: MarketMetadata) -> tuple[str, str] | None:
    raw_tokens = meta.extra.get("clob_token_ids")
    if not isinstance(raw_tokens, tuple | list) or len(raw_tokens) != 2:
        return None
    t0 = _non_empty_str(raw_tokens[0])
    t1 = _non_empty_str(raw_tokens[1])
    if t0 is None or t1 is None:
        return None
    return (t0, t1)


def _canonical_outcome_index(meta: MarketMetadata) -> int:
    if meta.is_yes_no:
        for idx, label in enumerate(meta.outcomes):
            if label.strip().lower() == "yes":
                return idx
    return 0


def _iter_monthly_windows(
    start_utc: datetime, end_utc: datetime
) -> list[tuple[datetime, datetime]]:
    """Split ``[start_utc, end_utc]`` into contiguous calendar-month UTC slices."""
    start = _ensure_utc(start_utc, "start_utc")
    end = _ensure_utc(end_utc, "end_utc")
    if start >= end:
        raise ValueError(f"start_utc ({start.isoformat()}) must be < end_utc ({end.isoformat()})")

    windows: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        if cursor.month == 12:
            next_month = datetime(cursor.year + 1, 1, 1, tzinfo=UTC)
        else:
            next_month = datetime(cursor.year, cursor.month + 1, 1, tzinfo=UTC)
        window_end = min(next_month, end)
        windows.append((cursor, window_end))
        cursor = window_end
    return windows


def _allocate_gamma_pages(num_windows: int, max_gamma_pages: int) -> list[int]:
    """Distribute ``max_gamma_pages`` evenly across ``num_windows`` monthly slices."""
    if num_windows <= 0 or max_gamma_pages <= 0:
        return [0] * max(num_windows, 0)
    base, remainder = divmod(max_gamma_pages, num_windows)
    return [base + (1 if idx < remainder else 0) for idx in range(num_windows)]


# ---------------------------------------------------------------------------
# Pure eligibility, stratification, and point-in-time extraction (zero I/O)
# ---------------------------------------------------------------------------


def is_eligible_resolved_polymarket(market: ResolvedMarket) -> bool:
    """Return ``True`` if ``market`` is a closed binary non-combo Polymarket CLOB market.

    Compatible with ``status='closed'``/``'archived'``/``'resolved'`` (unlike live
    screening's ``is_binary_non_combo``), while preserving identical exclusions for
    multi-outcome markets, combo/parlay/MVE contracts, and 5m/15m intraday crypto
    contracts via :func:`pm_quant.ingest.universe.is_short_term_crypto`.
    """
    meta = market.metadata
    if meta.venue != "polymarket":
        return False

    if len(meta.outcomes) != 2 or any(not outcome.strip() for outcome in meta.outcomes):
        return False

    market_type = _extra_str(meta, "market_type")
    if market_type is not None and market_type.lower() != "binary":
        return False

    # Exclude combo / parlay / MVE contracts.
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

    slug = _extra_str(meta, "slug")
    fee_type = _extra_str(meta, "fee_type")
    combo_fields = (
        meta.market_id,
        meta.event_id,
        series_ticker,
        slug,
        meta.category,
        fee_type,
        meta.title,
    )
    if any(field is not None and _COMBO_TEXT_RE.search(field) for field in combo_fields):
        return False

    if is_short_term_crypto(meta):
        return False

    if meta.extra.get("enable_order_book") is False:
        return False

    if _extract_clob_token_ids(meta) is None:
        return False

    return True


def stratify_resolved_markets(
    markets: Sequence[ResolvedMarket],
    perimeter: BackfillPerimeter,
    *,
    max_markets: int | None = None,
    eligibility_fn: Callable[[ResolvedMarket], bool] = is_eligible_resolved_polymarket,
) -> list[ResolvedMarket]:
    """Deterministically filter and stratify resolved markets by ``(category, YYYY-MM)``.

    Steps:
    1. Deduplicate by ``market.metadata.market_id`` (preserving first occurrence).
    2. Retain only binary non-combo markets passing ``eligibility_fn``
       (defaulting to :func:`is_eligible_resolved_polymarket`) with
       ``end_time_utc`` inside ``[perimeter.start_date_utc, perimeter.end_date_utc]``.
    3. Group into ``(category_norm, YYYY-MM)`` strata, sort members and strata
       deterministically via SHA-256 keyed by ``perimeter.seed``, and select
       round-robin across strata up to ``max_markets`` (defaulting to
       ``perimeter.max_markets_per_venue``).
    """
    target_cap = perimeter.max_markets_per_venue if max_markets is None else max_markets
    if isinstance(target_cap, bool) or target_cap < 1:
        raise ValueError(f"max_markets must be an integer >= 1, got {target_cap!r}")

    deduped: dict[str, ResolvedMarket] = {}
    for market in markets:
        market_id = market.metadata.market_id
        if market_id not in deduped:
            deduped[market_id] = market

    strata: dict[tuple[str, str], list[ResolvedMarket]] = {}
    for market in deduped.values():
        if not eligibility_fn(market):
            continue
        end_time = market.metadata.end_time_utc
        if end_time is None:
            continue
        if end_time < perimeter.start_date_utc or end_time > perimeter.end_date_utc:
            continue

        raw_category = market.metadata.category
        category_norm = (
            raw_category.strip().lower()
            if isinstance(raw_category, str) and raw_category.strip()
            else "uncategorized"
        )
        month_key = end_time.astimezone(UTC).strftime("%Y-%m")
        strata.setdefault((category_norm, month_key), []).append(market)

    if not strata:
        return []

    ordered_stratum_keys = sorted(
        strata.keys(),
        key=lambda key: (
            _sha256_hex(f"{perimeter.seed}:stratum:{key[0]}:{key[1]}"),
            key[0],
            key[1],
        ),
    )

    queues: list[list[ResolvedMarket]] = []
    for key in ordered_stratum_keys:
        members = sorted(
            strata[key],
            key=lambda m: (
                _sha256_hex(f"{perimeter.seed}:market:{m.metadata.venue}:{m.metadata.market_id}"),
                m.metadata.market_id,
            ),
        )
        queues.append(members)

    selected: list[ResolvedMarket] = []
    round_idx = 0
    while len(selected) < target_cap:
        added_in_round = False
        for queue in queues:
            if round_idx < len(queue):
                selected.append(queue[round_idx])
                added_in_round = True
                if len(selected) >= target_cap:
                    break
        if not added_in_round:
            break
        round_idx += 1

    return selected


def extract_asof_price(
    points: Sequence[Mapping[str, Any]],
    snapshot_ts_utc: datetime,
    *,
    ts_key: str = "timestamp",
    price_key: str = "price",
) -> Decimal | None:
    """Return the latest valid price with ``point_ts <= snapshot_ts_utc`` (strict as-of).

    Never performs nearest-neighbor matching and strictly ignores any point with
    ``point_ts > snapshot_ts_utc`` or on-chain settlement points
    (``resolution_seconds == 0``). Handles unsorted inputs safely.
    """
    snap_utc = _ensure_utc(snapshot_ts_utc, "snapshot_ts_utc")
    best_ts: datetime | None = None
    best_price: Decimal | None = None

    for point in points:
        if _is_settlement_point(point):
            continue
        point_ts = _parse_point_timestamp_utc(point.get(ts_key))
        if point_ts is None or point_ts > snap_utc:
            continue
        price = _to_decimal(point.get(price_key))
        if price is None or price < _ZERO or price > _ONE:
            continue
        if best_ts is None or point_ts > best_ts:
            best_ts = point_ts
            best_price = price

    return best_price


def compute_point_in_time_volume_usd(
    trades: Sequence[TradeRecord],
    condition_id: str,
    snapshot_ts_utc: datetime,
) -> tuple[Decimal, int]:
    """Sum ``notional_usd`` and count trades for ``condition_id`` with ``ts_utc <= t_i``."""
    clean_condition_id = _non_empty_str(condition_id)
    if clean_condition_id is None:
        raise ValueError("condition_id must be a non-empty string")
    snap_utc = _ensure_utc(snapshot_ts_utc, "snapshot_ts_utc")

    total_usd = _ZERO
    count = 0
    for trade in trades:
        if trade.market_id != clean_condition_id:
            continue
        if trade.ts_utc <= snap_utc:
            total_usd += trade.notional_usd
            count += 1
    return total_usd, count


# ---------------------------------------------------------------------------
# Perimeter hashing & resumable append-only JSONL checkpoint I/O
# ---------------------------------------------------------------------------


def compute_perimeter_hash(perimeter: BackfillPerimeter) -> str:
    """Return a deterministic 16-char SHA-256 hex digest uniquely identifying ``perimeter``."""
    canonical = json.dumps(
        {
            "start_date_utc": perimeter.start_date_utc.isoformat(),
            "end_date_utc": perimeter.end_date_utc.isoformat(),
            "max_markets_per_venue": perimeter.max_markets_per_venue,
            "min_vol_usd": str(perimeter.min_vol_usd),
            "min_price": str(perimeter.min_price),
            "max_price": str(perimeter.max_price),
            "pre_resolution_buffer_hours": perimeter.pre_resolution_buffer_hours,
            "seed": perimeter.seed,
            "short_tau_target_share": perimeter.short_tau_target_share,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_hex(canonical)[:16]


def default_checkpoint_path(
    venue: Venue,
    perimeter: BackfillPerimeter,
    *,
    base_dir: Path = DEFAULT_BACKFILL_CACHE_DIR,
) -> Path:
    """Build the default append-only JSONL checkpoint path for ``(venue, perimeter)``."""
    return base_dir / f"{venue}_{compute_perimeter_hash(perimeter)}.jsonl"


def _serialize_record(record: HistoricalSnapshotRecord) -> dict[str, Any]:
    return {
        "schema_version": record.schema_version,
        "venue": record.venue,
        "market_id": record.market_id,
        "token_id": record.token_id,
        "event_id": record.event_id,
        "title": record.title,
        "category": record.category,
        "outcome_side": record.outcome_side,
        "outcome_label": record.outcome_label,
        "is_yes_no": record.is_yes_no,
        "neg_risk": record.neg_risk,
        "fee_type": record.fee_type,
        "open_time_utc": (
            record.open_time_utc.isoformat().replace("+00:00", "Z")
            if record.open_time_utc is not None
            else None
        ),
        "snapshot_ts_utc": record.snapshot_ts_utc.isoformat().replace("+00:00", "Z"),
        "end_time_utc": record.end_time_utc.isoformat().replace("+00:00", "Z"),
        "tau_days": str(record.tau_days),
        "log1p_tau": record.log1p_tau,
        "p_mkt": str(record.p_mkt),
        "vol_usd_le_t": str(record.vol_usd_le_t),
        "vol_contracts_le_t": (
            str(record.vol_contracts_le_t) if record.vol_contracts_le_t is not None else None
        ),
        "vol_unit": record.vol_unit,
        "trade_count_le_t": record.trade_count_le_t,
        "trades_truncated": record.trades_truncated,
        "price_source": record.price_source,
        "outcome_y": record.outcome_y,
        "hist_yes_bid": str(record.hist_yes_bid) if record.hist_yes_bid is not None else None,
        "hist_yes_ask": str(record.hist_yes_ask) if record.hist_yes_ask is not None else None,
    }


def _deserialize_record(data: Mapping[str, Any]) -> HistoricalSnapshotRecord:
    def _opt_dt(raw: Any) -> datetime | None:
        if raw is None:
            return None
        if not isinstance(raw, str):
            raise ValueError(f"Expected ISO timestamp string or None, got {raw!r}")
        return _ensure_utc(datetime.fromisoformat(raw), "timestamp")

    def _req_dt(raw: Any, name: str) -> datetime:
        if not isinstance(raw, str):
            raise ValueError(f"Expected ISO timestamp string for {name}, got {raw!r}")
        return _ensure_utc(datetime.fromisoformat(raw), name)

    def _opt_dec(raw: Any) -> Decimal | None:
        if raw is None:
            return None
        dec = _to_decimal(raw)
        if dec is None:
            raise ValueError(f"Expected Decimal string or None, got {raw!r}")
        return dec

    def _req_dec(raw: Any, name: str) -> Decimal:
        dec = _to_decimal(raw)
        if dec is None:
            raise ValueError(f"Expected valid Decimal for {name}, got {raw!r}")
        return dec

    outcome_y_raw = data["outcome_y"]
    if isinstance(outcome_y_raw, bool) or outcome_y_raw not in (0, 1):
        raise ValueError(f"Invalid outcome_y in checkpoint: {outcome_y_raw!r}")
    outcome_y: BinaryOutcome = cast(BinaryOutcome, int(outcome_y_raw))

    return HistoricalSnapshotRecord(
        schema_version=str(data["schema_version"]),
        venue=cast(Venue, str(data["venue"])),
        market_id=str(data["market_id"]),
        token_id=str(data["token_id"]) if data.get("token_id") is not None else None,
        event_id=str(data["event_id"]) if data.get("event_id") is not None else None,
        title=str(data["title"]),
        category=str(data["category"]) if data.get("category") is not None else None,
        outcome_side=cast(HistoricalOutcomeSide, str(data["outcome_side"])),
        outcome_label=(
            str(data["outcome_label"]) if data.get("outcome_label") is not None else None
        ),
        is_yes_no=bool(data["is_yes_no"]),
        neg_risk=bool(data["neg_risk"]) if data.get("neg_risk") is not None else None,
        fee_type=str(data["fee_type"]) if data.get("fee_type") is not None else None,
        open_time_utc=_opt_dt(data.get("open_time_utc")),
        snapshot_ts_utc=_req_dt(data["snapshot_ts_utc"], "snapshot_ts_utc"),
        end_time_utc=_req_dt(data["end_time_utc"], "end_time_utc"),
        tau_days=_req_dec(data["tau_days"], "tau_days"),
        log1p_tau=float(data["log1p_tau"]),
        p_mkt=_req_dec(data["p_mkt"], "p_mkt"),
        vol_usd_le_t=_req_dec(data["vol_usd_le_t"], "vol_usd_le_t"),
        vol_contracts_le_t=_opt_dec(data.get("vol_contracts_le_t")),
        vol_unit=cast(HistoricalVolumeUnit, str(data["vol_unit"])),
        trade_count_le_t=(
            int(data["trade_count_le_t"]) if data.get("trade_count_le_t") is not None else None
        ),
        trades_truncated=bool(data.get("trades_truncated", False)),
        price_source=str(data["price_source"]),
        outcome_y=outcome_y,
        hist_yes_bid=_opt_dec(data.get("hist_yes_bid")),
        hist_yes_ask=_opt_dec(data.get("hist_yes_ask")),
    )


def _load_checkpoint(
    checkpoint_path: Path,
    expected_perimeter_hash: str,
) -> tuple[dict[str, HistoricalSnapshotRecord], set[str]]:
    """Load completed ``ok`` records and ``skipped`` market IDs from a JSONL checkpoint.

    Safely ignores a truncated/corrupt trailing line left behind by an interrupted
    write, and raises ``ValueError`` if valid entries belong to a different
    ``perimeter_hash`` or ``checkpoint_version``.
    """
    ok_records: dict[str, HistoricalSnapshotRecord] = {}
    skipped_ids: set[str] = set()

    if not checkpoint_path.exists():
        return ok_records, skipped_ids

    raw_lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
    non_empty_indices = [idx for idx, line in enumerate(raw_lines) if line.strip()]
    last_non_empty_idx = non_empty_indices[-1] if non_empty_indices else -1
    valid_lines: list[str] = []
    dropped_trailing = False

    for idx in non_empty_indices:
        line = raw_lines[idx]
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            if idx == last_non_empty_idx:
                logger.warning(
                    "Ignoring truncated trailing line %d in checkpoint %s: %s",
                    idx + 1,
                    checkpoint_path,
                    exc,
                )
                dropped_trailing = True
                break
            raise ValueError(
                f"Corrupt JSON on line {idx + 1} of checkpoint {checkpoint_path}"
            ) from exc

        if not isinstance(entry, Mapping):
            if idx == last_non_empty_idx:
                logger.warning(
                    "Ignoring non-object trailing line %d in checkpoint %s",
                    idx + 1,
                    checkpoint_path,
                )
                dropped_trailing = True
                break
            raise ValueError(f"Expected JSON object on line {idx + 1} of {checkpoint_path}")

        version = entry.get("checkpoint_version")
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint_version {version!r} in {checkpoint_path} "
                f"(expected {CHECKPOINT_VERSION})"
            )

        entry_hash = entry.get("perimeter_hash")
        if entry_hash != expected_perimeter_hash:
            raise ValueError(
                f"Checkpoint {checkpoint_path} perimeter_hash mismatch: "
                f"found {entry_hash!r}, expected {expected_perimeter_hash!r}"
            )

        market_id = _non_empty_str(entry.get("market_id"))
        status = entry.get("status")
        if market_id is None or status not in ("ok", "skipped"):
            if idx == last_non_empty_idx:
                logger.warning(
                    "Ignoring incomplete trailing entry on line %d in checkpoint %s",
                    idx + 1,
                    checkpoint_path,
                )
                dropped_trailing = True
                break
            raise ValueError(f"Invalid checkpoint entry on line {idx + 1} of {checkpoint_path}")

        if status == "ok":
            raw_record = entry.get("record")
            if not isinstance(raw_record, Mapping):
                if idx == last_non_empty_idx:
                    logger.warning(
                        "Ignoring incomplete trailing record on line %d in checkpoint %s",
                        idx + 1,
                        checkpoint_path,
                    )
                    dropped_trailing = True
                    break
                raise ValueError(
                    f"Missing 'record' payload for ok entry on line {idx + 1} of {checkpoint_path}"
                )
            record = _deserialize_record(raw_record)
            ok_records[market_id] = record
            skipped_ids.discard(market_id)
        else:
            skipped_ids.add(market_id)
            ok_records.pop(market_id, None)

        valid_lines.append(line)

    if dropped_trailing:
        cleaned_text = ("\n".join(valid_lines) + "\n") if valid_lines else ""
        checkpoint_path.write_text(cleaned_text, encoding="utf-8")

    return ok_records, skipped_ids


def _append_checkpoint_entry(
    checkpoint_path: Path | None,
    *,
    perimeter_hash: str,
    market_id: str,
    status: CheckpointStatus,
    skip_reason: str | None = None,
    record: HistoricalSnapshotRecord | None = None,
) -> None:
    """Append a single durable JSONL checkpoint entry (no-op if ``checkpoint_path`` is ``None``)."""
    if checkpoint_path is None:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "perimeter_hash": perimeter_hash,
        "market_id": market_id,
        "status": status,
        "skip_reason": skip_reason,
        "fetched_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "record": _serialize_record(record) if record is not None else None,
    }
    line = json.dumps(entry, separators=(",", ":"), sort_keys=True)
    with checkpoint_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# Internal orchestration stages
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _PricedCandidate:
    market: ResolvedMarket
    snapshot_ts_utc: datetime
    canonical_idx: int
    token_id: str
    p_mkt: Decimal
    price_source: Literal["poly_v2", "poly_clob"]


@dataclass(frozen=True, slots=True)
class _PriceStageOutcome:
    market_id: str
    priced: _PricedCandidate | None = None
    skip_reason: str | None = None
    transient_error: bool = False


async def _fetch_asof_price_for_market(
    client: PolymarketClient,
    market: ResolvedMarket,
    perimeter: BackfillPerimeter,
    semaphore: asyncio.Semaphore,
) -> _PriceStageOutcome:
    """Sample ``t_i`` and fetch point-in-time price (Data v2 -> CLOB fallback) before trades."""
    meta = market.metadata
    market_id = meta.market_id

    snapshot_ts = sample_variable_tau_snapshot_ts(market, perimeter)
    if snapshot_ts is None:
        return _PriceStageOutcome(market_id=market_id, skip_reason="no_valid_snapshot_ts")

    token_pair = _extract_clob_token_ids(meta)
    if token_pair is None:
        return _PriceStageOutcome(market_id=market_id, skip_reason="missing_clob_tokens")

    canonical_idx = _canonical_outcome_index(meta)
    token_id = token_pair[canonical_idx]

    v2_start = snapshot_ts - timedelta(days=DEFAULT_V2_PRICE_LOOKBACK_DAYS)
    clob_start = snapshot_ts - timedelta(days=DEFAULT_CLOB_PRICE_LOOKBACK_DAYS)

    p_mkt: Decimal | None = None
    price_source: Literal["poly_v2", "poly_clob"] = "poly_v2"

    async with semaphore:
        try:
            v2_points = await client.get_v2_price_history(
                token_id,
                start_ts=v2_start,
                end_ts=snapshot_ts,
                exclude_settlement=True,
            )
            p_mkt = extract_asof_price(
                v2_points,
                snapshot_ts,
                ts_key="timestamp",
                price_key="price",
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                logger.warning(
                    "Transient HTTP %d fetching Data v2 price history for market=%s token=%s: %s",
                    exc.response.status_code,
                    market_id,
                    token_id,
                    exc,
                )
                return _PriceStageOutcome(market_id=market_id, transient_error=True)
            logger.debug(
                "Data v2 price history returned 404 for market=%s token=%s; falling back to CLOB",
                market_id,
                token_id,
            )
            p_mkt = None
        except httpx.HTTPError as exc:
            logger.warning(
                "Transient HTTP error fetching Data v2 price history for market=%s token=%s: %s",
                market_id,
                token_id,
                exc,
            )
            return _PriceStageOutcome(market_id=market_id, transient_error=True)

        if p_mkt is None:
            try:
                clob_points = await client.get_clob_price_history(
                    token_id,
                    start_ts=clob_start,
                    end_ts=snapshot_ts,
                )
            except httpx.HTTPError as exc:
                logger.warning(
                    "Transient HTTP error fetching CLOB fallback price history for market=%s "
                    "token=%s: %s",
                    market_id,
                    token_id,
                    exc,
                )
                return _PriceStageOutcome(market_id=market_id, transient_error=True)

            p_mkt = extract_asof_price(
                clob_points,
                snapshot_ts,
                ts_key="t",
                price_key="p",
            )
            if p_mkt is not None:
                price_source = "poly_clob"

    if p_mkt is None:
        return _PriceStageOutcome(market_id=market_id, skip_reason="no_price_history")

    if p_mkt < perimeter.min_price or p_mkt > perimeter.max_price:
        return _PriceStageOutcome(market_id=market_id, skip_reason="price_out_of_bounds")

    return _PriceStageOutcome(
        market_id=market_id,
        priced=_PricedCandidate(
            market=market,
            snapshot_ts_utc=snapshot_ts,
            canonical_idx=canonical_idx,
            token_id=token_id,
            p_mkt=p_mkt,
            price_source=price_source,
        ),
    )


async def _fetch_trades_batch(
    client: PolymarketClient,
    condition_ids: Sequence[str],
    *,
    max_trade_pages_per_batch: int,
    semaphore: asyncio.Semaphore,
) -> tuple[list[TradeRecord], bool] | None:
    """Fetch taker-only trades for a mini-batch (<= 20 IDs), or ``None`` on transient error."""
    async with semaphore:
        try:
            use_truncation_method = (
                isinstance(client, PolymarketClient) and "get_trades" not in client.__dict__
            )
            if use_truncation_method:
                return await client.get_trades_with_truncation(
                    condition_ids,
                    max_pages=max_trade_pages_per_batch,
                    limit=POLYMARKET_MAX_TRADES_LIMIT,
                )
            trades = await client.get_trades(
                condition_ids,
                max_pages=max_trade_pages_per_batch,
                limit=POLYMARKET_MAX_TRADES_LIMIT,
            )
            inferred_truncated = (
                len(trades) >= max_trade_pages_per_batch * POLYMARKET_MAX_TRADES_LIMIT
            )
            return trades, inferred_truncated
        except httpx.HTTPError as exc:
            logger.warning(
                "Transient HTTP error fetching Data v2 trades for batch (%d markets): %s",
                len(condition_ids),
                exc,
            )
            return None


def _build_snapshot_record(
    candidate: _PricedCandidate,
    *,
    vol_usd_le_t: Decimal,
    trade_count_le_t: int,
    trades_truncated: bool,
) -> HistoricalSnapshotRecord:
    market = candidate.market
    meta = market.metadata
    end_time_utc = meta.end_time_utc
    if end_time_utc is None:
        raise ValueError(f"Market {meta.market_id} missing required end_time_utc")

    tau_days, log1p_tau = compute_tau_metrics(candidate.snapshot_ts_utc, end_time_utc)
    outcome_side: HistoricalOutcomeSide = "yes" if meta.is_yes_no else "outcome_0"
    outcome_label = meta.outcomes[candidate.canonical_idx]
    neg_risk_raw = meta.extra.get("neg_risk")
    neg_risk = neg_risk_raw if isinstance(neg_risk_raw, bool) else None
    fee_type = _extra_str(meta, "fee_type")

    return HistoricalSnapshotRecord(
        venue="polymarket",
        market_id=meta.market_id,
        token_id=candidate.token_id,
        event_id=meta.event_id,
        title=meta.title,
        category=meta.category,
        outcome_side=outcome_side,
        outcome_label=outcome_label,
        is_yes_no=meta.is_yes_no,
        neg_risk=neg_risk,
        fee_type=fee_type,
        open_time_utc=market.open_time_utc,
        snapshot_ts_utc=candidate.snapshot_ts_utc,
        end_time_utc=end_time_utc,
        tau_days=tau_days,
        log1p_tau=log1p_tau,
        p_mkt=candidate.p_mkt,
        vol_usd_le_t=vol_usd_le_t,
        vol_contracts_le_t=None,
        vol_unit="usd",
        trade_count_le_t=trade_count_le_t,
        trades_truncated=trades_truncated,
        price_source=candidate.price_source,
        outcome_y=market.outcome_y,
        hist_yes_bid=None,
        hist_yes_ask=None,
    )


async def _discover_resolved_markets_chronological(
    client: PolymarketClient,
    perimeter: BackfillPerimeter,
    *,
    max_gamma_pages: int,
    semaphore: asyncio.Semaphore,
) -> list[ResolvedMarket]:
    """Fetch closed Gamma markets via monthly chronological slicing (never ``order=volume*``)."""
    if max_gamma_pages == 0:
        return []

    windows = _iter_monthly_windows(perimeter.start_date_utc, perimeter.end_date_utc)
    page_budgets = _allocate_gamma_pages(len(windows), max_gamma_pages)

    async def _fetch_window(
        win_start: datetime,
        win_end: datetime,
        pages: int,
    ) -> list[ResolvedMarket]:
        if pages <= 0:
            return []
        async with semaphore:
            return await client.get_resolved_markets(
                closed=True,
                limit=POLYMARKET_GAMMA_MAX_LIMIT,
                max_pages=pages,
                end_date_min=win_start,
                end_date_max=win_end,
            )

    window_results = await asyncio.gather(
        *(
            _fetch_window(win_start, win_end, pages)
            for (win_start, win_end), pages in zip(windows, page_budgets, strict=True)
        )
    )

    deduped: list[ResolvedMarket] = []
    seen_ids: set[str] = set()
    for chunk in window_results:
        for market in chunk:
            market_id = market.metadata.market_id
            if market_id not in seen_ids:
                seen_ids.add(market_id)
                deduped.append(market)
    return deduped


# ---------------------------------------------------------------------------
# Public async orchestrator
# ---------------------------------------------------------------------------


async def backfill_polymarket(
    perimeter: BackfillPerimeter,
    *,
    client: PolymarketClient | None = None,
    checkpoint_path: Path | str | None = None,
    max_gamma_pages: int = DEFAULT_MAX_GAMMA_PAGES,
    max_trade_pages_per_batch: int = DEFAULT_MAX_TRADE_PAGES_PER_BATCH,
    max_concurrency: int = DEFAULT_BACKFILL_MAX_CONCURRENCY,
    trade_batch_size: int = DEFAULT_TRADE_BATCH_SIZE,
    oversample_factor: float = DEFAULT_OVERSAMPLE_FACTOR,
) -> list[HistoricalSnapshotRecord]:
    """Run point-in-time historical backfill for resolved Polymarket binary markets.

    Orchestrates monthly chronological Gamma discovery, balanced SHA-256
    stratification with oversampling before guards, variable-``tau`` sampling,
    strict as-of price extraction (Data API v2 with CLOB fallback) **before**
    taker-only ``/v2/trades`` volume aggregation, and append-only resumable
    JSONL checkpointing.
    """
    if isinstance(max_gamma_pages, bool) or max_gamma_pages < 0:
        raise ValueError(f"max_gamma_pages must be an integer >= 0, got {max_gamma_pages!r}")
    if isinstance(max_trade_pages_per_batch, bool) or max_trade_pages_per_batch < 1:
        raise ValueError(
            f"max_trade_pages_per_batch must be an integer >= 1, got {max_trade_pages_per_batch!r}"
        )
    if isinstance(max_concurrency, bool) or max_concurrency < 1:
        raise ValueError(f"max_concurrency must be an integer >= 1, got {max_concurrency!r}")
    if (
        isinstance(trade_batch_size, bool)
        or trade_batch_size < 1
        or trade_batch_size > POLYMARKET_MAX_CONDITION_BATCH
    ):
        raise ValueError(
            f"trade_batch_size must be in [1, {POLYMARKET_MAX_CONDITION_BATCH}], "
            f"got {trade_batch_size!r}"
        )
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

    owns_client = client is None
    active_client = PolymarketClient(max_concurrency=max_concurrency) if client is None else client
    semaphore = asyncio.Semaphore(max_concurrency)

    try:
        if owns_client:
            await active_client.__aenter__()

        discovered = await _discover_resolved_markets_chronological(
            active_client,
            perimeter,
            max_gamma_pages=max_gamma_pages,
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

            # Stage 1: Fetch point-in-time prices BEFORE trades and apply price bounds guard.
            price_outcomes = await asyncio.gather(
                *(
                    _fetch_asof_price_for_market(active_client, market, perimeter, semaphore)
                    for market in wave
                )
            )

            priced_candidates: list[_PricedCandidate] = []
            for outcome in price_outcomes:
                if outcome.transient_error:
                    continue
                if outcome.skip_reason is not None:
                    skipped_ids.add(outcome.market_id)
                    _append_checkpoint_entry(
                        resolved_ckpt_path,
                        perimeter_hash=perimeter_hash,
                        market_id=outcome.market_id,
                        status="skipped",
                        skip_reason=outcome.skip_reason,
                    )
                elif outcome.priced is not None:
                    priced_candidates.append(outcome.priced)

            if not priced_candidates:
                continue

            # Stage 2: Fetch taker-only /v2/trades in bounded mini-batches (<= 20 condition IDs).
            batches = [
                priced_candidates[idx : idx + trade_batch_size]
                for idx in range(0, len(priced_candidates), trade_batch_size)
            ]
            batch_results = await asyncio.gather(
                *(
                    _fetch_trades_batch(
                        active_client,
                        [cand.market.metadata.market_id for cand in batch],
                        max_trade_pages_per_batch=max_trade_pages_per_batch,
                        semaphore=semaphore,
                    )
                    for batch in batches
                )
            )

            for batch, trade_result in zip(batches, batch_results, strict=True):
                if trade_result is None:
                    # Transient HTTP error: never cache so the batch is retried on resume.
                    continue
                batch_trades, batch_truncated = trade_result
                for cand in batch:
                    if len(ok_records) >= perimeter.max_markets_per_venue:
                        break
                    condition_id = cand.market.metadata.market_id
                    vol_usd_le_t, trade_count_le_t = compute_point_in_time_volume_usd(
                        batch_trades,
                        condition_id,
                        cand.snapshot_ts_utc,
                    )
                    if vol_usd_le_t < perimeter.min_vol_usd:
                        skipped_ids.add(condition_id)
                        _append_checkpoint_entry(
                            resolved_ckpt_path,
                            perimeter_hash=perimeter_hash,
                            market_id=condition_id,
                            status="skipped",
                            skip_reason="vol_below_min",
                        )
                        continue

                    record = _build_snapshot_record(
                        cand,
                        vol_usd_le_t=vol_usd_le_t,
                        trade_count_le_t=trade_count_le_t,
                        trades_truncated=batch_truncated,
                    )
                    ok_records[condition_id] = record
                    _append_checkpoint_entry(
                        resolved_ckpt_path,
                        perimeter_hash=perimeter_hash,
                        market_id=condition_id,
                        status="ok",
                        record=record,
                    )

        return _sorted_final()
    finally:
        if owns_client:
            await active_client.__aexit__(None, None, None)
