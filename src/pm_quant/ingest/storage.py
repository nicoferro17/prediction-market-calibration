"""Parquet snapshot storage with schema versioning, defensive L2 depth cap, and DuckDB reads.

This module serializes normalized L2 orderbook snapshots into daily Zstandard-compressed
Parquet files with an explicit Polars schema (``L2_SNAPSHOT_SCHEMA_V1``) and reads them
back across schema versions via DuckDB ``read_parquet(..., union_by_name=true)``.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final

import duckdb
import polars as pl

from pm_quant.ingest.schemas import SCHEMA_VERSION, MarketMetadata, NormalizedOrderBook
from pm_quant.ingest.universe import SelectionReason, VolumeUnit

DEFAULT_L2_MAX_DEPTH: Final[int] = 20
MIN_L2_MAX_DEPTH: Final[int] = 15
MAX_L2_MAX_DEPTH: Final[int] = 20

_ZERO: Final[Decimal] = Decimal("0")
_TWO: Final[Decimal] = Decimal("2")
_VALID_SELECTION_REASONS: Final[frozenset[str]] = frozenset({"target_series", "volume_core"})
_VALID_VOLUME_UNITS: Final[frozenset[str]] = frozenset({"usd", "contracts"})

L2_SNAPSHOT_SCHEMA_V1: Final[pl.Schema] = pl.Schema(
    {
        "schema_version": pl.String(),
        "snapshot_date": pl.Date(),
        "ts_utc": pl.Datetime("us", "UTC"),
        "venue": pl.String(),
        "market_id": pl.String(),
        "token_id": pl.String(),
        "event_id": pl.String(),
        "title": pl.String(),
        "category": pl.String(),
        "outcome_side": pl.String(),
        "outcome_label": pl.String(),
        "selection_reason": pl.String(),
        "fee_type": pl.String(),
        "is_yes_no": pl.Boolean(),
        "neg_risk": pl.Boolean(),
        "end_time_utc": pl.Datetime("us", "UTC"),
        "volume_24h": pl.Decimal(18, 2),
        "volume_24h_unit": pl.String(),
        "open_interest": pl.Decimal(18, 2),
        "tick_size": pl.Decimal(12, 6),
        "best_bid": pl.Decimal(12, 6),
        "best_ask": pl.Decimal(12, 6),
        "mid": pl.Decimal(12, 6),
        "spread": pl.Decimal(12, 6),
        "is_crossed": pl.Boolean(),
        "bid_depth": pl.UInt16(),
        "ask_depth": pl.UInt16(),
        "bid_prices": pl.List(pl.Decimal(12, 6)),
        "ask_prices": pl.List(pl.Decimal(12, 6)),
        "bid_sizes": pl.List(pl.Decimal(18, 6)),
        "ask_sizes": pl.List(pl.Decimal(18, 6)),
    }
)


def _validate_non_negative_optional(value: Decimal | None, field_name: str) -> None:
    if value is not None and (not value.is_finite() or value < _ZERO):
        raise ValueError(
            f"{field_name} must be None or a finite non-negative Decimal, got {value!r}"
        )


def _validate_max_depth(max_depth: int) -> None:
    if isinstance(max_depth, bool) or not (MIN_L2_MAX_DEPTH <= max_depth <= MAX_L2_MAX_DEPTH):
        raise ValueError(
            f"max_depth must be an integer in [{MIN_L2_MAX_DEPTH}, {MAX_L2_MAX_DEPTH}], "
            f"got {max_depth!r}"
        )


def _extract_optional_str(metadata: MarketMetadata, key: str) -> str | None:
    value = metadata.extra.get(key)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return None


def _extract_optional_bool(metadata: MarketMetadata, key: str) -> bool | None:
    value = metadata.extra.get(key)
    if isinstance(value, bool):
        return value
    return None


def _extract_optional_decimal(metadata: MarketMetadata, key: str) -> Decimal | None:
    value = metadata.extra.get(key)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, (int, float, str)):
        try:
            parsed = Decimal(str(value).strip())
        except InvalidOperation:
            return None
        return parsed if parsed.is_finite() else None
    return None


@dataclass(frozen=True, slots=True)
class L2SnapshotRecord:
    """Immutable L2 orderbook snapshot paired with market metadata and screening fields."""

    orderbook: NormalizedOrderBook
    metadata: MarketMetadata
    selection_reason: SelectionReason
    volume_24h: Decimal | None = None
    volume_24h_unit: VolumeUnit | None = None
    open_interest: Decimal | None = None
    outcome_label: str | None = None

    def __post_init__(self) -> None:
        if self.orderbook.venue != self.metadata.venue:
            raise ValueError(
                f"orderbook.venue ({self.orderbook.venue!r}) does not match "
                f"metadata.venue ({self.metadata.venue!r})"
            )
        if self.orderbook.market_id != self.metadata.market_id:
            raise ValueError(
                f"orderbook.market_id ({self.orderbook.market_id!r}) does not match "
                f"metadata.market_id ({self.metadata.market_id!r})"
            )
        if self.selection_reason not in _VALID_SELECTION_REASONS:
            raise ValueError(
                f"selection_reason must be one of {sorted(_VALID_SELECTION_REASONS)}, "
                f"got {self.selection_reason!r}"
            )
        _validate_non_negative_optional(self.volume_24h, "volume_24h")
        if self.volume_24h_unit is not None and self.volume_24h_unit not in _VALID_VOLUME_UNITS:
            raise ValueError(
                f"volume_24h_unit must be None or one of {sorted(_VALID_VOLUME_UNITS)}, "
                f"got {self.volume_24h_unit!r}"
            )
        _validate_non_negative_optional(self.open_interest, "open_interest")


def orderbooks_to_frame(
    records: Iterable[L2SnapshotRecord],
    *,
    max_depth: int = DEFAULT_L2_MAX_DEPTH,
    snapshot_date: date | None = None,
) -> pl.DataFrame:
    """Convert L2 snapshot records into a strictly typed Polars DataFrame.

    Enforces ``MIN_L2_MAX_DEPTH <= max_depth <= MAX_L2_MAX_DEPTH``, defensively
    re-sorts bids descending and asks ascending before truncating each side to
    ``max_depth``, handles empty/one-sided books and empty input sequences, and
    orders rows deterministically by ``(venue, market_id, outcome_side)``.
    When ``snapshot_date`` is provided, every row's ``snapshot_date`` column is
    stamped with it so the column stays consistent with daily file partitioning.
    """
    _validate_max_depth(max_depth)
    if snapshot_date is not None and isinstance(snapshot_date, datetime):
        raise TypeError(f"snapshot_date must be a datetime.date, not datetime: {snapshot_date!r}")

    sorted_records = sorted(
        records,
        key=lambda rec: (
            rec.orderbook.venue,
            rec.orderbook.market_id,
            rec.orderbook.outcome_side,
        ),
    )

    if not sorted_records:
        return pl.DataFrame(schema=L2_SNAPSHOT_SCHEMA_V1)

    seen_keys: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []

    for rec in sorted_records:
        book = rec.orderbook
        meta = rec.metadata
        row_key = (book.venue, book.market_id, book.outcome_side)
        if row_key in seen_keys:
            raise ValueError(
                f"Duplicate L2 snapshot key (venue, market_id, outcome_side): {row_key!r}"
            )
        seen_keys.add(row_key)

        capped_bids = sorted(book.bids, key=lambda lv: lv.price, reverse=True)[:max_depth]
        capped_asks = sorted(book.asks, key=lambda lv: lv.price)[:max_depth]

        best_bid = capped_bids[0].price if capped_bids else None
        best_ask = capped_asks[0].price if capped_asks else None
        if best_bid is not None and best_ask is not None:
            mid: Decimal | None = (best_bid + best_ask) / _TWO
            spread: Decimal | None = best_ask - best_bid
            is_crossed = best_bid > best_ask
        else:
            mid = None
            spread = None
            is_crossed = False

        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "snapshot_date": snapshot_date if snapshot_date is not None else book.ts_utc.date(),
                "ts_utc": book.ts_utc,
                "venue": book.venue,
                "market_id": book.market_id,
                "token_id": book.token_id,
                "event_id": meta.event_id,
                "title": meta.title,
                "category": meta.category,
                "outcome_side": book.outcome_side,
                "outcome_label": rec.outcome_label,
                "selection_reason": rec.selection_reason,
                "fee_type": _extract_optional_str(meta, "fee_type"),
                "is_yes_no": meta.is_yes_no,
                "neg_risk": _extract_optional_bool(meta, "neg_risk"),
                "end_time_utc": meta.end_time_utc,
                "volume_24h": rec.volume_24h,
                "volume_24h_unit": rec.volume_24h_unit,
                "open_interest": rec.open_interest,
                "tick_size": _extract_optional_decimal(meta, "tick_size"),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid": mid,
                "spread": spread,
                "is_crossed": is_crossed,
                "bid_depth": len(capped_bids),
                "ask_depth": len(capped_asks),
                "bid_prices": [lv.price for lv in capped_bids],
                "ask_prices": [lv.price for lv in capped_asks],
                "bid_sizes": [lv.size for lv in capped_bids],
                "ask_sizes": [lv.size for lv in capped_asks],
            }
        )

    return pl.DataFrame(rows, schema=L2_SNAPSHOT_SCHEMA_V1)


def write_daily_parquet(
    records: Iterable[L2SnapshotRecord],
    output_dir: str | Path,
    snapshot_date: date,
    *,
    max_depth: int = DEFAULT_L2_MAX_DEPTH,
) -> Path:
    """Atomically write daily L2 snapshots to ``<output_dir>/<YYYY-MM-DD>.parquet``.

    Serializes ``records`` with Zstandard compression to a temporary file inside
    ``output_dir`` and replaces the target path via :func:`os.replace` after
    verifying ``schema_version`` is non-null.
    """
    if isinstance(snapshot_date, datetime):
        raise TypeError(f"snapshot_date must be a datetime.date, not datetime: {snapshot_date!r}")

    frame = orderbooks_to_frame(records, max_depth=max_depth, snapshot_date=snapshot_date)
    if frame["schema_version"].null_count() > 0:
        raise ValueError("schema_version column must not contain null values")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / f"{snapshot_date.isoformat()}.parquet"

    fd, tmp_name = tempfile.mkstemp(
        dir=out_dir,
        prefix=f".{snapshot_date.isoformat()}.",
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


def read_snapshots_duckdb(parquet_glob: str | Path) -> pl.DataFrame:
    """Read one or more snapshot Parquet files via DuckDB ``union_by_name=true``."""
    with duckdb.connect(":memory:") as con:
        con.execute("SET TimeZone = 'UTC'")
        rel = con.sql(
            "SELECT * FROM read_parquet(?, union_by_name=true)",
            params=[str(parquet_glob)],
        )
        return pl.DataFrame(rel)
