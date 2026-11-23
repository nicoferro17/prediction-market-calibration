"""Asynchronous daily L2 orderbook snapshot collector orchestrator.

Coordinates live market discovery across Polymarket and Kalshi, deterministic
universe selection (:func:`pm_quant.ingest.universe.select_universe`), batched
L2 orderbook downloads with depth capping, per-venue fault isolation, and
atomic Zstandard-compressed daily Parquet writes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pm_quant.ingest.http import NowFn
from pm_quant.ingest.kalshi import KalshiClient
from pm_quant.ingest.polymarket import PolymarketClient
from pm_quant.ingest.schemas import MarketMetadata, OutcomeSide, Venue
from pm_quant.ingest.storage import (
    DEFAULT_L2_MAX_DEPTH,
    MAX_L2_MAX_DEPTH,
    MIN_L2_MAX_DEPTH,
    L2SnapshotRecord,
    write_daily_parquet,
)
from pm_quant.ingest.universe import (
    LiveMarketCandidate,
    SelectedMarket,
    UniversePolicy,
    select_universe,
)

logger = logging.getLogger(__name__)

DEFAULT_KALSHI_MAX_PAGES: Final[int] = 10
DEFAULT_POLYMARKET_MAX_PAGES: Final[int] = 10
DEFAULT_TARGET_SERIES_MAX_PAGES: Final[int] = 3

_VENUE_ORDER: Final[tuple[Venue, ...]] = ("polymarket", "kalshi")
_REASON_ORDER: Final[tuple[str, ...]] = ("target_series", "volume_core")


def _ensure_utc(dt: datetime, field_name: str) -> datetime:
    """Validate that ``dt`` is timezone-aware and normalize it to UTC."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware (UTC), got naive datetime {dt!r}")
    return dt.astimezone(UTC)


def _validate_max_depth(max_depth: int) -> None:
    if isinstance(max_depth, bool) or not (MIN_L2_MAX_DEPTH <= max_depth <= MAX_L2_MAX_DEPTH):
        raise ValueError(
            f"max_depth must be an integer in [{MIN_L2_MAX_DEPTH}, {MAX_L2_MAX_DEPTH}], "
            f"got {max_depth!r}"
        )


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative int, got {value!r}")


def _normalize_unique_non_empty(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in values:
        text = raw.strip()
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    return tuple(cleaned)


@dataclass(frozen=True, slots=True)
class CollectionSummary:
    """Immutable summary of a completed daily L2 orderbook snapshot collection."""

    snapshot_date: date
    ts_utc: datetime
    parquet_path: Path
    file_size_bytes: int
    total_rows: int
    rows_by_venue: Mapping[str, int]
    selected_by_reason: Mapping[str, int]
    empty_books_count: int
    one_sided_books_count: int
    failed_venues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.snapshot_date, datetime):
            raise TypeError(
                f"snapshot_date must be a datetime.date, not datetime: {self.snapshot_date!r}"
            )
        object.__setattr__(self, "ts_utc", _ensure_utc(self.ts_utc, "ts_utc"))
        if not isinstance(self.parquet_path, Path):
            object.__setattr__(self, "parquet_path", Path(self.parquet_path))
        _validate_non_negative_int(self.file_size_bytes, "file_size_bytes")
        _validate_non_negative_int(self.total_rows, "total_rows")
        _validate_non_negative_int(self.empty_books_count, "empty_books_count")
        _validate_non_negative_int(self.one_sided_books_count, "one_sided_books_count")
        if not isinstance(self.rows_by_venue, MappingProxyType):
            object.__setattr__(self, "rows_by_venue", MappingProxyType(dict(self.rows_by_venue)))
        if not isinstance(self.selected_by_reason, MappingProxyType):
            object.__setattr__(
                self,
                "selected_by_reason",
                MappingProxyType(dict(self.selected_by_reason)),
            )
        object.__setattr__(self, "failed_venues", tuple(dict.fromkeys(self.failed_venues)))


def _resolve_polymarket_token_specs(
    metadata: MarketMetadata,
) -> tuple[tuple[str, OutcomeSide, str], tuple[str, OutcomeSide, str]]:
    """Pair Polymarket binary ``clob_token_ids`` with canonical ``OutcomeSide`` and labels."""
    raw_tokens = metadata.extra.get("clob_token_ids")
    if not isinstance(raw_tokens, tuple | list) or len(raw_tokens) != 2:
        raise ValueError(f"Polymarket market {metadata.market_id!r} missing 2 clob_token_ids")
    if len(metadata.outcomes) != 2:
        raise ValueError(f"Polymarket market {metadata.market_id!r} missing 2 outcomes")

    tok_0 = str(raw_tokens[0]).strip()
    tok_1 = str(raw_tokens[1]).strip()
    if not tok_0 or not tok_1:
        raise ValueError(f"Polymarket market {metadata.market_id!r} has empty clob_token_id")

    out_0 = metadata.outcomes[0].strip()
    out_1 = metadata.outcomes[1].strip()
    if out_0.lower() == "no" and out_1.lower() == "yes":
        sides: tuple[OutcomeSide, OutcomeSide] = ("no", "yes")
    else:
        sides = ("yes", "no")

    return (
        (tok_0, sides[0], out_0),
        (tok_1, sides[1], out_1),
    )


async def _discover_kalshi_candidates(
    client: KalshiClient,
    *,
    max_pages: int,
    target_series: tuple[str, ...],
    target_series_max_pages: int,
) -> list[LiveMarketCandidate]:
    """Fetch open non-MVE Kalshi candidates plus explicit target calibration series."""
    candidates = await client.get_live_market_candidates(
        status="open",
        mve_filter="exclude",
        max_pages=max_pages,
    )
    if not target_series or target_series_max_pages == 0:
        return candidates

    series_results = await asyncio.gather(
        *(
            client.get_live_market_candidates(
                status="open",
                series_ticker=series_ticker,
                mve_filter="exclude",
                max_pages=target_series_max_pages,
            )
            for series_ticker in target_series
        ),
        return_exceptions=True,
    )

    first_error: BaseException | None = None
    series_successes = 0
    for series_ticker, outcome in zip(target_series, series_results, strict=True):
        if isinstance(outcome, BaseException):
            if not isinstance(outcome, Exception):
                raise outcome
            if first_error is None:
                first_error = outcome
            logger.warning(
                "Kalshi target series discovery failed for series_ticker=%s: %s",
                series_ticker,
                outcome,
            )
            continue
        series_successes += 1
        candidates.extend(outcome)

    if not candidates and series_successes == 0 and first_error is not None:
        raise first_error

    return candidates


async def _discover_polymarket_candidates(
    client: PolymarketClient,
    *,
    max_pages: int,
    target_tag_ids: tuple[str, ...],
    target_series_max_pages: int,
) -> list[LiveMarketCandidate]:
    """Fetch active Polymarket candidates sorted by 24h volume plus targeted tag scans."""
    candidates = await client.get_live_market_candidates(
        active=True,
        closed=False,
        order="volume24hr",
        ascending=False,
        max_pages=max_pages,
    )
    if not target_tag_ids or target_series_max_pages == 0:
        return candidates

    tag_results = await asyncio.gather(
        *(
            client.get_live_market_candidates(
                active=True,
                closed=False,
                order="volume24hr",
                ascending=False,
                tag_id=tag_id,
                max_pages=target_series_max_pages,
            )
            for tag_id in target_tag_ids
        ),
        return_exceptions=True,
    )

    first_error: BaseException | None = None
    tag_successes = 0
    for tag_id, outcome in zip(target_tag_ids, tag_results, strict=True):
        if isinstance(outcome, BaseException):
            if not isinstance(outcome, Exception):
                raise outcome
            if first_error is None:
                first_error = outcome
            logger.warning(
                "Polymarket target tag discovery failed for tag_id=%s: %s",
                tag_id,
                outcome,
            )
            continue
        tag_successes += 1
        candidates.extend(outcome)

    if not candidates and tag_successes == 0 and first_error is not None:
        raise first_error

    return candidates


async def _fetch_kalshi_records(
    client: KalshiClient,
    selected_markets: Sequence[SelectedMarket],
    *,
    max_depth: int,
    ts_utc: datetime,
) -> list[L2SnapshotRecord]:
    """Download batched Kalshi orderbooks and emit BOTH ``yes`` and ``no`` snapshot rows."""
    if not selected_markets:
        return []

    selected_by_ticker: dict[str, SelectedMarket] = {
        sel.candidate.metadata.market_id: sel for sel in selected_markets
    }
    books_by_ticker = await client.get_orderbooks_batch(
        list(selected_by_ticker.keys()),
        depth=max_depth,
        ts_utc=ts_utc,
    )

    records: list[L2SnapshotRecord] = []
    missing_tickers: list[str] = []
    for ticker, sel in selected_by_ticker.items():
        book_pair = books_by_ticker.get(ticker)
        if book_pair is None:
            missing_tickers.append(ticker)
            continue
        yes_book, no_book = book_pair
        candidate = sel.candidate
        meta = candidate.metadata
        yes_label = meta.outcomes[0] if len(meta.outcomes) >= 1 else "Yes"
        no_label = meta.outcomes[1] if len(meta.outcomes) >= 2 else "No"
        for book, outcome_label in ((yes_book, yes_label), (no_book, no_label)):
            records.append(
                L2SnapshotRecord(
                    orderbook=book,
                    metadata=meta,
                    selection_reason=sel.selection_reason,
                    volume_24h=candidate.volume_24h,
                    volume_24h_unit=candidate.volume_24h_unit,
                    open_interest=candidate.open_interest,
                    outcome_label=outcome_label,
                )
            )

    if missing_tickers:
        logger.warning(
            "Kalshi /markets/orderbooks omitted %d/%d selected tickers (sample=%s)",
            len(missing_tickers),
            len(selected_by_ticker),
            missing_tickers[:5],
        )

    return records


async def _fetch_polymarket_records(
    client: PolymarketClient,
    selected_markets: Sequence[SelectedMarket],
    *,
    max_depth: int,
    ts_utc: datetime,
) -> list[L2SnapshotRecord]:
    """Download batched Polymarket CLOB orderbooks for BOTH outcome tokens per market."""
    if not selected_markets:
        return []

    all_token_ids: list[str] = []
    outcome_sides: dict[str, OutcomeSide] = {}
    market_ids: dict[str, str] = {}
    token_context: dict[str, tuple[SelectedMarket, str]] = {}

    for sel in selected_markets:
        meta = sel.candidate.metadata
        token_specs = _resolve_polymarket_token_specs(meta)
        for token_id, outcome_side, outcome_label in token_specs:
            if token_id in token_context:
                logger.warning(
                    "Duplicate Polymarket token_id=%s encountered on market_id=%s; skipping",
                    token_id,
                    meta.market_id,
                )
                continue
            all_token_ids.append(token_id)
            outcome_sides[token_id] = outcome_side
            market_ids[token_id] = meta.market_id
            token_context[token_id] = (sel, outcome_label)

    if not all_token_ids:
        return []

    books_by_token = await client.get_orderbooks_batch(
        all_token_ids,
        max_depth=max_depth,
        outcome_sides=outcome_sides,
        market_ids=market_ids,
        ts_utc=ts_utc,
    )

    records: list[L2SnapshotRecord] = []
    missing_tokens: list[str] = []
    for token_id in all_token_ids:
        book = books_by_token.get(token_id)
        if book is None:
            missing_tokens.append(token_id)
            continue
        sel, outcome_label = token_context[token_id]
        candidate = sel.candidate
        records.append(
            L2SnapshotRecord(
                orderbook=book,
                metadata=candidate.metadata,
                selection_reason=sel.selection_reason,
                volume_24h=candidate.volume_24h,
                volume_24h_unit=candidate.volume_24h_unit,
                open_interest=candidate.open_interest,
                outcome_label=outcome_label,
            )
        )

    if missing_tokens:
        logger.warning(
            "Polymarket CLOB /books omitted %d/%d selected token_ids (sample=%s)",
            len(missing_tokens),
            len(all_token_ids),
            missing_tokens[:5],
        )

    return records


async def collect_daily_l2(
    *,
    output_dir: str | Path,
    policy: UniversePolicy | None = None,
    max_depth: int = DEFAULT_L2_MAX_DEPTH,
    snapshot_date: date | None = None,
    kalshi_client: KalshiClient | None = None,
    polymarket_client: PolymarketClient | None = None,
    now_fn: NowFn | None = None,
    kalshi_max_pages: int = DEFAULT_KALSHI_MAX_PAGES,
    polymarket_max_pages: int = DEFAULT_POLYMARKET_MAX_PAGES,
    target_series_max_pages: int = DEFAULT_TARGET_SERIES_MAX_PAGES,
    kalshi_target_series: Sequence[str] | None = None,
) -> CollectionSummary:
    """Collect daily L2 orderbook snapshots from Polymarket and Kalshi into Parquet.

    Pipeline:
    1. Concurrently discover live market candidates from Kalshi (``status='open'``,
       ``mve_filter='exclude'``, plus explicit target series queries) and Polymarket
       (``active=True, closed=False, order='volume24hr', ascending=False``, plus
       targeted queries for ``policy.polymarket_target_tag_ids``).
    2. Select the deterministic bounded universe via :func:`select_universe`.
    3. Concurrently download batched L2 orderbooks capped to ``max_depth`` (emitting
       both YES and NO sides on Kalshi and both outcome tokens on Polymarket with
       ``outcome_label`` populated from ``metadata.outcomes``).
    4. Isolate single-venue errors (logging a warning and recording ``failed_venues``),
       raising :class:`RuntimeError` if total collected rows equal 0.
    5. Atomically write ``<output_dir>/<YYYY-MM-DD>.parquet`` via
       :func:`write_daily_parquet`.
    """
    _validate_max_depth(max_depth)
    if snapshot_date is not None and isinstance(snapshot_date, datetime):
        raise TypeError(f"snapshot_date must be a datetime.date, not datetime: {snapshot_date!r}")
    _validate_non_negative_int(kalshi_max_pages, "kalshi_max_pages")
    _validate_non_negative_int(polymarket_max_pages, "polymarket_max_pages")
    _validate_non_negative_int(target_series_max_pages, "target_series_max_pages")

    active_policy = policy if policy is not None else UniversePolicy()
    resolved_now_fn: NowFn = now_fn if now_fn is not None else (lambda: datetime.now(UTC))
    ts_utc = _ensure_utc(resolved_now_fn(), "ts_utc")
    resolved_snapshot_date = snapshot_date if snapshot_date is not None else ts_utc.date()
    resolved_kalshi_series = (
        _normalize_unique_non_empty(kalshi_target_series)
        if kalshi_target_series is not None
        else active_policy.target_series_prefixes
    )

    failed_venue_set: set[Venue] = set()

    async with AsyncExitStack() as stack:
        active_kalshi = (
            kalshi_client
            if kalshi_client is not None
            else await stack.enter_async_context(KalshiClient(now_fn=resolved_now_fn))
        )
        active_polymarket = (
            polymarket_client
            if polymarket_client is not None
            else await stack.enter_async_context(PolymarketClient(now_fn=resolved_now_fn))
        )

        # Stage 1: Discover live market candidates from both venues concurrently.
        poly_disc_result, kalshi_disc_result = await asyncio.gather(
            _discover_polymarket_candidates(
                active_polymarket,
                max_pages=polymarket_max_pages,
                target_tag_ids=active_policy.polymarket_target_tag_ids,
                target_series_max_pages=target_series_max_pages,
            ),
            _discover_kalshi_candidates(
                active_kalshi,
                max_pages=kalshi_max_pages,
                target_series=resolved_kalshi_series,
                target_series_max_pages=target_series_max_pages,
            ),
            return_exceptions=True,
        )

        all_candidates: list[LiveMarketCandidate] = []
        discovery_outcomes: tuple[tuple[Venue, list[LiveMarketCandidate] | BaseException], ...] = (
            ("polymarket", poly_disc_result),
            ("kalshi", kalshi_disc_result),
        )
        for venue, disc_outcome in discovery_outcomes:
            if isinstance(disc_outcome, BaseException):
                if not isinstance(disc_outcome, Exception):
                    raise disc_outcome
                failed_venue_set.add(venue)
                logger.warning(
                    "Venue %s candidate discovery failed; isolating venue: %s",
                    venue,
                    disc_outcome,
                    exc_info=disc_outcome,
                )
            else:
                all_candidates.extend(disc_outcome)

        # Stage 2: Deterministic universe selection.
        selected = select_universe(all_candidates, active_policy, now_utc=ts_utc)
        selected_poly = [s for s in selected if s.candidate.metadata.venue == "polymarket"]
        selected_kalshi = [s for s in selected if s.candidate.metadata.venue == "kalshi"]

        # Stage 3: Batched L2 orderbook collection with per-venue isolation.
        poly_books_result, kalshi_books_result = await asyncio.gather(
            _fetch_polymarket_records(
                active_polymarket,
                selected_poly,
                max_depth=max_depth,
                ts_utc=ts_utc,
            ),
            _fetch_kalshi_records(
                active_kalshi,
                selected_kalshi,
                max_depth=max_depth,
                ts_utc=ts_utc,
            ),
            return_exceptions=True,
        )

        records: list[L2SnapshotRecord] = []
        book_outcomes: tuple[tuple[Venue, list[L2SnapshotRecord] | BaseException], ...] = (
            ("polymarket", poly_books_result),
            ("kalshi", kalshi_books_result),
        )
        for venue, books_outcome in book_outcomes:
            if isinstance(books_outcome, BaseException):
                if not isinstance(books_outcome, Exception):
                    raise books_outcome
                failed_venue_set.add(venue)
                logger.warning(
                    "Venue %s orderbook batch fetch failed; isolating venue: %s",
                    venue,
                    books_outcome,
                    exc_info=books_outcome,
                )
            else:
                records.extend(books_outcome)

    failed_venues = tuple(venue for venue in _VENUE_ORDER if venue in failed_venue_set)

    if not records:
        detail = f" (failed_venues={list(failed_venues)})" if failed_venues else ""
        raise RuntimeError(f"Daily L2 collection produced 0 rows across all venues{detail}")

    # Stage 4: Atomic daily Parquet write and summary aggregation.
    parquet_path = write_daily_parquet(
        records,
        output_dir,
        resolved_snapshot_date,
        max_depth=max_depth,
    )
    file_size_bytes = parquet_path.stat().st_size

    rows_by_venue: dict[str, int] = {venue: 0 for venue in _VENUE_ORDER}
    empty_books_count = 0
    one_sided_books_count = 0
    for rec in records:
        rows_by_venue[rec.orderbook.venue] = rows_by_venue.get(rec.orderbook.venue, 0) + 1
        has_bids = bool(rec.orderbook.bids)
        has_asks = bool(rec.orderbook.asks)
        if not has_bids and not has_asks:
            empty_books_count += 1
        elif has_bids != has_asks:
            one_sided_books_count += 1

    selected_by_reason: dict[str, int] = {reason: 0 for reason in _REASON_ORDER}
    for sel in selected:
        if sel.candidate.metadata.venue in failed_venue_set:
            continue
        selected_by_reason[sel.selection_reason] = (
            selected_by_reason.get(sel.selection_reason, 0) + 1
        )

    return CollectionSummary(
        snapshot_date=resolved_snapshot_date,
        ts_utc=ts_utc,
        parquet_path=parquet_path,
        file_size_bytes=file_size_bytes,
        total_rows=len(records),
        rows_by_venue=rows_by_venue,
        selected_by_reason=selected_by_reason,
        empty_books_count=empty_books_count,
        one_sided_books_count=one_sided_books_count,
        failed_venues=failed_venues,
    )
