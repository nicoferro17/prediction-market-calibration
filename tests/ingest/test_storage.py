"""Offline unit tests for L2 snapshot Parquet storage and DuckDB union_by_name reads."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from pm_quant.ingest import (
    DEFAULT_L2_MAX_DEPTH,
    L2_SNAPSHOT_SCHEMA_V1,
    MAX_L2_MAX_DEPTH,
    MIN_L2_MAX_DEPTH,
    SCHEMA_VERSION,
    L2SnapshotRecord,
    MarketMetadata,
    NormalizedOrderBook,
    OutcomeSide,
    PriceLevel,
    SelectionReason,
    Venue,
    VolumeUnit,
    orderbooks_to_frame,
    read_snapshots_duckdb,
    write_daily_parquet,
)


def _make_record(
    *,
    venue: Venue = "polymarket",
    market_id: str = "0xcond_001",
    outcome_side: OutcomeSide = "yes",
    token_id: str | None = "tok_yes_1",
    ts_utc: datetime = datetime(2026, 12, 1, 14, 30, 0, tzinfo=UTC),
    bids: tuple[PriceLevel, ...] = (
        PriceLevel(Decimal("0.47"), Decimal("150.25")),
        PriceLevel(Decimal("0.46"), Decimal("300.00")),
    ),
    asks: tuple[PriceLevel, ...] = (
        PriceLevel(Decimal("0.49"), Decimal("120.50")),
        PriceLevel(Decimal("0.50"), Decimal("250.00")),
    ),
    event_id: str | None = "evt_001",
    title: str = "Will CPI exceed 3.0%?",
    category: str | None = "economics",
    is_yes_no: bool = True,
    end_time_utc: datetime | None = datetime(2026, 12, 15, 13, 30, 0, tzinfo=UTC),
    extra: dict[str, Any] | None = None,
    selection_reason: SelectionReason = "target_series",
    volume_24h: Decimal | None = Decimal("12500.50"),
    volume_24h_unit: VolumeUnit | None = "usd",
    open_interest: Decimal | None = Decimal("45000.00"),
    outcome_label: str | None = "Yes",
) -> L2SnapshotRecord:
    default_extra: dict[str, Any] = (
        {"fee_type": "crypto_fees", "neg_risk": False, "tick_size": "0.01"}
        if venue == "polymarket"
        else {"tick_size": "0.01"}
    )
    meta = MarketMetadata(
        venue=venue,
        market_id=market_id,
        event_id=event_id,
        title=title,
        category=category,
        outcomes=("Yes", "No") if is_yes_no else ("Team A", "Team B"),
        is_yes_no=is_yes_no,
        end_time_utc=end_time_utc,
        status="active",
        extra=default_extra if extra is None else extra,
    )
    book = NormalizedOrderBook(
        venue=venue,
        market_id=market_id,
        outcome_side=outcome_side,
        ts_utc=ts_utc,
        bids=bids,
        asks=asks,
        token_id=token_id,
    )
    return L2SnapshotRecord(
        orderbook=book,
        metadata=meta,
        selection_reason=selection_reason,
        volume_24h=volume_24h,
        volume_24h_unit=volume_24h_unit,
        open_interest=open_interest,
        outcome_label=outcome_label,
    )


def test_constants_and_schema_v1_exact_columns_and_dtypes() -> None:
    assert DEFAULT_L2_MAX_DEPTH == 20
    assert MIN_L2_MAX_DEPTH == 15
    assert MAX_L2_MAX_DEPTH == 20

    expected_schema = pl.Schema(
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

    assert L2_SNAPSHOT_SCHEMA_V1 == expected_schema
    assert list(L2_SNAPSHOT_SCHEMA_V1.keys()) == list(expected_schema.keys())

    poly_rec = _make_record(venue="polymarket", market_id="0xcond_b", outcome_side="yes")
    kalshi_rec = _make_record(
        venue="kalshi",
        market_id="KXCPI-26DEC",
        outcome_side="yes",
        token_id=None,
        volume_24h_unit="contracts",
        extra={},
        outcome_label=None,
    )
    frame = orderbooks_to_frame([poly_rec, kalshi_rec])

    assert frame.schema == expected_schema
    assert frame.columns == list(expected_schema.keys())
    assert frame["schema_version"].to_list() == [SCHEMA_VERSION, SCHEMA_VERSION]
    assert frame["schema_version"].null_count() == 0

    # Deterministic ordering by (venue, market_id, outcome_side): kalshi before polymarket.
    assert frame["venue"].to_list() == ["kalshi", "polymarket"]
    assert frame["neg_risk"].to_list() == [None, False]
    assert frame["fee_type"].to_list() == [None, "crypto_fees"]
    assert frame["tick_size"].to_list() == [None, Decimal("0.010000")]
    assert frame["best_bid"][1] == Decimal("0.470000")
    assert frame["best_ask"][1] == Decimal("0.490000")
    assert frame["mid"][1] == Decimal("0.480000")
    assert frame["spread"][1] == Decimal("0.020000")
    assert frame["is_crossed"][1] is False


@pytest.mark.parametrize("max_depth", [15, 20])
def test_defensive_l2_depth_cap_sorts_and_keeps_best_levels(tmp_path: Path, max_depth: int) -> None:
    # Build 35 bids in [0.13, 0.47] and 35 asks in [0.51, 0.85] in scrambled order.
    ordered_bids = [
        PriceLevel(Decimal(f"0.{47 - i:02d}"), Decimal(f"{100 + i}.125")) for i in range(35)
    ]
    ordered_asks = [
        PriceLevel(Decimal(f"0.{51 + i:02d}"), Decimal(f"{200 + i}.250")) for i in range(35)
    ]

    # Deterministic non-monotonic scramble (stride 7 mod 35, gcd(7, 35) != 1 so use stride 11).
    scrambled_bids = tuple(ordered_bids[(i * 11) % 35] for i in range(35))
    scrambled_asks = tuple(ordered_asks[(i * 13) % 35] for i in range(35))
    assert len(scrambled_bids) == 35
    assert len(scrambled_asks) == 35

    rec = _make_record(bids=scrambled_bids, asks=scrambled_asks)
    parquet_path = write_daily_parquet([rec], tmp_path, date(2026, 12, 1), max_depth=max_depth)
    df = read_snapshots_duckdb(parquet_path)

    assert df.schema == L2_SNAPSHOT_SCHEMA_V1
    assert df.height == 1
    assert df["bid_depth"][0] == max_depth
    assert df["ask_depth"][0] == max_depth

    expected_bid_prices = [
        lv.price.quantize(Decimal("0.000001")) for lv in ordered_bids[:max_depth]
    ]
    expected_bid_sizes = [lv.size.quantize(Decimal("0.000001")) for lv in ordered_bids[:max_depth]]
    expected_ask_prices = [
        lv.price.quantize(Decimal("0.000001")) for lv in ordered_asks[:max_depth]
    ]
    expected_ask_sizes = [lv.size.quantize(Decimal("0.000001")) for lv in ordered_asks[:max_depth]]

    actual_bid_prices = df["bid_prices"][0].to_list()
    actual_bid_sizes = df["bid_sizes"][0].to_list()
    actual_ask_prices = df["ask_prices"][0].to_list()
    actual_ask_sizes = df["ask_sizes"][0].to_list()

    assert actual_bid_prices == expected_bid_prices
    assert actual_bid_sizes == expected_bid_sizes
    assert actual_ask_prices == expected_ask_prices
    assert actual_ask_sizes == expected_ask_sizes
    assert actual_bid_prices[0] == Decimal("0.470000")
    assert actual_ask_prices[0] == Decimal("0.510000")
    assert df["best_bid"][0] == Decimal("0.470000")
    assert df["best_ask"][0] == Decimal("0.510000")
    assert df["mid"][0] == Decimal("0.490000")
    assert df["spread"][0] == Decimal("0.040000")


@pytest.mark.parametrize("bad_depth", [-1, 0, 14, 21, 50, True, False])
def test_invalid_max_depth_raises_value_error(tmp_path: Path, bad_depth: int) -> None:
    rec = _make_record()
    with pytest.raises(ValueError, match="max_depth"):
        orderbooks_to_frame([rec], max_depth=bad_depth)

    with pytest.raises(ValueError, match="max_depth"):
        write_daily_parquet([rec], tmp_path, date(2026, 12, 1), max_depth=bad_depth)


def test_round_trip_empty_one_sided_and_empty_records(tmp_path: Path) -> None:
    empty_book_rec = _make_record(market_id="m_empty", bids=(), asks=())
    bids_only_rec = _make_record(
        market_id="m_bids_only",
        bids=(PriceLevel(Decimal("0.40"), Decimal("50")),),
        asks=(),
    )
    asks_only_rec = _make_record(
        market_id="m_asks_only",
        bids=(),
        asks=(PriceLevel(Decimal("0.60"), Decimal("75")),),
    )

    path = write_daily_parquet(
        [asks_only_rec, empty_book_rec, bids_only_rec],
        tmp_path,
        date(2026, 12, 1),
    )
    assert path == tmp_path / "2026-12-01.parquet"
    assert list(tmp_path.glob("*.tmp")) == []

    df = read_snapshots_duckdb(path)
    assert df.schema == L2_SNAPSHOT_SCHEMA_V1
    assert df["market_id"].to_list() == ["m_asks_only", "m_bids_only", "m_empty"]

    # Asks-only row
    assert df["best_bid"][0] is None
    assert df["best_ask"][0] == Decimal("0.600000")
    assert df["mid"][0] is None
    assert df["spread"][0] is None
    assert df["is_crossed"][0] is False
    assert df["bid_depth"][0] == 0
    assert df["ask_depth"][0] == 1
    assert df["bid_prices"][0].to_list() == []
    assert df["ask_prices"][0].to_list() == [Decimal("0.600000")]

    # Bids-only row
    assert df["best_bid"][1] == Decimal("0.400000")
    assert df["best_ask"][1] is None
    assert df["mid"][1] is None
    assert df["spread"][1] is None
    assert df["is_crossed"][1] is False
    assert df["bid_depth"][1] == 1
    assert df["ask_depth"][1] == 0
    assert df["bid_prices"][1].to_list() == [Decimal("0.400000")]
    assert df["ask_prices"][1].to_list() == []

    # Empty-book row
    assert df["best_bid"][2] is None
    assert df["best_ask"][2] is None
    assert df["mid"][2] is None
    assert df["spread"][2] is None
    assert df["is_crossed"][2] is False
    assert df["bid_depth"][2] == 0
    assert df["ask_depth"][2] == 0
    assert df["bid_prices"][2].to_list() == []
    assert df["ask_prices"][2].to_list() == []

    # Empty records list round-trip
    empty_path = write_daily_parquet([], tmp_path, date(2026, 12, 2))
    empty_df = read_snapshots_duckdb(empty_path)
    assert empty_df.height == 0
    assert empty_df.schema == L2_SNAPSHOT_SCHEMA_V1


def test_duckdb_union_by_name_reads_v1_and_v2_with_extra_column(tmp_path: Path) -> None:
    rec_v1 = _make_record(
        market_id="0xcond_v1",
        ts_utc=datetime(2026, 12, 1, 12, 0, 0, tzinfo=UTC),
        bids=(PriceLevel(Decimal("0.47"), Decimal("100.50")),),
        asks=(PriceLevel(Decimal("0.49"), Decimal("200.25")),),
    )
    write_daily_parquet([rec_v1], tmp_path, date(2026, 12, 1))

    rec_v2 = _make_record(
        market_id="0xcond_v2",
        ts_utc=datetime(2026, 12, 2, 12, 0, 0, tzinfo=UTC),
        bids=(PriceLevel(Decimal("0.62"), Decimal("300.75")),),
        asks=(PriceLevel(Decimal("0.64"), Decimal("150.00")),),
    )
    frame_v2 = orderbooks_to_frame([rec_v2]).with_columns(
        pl.lit("v2").alias("schema_version"),
        pl.lit("tier_maker_rebate").cast(pl.String).alias("fee_tier"),
    )
    frame_v2.write_parquet(tmp_path / "2026-12-02.parquet", compression="zstd")

    combined = read_snapshots_duckdb(f"{tmp_path}/*.parquet").sort("snapshot_date")

    assert combined.height == 2
    assert combined["schema_version"].to_list() == ["v1", "v2"]
    assert "fee_tier" in combined.columns
    assert combined["fee_tier"][0] is None
    assert combined["fee_tier"][1] == "tier_maker_rebate"

    # Decimal scalar and list columns remain intact across schema evolution.
    assert combined.schema["best_bid"] == pl.Decimal(12, 6)
    assert combined.schema["bid_prices"] == pl.List(pl.Decimal(12, 6))
    assert combined.schema["bid_sizes"] == pl.List(pl.Decimal(18, 6))
    assert combined["best_bid"].to_list() == [Decimal("0.470000"), Decimal("0.620000")]
    assert combined["best_ask"].to_list() == [Decimal("0.490000"), Decimal("0.640000")]
    assert combined["bid_prices"][0].to_list() == [Decimal("0.470000")]
    assert combined["bid_prices"][1].to_list() == [Decimal("0.620000")]
    assert combined["bid_sizes"][0].to_list() == [Decimal("100.500000")]
    assert combined["bid_sizes"][1].to_list() == [Decimal("300.750000")]


def test_record_immutability_and_validation_guards(tmp_path: Path) -> None:
    rec = _make_record()
    with pytest.raises(FrozenInstanceError):
        rec.outcome_label = "No"  # type: ignore[misc]

    # Mismatched venue
    with pytest.raises(ValueError, match="venue"):
        L2SnapshotRecord(
            orderbook=rec.orderbook,
            metadata=MarketMetadata(
                venue="kalshi",
                market_id=rec.metadata.market_id,
                event_id=None,
                title="Mismatch",
                category=None,
                outcomes=("Yes", "No"),
                is_yes_no=True,
                end_time_utc=None,
                status="open",
            ),
            selection_reason="volume_core",
        )

    # Duplicate (venue, market_id, outcome_side)
    with pytest.raises(ValueError, match="Duplicate"):
        orderbooks_to_frame([rec, rec])

    # Reject datetime passed as snapshot_date
    with pytest.raises(TypeError, match="datetime.date"):
        write_daily_parquet([rec], tmp_path, datetime(2026, 12, 1, 0, 0, tzinfo=UTC))
