"""Offline unit tests for Week 3 Block 1 historical schema, parsers, and tau sampler."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from pm_quant.ingest import (
    DEFAULT_BACKFILL_SEED,
    DEFAULT_FALLBACK_LOOKBACK_DAYS,
    DEFAULT_MAX_MARKETS_PER_VENUE,
    DEFAULT_MAX_PRICE,
    DEFAULT_MIN_PRICE,
    DEFAULT_MIN_VOL_USD,
    DEFAULT_PRE_RESOLUTION_BUFFER_HOURS,
    DEFAULT_SHORT_TAU_TARGET_SHARE,
    HISTORICAL_CALIBRATION_SCHEMA_V1,
    SCHEMA_VERSION,
    BackfillPerimeter,
    BinaryOutcome,
    HistoricalOutcomeSide,
    HistoricalSnapshotRecord,
    HistoricalVolumeUnit,
    MarketMetadata,
    ResolvedMarket,
    Venue,
    compute_tau_metrics,
    historical_records_to_frame,
    parse_gamma_market,
    parse_gamma_resolved_market,
    parse_kalshi_resolved_market,
    read_historical_duckdb,
    sample_variable_tau_snapshot_ts,
    write_historical_parquet,
)

BASE_START = datetime(2026, 5, 1, 0, 0, 0, tzinfo=UTC)
BASE_END = datetime(2026, 11, 1, 0, 0, 0, tzinfo=UTC)

FORBIDDEN_GAMMA_KEYS = {
    "volume",
    "volumeNum",
    "volume_num",
    "volumeClob",
    "volume_clob",
    "volume24hr",
    "liquidity",
    "outcomePrices",
    "outcome_prices",
    "bestBid",
    "bestAsk",
    "lastTradePrice",
}


def _make_perimeter(**overrides: Any) -> BackfillPerimeter:
    defaults: dict[str, Any] = {
        "start_date_utc": BASE_START,
        "end_date_utc": BASE_END,
    }
    defaults.update(overrides)
    return BackfillPerimeter(**defaults)


def _make_resolved_market(
    *,
    venue: Venue = "polymarket",
    market_id: str = "0xcond_001",
    open_time_utc: datetime | None = datetime(2026, 8, 1, 0, 0, 0, tzinfo=UTC),
    end_time_utc: datetime | None = datetime(2026, 9, 30, 0, 0, 0, tzinfo=UTC),
    settlement_ts_utc: datetime | None = None,
    outcome_y: BinaryOutcome = 1,
    is_yes_no: bool = True,
) -> ResolvedMarket:
    meta = MarketMetadata(
        venue=venue,
        market_id=market_id,
        event_id=f"evt_{market_id}",
        title=f"Market {market_id}",
        category="economics",
        outcomes=("Yes", "No") if is_yes_no else ("Team A", "Team B"),
        is_yes_no=is_yes_no,
        end_time_utc=end_time_utc,
        status="closed" if venue == "polymarket" else "settled",
    )
    return ResolvedMarket(
        metadata=meta,
        outcome_y=outcome_y,
        open_time_utc=open_time_utc,
        settlement_ts_utc=settlement_ts_utc,
    )


def _make_snapshot_record(
    *,
    venue: Venue = "polymarket",
    market_id: str = "0xcond_001",
    token_id: str | None = "tok_yes_1",
    event_id: str | None = "evt_001",
    title: str = "Will CPI exceed 3.0%?",
    category: str | None = "economics",
    outcome_side: HistoricalOutcomeSide = "yes",
    outcome_label: str | None = "Yes",
    is_yes_no: bool = True,
    neg_risk: bool | None = False,
    fee_type: str | None = "economics",
    open_time_utc: datetime | None = datetime(2026, 8, 1, 0, 0, 0, tzinfo=UTC),
    snapshot_ts_utc: datetime = datetime(2026, 9, 29, 18, 0, 0, tzinfo=UTC),
    end_time_utc: datetime = datetime(2026, 9, 30, 0, 0, 0, tzinfo=UTC),
    p_mkt: Decimal = Decimal("0.625000"),
    vol_usd_le_t: Decimal = Decimal("4520.50"),
    vol_contracts_le_t: Decimal | None = None,
    vol_unit: HistoricalVolumeUnit = "usd",
    trade_count_le_t: int | None = 42,
    trades_truncated: bool = False,
    price_source: str = "poly_v2",
    outcome_y: BinaryOutcome = 1,
    hist_yes_bid: Decimal | None = None,
    hist_yes_ask: Decimal | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> HistoricalSnapshotRecord:
    tau_days, log1p_tau = compute_tau_metrics(snapshot_ts_utc, end_time_utc)
    return HistoricalSnapshotRecord(
        venue=venue,
        market_id=market_id,
        token_id=token_id,
        event_id=event_id,
        title=title,
        category=category,
        outcome_side=outcome_side,
        outcome_label=outcome_label,
        is_yes_no=is_yes_no,
        neg_risk=neg_risk,
        open_time_utc=open_time_utc,
        snapshot_ts_utc=snapshot_ts_utc,
        end_time_utc=end_time_utc,
        tau_days=tau_days,
        log1p_tau=log1p_tau,
        p_mkt=p_mkt,
        vol_usd_le_t=vol_usd_le_t,
        vol_contracts_le_t=vol_contracts_le_t,
        vol_unit=vol_unit,
        trade_count_le_t=trade_count_le_t,
        price_source=price_source,
        outcome_y=outcome_y,
        fee_type=fee_type,
        trades_truncated=trades_truncated,
        hist_yes_bid=hist_yes_bid,
        hist_yes_ask=hist_yes_ask,
        schema_version=schema_version,
    )


# ---------------------------------------------------------------------------
# 1. Dataclass defaults, immutability, and validation
# ---------------------------------------------------------------------------


def test_perimeter_defaults_normalization_and_immutability() -> None:
    tz_plus2 = timezone(timedelta(hours=2))
    perimeter = BackfillPerimeter(
        start_date_utc=datetime(2026, 5, 1, 2, 0, 0, tzinfo=tz_plus2),
        end_date_utc=datetime(2026, 11, 1, 2, 0, 0, tzinfo=tz_plus2),
    )
    assert perimeter.start_date_utc == BASE_START
    assert perimeter.end_date_utc == BASE_END
    assert perimeter.max_markets_per_venue == DEFAULT_MAX_MARKETS_PER_VENUE == 500
    assert perimeter.min_vol_usd == DEFAULT_MIN_VOL_USD == Decimal("1000")
    assert perimeter.min_price == DEFAULT_MIN_PRICE == Decimal("0.01")
    assert perimeter.max_price == DEFAULT_MAX_PRICE == Decimal("0.99")
    assert perimeter.pre_resolution_buffer_hours == DEFAULT_PRE_RESOLUTION_BUFFER_HOURS == 2
    assert perimeter.seed == DEFAULT_BACKFILL_SEED == 42
    assert perimeter.short_tau_target_share == DEFAULT_SHORT_TAU_TARGET_SHARE == 0.25

    with pytest.raises(FrozenInstanceError):
        perimeter.seed = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_date_utc": datetime(2026, 5, 1, 0, 0, 0)},  # naive start
        {"end_date_utc": datetime(2026, 11, 1, 0, 0, 0)},  # naive end
        {"start_date_utc": BASE_END, "end_date_utc": BASE_START},  # reversed
        {"max_markets_per_venue": 0},
        {"max_markets_per_venue": True},
        {"min_vol_usd": Decimal("-1")},
        {"min_price": Decimal("0.99"), "max_price": Decimal("0.99")},
        {"min_price": Decimal("-0.01")},
        {"max_price": Decimal("1.01")},
        {"pre_resolution_buffer_hours": 0},
        {"pre_resolution_buffer_hours": True},
        {"seed": True},
        {"short_tau_target_share": -0.1},
        {"short_tau_target_share": 1.1},
        {"short_tau_target_share": math.nan},
        {"short_tau_target_share": True},
    ],
)
def test_perimeter_validation_rejects_invalid_inputs(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _make_perimeter(**kwargs)


def test_resolved_market_and_snapshot_record_validation() -> None:
    meta = MarketMetadata(
        venue="kalshi",
        market_id="KXCPI-26OCT",
        event_id="KXCPI",
        title="CPI",
        category="economics",
        outcomes=("Yes", "No"),
        is_yes_no=True,
        end_time_utc=BASE_END,
        status="settled",
    )
    with pytest.raises(ValueError, match="outcome_y"):
        ResolvedMarket(metadata=meta, outcome_y=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outcome_y"):
        ResolvedMarket(metadata=meta, outcome_y=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="naive"):
        ResolvedMarket(metadata=meta, outcome_y=1, open_time_utc=datetime(2026, 10, 1))
    with pytest.raises(ValueError, match="naive"):
        ResolvedMarket(metadata=meta, outcome_y=1, settlement_ts_utc=datetime(2026, 11, 2))

    tau_6h, log1p_6h = compute_tau_metrics(
        datetime(2026, 9, 29, 18, 0, 0, tzinfo=UTC),
        datetime(2026, 9, 30, 0, 0, 0, tzinfo=UTC),
    )
    assert tau_6h == Decimal("0.250000")
    assert log1p_6h == pytest.approx(math.log1p(0.25))

    with pytest.raises(ValueError, match="strictly before"):
        compute_tau_metrics(BASE_END, BASE_END)

    rec = _make_snapshot_record()
    with pytest.raises(FrozenInstanceError):
        rec.outcome_y = 0  # type: ignore[misc]

    # Invalid HistoricalSnapshotRecord guards
    with pytest.raises(ValueError, match="strictly before"):
        HistoricalSnapshotRecord(
            venue="polymarket",
            market_id="m1",
            token_id="t1",
            event_id=None,
            title="t",
            category=None,
            outcome_side="yes",
            outcome_label="Yes",
            is_yes_no=True,
            neg_risk=False,
            open_time_utc=None,
            snapshot_ts_utc=BASE_END,
            end_time_utc=BASE_END,
            tau_days=Decimal("1"),
            log1p_tau=math.log1p(1.0),
            p_mkt=Decimal("0.5"),
            vol_usd_le_t=Decimal("1000"),
            vol_contracts_le_t=None,
            vol_unit="usd",
            trade_count_le_t=10,
            price_source="poly_v2",
            outcome_y=1,
        )

    with pytest.raises(ValueError, match="p_mkt"):
        _make_snapshot_record(p_mkt=Decimal("1.05"))
    with pytest.raises(ValueError, match="vol_usd_le_t"):
        _make_snapshot_record(vol_usd_le_t=Decimal("-1"))
    with pytest.raises(ValueError, match="trade_count_le_t"):
        _make_snapshot_record(trade_count_le_t=-1)
    with pytest.raises(ValueError, match="price_source"):
        _make_snapshot_record(price_source="   ")
    with pytest.raises(ValueError, match="hist_yes_bid"):
        _make_snapshot_record(hist_yes_bid=Decimal("-0.01"))
    with pytest.raises(ValueError, match="hist_yes_ask"):
        _make_snapshot_record(hist_yes_ask=Decimal("1.5"))


# ---------------------------------------------------------------------------
# 2. Pure resolved-market parsers (Gamma & Kalshi) + anti-leakage checks
# ---------------------------------------------------------------------------


def test_parse_gamma_resolved_market_binary_resolutions_and_anti_leakage() -> None:
    raw_yes_win: dict[str, Any] = {
        "id": "77100",
        "conditionId": "0xresolved_yes",
        "question": "Will CPI exceed 3.0% in September 2026?",
        "slug": "cpi-sep-2026",
        "category": "Economics",
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": '["101", "102"]',
        "outcomePrices": '["1", "0"]',
        "startDate": "2026-08-15T12:00:00Z",
        "endDate": "2026-09-15T12:30:00Z",
        "closed": True,
        "active": True,  # Gamma frequently leaves active=True alongside closed=True
        "negRisk": False,
        "feeType": "economics",
        # Poison final lifetime fields: must NEVER leak into MarketMetadata.extra.
        "volume": "9876543.21",
        "volumeNum": 9876543.21,
        "volumeClob": 9800000.0,
        "volume24hr": 120000.0,
        "liquidity": "500000.00",
        "bestBid": 0.99,
        "bestAsk": 1.0,
        "lastTradePrice": 1.0,
    }

    resolved_yes = parse_gamma_resolved_market(raw_yes_win)
    assert resolved_yes is not None
    assert resolved_yes.outcome_y == 1
    assert resolved_yes.open_time_utc == datetime(2026, 8, 15, 12, 0, 0, tzinfo=UTC)
    assert resolved_yes.metadata.end_time_utc == datetime(2026, 9, 15, 12, 30, 0, tzinfo=UTC)
    assert resolved_yes.settlement_ts_utc is None
    assert resolved_yes.metadata == parse_gamma_market(raw_yes_win)
    assert FORBIDDEN_GAMMA_KEYS.isdisjoint(resolved_yes.metadata.extra.keys())

    # 0/1 -> y=0
    raw_no_win = {**raw_yes_win, "conditionId": "0xresolved_no", "outcomePrices": ["0.0", "1.0"]}
    resolved_no = parse_gamma_resolved_market(raw_no_win)
    assert resolved_no is not None
    assert resolved_no.outcome_y == 0

    # Reversed YES/NO ordering: ["No", "Yes"] with ["0", "1"] means Yes won -> y=1.
    raw_reversed = {
        **raw_yes_win,
        "conditionId": "0xreversed_yes_no",
        "outcomes": ["No", "Yes"],
        "outcomePrices": ["0", "1"],
    }
    resolved_reversed = parse_gamma_resolved_market(raw_reversed)
    assert resolved_reversed is not None
    assert resolved_reversed.metadata.is_yes_no is True
    assert resolved_reversed.outcome_y == 1

    # Symmetric contract ["Lakers", "Celtics"]: tracks outcome index 0.
    raw_sym_win = {
        **raw_yes_win,
        "conditionId": "0xsym_0",
        "outcomes": ["Lakers", "Celtics"],
        "outcomePrices": ["1", "0"],
    }
    resolved_sym = parse_gamma_resolved_market(raw_sym_win)
    assert resolved_sym is not None
    assert resolved_sym.metadata.is_yes_no is False
    assert resolved_sym.outcome_y == 1

    # Fallback to tokens[*].winner when outcomePrices is absent.
    raw_tokens_only: dict[str, Any] = {
        "conditionId": "0xtokens_only",
        "question": "Token winner fallback",
        "outcomes": ["Yes", "No"],
        "clobTokenIds": ["201", "202"],
        "endDate": "2026-09-20T00:00:00Z",
        "createdAt": "2026-09-01T00:00:00Z",
        "closed": True,
        "tokens": [
            {"outcome": "No", "winner": False},
            {"outcome": "Yes", "winner": True},
        ],
    }
    resolved_tokens = parse_gamma_resolved_market(raw_tokens_only)
    assert resolved_tokens is not None
    assert resolved_tokens.outcome_y == 1
    assert resolved_tokens.open_time_utc == datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "overrides",
    [
        {"closed": False, "active": True, "outcomePrices": '["1", "0"]'},  # still open
        {"closed": True, "outcomePrices": '["0.5", "0.5"]'},  # 50-50 void split
        {"closed": True, "outcomePrices": '["0.99", "0.01"]'},  # unsettled price
        {"closed": True, "outcomePrices": '["0", "0"]'},  # void / cancelled
        {"closed": True, "outcomePrices": None, "tokens": None},  # missing resolution
        {
            "closed": True,
            "outcomes": '["A", "B", "C"]',
            "outcomePrices": '["1", "0", "0"]',
        },  # 3-outcome
        {
            "closed": True,
            "outcomePrices": None,
            "tokens": [{"outcome": "Yes", "winner": True}, {"outcome": "No", "winner": True}],
        },  # ambiguous double winner
        {
            "closed": True,
            "outcomePrices": '["0.5", "0.5"]',
            "tokens": [{"outcome": "Yes", "winner": True}, {"outcome": "No", "winner": False}],
        },  # outcomePrices 50-50 must NOT be overridden by tokens
    ],
)
def test_parse_gamma_resolved_market_discards_open_void_or_ambiguous(
    overrides: dict[str, Any],
) -> None:
    base: dict[str, Any] = {
        "conditionId": "0xdiscard",
        "question": "Discard test",
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": '["301", "302"]',
        "endDate": "2026-09-10T00:00:00Z",
        "closed": True,
    }
    base.update(overrides)
    assert parse_gamma_resolved_market(base) is None


def test_parse_kalshi_resolved_market_and_settlement_ts_isolation() -> None:
    close_ts = "2026-09-10T18:00:00Z"
    settlement_a = "2026-09-10T20:00:00Z"
    settlement_b = "2026-09-25T20:00:00Z"  # +15 days settlement lag

    raw_base: dict[str, Any] = {
        "ticker": "KXHIGHNY-26SEP10-T75",
        "event_ticker": "KXHIGHNY-26SEP10",
        "series_ticker": "KXHIGHNY",
        "title": "NYC high temp > 75F?",
        "market_type": "binary",
        "status": "finalized",
        "result": "yes",
        "open_time": "2026-09-09T14:00:00Z",
        "close_time": close_ts,
        "settlement_ts": settlement_a,
        "volume": 250000,
    }

    res_yes = parse_kalshi_resolved_market(raw_base)
    assert res_yes is not None
    assert res_yes.outcome_y == 1
    assert res_yes.open_time_utc == datetime(2026, 9, 9, 14, 0, 0, tzinfo=UTC)
    assert res_yes.metadata.end_time_utc == datetime(2026, 9, 10, 18, 0, 0, tzinfo=UTC)
    assert res_yes.settlement_ts_utc == datetime(2026, 9, 10, 20, 0, 0, tzinfo=UTC)

    res_delayed = parse_kalshi_resolved_market({**raw_base, "settlement_ts": settlement_b})
    assert res_delayed is not None
    assert res_delayed.metadata.end_time_utc == res_yes.metadata.end_time_utc
    assert res_delayed.settlement_ts_utc == datetime(2026, 9, 25, 20, 0, 0, tzinfo=UTC)

    # Shifting settlement_ts by +15d MUST NOT alter sampled snapshot timestamp or tau_days.
    perimeter = _make_perimeter()
    ts_a = sample_variable_tau_snapshot_ts(res_yes, perimeter)
    ts_b = sample_variable_tau_snapshot_ts(res_delayed, perimeter)
    assert ts_a is not None
    assert ts_a == ts_b

    # result='no' -> outcome_y=0
    res_no = parse_kalshi_resolved_market({**raw_base, "result": "NO"})
    assert res_no is not None
    assert res_no.outcome_y == 0

    # Void / scalar / blank results -> None
    for bad_result in ("", "  ", None, "void", "scalar", "all_no", "all_yes"):
        assert parse_kalshi_resolved_market({**raw_base, "result": bad_result}) is None

    # Non-binary market_type -> None
    assert parse_kalshi_resolved_market({**raw_base, "market_type": "scalar"}) is None


# ---------------------------------------------------------------------------
# 3. Deterministic SHA-256 variable-tau sampler (including tau < 1d)
# ---------------------------------------------------------------------------


def test_sampler_is_deterministic_and_order_independent() -> None:
    perimeter = _make_perimeter(seed=42)
    other_seed = _make_perimeter(seed=1337)

    markets = [
        _make_resolved_market(
            venue="polymarket" if i % 2 == 0 else "kalshi",
            market_id=f"mkt_{i:03d}",
            open_time_utc=datetime(2026, 7, 1, 0, 0, 0, tzinfo=UTC),
            end_time_utc=datetime(2026, 9, 15, 0, 0, 0, tzinfo=UTC),
        )
        for i in range(40)
    ]

    forward = {
        (m.metadata.venue, m.metadata.market_id): sample_variable_tau_snapshot_ts(m, perimeter)
        for m in markets
    }
    reversed_run = {
        (m.metadata.venue, m.metadata.market_id): sample_variable_tau_snapshot_ts(m, perimeter)
        for m in reversed(markets)
    }
    assert forward == reversed_run
    assert all(ts is not None for ts in forward.values())

    different_seed = {
        (m.metadata.venue, m.metadata.market_id): sample_variable_tau_snapshot_ts(m, other_seed)
        for m in markets
    }
    assert different_seed != forward


def test_sampler_respects_buffer_and_covers_tau_lt_1d_on_daily_and_multiday() -> None:
    perimeter = _make_perimeter(pre_resolution_buffer_hours=2, short_tau_target_share=0.25)
    buffer = timedelta(hours=perimeter.pre_resolution_buffer_hours)

    # A. Short-lived / daily markets (lifetime <= 24h): 100% MUST land in 0 < tau_days < 1.
    daily_end = datetime(2026, 9, 20, 22, 0, 0, tzinfo=UTC)
    for idx, lifetime_hours in enumerate((6, 12, 18, 24)):
        daily_open = daily_end - timedelta(hours=lifetime_hours)
        market = _make_resolved_market(
            venue="kalshi",
            market_id=f"KXHIGHNY-DAILY-{idx}",
            open_time_utc=daily_open,
            end_time_utc=daily_end,
        )
        snap = sample_variable_tau_snapshot_ts(market, perimeter)
        assert snap is not None
        assert daily_open <= snap <= daily_end - buffer
        tau_days, log1p_tau = compute_tau_metrics(snap, daily_end)
        assert Decimal("0") < tau_days < Decimal("1")
        assert log1p_tau > 0.0 and math.isfinite(log1p_tau)

    # B. Multi-day markets (60d lifetime): verified coverage across <1d and multi-day buckets.
    multi_open = datetime(2026, 7, 1, 0, 0, 0, tzinfo=UTC)
    multi_end = datetime(2026, 8, 30, 0, 0, 0, tzinfo=UTC)
    buckets = {"<1d": 0, "1-3d": 0, "3-7d": 0, "7-30d": 0, ">30d": 0}
    n_markets = 200

    for i in range(n_markets):
        market = _make_resolved_market(
            venue="polymarket",
            market_id=f"0xmulti_{i:04d}",
            open_time_utc=multi_open,
            end_time_utc=multi_end,
        )
        snap = sample_variable_tau_snapshot_ts(market, perimeter)
        assert snap is not None
        assert multi_open <= snap <= multi_end - buffer

        tau_days, log1p_tau = compute_tau_metrics(snap, multi_end)
        assert tau_days >= Decimal("2") / Decimal("24") - Decimal("0.000001")
        assert log1p_tau > 0.0

        if tau_days < Decimal("1"):
            buckets["<1d"] += 1
        elif tau_days < Decimal("3"):
            buckets["1-3d"] += 1
        elif tau_days < Decimal("7"):
            buckets["3-7d"] += 1
        elif tau_days < Decimal("30"):
            buckets["7-30d"] += 1
        else:
            buckets[">30d"] += 1

    # Target short_tau_target_share=0.25 -> expect solid representation around ~50/200.
    assert 25 <= buckets["<1d"] <= 80
    assert all(count > 10 for count in buckets.values())


def test_sampler_edge_cases_and_fallback_lookback() -> None:
    perimeter = _make_perimeter(pre_resolution_buffer_hours=2)
    end_ts = datetime(2026, 9, 15, 12, 0, 0, tzinfo=UTC)

    # Missing end_time_utc -> None
    assert (
        sample_variable_tau_snapshot_ts(_make_resolved_market(end_time_utc=None), perimeter) is None
    )

    # open_time_utc >= t_max (opened within 2h pre-resolution buffer) -> None
    assert (
        sample_variable_tau_snapshot_ts(
            _make_resolved_market(
                open_time_utc=end_ts - timedelta(hours=2),
                end_time_utc=end_ts,
            ),
            perimeter,
        )
        is None
    )
    assert (
        sample_variable_tau_snapshot_ts(
            _make_resolved_market(
                open_time_utc=end_ts - timedelta(minutes=30),
                end_time_utc=end_ts,
            ),
            perimeter,
        )
        is None
    )

    # Missing open_time_utc -> falls back to conservative DEFAULT_FALLBACK_LOOKBACK_DAYS window.
    snap_fallback = sample_variable_tau_snapshot_ts(
        _make_resolved_market(open_time_utc=None, end_time_utc=end_ts),
        perimeter,
    )
    assert snap_fallback is not None
    expected_t_max = end_ts - timedelta(hours=perimeter.pre_resolution_buffer_hours)
    expected_t_min = expected_t_max - timedelta(days=DEFAULT_FALLBACK_LOOKBACK_DAYS)
    assert expected_t_min <= snap_fallback <= expected_t_max

    # short_tau_target_share=0.0 -> never picks <1d on multi-day markets.
    zero_short = _make_perimeter(short_tau_target_share=0.0)
    for i in range(30):
        m = _make_resolved_market(
            market_id=f"zero_short_{i}",
            open_time_utc=end_ts - timedelta(days=15),
            end_time_utc=end_ts,
        )
        snap = sample_variable_tau_snapshot_ts(m, zero_short)
        assert snap is not None
        tau_days, _ = compute_tau_metrics(snap, end_ts)
        assert tau_days >= Decimal("1")

    # short_tau_target_share=1.0 -> always picks <1d on multi-day markets.
    full_short = _make_perimeter(short_tau_target_share=1.0)
    for i in range(30):
        m = _make_resolved_market(
            market_id=f"full_short_{i}",
            open_time_utc=end_ts - timedelta(days=15),
            end_time_utc=end_ts,
        )
        snap = sample_variable_tau_snapshot_ts(m, full_short)
        assert snap is not None
        tau_days, _ = compute_tau_metrics(snap, end_ts)
        assert Decimal("0") < tau_days < Decimal("1")

    # Large buffer (>= 24h) skips short_bucket and samples multi-day cleanly.
    big_buffer = _make_perimeter(pre_resolution_buffer_hours=24)
    snap_big = sample_variable_tau_snapshot_ts(
        _make_resolved_market(
            market_id="big_buf_1",
            open_time_utc=end_ts - timedelta(days=10),
            end_time_utc=end_ts,
        ),
        big_buffer,
    )
    assert snap_big is not None
    assert snap_big <= end_ts - timedelta(hours=24)


# ---------------------------------------------------------------------------
# 4. Polars schema, Parquet Zstd round-trip, DuckDB union_by_name, duplicates
# ---------------------------------------------------------------------------


def test_historical_schema_v1_exact_columns_dtypes_and_empty_frame() -> None:
    expected = pl.Schema(
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
    assert HISTORICAL_CALIBRATION_SCHEMA_V1 == expected
    assert list(HISTORICAL_CALIBRATION_SCHEMA_V1.keys()) == list(expected.keys())

    # Forbidden leakage columns must never exist in schema.
    forbidden_cols = {
        "gamma_volume",
        "volume",
        "volume_final",
        "volume24hr",
        "liquidity",
        "resolved_time",
        "uma_resolution_time",
        "settlement_ts",
        "future_spread",
    }
    assert forbidden_cols.isdisjoint(HISTORICAL_CALIBRATION_SCHEMA_V1.keys())

    empty_df = historical_records_to_frame([])
    assert empty_df.height == 0
    assert empty_df.schema == expected


def test_parquet_zstd_roundtrip_and_duckdb_union_by_name_v1_v2(tmp_path: Path) -> None:
    poly_sym = _make_snapshot_record(
        venue="polymarket",
        market_id="0xsym_001",
        token_id="tok_lakers",
        outcome_side="outcome_0",
        outcome_label="Lakers",
        is_yes_no=False,
        neg_risk=False,
        fee_type="sports",
        p_mkt=Decimal("0.44123456"),  # >6dp -> quantized cleanly to 6dp
        vol_usd_le_t=Decimal("12345.678"),  # >2dp -> quantized cleanly to 2dp
        vol_contracts_le_t=None,
        vol_unit="usd",
        trade_count_le_t=85,
        trades_truncated=True,
        price_source="poly_clob",
        outcome_y=0,
    )
    kalshi_rec = _make_snapshot_record(
        venue="kalshi",
        market_id="KXHIGHNY-26SEP20-T75",
        token_id=None,
        event_id="KXHIGHNY-26SEP20",
        title="NYC high temp > 75F?",
        category="KXHIGHNY",
        outcome_side="yes",
        outcome_label="Yes",
        is_yes_no=True,
        neg_risk=None,
        fee_type=None,
        snapshot_ts_utc=datetime(2026, 9, 20, 14, 0, 0, tzinfo=UTC),
        end_time_utc=datetime(2026, 9, 20, 20, 0, 0, tzinfo=UTC),
        p_mkt=Decimal("0.550000"),
        vol_usd_le_t=Decimal("1650.00"),
        vol_contracts_le_t=Decimal("3000.00"),
        vol_unit="contracts_notional",
        trade_count_le_t=12,
        trades_truncated=False,
        price_source="kalshi_mid",
        outcome_y=1,
        hist_yes_bid=Decimal("0.540000"),
        hist_yes_ask=Decimal("0.560000"),
    )

    v1_path = write_historical_parquet([poly_sym, kalshi_rec], tmp_path / "historical_v1.parquet")
    assert v1_path.exists()
    # No stray atomic temp files remain.
    assert not list(tmp_path.glob("*.tmp"))

    df_v1 = read_historical_duckdb(v1_path)
    assert df_v1.schema == HISTORICAL_CALIBRATION_SCHEMA_V1
    assert df_v1.height == 2
    assert df_v1["schema_version"].to_list() == [SCHEMA_VERSION, SCHEMA_VERSION]
    # Deterministic sort by (venue, end_time_utc, market_id): kalshi before polymarket.
    assert df_v1["venue"].to_list() == ["kalshi", "polymarket"]
    assert df_v1["outcome_side"].to_list() == ["yes", "outcome_0"]
    assert df_v1["outcome_y"].to_list() == [1, 0]
    assert df_v1["hist_yes_bid"].to_list() == [Decimal("0.540000"), None]
    assert df_v1["hist_yes_ask"].to_list() == [Decimal("0.560000"), None]
    assert df_v1["vol_contracts_le_t"].to_list() == [Decimal("3000.00"), None]
    assert df_v1["p_mkt"][1] == Decimal("0.441235")
    assert df_v1["vol_usd_le_t"][1] == Decimal("12345.68")
    assert df_v1["trades_truncated"].to_list() == [False, True]

    # Simulate a future v2 Parquet file with an added column and verify union_by_name=true.
    v2_base = historical_records_to_frame(
        [
            _make_snapshot_record(
                venue="polymarket",
                market_id="0xnew_v2",
                schema_version="v2",
            )
        ]
    )
    v2_frame = v2_base.with_columns(pl.lit("macro_tier_1").alias("liquidity_tier_v2"))
    v2_path = tmp_path / "historical_v2.parquet"
    v2_frame.write_parquet(v2_path, compression="zstd")

    combined = read_historical_duckdb(tmp_path / "*.parquet").sort(["schema_version", "venue"])
    assert combined.height == 3
    assert "liquidity_tier_v2" in combined.columns
    assert combined["schema_version"].to_list() == ["v1", "v1", "v2"]
    assert combined["liquidity_tier_v2"].to_list() == [None, None, "macro_tier_1"]


def test_duplicate_venue_market_id_raises_immediately(tmp_path: Path) -> None:
    snap_1d = _make_snapshot_record(
        venue="polymarket",
        market_id="0xdup_market",
        snapshot_ts_utc=datetime(2026, 9, 29, 0, 0, 0, tzinfo=UTC),
        end_time_utc=datetime(2026, 9, 30, 0, 0, 0, tzinfo=UTC),
    )
    snap_7d = _make_snapshot_record(
        venue="polymarket",
        market_id="0xdup_market",
        snapshot_ts_utc=datetime(2026, 9, 23, 0, 0, 0, tzinfo=UTC),
        end_time_utc=datetime(2026, 9, 30, 0, 0, 0, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="Duplicate historical snapshot key"):
        historical_records_to_frame([snap_1d, snap_7d])

    with pytest.raises(ValueError, match="Duplicate historical snapshot key"):
        write_historical_parquet([snap_1d, snap_7d], tmp_path / "dup.parquet")

    # Same market_id across DIFFERENT venues is distinct and permitted.
    other_venue = _make_snapshot_record(
        venue="kalshi",
        market_id="0xdup_market",
        token_id=None,
        vol_unit="contracts_notional",
        price_source="kalshi_close",
    )
    frame = historical_records_to_frame([snap_1d, other_venue])
    assert frame.height == 2
