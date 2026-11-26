"""Offline unit and integration tests for Kalshi point-in-time historical backfill."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from pm_quant.ingest import (
    BackfillPerimeter,
    KalshiCandle,
    KalshiClient,
    backfill_kalshi,
    extract_kalshi_point_in_time,
    is_eligible_resolved_kalshi,
    parse_kalshi_candlestick,
    parse_kalshi_resolved_market,
    read_historical_duckdb,
    sample_variable_tau_snapshot_ts,
    stratify_resolved_markets,
    write_historical_parquet,
)
from pm_quant.ingest.backfill_kalshi import (
    _choose_historical_interval_and_start,
    _KalshiCandidate,
    _plan_live_candlestick_batches,
)


def _make_kalshi_settled_payload(
    ticker: str,
    *,
    event_ticker: str | None = None,
    series_ticker: str | None = None,
    title: str = "Will CPI exceed 3.0%?",
    category: str | None = "Economics",
    result: str = "yes",
    status: str = "settled",
    market_type: str = "binary",
    open_time: str = "2026-05-01T00:00:00Z",
    close_time: str = "2026-06-15T12:00:00Z",
    settlement_ts: str | None = "2026-06-15T14:00:00Z",
    volume: int = 999_999_999,
    volume_fp: str = "999999999.00",
    mve_collection_ticker: str | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ticker": ticker,
        "event_ticker": event_ticker or f"EVT-{ticker}",
        "title": title,
        "category": category,
        "status": status,
        "market_type": market_type,
        "result": result,
        "open_time": open_time,
        "close_time": close_time,
        # Huge final lifetime volume included deliberately to assert anti-leakage!
        "volume": volume,
        "volume_fp": volume_fp,
    }
    if series_ticker is not None:
        payload["series_ticker"] = series_ticker
    if settlement_ts is not None:
        payload["settlement_ts"] = settlement_ts
    if mve_collection_ticker is not None:
        payload["mve_collection_ticker"] = mve_collection_ticker
    if extra_fields:
        payload.update(extra_fields)
    return payload


def _make_candle_dollars(
    end_ts: datetime,
    *,
    yes_bid: str | None = "0.5800",
    yes_ask: str | None = "0.6200",
    price_close: str | None = "0.6000",
    volume_fp: str | None = "1000.00",
) -> dict[str, Any]:
    candle: dict[str, Any] = {"end_period_ts": int(end_ts.timestamp())}
    if yes_bid is not None:
        candle["yes_bid"] = {"close_dollars": yes_bid}
    if yes_ask is not None:
        candle["yes_ask"] = {"close_dollars": yes_ask}
    if price_close is not None:
        candle["price"] = {"close_dollars": price_close}
    if volume_fp is not None:
        candle["volume_fp"] = volume_fp
    return candle


# ---------------------------------------------------------------------------
# 1. KalshiClient extensions: min_close_ts/max_close_ts & get_resolved_markets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kalshi_client_close_ts_and_get_resolved_markets() -> None:
    """Applies min/max_close_ts and status=settled only on /markets, never /historical/markets."""
    seen_requests: list[tuple[str, dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append((request.url.path, dict(request.url.params)))
        if request.url.path.endswith("/historical/markets"):
            return httpx.Response(
                200,
                json={
                    "markets": [
                        _make_kalshi_settled_payload(
                            "KXCPI-26MAY-B",
                            result="no",
                            close_time="2026-05-15T12:00:00Z",
                            settlement_ts="2026-05-15T13:00:00Z",
                        ),
                    ],
                    "cursor": "",
                },
            )
        if request.url.path.endswith("/markets"):
            return httpx.Response(
                200,
                json={
                    "markets": [
                        _make_kalshi_settled_payload("KXCPI-26JUN-A", result="yes"),
                        _make_kalshi_settled_payload("KXCPI-26JUN-VOID", result="void"),
                        _make_kalshi_settled_payload("KXCPI-26JUN-A", result="yes"),  # dup
                    ],
                    "cursor": "",
                },
            )
        return httpx.Response(404, json={"error": "not found"})

    async with KalshiClient(transport=httpx.MockTransport(handler)) as client:
        live = await client.get_resolved_markets(
            archived=False,
            mve_filter="exclude",
            min_close_ts=1_777_593_600,
            max_close_ts=1_782_864_000,
        )
        archived = await client.get_resolved_markets(
            archived=True,
            mve_filter="exclude",
            min_close_ts=1_777_593_600,
            max_close_ts=1_782_864_000,
        )

        # Mutual exclusivity guard on /historical/markets remains intact!
        with pytest.raises(ValueError, match="mutually exclusive"):
            await client.get_resolved_markets(
                archived=True,
                series_ticker="KXCPI",
                mve_filter="exclude",
            )

        with pytest.raises(ValueError, match="min_close_ts"):
            await client.get_markets(min_close_ts=200, max_close_ts=100)

    assert [m.metadata.market_id for m in live] == ["KXCPI-26JUN-A"]
    assert live[0].outcome_y == 1
    assert [m.metadata.market_id for m in archived] == ["KXCPI-26MAY-B"]
    assert archived[0].outcome_y == 0

    live_path, live_params = seen_requests[0]
    assert live_path.endswith("/markets")
    assert live_params["status"] == "settled"
    assert live_params["mve_filter"] == "exclude"
    assert live_params["min_close_ts"] == "1777593600"
    assert live_params["max_close_ts"] == "1782864000"

    arch_path, arch_params = seen_requests[1]
    assert arch_path.endswith("/historical/markets")
    assert "status" not in arch_params
    assert "min_close_ts" not in arch_params
    assert "max_close_ts" not in arch_params
    assert arch_params["mve_filter"] == "exclude"


# ---------------------------------------------------------------------------
# 2. Pure candlestick parsing & point-in-time extraction (zero I/O)
# ---------------------------------------------------------------------------


def test_parse_kalshi_candlestick_dollars_and_integer_cents() -> None:
    """Supports both modern *_dollars strings and legacy integer cents (0..100)."""
    ts = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)

    # Modern *_dollars format + volume_fp
    raw_dollars = {
        "end_period_ts": int(ts.timestamp()),
        "yes_bid": {"close_dollars": "0.4200"},
        "yes_ask": {"close_dollars": "0.4600"},
        "price": {"close_dollars": "0.4400"},
        "volume_fp": "1250.50",
    }
    c_dollars = parse_kalshi_candlestick(raw_dollars)
    assert c_dollars is not None
    assert c_dollars.end_period_ts == ts
    assert c_dollars.yes_bid_close == Decimal("0.4200")
    assert c_dollars.yes_ask_close == Decimal("0.4600")
    assert c_dollars.price_close == Decimal("0.4400")
    assert c_dollars.volume_contracts == Decimal("1250.50")
    assert c_dollars.mid_price == Decimal("0.44")

    # Legacy integer cents 0..100 + integer volume
    raw_cents = {
        "end_period_ts": int(ts.timestamp()),
        "yes_bid": {"close": 51},
        "yes_ask": {"close": 55},
        "price": {"close": 53},
        "volume": 800,
    }
    c_cents = parse_kalshi_candlestick(raw_cents)
    assert c_cents is not None
    assert c_cents.yes_bid_close == Decimal("0.51")
    assert c_cents.yes_ask_close == Decimal("0.55")
    assert c_cents.price_close == Decimal("0.53")
    assert c_cents.volume_contracts == Decimal("800")
    assert c_cents.mid_price == Decimal("0.53")

    # Edge cents: 0 and 100
    c_bounds = parse_kalshi_candlestick(
        {
            "end_period_ts": int(ts.timestamp()),
            "yes_bid": {"close": 0},
            "yes_ask": {"close": 100},
            "price": {"close": 1},
            "volume": 10,
        }
    )
    assert c_bounds is not None
    assert c_bounds.yes_bid_close == Decimal("0")
    assert c_bounds.yes_ask_close == Decimal("1")
    assert c_bounds.price_close == Decimal("0.01")

    # Malformed candles return None
    assert parse_kalshi_candlestick({"yes_bid": {"close": 50}}) is None
    assert parse_kalshi_candlestick({"end_period_ts": int(ts.timestamp()), "volume": -5}) is None


def test_extract_kalshi_point_in_time_mid_fallback_and_anti_leakage() -> None:
    """Computes bivariate mid, falls back to price_close on one-sided candles, and ignores > t_i."""
    t_i = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)

    # Case 1: Bivariate mid + future candle strictly ignored for both p_mkt and volume!
    candles_bivariate = [
        _make_candle_dollars(
            t_i - timedelta(hours=2),
            yes_bid="0.40",
            yes_ask="0.44",
            price_close="0.41",
            volume_fp="1000",
        ),
        _make_candle_dollars(
            t_i,
            yes_bid="0.58",
            yes_ask="0.62",
            price_close="0.59",
            volume_fp="500",
        ),
        # Future candle (> t_i): MUST be ignored for both p_mkt and vol_usd_le_t!
        _make_candle_dollars(
            t_i + timedelta(seconds=1),
            yes_bid="0.98",
            yes_ask="0.99",
            price_close="0.99",
            volume_fp="500000",
        ),
    ]
    pit_mid = extract_kalshi_point_in_time(candles_bivariate, t_i)
    assert pit_mid is not None
    assert pit_mid.price_source == "kalshi_mid"
    assert pit_mid.p_mkt == Decimal("0.60")
    assert pit_mid.hist_yes_bid == Decimal("0.58")
    assert pit_mid.hist_yes_ask == Decimal("0.62")
    assert pit_mid.vol_contracts_le_t == Decimal("1500")
    # 1000 * 0.42 + 500 * 0.60 = 420 + 300 = 720
    assert pit_mid.vol_usd_le_t == Decimal("720.00")
    assert pit_mid.candle_count_le_t == 2

    # Case 2: One-sided candle falls back to price_close (kalshi_close), hist_yes_bid/ask are None
    candles_one_sided = [
        _make_candle_dollars(
            t_i - timedelta(hours=1),
            yes_bid="0.45",
            yes_ask=None,
            price_close="0.48",
            volume_fp="2000",
        ),
        # Unpriced volume candle -> falls back to Decimal("1") for notional
        _make_candle_dollars(
            t_i - timedelta(minutes=30),
            yes_bid=None,
            yes_ask=None,
            price_close=None,
            volume_fp="100",
        ),
    ]
    pit_close = extract_kalshi_point_in_time(candles_one_sided, t_i)
    assert pit_close is not None
    assert pit_close.price_source == "kalshi_close"
    assert pit_close.p_mkt == Decimal("0.48")
    assert pit_close.hist_yes_bid is None
    assert pit_close.hist_yes_ask is None
    assert pit_close.vol_contracts_le_t == Decimal("2100")
    # 2000 * 0.48 + 100 * 1 = 960 + 100 = 1060
    assert pit_close.vol_usd_le_t == Decimal("1060.00")

    # Case 3: Only future candles -> returns None
    assert extract_kalshi_point_in_time([candles_bivariate[-1]], t_i) is None


def test_historical_interval_and_live_batch_period_caps() -> None:
    """Switches historical interval 60 -> 1440 above 5,000h and calibrates live batch <= 10,000."""
    end_ts = int(datetime(2026, 6, 15, 0, 0, tzinfo=UTC).timestamp())
    # 6,000 hours > 5,000 period cap -> must step up to 1440 (daily)
    long_start = end_ts - 6_000 * 3600
    eff_interval, eff_start = _choose_historical_interval_and_start(long_start, end_ts, 60)
    assert eff_interval == 1440
    assert eff_start == long_start

    # Short window (100 hours) stays at 60m
    short_start = end_ts - 100 * 3600
    eff_interval_short, _ = _choose_historical_interval_and_start(short_start, end_ts, 60)
    assert eff_interval_short == 60

    # Live batch packing: 5 markets each spanning 3,000 hours -> max 3 per batch (3*3001 <= 10,000)
    cands: list[_KalshiCandidate] = []
    for i in range(5):
        resolved = parse_kalshi_resolved_market(_make_kalshi_settled_payload(f"KXMACRO-{i}"))
        assert resolved is not None
        snap = datetime(2026, 6, 10, 0, 0, tzinfo=UTC)
        cands.append(
            _KalshiCandidate(
                market=resolved,
                snapshot_ts_utc=snap,
                start_ts=int(snap.timestamp()) - 3_000 * 3600,
                end_ts=int(snap.timestamp()),
                pre_cutoff=False,
            )
        )
    batches = _plan_live_candlestick_batches(cands, 60)
    assert len(batches) == 2
    assert len(batches[0].candidates) == 3
    assert len(batches[1].candidates) == 2
    for batch in batches:
        assert batch.max_batch_size <= 3
        assert len(batch.candidates) <= batch.max_batch_size


def test_kalshi_eligibility_and_stratification() -> None:
    """Excludes MVE/combo and 5m/15m crypto markets while stratifying eligible Kalshi markets."""
    perimeter = BackfillPerimeter(
        start_date_utc=datetime(2026, 5, 1, tzinfo=UTC),
        end_date_utc=datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC),
        max_markets_per_venue=4,
    )
    good_econ = parse_kalshi_resolved_market(
        _make_kalshi_settled_payload("KXCPI-26JUN-01", category="Economics")
    )
    good_wx = parse_kalshi_resolved_market(
        _make_kalshi_settled_payload("KXHIGHNY-26JUN-01", category="Weather")
    )
    mve_market = parse_kalshi_resolved_market(
        _make_kalshi_settled_payload(
            "KXMVE-PARLAY-01",
            category="Sports",
            mve_collection_ticker="KXMVE-COL-1",
        )
    )
    crypto_15m = parse_kalshi_resolved_market(
        _make_kalshi_settled_payload(
            "KXBTC15M-26JUN-01",
            series_ticker="KXBTC15M",
            title="Bitcoin price up in next 15 minutes?",
            category="Crypto",
        )
    )
    assert good_econ is not None and good_wx is not None
    assert mve_market is not None and crypto_15m is not None

    assert is_eligible_resolved_kalshi(good_econ)
    assert is_eligible_resolved_kalshi(good_wx)
    assert not is_eligible_resolved_kalshi(mve_market)
    assert not is_eligible_resolved_kalshi(crypto_15m)

    stratified = stratify_resolved_markets(
        [good_econ, good_wx, mve_market, crypto_15m],
        perimeter,
        eligibility_fn=is_eligible_resolved_kalshi,
    )
    assert {m.metadata.market_id for m in stratified} == {"KXCPI-26JUN-01", "KXHIGHNY-26JUN-01"}


# ---------------------------------------------------------------------------
# 3. End-to-end async backfill: cutoff routing, anti-leakage, MVE, checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_kalshi_cutoff_routing_anti_leakage_and_tau_invariance(
    tmp_path: Path,
) -> None:
    """Verifies /historical/cutoff routing, <= t_i anti-leakage, and settlement_ts invariance."""
    perimeter = BackfillPerimeter(
        start_date_utc=datetime(2026, 5, 1, tzinfo=UTC),
        end_date_utc=datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC),
        max_markets_per_venue=10,
        min_vol_usd=Decimal("500"),
        seed=42,
    )
    cutoff_iso = "2026-06-01T00:00:00Z"

    # Pre-cutoff market (settled May 20 < June 1 cutoff)
    pre_raw = _make_kalshi_settled_payload(
        "KXCPI-26MAY-PRE",
        category="Economics",
        result="no",
        open_time="2026-05-01T00:00:00Z",
        close_time="2026-05-20T12:00:00Z",
        settlement_ts="2026-05-20T14:00:00Z",
        volume=888_888_888,
        volume_fp="888888888.00",
    )
    # Post-cutoff market (settled June 15 >= June 1 cutoff)
    post_raw = _make_kalshi_settled_payload(
        "KXHIGHNY-26JUN-POST",
        category="Weather",
        result="yes",
        open_time="2026-06-01T00:00:00Z",
        close_time="2026-06-15T12:00:00Z",
        settlement_ts="2026-06-15T14:00:00Z",
        volume=777_777_777,
        volume_fp="777777777.00",
    )
    # MVE market that should be filtered out even if leaked by server
    mve_raw = _make_kalshi_settled_payload(
        "KXMVE-26JUN-COMBO",
        category="Sports",
        result="yes",
        close_time="2026-06-18T12:00:00Z",
        settlement_ts="2026-06-18T14:00:00Z",
        mve_collection_ticker="KXMVE-COL",
    )

    pre_resolved = parse_kalshi_resolved_market(pre_raw)
    post_resolved = parse_kalshi_resolved_market(post_raw)
    assert pre_resolved is not None and post_resolved is not None
    t_pre = sample_variable_tau_snapshot_ts(pre_resolved, perimeter)
    t_post = sample_variable_tau_snapshot_ts(post_resolved, perimeter)
    assert t_pre is not None and t_post is not None

    async def _run_with_settlement_shift(shift_days: int) -> tuple[list[Any], dict[str, int]]:
        counts = {
            "cutoff": 0,
            "live_markets": 0,
            "hist_markets": 0,
            "live_candles_batch": 0,
            "hist_candles_pre": 0,
        }
        shifted_post = dict(post_raw)
        base_settle = datetime(2026, 6, 15, 14, 0, tzinfo=UTC) + timedelta(days=shift_days)
        shifted_post["settlement_ts"] = base_settle.isoformat().replace("+00:00", "Z")

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            params = request.url.params

            if path.endswith("/historical/cutoff"):
                counts["cutoff"] += 1
                return httpx.Response(200, json={"market_settled_ts": cutoff_iso})

            if path.endswith("/historical/markets"):
                counts["hist_markets"] += 1
                assert params.get("mve_filter") == "exclude"
                assert "status" not in params
                assert "min_close_ts" not in params
                return httpx.Response(200, json={"markets": [pre_raw], "cursor": ""})

            if path.endswith("/markets"):
                counts["live_markets"] += 1
                assert params.get("mve_filter") == "exclude"
                assert params.get("status") == "settled"
                return httpx.Response(200, json={"markets": [shifted_post, mve_raw], "cursor": ""})

            if path.endswith("/markets/candlesticks"):
                counts["live_candles_batch"] += 1
                tickers = params.get("market_tickers", "").split(",")
                assert tickers == ["KXHIGHNY-26JUN-POST"]
                return httpx.Response(
                    200,
                    json={
                        "markets": [
                            {
                                "market_ticker": "KXHIGHNY-26JUN-POST",
                                "candlesticks": [
                                    _make_candle_dollars(
                                        t_post - timedelta(hours=1),
                                        yes_bid="0.6800",
                                        yes_ask="0.7200",
                                        price_close="0.7000",
                                        volume_fp="2000.00",
                                    ),
                                    # Post-t_i candle MUST be ignored!
                                    _make_candle_dollars(
                                        t_post + timedelta(hours=1),
                                        yes_bid="0.9800",
                                        yes_ask="0.9900",
                                        price_close="0.9900",
                                        volume_fp="999999.00",
                                    ),
                                ],
                            }
                        ]
                    },
                )

            if path.endswith("/historical/markets/KXCPI-26MAY-PRE/candlesticks"):
                counts["hist_candles_pre"] += 1
                # Legacy integer cents + one-sided fallback to price_close
                return httpx.Response(
                    200,
                    json={
                        "candlesticks": [
                            {
                                "end_period_ts": int((t_pre - timedelta(hours=1)).timestamp()),
                                "yes_bid": {"close": 34},
                                "yes_ask": {"close": None},
                                "price": {"close": 35},
                                "volume": 3000,
                            },
                            # Post-t_i candle MUST be ignored!
                            {
                                "end_period_ts": int((t_pre + timedelta(hours=1)).timestamp()),
                                "yes_bid": {"close": 1},
                                "yes_ask": {"close": 2},
                                "price": {"close": 1},
                                "volume": 999_999,
                            },
                        ]
                    },
                )

            return httpx.Response(404, json={"error": f"unexpected path {path}"})

        async with KalshiClient(transport=httpx.MockTransport(handler)) as client:
            recs = await backfill_kalshi(
                perimeter,
                client=client,
                max_live_pages=1,
                max_archived_pages=1,
                target_series=(),
            )
        return recs, counts

    records_base, counts_base = await _run_with_settlement_shift(shift_days=0)
    records_shifted, _ = await _run_with_settlement_shift(shift_days=7)

    # Cutoff called strictly once; routing verified!
    assert counts_base["cutoff"] == 1
    assert counts_base["live_candles_batch"] == 1
    assert counts_base["hist_candles_pre"] == 1
    assert len(records_base) == 2

    by_id_base = {r.market_id: r for r in records_base}
    by_id_shifted = {r.market_id: r for r in records_shifted}

    # Pre-cutoff archived market: integer cents + fallback to kalshi_close + strict <= t_i vol
    rec_pre = by_id_base["KXCPI-26MAY-PRE"]
    assert rec_pre.venue == "kalshi"
    assert rec_pre.price_source == "kalshi_close"
    assert rec_pre.p_mkt == Decimal("0.35")
    assert rec_pre.hist_yes_bid is None
    assert rec_pre.hist_yes_ask is None
    assert rec_pre.vol_contracts_le_t == Decimal("3000")
    assert rec_pre.vol_usd_le_t == Decimal("1050.00")  # 3000 * 0.35 (NOT 888,888,888!)
    assert rec_pre.vol_unit == "contracts_notional"
    assert rec_pre.trade_count_le_t is None
    assert rec_pre.outcome_y == 0

    # Post-cutoff live market: bivariate mid + Pillar 2 bid/ask + strict <= t_i vol
    rec_post = by_id_base["KXHIGHNY-26JUN-POST"]
    assert rec_post.price_source == "kalshi_mid"
    assert rec_post.p_mkt == Decimal("0.70")
    assert rec_post.hist_yes_bid == Decimal("0.6800")
    assert rec_post.hist_yes_ask == Decimal("0.7200")
    assert rec_post.vol_contracts_le_t == Decimal("2000.00")
    assert rec_post.vol_usd_le_t == Decimal("1400.00")  # 2000 * 0.70 (NOT 777,777,777!)
    assert rec_post.outcome_y == 1

    # tau_days computed from close_time and 100% invariant to +7d settlement_ts shift!
    rec_post_shifted = by_id_shifted["KXHIGHNY-26JUN-POST"]
    assert rec_post.end_time_utc == rec_post_shifted.end_time_utc
    assert rec_post.tau_days == rec_post_shifted.tau_days
    assert rec_post.log1p_tau == rec_post_shifted.log1p_tau

    # Parquet + DuckDB round-trip check
    parquet_file = tmp_path / "kalshi_backfill.parquet"
    write_historical_parquet(records_base, parquet_file)
    df = read_historical_duckdb(parquet_file)
    assert df.height == 2
    assert set(df["market_id"].to_list()) == {"KXCPI-26MAY-PRE", "KXHIGHNY-26JUN-POST"}


@pytest.mark.asyncio
async def test_backfill_kalshi_checkpoint_resumability_and_targeted_series(tmp_path: Path) -> None:
    """Resumes from JSONL checkpoint (ignoring truncated tail, retrying transient errors)."""
    perimeter = BackfillPerimeter(
        start_date_utc=datetime(2026, 6, 1, tzinfo=UTC),
        end_date_utc=datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC),
        max_markets_per_venue=2,
        min_vol_usd=Decimal("100"),
        seed=7,
    )
    ckpt_path = tmp_path / "kalshi_ckpt.jsonl"

    m1_raw = _make_kalshi_settled_payload(
        "KXCPI-26JUN-M1",
        category="Economics",
        close_time="2026-06-15T12:00:00Z",
        settlement_ts="2026-06-15T13:00:00Z",
    )
    m2_raw = _make_kalshi_settled_payload(
        "KXHIGHNY-26JUN-M2",
        series_ticker="KXHIGHNY",
        category="Weather",
        close_time="2026-06-16T12:00:00Z",
        settlement_ts="2026-06-16T13:00:00Z",
    )
    # Pre-cutoff market to test transient 500 on first run -> retried on resume
    m3_raw = _make_kalshi_settled_payload(
        "KXCPI-26JUN-M3",
        category="Economics",
        close_time="2026-06-10T12:00:00Z",
        settlement_ts="2026-05-10T12:00:00Z",
    )

    m1_res = parse_kalshi_resolved_market(m1_raw)
    m2_res = parse_kalshi_resolved_market(m2_raw)
    m3_res = parse_kalshi_resolved_market(m3_raw)
    assert m1_res is not None and m2_res is not None and m3_res is not None
    t_m1 = sample_variable_tau_snapshot_ts(m1_res, perimeter)
    t_m2 = sample_variable_tau_snapshot_ts(m2_res, perimeter)
    t_m3 = sample_variable_tau_snapshot_ts(m3_res, perimeter)
    assert t_m1 is not None and t_m2 is not None and t_m3 is not None

    fail_m3 = True
    series_seen: list[str] = []
    m1_batch_calls = 0
    m3_hist_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal m1_batch_calls, m3_hist_calls
        path = request.url.path
        params = request.url.params

        if path.endswith("/historical/cutoff"):
            return httpx.Response(200, json={"market_settled_ts": "2026-06-01T00:00:00Z"})

        if path.endswith("/historical/markets"):
            return httpx.Response(200, json={"markets": [m3_raw], "cursor": ""})

        if path.endswith("/markets"):
            series = params.get("series_ticker")
            if series:
                series_seen.append(series)
                if series == "KXHIGHNY":
                    return httpx.Response(200, json={"markets": [m2_raw], "cursor": ""})
                return httpx.Response(200, json={"markets": [], "cursor": ""})
            return httpx.Response(200, json={"markets": [m1_raw], "cursor": ""})

        if path.endswith("/markets/candlesticks"):
            tickers = params.get("market_tickers", "").split(",")
            markets_out: list[dict[str, Any]] = []
            for tk in tickers:
                if tk == "KXCPI-26JUN-M1":
                    m1_batch_calls += 1
                    markets_out.append(
                        {
                            "market_ticker": tk,
                            "candlesticks": [
                                _make_candle_dollars(
                                    t_m1 - timedelta(hours=1),
                                    yes_bid="0.48",
                                    yes_ask="0.52",
                                    volume_fp="500",
                                )
                            ],
                        }
                    )
                elif tk == "KXHIGHNY-26JUN-M2":
                    # Below min_vol_usd -> skipped & checkpointed
                    markets_out.append(
                        {
                            "market_ticker": tk,
                            "candlesticks": [
                                _make_candle_dollars(
                                    t_m2 - timedelta(hours=1),
                                    yes_bid="0.48",
                                    yes_ask="0.52",
                                    volume_fp="10",
                                )
                            ],
                        }
                    )
            return httpx.Response(200, json={"markets": markets_out})

        if path.endswith("/historical/markets/KXCPI-26JUN-M3/candlesticks"):
            m3_hist_calls += 1
            if fail_m3:
                return httpx.Response(500, json={"error": "temporary outage"})
            return httpx.Response(
                200,
                json={
                    "candlesticks": [
                        _make_candle_dollars(
                            t_m3 - timedelta(hours=1),
                            yes_bid="0.38",
                            yes_ask="0.42",
                            volume_fp="1000",
                        )
                    ]
                },
            )

        return httpx.Response(404, json={"error": "not found"})

    # Run 1: M1 succeeds, M2 skipped (low vol), M3 fails transiently (500 -> NOT cached!)
    async with KalshiClient(transport=httpx.MockTransport(handler), max_retries=0) as client:
        run1 = await backfill_kalshi(
            perimeter,
            client=client,
            checkpoint_path=ckpt_path,
            max_live_pages=2,
            max_archived_pages=1,
            target_series=("KXHIGHNY",),
        )
    assert [r.market_id for r in run1] == ["KXCPI-26JUN-M1"]
    assert series_seen == ["KXHIGHNY"]
    assert m1_batch_calls == 1
    assert m3_hist_calls == 1

    # Simulate crash by appending truncated trailing JSON line
    with ckpt_path.open("a", encoding="utf-8") as fh:
        fh.write('{"checkpoint_version":1,"truncated')

    # Run 2: M3 succeeds; M1 & M2 are NOT re-fetched!
    fail_m3 = False
    async with KalshiClient(transport=httpx.MockTransport(handler), max_retries=0) as client:
        run2 = await backfill_kalshi(
            perimeter,
            client=client,
            checkpoint_path=ckpt_path,
            max_live_pages=2,
            max_archived_pages=1,
            target_series=("KXHIGHNY",),
        )
    assert {r.market_id for r in run2} == {"KXCPI-26JUN-M1", "KXCPI-26JUN-M3"}
    assert m1_batch_calls == 1  # never re-fetched!
    assert m3_hist_calls == 2  # retried cleanly!

    # Run 3: Already at max_markets_per_venue=2 -> zero HTTP calls!
    def fail_transport(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Should not make HTTP calls when checkpoint is full: {request.url}")

    async with KalshiClient(transport=httpx.MockTransport(fail_transport)) as client:
        run3 = await backfill_kalshi(
            perimeter,
            client=client,
            checkpoint_path=ckpt_path,
            max_live_pages=2,
            max_archived_pages=1,
        )
    assert [r.market_id for r in run3] == [r.market_id for r in run2]


@pytest.mark.asyncio
async def test_backfill_kalshi_parameter_validation() -> None:
    """Validates invalid orchestrator arguments eagerly before making HTTP calls."""
    perimeter = BackfillPerimeter(
        start_date_utc=datetime(2026, 6, 1, tzinfo=UTC),
        end_date_utc=datetime(2026, 6, 30, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="max_live_pages"):
        await backfill_kalshi(perimeter, max_live_pages=-1)
    with pytest.raises(ValueError, match="max_archived_pages"):
        await backfill_kalshi(perimeter, max_archived_pages=-1)
    with pytest.raises(ValueError, match="period_interval"):
        await backfill_kalshi(perimeter, period_interval=1)
    with pytest.raises(ValueError, match="max_concurrency"):
        await backfill_kalshi(perimeter, max_concurrency=0)
    with pytest.raises(ValueError, match="oversample_factor"):
        await backfill_kalshi(perimeter, oversample_factor=0.5)

    # Naive timestamp validation on pure helper
    with pytest.raises(ValueError, match="timezone-aware"):
        extract_kalshi_point_in_time([], datetime(2026, 6, 1, 12, 0))
    with pytest.raises(ValueError, match="volume_contracts"):
        KalshiCandle(
            end_period_ts=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
            yes_bid_close=None,
            yes_ask_close=None,
            price_close=None,
            volume_contracts=Decimal("-1"),
        )
