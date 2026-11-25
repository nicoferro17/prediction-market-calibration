"""Offline unit and integration tests for Polymarket point-in-time historical backfill."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from pm_quant.ingest import (
    CHECKPOINT_VERSION,
    POLYMARKET_MAX_CONDITION_BATCH,
    BackfillPerimeter,
    MarketMetadata,
    PolymarketClient,
    ResolvedMarket,
    TradeRecord,
    backfill_polymarket,
    compute_perimeter_hash,
    compute_point_in_time_volume_usd,
    default_checkpoint_path,
    extract_asof_price,
    is_eligible_resolved_polymarket,
    parse_gamma_resolved_market,
    read_historical_duckdb,
    sample_variable_tau_snapshot_ts,
    stratify_resolved_markets,
    write_historical_parquet,
)


def _make_gamma_closed_payload(
    condition_id: str,
    *,
    question: str = "Will the Fed cut rates?",
    slug: str | None = None,
    category: str | None = "Economics",
    outcomes: list[str] | str = '["Yes", "No"]',
    outcome_prices: list[str] | str = '["1", "0"]',
    clob_token_ids: list[str] | str | None = None,
    start_date: str = "2026-05-01T00:00:00Z",
    end_date: str = "2026-06-15T12:00:00Z",
    closed: bool = True,
    active: bool = False,
    enable_order_book: bool = True,
    neg_risk: bool = False,
    fee_type: str | None = "economics",
    volume: str = "999999999",
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tokens = (
        clob_token_ids
        if clob_token_ids is not None
        else json.dumps([f"tok_yes_{condition_id}", f"tok_no_{condition_id}"])
    )
    payload: dict[str, Any] = {
        "id": f"gamma_{condition_id}",
        "conditionId": condition_id,
        "question": question,
        "slug": slug or f"slug-{condition_id}",
        "category": category,
        "outcomes": outcomes,
        "outcomePrices": outcome_prices,
        "clobTokenIds": tokens,
        "startDate": start_date,
        "endDate": end_date,
        "closed": closed,
        "active": active,
        "enableOrderBook": enable_order_book,
        "acceptingOrders": False,
        "negRisk": neg_risk,
        # Huge Gamma lifetime volume included deliberately to assert anti-leakage!
        "volume": volume,
        "volumeNum": float(volume),
        "liquidity": "500000",
        "events": [{"id": f"evt_{condition_id}", "slug": f"evt-slug-{condition_id}"}],
    }
    if fee_type is not None:
        payload["feeType"] = fee_type
    if extra_fields:
        payload.update(extra_fields)
    return payload


def _make_resolved(
    condition_id: str,
    *,
    category: str | None = "Economics",
    end_date: str = "2026-06-15T12:00:00Z",
    start_date: str = "2026-05-01T00:00:00Z",
    question: str = "Will event X happen?",
    slug: str | None = None,
    outcomes: list[str] | str = '["Yes", "No"]',
    outcome_prices: list[str] | str = '["1", "0"]',
    clob_token_ids: list[str] | str | None = None,
    enable_order_book: bool = True,
    fee_type: str | None = "economics",
) -> ResolvedMarket:
    raw = _make_gamma_closed_payload(
        condition_id,
        question=question,
        slug=slug,
        category=category,
        outcomes=outcomes,
        outcome_prices=outcome_prices,
        clob_token_ids=clob_token_ids,
        start_date=start_date,
        end_date=end_date,
        enable_order_book=enable_order_book,
        fee_type=fee_type,
    )
    resolved = parse_gamma_resolved_market(raw)
    assert resolved is not None
    return resolved


# ---------------------------------------------------------------------------
# 1. Gamma end_date_min / end_date_max & get_resolved_markets()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gamma_end_date_bounds_and_get_resolved_markets() -> None:
    """Gamma end_date_min/max serialize to ISO-8601 UTC and get_resolved_markets deduplicates."""
    recorded_params: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        call_idx = len(recorded_params)
        recorded_params.append(dict(request.url.params))
        items: list[dict[str, Any]] = [
            _make_gamma_closed_payload("0xcond_1"),
            _make_gamma_closed_payload("0xcond_1"),  # Duplicate -> deduplicated
            _make_gamma_closed_payload(
                "0xcond_void",
                outcome_prices='["0.5", "0.5"]',
            ),  # Ambiguous -> discarded by get_resolved_markets
            _make_gamma_closed_payload(
                "0xcond_open",
                closed=False,
                active=True,
            ),  # Open -> discarded by get_resolved_markets
            _make_gamma_closed_payload(
                "0xcond_2",
                outcomes='["Lakers", "Celtics"]',
                outcome_prices='["0", "1"]',
            ),
        ]
        if call_idx >= 1:
            items.append({"question": "Malformed missing conditionId", "closed": True})
        return httpx.Response(200, json=items)

    transport = httpx.MockTransport(handler)
    async with PolymarketClient(transport=transport, max_retries=0) as client:
        dt_min = datetime(2026, 6, 1, 2, 0, 0, tzinfo=timezone(timedelta(hours=2)))
        dt_max = datetime(2026, 7, 1, 0, 0, 0, tzinfo=UTC)

        # 1) get_markets forwards end_date_min / end_date_max in UTC ISO-8601.
        markets = await client.get_markets(
            closed=True,
            active=None,
            end_date_min=dt_min,
            end_date_max="2026-07-01T00:00:00Z",
            limit=50,
            max_pages=1,
        )
        assert len(markets) >= 1
        assert recorded_params[0]["end_date_min"] == "2026-06-01T00:00:00Z"
        assert recorded_params[0]["end_date_max"] == "2026-07-01T00:00:00Z"

        # 2) get_resolved_markets sends closed=true, deduplicates, and preserves anti-leakage.
        resolved = await client.get_resolved_markets(
            end_date_min=dt_min,
            end_date_max=dt_max,
            limit=50,
            max_pages=1,
        )
        assert recorded_params[1]["closed"] == "true"
        assert recorded_params[1]["end_date_min"] == "2026-06-01T00:00:00Z"
        assert recorded_params[1]["end_date_max"] == "2026-07-01T00:00:00Z"
        assert [m.metadata.market_id for m in resolved] == ["0xcond_1", "0xcond_2"]
        assert resolved[0].outcome_y == 1
        assert resolved[1].outcome_y == 0
        assert "volume" not in resolved[0].metadata.extra
        assert "volumeNum" not in resolved[0].metadata.extra

        # 3) Validation rejects naive datetimes and inverted ranges.
        with pytest.raises(ValueError, match="timezone-aware"):
            await client.get_resolved_markets(end_date_min=datetime(2026, 6, 1))
        with pytest.raises(ValueError, match="must be <="):
            await client.get_resolved_markets(
                end_date_min="2026-07-02T00:00:00Z",
                end_date_max="2026-07-01T00:00:00Z",
            )


# ---------------------------------------------------------------------------
# 2. Pure eligibility & balanced SHA-256 stratification
# ---------------------------------------------------------------------------


def test_stratify_resolved_markets_filters_and_balances_deterministically() -> None:
    """Filters combos/5m-crypto/out-of-window/bad-tokens and balances (category, YYYY-MM) strata."""
    perimeter = BackfillPerimeter(
        start_date_utc=datetime(2026, 5, 1, tzinfo=UTC),
        end_date_utc=datetime(2026, 7, 31, 23, 59, 59, tzinfo=UTC),
        max_markets_per_venue=6,
        seed=42,
    )

    pool: list[ResolvedMarket] = []
    # Dominant stratum: 15 Politics markets in 2026-06
    for i in range(15):
        pool.append(
            _make_resolved(f"0xpol_{i:02d}", category="Politics", end_date="2026-06-15T12:00:00Z")
        )
    # Minority strata: 3 Weather in 2026-06, 3 Economics in 2026-07
    for i in range(3):
        pool.append(
            _make_resolved(f"0xwth_{i:02d}", category="Weather", end_date="2026-06-20T12:00:00Z")
        )
        pool.append(
            _make_resolved(f"0xeco_{i:02d}", category="Economics", end_date="2026-07-10T12:00:00Z")
        )

    # Disqualified markets:
    # 1) 5m/15m crypto short-term
    short_crypto = _make_resolved(
        "0xbtc15m",
        category="Crypto",
        question="Bitcoin 15m Up or Down 14:15?",
        slug="btc-15m-updown",
        end_date="2026-06-12T14:15:00Z",
    )
    # 2) Combo / parlay
    parlay = _make_resolved(
        "0xparlay",
        category="Sports",
        question="NBA Same Game Parlay: Lakers + Over 220",
        end_date="2026-06-12T18:00:00Z",
    )
    # 3) Missing 2 clob_token_ids
    one_token = _make_resolved(
        "0xonetok",
        category="Economics",
        clob_token_ids='["only_one_token"]',
        end_date="2026-06-12T18:00:00Z",
    )
    # 4) Order book disabled
    no_clob = _make_resolved(
        "0xnoclob",
        category="Economics",
        enable_order_book=False,
        end_date="2026-06-12T18:00:00Z",
    )
    # 5) Outside perimeter window
    out_of_window = _make_resolved(
        "0xold",
        category="Economics",
        end_date="2026-04-01T12:00:00Z",
    )
    # 6) Non-polymarket venue
    kalshi_resolved = ResolvedMarket(
        metadata=MarketMetadata(
            venue="kalshi",
            market_id="KXFED-26JUN",
            event_id=None,
            title="Fed cut?",
            category="Economics",
            outcomes=("Yes", "No"),
            is_yes_no=True,
            end_time_utc=datetime(2026, 6, 15, tzinfo=UTC),
            status="finalized",
            extra={"clob_token_ids": ("a", "b")},
        ),
        outcome_y=1,
    )

    # Disqualified checks
    for bad in (short_crypto, parlay, one_token, no_clob, kalshi_resolved):
        assert not is_eligible_resolved_polymarket(bad)

    # Medium-term crypto IS eligible (no false positive on long-term crypto)
    long_crypto = _make_resolved(
        "0xbtclong",
        category="Crypto",
        question="Will Bitcoin exceed $150k by July 2026?",
        slug="btc-150k-july-2026",
        end_date="2026-07-25T12:00:00Z",
    )
    assert is_eligible_resolved_polymarket(long_crypto)

    all_inputs = pool + [short_crypto, parlay, one_token, no_clob, out_of_window, kalshi_resolved]
    selected = stratify_resolved_markets(all_inputs, perimeter)
    selected_reversed = stratify_resolved_markets(list(reversed(all_inputs)), perimeter)

    assert len(selected) == 6
    # Deterministic & input-order invariant
    assert [m.metadata.market_id for m in selected] == [
        m.metadata.market_id for m in selected_reversed
    ]
    # Balanced: 3 strata -> exactly 2 markets from each stratum!
    strata_counts: dict[tuple[str, str], int] = {}
    for m in selected:
        assert m.metadata.end_time_utc is not None
        key = ((m.metadata.category or "").lower(), m.metadata.end_time_utc.strftime("%Y-%m"))
        strata_counts[key] = strata_counts.get(key, 0) + 1
    assert strata_counts == {
        ("politics", "2026-06"): 2,
        ("weather", "2026-06"): 2,
        ("economics", "2026-07"): 2,
    }


# ---------------------------------------------------------------------------
# 3. Pure point-in-time extraction: extract_asof_price & volume
# ---------------------------------------------------------------------------


def test_extract_asof_price_anti_leakage_and_settlement_exclusion() -> None:
    """Enforces point_ts <= t_i (never t_i + 1s / nearest-neighbor) and ignores settlement."""
    snap_ts = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
    t_epoch = int(snap_ts.timestamp())

    # Only future point (t_i + 1s) -> MUST return None (never nearest-neighbor).
    future_only = [{"timestamp": t_epoch + 1, "price": "0.95", "resolution_seconds": 60}]
    assert extract_asof_price(future_only, snap_ts, ts_key="timestamp", price_key="price") is None

    # Unsorted points with future point (t_i + 1s) and settlement point (resolution_seconds=0).
    mixed_points = [
        {"timestamp": t_epoch + 1, "price": "0.99", "resolution_seconds": 60},  # Future -> ignored
        {"timestamp": t_epoch, "price": "1.00", "resolution_seconds": 0},  # Settlement -> ignored
        {"timestamp": t_epoch - 300, "price": "0.41", "resolution_seconds": 60},
        {"timestamp": t_epoch - 60, "price": "0.63", "resolution_seconds": 60},  # Latest valid!
        {"timestamp": t_epoch - 600, "price": "0.35", "resolution_seconds": 60},
    ]
    assert extract_asof_price(
        mixed_points,
        snap_ts,
        ts_key="timestamp",
        price_key="price",
    ) == Decimal("0.63")

    # CLOB keys ('t', 'p') with exact t_i boundary match.
    clob_points = [
        {"t": t_epoch - 120, "p": Decimal("0.51")},
        {"t": t_epoch, "p": Decimal("0.54")},
        {"t": t_epoch + 1, "p": Decimal("0.88")},
    ]
    assert extract_asof_price(clob_points, snap_ts, ts_key="t", price_key="p") == Decimal("0.54")

    with pytest.raises(ValueError, match="timezone-aware"):
        extract_asof_price(clob_points, datetime(2026, 6, 10, 12, 0, 0))


def test_compute_point_in_time_volume_usd_excludes_future_and_other_markets() -> None:
    """Strictly sums notional_usd for market_id == condition_id and ts_utc <= t_i."""
    snap_ts = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
    trades = [
        TradeRecord(
            venue="polymarket",
            market_id="0xtarget",
            token_id="t1",
            price=Decimal("0.50"),
            size=Decimal("200"),
            notional_usd=Decimal("100"),
            ts_utc=snap_ts - timedelta(hours=2),
            side="buy",
        ),
        TradeRecord(
            venue="polymarket",
            market_id="0xtarget",
            token_id="t1",
            price=Decimal("0.50"),
            size=Decimal("300"),
            notional_usd=Decimal("150"),
            ts_utc=snap_ts,  # Exact boundary -> included
            side="sell",
        ),
        TradeRecord(
            venue="polymarket",
            market_id="0xtarget",
            token_id="t1",
            price=Decimal("0.90"),
            size=Decimal("5000"),
            notional_usd=Decimal("4500"),
            ts_utc=snap_ts + timedelta(seconds=1),  # Future -> MUST be excluded!
            side="buy",
        ),
        TradeRecord(
            venue="polymarket",
            market_id="0xother",
            token_id="t2",
            price=Decimal("0.50"),
            size=Decimal("1000"),
            notional_usd=Decimal("500"),
            ts_utc=snap_ts - timedelta(minutes=10),  # Different market -> excluded
            side="buy",
        ),
    ]

    vol_usd, count = compute_point_in_time_volume_usd(trades, "0xtarget", snap_ts)
    assert vol_usd == Decimal("250")
    assert count == 2


# ---------------------------------------------------------------------------
# 4. End-to-end offline backfill_polymarket() via httpx.MockTransport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_polymarket_anti_leakage_clob_fallback_and_volume_isolation(
    tmp_path: Path,
) -> None:
    """Verifies anti-leakage prices/volume, CLOB fallback, outcome_0, fee_type, and HTTP params."""
    perimeter = BackfillPerimeter(
        start_date_utc=datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC),
        end_date_utc=datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC),
        max_markets_per_venue=5,
        min_vol_usd=Decimal("200"),
        min_price=Decimal("0.01"),
        max_price=Decimal("0.99"),
        seed=42,
    )

    # Market A: Data v2 returns settlement + future point + valid past point -> poly_v2.
    # Also has huge Gamma volume="999999999" that MUST be ignored (actual trades <= t_i sum $250).
    raw_a = _make_gamma_closed_payload(
        "0xcond_a",
        category="Economics",
        outcomes='["No", "Yes"]',  # Reversed order -> canonical YES is index 1!
        outcome_prices='["0", "1"]',  # YES won -> outcome_y = 1
        clob_token_ids='["tok_no_a", "tok_yes_a"]',
        fee_type="economics",
        volume="999999999",
    )
    # Market B: Data v2 returns data: [] (resolution decay) -> automatic CLOB fallback (poly_clob).
    # Non-YES/NO outcomes -> outcome_side="outcome_0", outcome_label="Lakers".
    raw_b = _make_gamma_closed_payload(
        "0xcond_b",
        category="Sports",
        outcomes='["Lakers", "Celtics"]',
        outcome_prices='["1", "0"]',
        clob_token_ids='["tok_lakers_b", "tok_celtics_b"]',
        fee_type="sports",
        volume="999999999",
    )
    # Market C: Price outside [0.01, 0.99] -> skipped BEFORE /v2/trades (must NEVER query trades!).
    raw_c = _make_gamma_closed_payload(
        "0xcond_c",
        category="Politics",
        clob_token_ids='["tok_yes_c", "tok_no_c"]',
        volume="999999999",
    )

    resolved_a = parse_gamma_resolved_market(raw_a)
    resolved_b = parse_gamma_resolved_market(raw_b)
    resolved_c = parse_gamma_resolved_market(raw_c)
    assert resolved_a is not None and resolved_b is not None and resolved_c is not None

    t_a = sample_variable_tau_snapshot_ts(resolved_a, perimeter)
    t_b = sample_variable_tau_snapshot_ts(resolved_b, perimeter)
    t_c = sample_variable_tau_snapshot_ts(resolved_c, perimeter)
    assert t_a is not None and t_b is not None and t_c is not None
    ta_s, tb_s, tc_s = int(t_a.timestamp()), int(t_b.timestamp()), int(t_c.timestamp())

    gamma_requests: list[dict[str, str]] = []
    v2_price_requests: list[dict[str, str]] = []
    clob_price_requests: list[dict[str, str]] = []
    trade_requests: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        path = request.url.path
        params = dict(request.url.params)

        if host == "gamma-api.polymarket.com" and path == "/markets":
            gamma_requests.append(params)
            assert "order" not in params
            return httpx.Response(200, json=[raw_a, raw_b, raw_c])

        if host == "data-api.polymarket.com" and path == "/v2/prices-history":
            v2_price_requests.append(params)
            tid = params["token_id"]
            if tid == "tok_yes_a":
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"timestamp": ta_s - 3600, "price": "0.62", "resolution_seconds": 60},
                            # Settlement point -> MUST be ignored!
                            {"timestamp": ta_s, "price": "1.00", "resolution_seconds": 0},
                            # Future point (t_i + 1s) -> MUST be ignored!
                            {"timestamp": ta_s + 1, "price": "0.97", "resolution_seconds": 60},
                        ],
                        "pagination": {"has_more": False, "next_cursor": None},
                    },
                )
            if tid == "tok_lakers_b":
                # Resolution decay -> empty list triggers CLOB fallback
                return httpx.Response(
                    200,
                    json={"data": [], "pagination": {"has_more": False, "next_cursor": None}},
                )
            if tid == "tok_yes_c":
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"timestamp": tc_s - 60, "price": "0.995", "resolution_seconds": 60},
                        ],
                        "pagination": {"has_more": False, "next_cursor": None},
                    },
                )
            return httpx.Response(200, json={"data": []})

        if host == "clob.polymarket.com" and path == "/prices-history":
            clob_price_requests.append(params)
            assert params["market"] == "tok_lakers_b"
            assert int(params["endTs"]) == tb_s
            assert int(params["startTs"]) == tb_s - 14 * 86_400
            return httpx.Response(
                200,
                json={
                    "history": [
                        {"t": tb_s - 7200, "p": 0.44},
                        {"t": tb_s + 1, "p": 0.91},  # Future -> MUST be ignored!
                    ]
                },
            )

        if host == "data-api.polymarket.com" and path == "/v2/trades":
            trade_requests.append(params)
            assert params["taker_only"] == "true"
            assert "market" not in params
            assert "offset" not in params
            requested_ids = params["condition"].split(",")
            assert 1 <= len(requested_ids) <= POLYMARKET_MAX_CONDITION_BATCH
            # Market C failed price guard -> MUST NEVER appear in /v2/trades!
            assert "0xcond_c" not in requested_ids

            cursor = params.get("cursor")
            if cursor is None:
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            # Market A: $100 <= t_a, plus $5000 at t_a + 1s (future -> excluded!)
                            {
                                "condition_id": "0xcond_a",
                                "token_id": "tok_yes_a",
                                "price": "0.50",
                                "size": "200",
                                "usdc_size": "100",
                                "timestamp": ta_s - 120,
                                "side": "BUY",
                            },
                            {
                                "condition_id": "0xcond_a",
                                "token_id": "tok_yes_a",
                                "price": "0.90",
                                "size": "5555",
                                "usdc_size": "5000",
                                "timestamp": ta_s + 1,
                                "side": "BUY",
                            },
                            # Market B: $300 <= t_b
                            {
                                "condition_id": "0xcond_b",
                                "token_id": "tok_lakers_b",
                                "price": "0.44",
                                "size": "681.818181",
                                "usdc_size": "300",
                                "timestamp": tb_s - 60,
                                "side": "BUY",
                            },
                        ],
                        "pagination": {"has_more": True, "next_cursor": "cursor_page_2"},
                    },
                )
            assert cursor == "cursor_page_2"
            return httpx.Response(
                200,
                json={
                    "data": [
                        # Second page adds $150 <= t_a -> total for Market A = $250
                        {
                            "condition_id": "0xcond_a",
                            "token_id": "tok_yes_a",
                            "price": "0.60",
                            "size": "250",
                            "usdc_size": "150",
                            "timestamp": ta_s,
                            "side": "SELL",
                        }
                    ],
                    "pagination": {"has_more": False, "next_cursor": None},
                },
            )

        raise AssertionError(f"Unexpected HTTP call: {request.url}")

    transport = httpx.MockTransport(handler)
    async with PolymarketClient(transport=transport, max_retries=0) as client:
        records = await backfill_polymarket(perimeter, client=client)

    assert len(gamma_requests) == 1
    assert gamma_requests[0]["closed"] == "true"
    assert gamma_requests[0]["end_date_min"] == "2026-06-01T00:00:00Z"
    assert gamma_requests[0]["end_date_max"] == "2026-06-30T23:59:59Z"

    # CLOB fallback called ONLY for Market B (not Market A)
    assert [req["market"] for req in clob_price_requests] == ["tok_lakers_b"]
    # Cursor pagination executed across 2 pages
    assert [req.get("cursor") for req in trade_requests] == [None, "cursor_page_2"]

    by_id = {rec.market_id: rec for rec in records}
    assert set(by_id.keys()) == {"0xcond_a", "0xcond_b"}

    rec_a = by_id["0xcond_a"]
    assert rec_a.token_id == "tok_yes_a"
    assert rec_a.outcome_side == "yes"
    assert rec_a.outcome_label == "Yes"
    assert rec_a.is_yes_no is True
    assert rec_a.fee_type == "economics"
    assert rec_a.p_mkt == Decimal("0.62")
    assert rec_a.price_source == "poly_v2"
    # Gamma volume "999999999" completely ignored; future trade $5000 excluded -> exactly $250!
    assert rec_a.vol_usd_le_t == Decimal("250")
    assert rec_a.trade_count_le_t == 2
    assert rec_a.trades_truncated is False
    assert rec_a.outcome_y == 1

    rec_b = by_id["0xcond_b"]
    assert rec_b.token_id == "tok_lakers_b"
    assert rec_b.outcome_side == "outcome_0"
    assert rec_b.outcome_label == "Lakers"
    assert rec_b.is_yes_no is False
    assert rec_b.fee_type == "sports"
    assert rec_b.p_mkt == Decimal("0.44")
    assert rec_b.price_source == "poly_clob"
    assert rec_b.vol_usd_le_t == Decimal("300")
    assert rec_b.trade_count_le_t == 1
    assert rec_b.trades_truncated is False
    assert rec_b.outcome_y == 1

    # Round-trip Parquet + DuckDB verification
    pq_path = write_historical_parquet(records, tmp_path / "poly_backfill.parquet")
    df = read_historical_duckdb(pq_path)
    assert df.height == 2


@pytest.mark.asyncio
async def test_backfill_polymarket_trade_chunking_pagination_cap_and_monthly_slicing() -> None:
    """Enforces <= 20 IDs per /v2/trades call, monthly Gamma slicing, and trades_truncated."""
    perimeter = BackfillPerimeter(
        start_date_utc=datetime(2026, 5, 1, 0, 0, 0, tzinfo=UTC),
        end_date_utc=datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC),
        max_markets_per_venue=23,
        min_vol_usd=Decimal("10"),
        seed=7,
    )

    may_payloads = [
        _make_gamma_closed_payload(
            f"0xmay_{i:02d}",
            category="Politics",
            start_date="2026-04-01T00:00:00Z",
            end_date="2026-05-20T12:00:00Z",
        )
        for i in range(12)
    ]
    jun_payloads = [
        _make_gamma_closed_payload(
            f"0xjun_{i:02d}",
            category="Economics",
            start_date="2026-05-01T00:00:00Z",
            end_date="2026-06-20T12:00:00Z",
        )
        for i in range(11)
    ]

    snap_by_cond: dict[str, int] = {}
    for raw in may_payloads + jun_payloads:
        res = parse_gamma_resolved_market(raw)
        assert res is not None
        ts = sample_variable_tau_snapshot_ts(res, perimeter)
        assert ts is not None
        snap_by_cond[res.metadata.market_id] = int(ts.timestamp())

    gamma_windows: list[tuple[str, str]] = []
    trade_batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        path = request.url.path
        params = dict(request.url.params)

        if host == "gamma-api.polymarket.com" and path == "/markets":
            win = (params["end_date_min"], params["end_date_max"])
            gamma_windows.append(win)
            if win[0].startswith("2026-05"):
                return httpx.Response(200, json=may_payloads)
            return httpx.Response(200, json=jun_payloads)

        if host == "data-api.polymarket.com" and path == "/v2/prices-history":
            end_s = int(params["end"])
            return httpx.Response(
                200,
                json={
                    "data": [{"timestamp": end_s - 60, "price": "0.55", "resolution_seconds": 60}],
                    "pagination": {"has_more": False, "next_cursor": None},
                },
            )

        if host == "data-api.polymarket.com" and path == "/v2/trades":
            assert params["taker_only"] == "true"
            assert "market" not in params
            conds = params["condition"].split(",")
            assert 1 <= len(conds) <= POLYMARKET_MAX_CONDITION_BATCH
            if params.get("cursor") is None:
                trade_batch_sizes.append(len(conds))
            page_num = 1 if params.get("cursor") is None else int(params["cursor"].split("_")[-1])
            data = [
                {
                    "condition_id": cid,
                    "token_id": f"tok_yes_{cid}",
                    "price": "0.55",
                    "size": "100",
                    "usdc_size": "55",
                    "timestamp": snap_by_cond[cid] - 30 * page_num,
                    "side": "BUY",
                }
                for cid in conds
            ]
            # Always claim another page -> hitting max_trade_pages_per_batch=2 sets truncated!
            return httpx.Response(
                200,
                json={
                    "data": data,
                    "pagination": {"has_more": True, "next_cursor": f"cur_{page_num + 1}"},
                },
            )

        raise AssertionError(f"Unexpected request: {request.url}")

    transport = httpx.MockTransport(handler)
    async with PolymarketClient(transport=transport, max_retries=0) as client:
        records = await backfill_polymarket(
            perimeter,
            client=client,
            max_gamma_pages=10,
            max_trade_pages_per_batch=2,
            trade_batch_size=20,
        )

    assert sorted(gamma_windows) == [
        ("2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z"),
        ("2026-06-01T00:00:00Z", "2026-06-30T23:59:59Z"),
    ]
    assert sorted(trade_batch_sizes) == [3, 20]
    assert len(records) == 23
    assert all(rec.trades_truncated is True for rec in records)
    assert all(rec.vol_usd_le_t == Decimal("110") for rec in records)
    assert all(rec.trade_count_le_t == 2 for rec in records)


# ---------------------------------------------------------------------------
# 5. Resumable JSONL checkpointing (full resume, partial resume, crash tail, transient errors)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_polymarket_checkpoint_resumability_and_crash_recovery(
    tmp_path: Path,
) -> None:
    """Second run skips completed markets without re-fetching; recovers from truncated tail line."""
    perimeter = BackfillPerimeter(
        start_date_utc=datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC),
        end_date_utc=datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC),
        max_markets_per_venue=2,
        min_vol_usd=Decimal("50"),
        seed=99,
    )
    ckpt = default_checkpoint_path("polymarket", perimeter, base_dir=tmp_path)
    assert ckpt.name == f"polymarket_{compute_perimeter_hash(perimeter)}.jsonl"

    raw_1 = _make_gamma_closed_payload("0xckpt_1", category="Economics")
    raw_2 = _make_gamma_closed_payload("0xckpt_2", category="Weather")

    snap_by_cond: dict[str, int] = {}
    for raw in (raw_1, raw_2):
        res = parse_gamma_resolved_market(raw)
        assert res is not None
        ts = sample_variable_tau_snapshot_ts(res, perimeter)
        assert ts is not None
        snap_by_cond[res.metadata.market_id] = int(ts.timestamp())

    call_log: list[tuple[str, str, dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        call_log.append((request.url.host, request.url.path, params))

        if request.url.host == "gamma-api.polymarket.com":
            return httpx.Response(200, json=[raw_1, raw_2])
        if request.url.path == "/v2/prices-history":
            end_s = int(params["end"])
            return httpx.Response(
                200,
                json={
                    "data": [{"timestamp": end_s - 60, "price": "0.52", "resolution_seconds": 60}],
                    "pagination": {"has_more": False, "next_cursor": None},
                },
            )
        if request.url.path == "/v2/trades":
            conds = params["condition"].split(",")
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "condition_id": cid,
                            "token_id": f"tok_yes_{cid}",
                            "price": "0.52",
                            "size": "200",
                            "usdc_size": "104",
                            "timestamp": snap_by_cond[cid] - 60,
                            "side": "BUY",
                        }
                        for cid in conds
                    ],
                    "pagination": {"has_more": False, "next_cursor": None},
                },
            )
        raise AssertionError(f"Unexpected call: {request.url}")

    transport = httpx.MockTransport(handler)
    async with PolymarketClient(transport=transport, max_retries=0) as client:
        # Run 1: populates both markets and appends to checkpoint.
        first_run = await backfill_polymarket(perimeter, client=client, checkpoint_path=ckpt)
        assert len(first_run) == 2
        assert len(call_log) > 0

        # Verify JSONL checkpoint schema on disk (Decimal stored as str).
        lines = ckpt.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first_entry = json.loads(lines[0])
        assert first_entry["checkpoint_version"] == CHECKPOINT_VERSION
        assert first_entry["perimeter_hash"] == compute_perimeter_hash(perimeter)
        assert first_entry["status"] == "ok"
        assert isinstance(first_entry["record"]["p_mkt"], str)
        assert isinstance(first_entry["record"]["vol_usd_le_t"], str)

        # Run 2 (full resume): quota already met -> ZERO HTTP calls!
        call_log.clear()
        second_run = await backfill_polymarket(perimeter, client=client, checkpoint_path=ckpt)
        assert call_log == []
        assert second_run == first_run

        # Run 3 (partial resume + truncated crash tail line): keep only first market + partial JSON.
        kept_market_id = first_entry["market_id"]
        missing_market_id = "0xckpt_2" if kept_market_id == "0xckpt_1" else "0xckpt_1"
        ckpt.write_text(
            lines[0] + '\n{"checkpoint_version":1,"status":"ok","truncated',
            encoding="utf-8",
        )
        call_log.clear()
        third_run = await backfill_polymarket(perimeter, client=client, checkpoint_path=ckpt)
        assert len(third_run) == 2
        assert {r.market_id for r in third_run} == {"0xckpt_1", "0xckpt_2"}

        # Kept market MUST NOT be re-queried for prices or trades!
        price_tokens = [p["token_id"] for _, path, p in call_log if path == "/v2/prices-history"]
        trade_conds = [p["condition"] for _, path, p in call_log if path == "/v2/trades"]
        assert price_tokens == [f"tok_yes_{missing_market_id}"]
        assert trade_conds == [missing_market_id]

        # Incompatible perimeter hash on existing checkpoint raises ValueError.
        different_perimeter = BackfillPerimeter(
            start_date_utc=perimeter.start_date_utc,
            end_date_utc=perimeter.end_date_utc,
            max_markets_per_venue=2,
            min_vol_usd=Decimal("50"),
            seed=12345,
        )
        with pytest.raises(ValueError, match="perimeter_hash mismatch"):
            await backfill_polymarket(different_perimeter, client=client, checkpoint_path=ckpt)


@pytest.mark.asyncio
async def test_backfill_polymarket_never_caches_transient_http_errors(tmp_path: Path) -> None:
    """Transient HTTP 503 errors are not cached as skipped and succeed on a subsequent resume."""
    perimeter = BackfillPerimeter(
        start_date_utc=datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC),
        end_date_utc=datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC),
        max_markets_per_venue=1,
        min_vol_usd=Decimal("10"),
        seed=42,
    )
    ckpt = tmp_path / "transient.jsonl"
    raw_1 = _make_gamma_closed_payload("0xtransient_1")
    res_1 = parse_gamma_resolved_market(raw_1)
    assert res_1 is not None
    snap_ts = sample_variable_tau_snapshot_ts(res_1, perimeter)
    assert snap_ts is not None
    snap_s = int(snap_ts.timestamp())

    fail_prices = True

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "gamma-api.polymarket.com":
            return httpx.Response(200, json=[raw_1])
        if request.url.path == "/v2/prices-history":
            if fail_prices:
                return httpx.Response(503, json={"error": "service unavailable"})
            return httpx.Response(
                200,
                json={
                    "data": [{"timestamp": snap_s - 60, "price": "0.60", "resolution_seconds": 60}],
                    "pagination": {"has_more": False, "next_cursor": None},
                },
            )
        if request.url.path == "/v2/trades":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "condition_id": "0xtransient_1",
                            "token_id": "tok_yes_0xtransient_1",
                            "price": "0.60",
                            "size": "100",
                            "usdc_size": "60",
                            "timestamp": snap_s - 60,
                            "side": "BUY",
                        }
                    ],
                    "pagination": {"has_more": False, "next_cursor": None},
                },
            )
        raise AssertionError(f"Unexpected call: {request.url}")

    transport = httpx.MockTransport(handler)
    async with PolymarketClient(transport=transport, max_retries=0) as client:
        # First run hits 503 -> returns empty and does NOT write a skipped checkpoint line.
        run_1 = await backfill_polymarket(perimeter, client=client, checkpoint_path=ckpt)
        assert run_1 == []
        assert not ckpt.exists()

        # Once transient outage clears, resume retries and succeeds!
        fail_prices = False
        run_2 = await backfill_polymarket(perimeter, client=client, checkpoint_path=ckpt)
        assert len(run_2) == 1
        assert run_2[0].market_id == "0xtransient_1"
        assert run_2[0].p_mkt == Decimal("0.60")
        assert run_2[0].vol_usd_le_t == Decimal("60")


@pytest.mark.asyncio
async def test_backfill_polymarket_guard_skips_checkpoint_and_v2_404_fallback(
    tmp_path: Path,
) -> None:
    """Persists deterministic guard skips and handles Data v2 404 -> CLOB fallback."""
    perimeter = BackfillPerimeter(
        start_date_utc=datetime(2025, 12, 1, 0, 0, 0, tzinfo=UTC),
        end_date_utc=datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC),
        max_markets_per_venue=4,
        min_vol_usd=Decimal("500"),
        pre_resolution_buffer_hours=2,
        seed=11,
    )
    ckpt = tmp_path / "guards.jsonl"

    # 1) Lifetime shorter than 2h buffer -> no_valid_snapshot_ts
    raw_short_life = _make_gamma_closed_payload(
        "0xshort_life",
        start_date="2025-12-15T11:00:00Z",
        end_date="2025-12-15T12:00:00Z",
    )
    # 2) Both v2 (404) and CLOB empty -> no_price_history (never calls /v2/trades)
    raw_no_price = _make_gamma_closed_payload(
        "0xno_price",
        start_date="2025-12-01T00:00:00Z",
        end_date="2025-12-20T12:00:00Z",
    )
    # 3) v2 returns 404 -> falls back to CLOB price, but trades <= t_i sum only $100 < $500
    #    despite Gamma volume="999999999" -> vol_below_min!
    raw_low_vol = _make_gamma_closed_payload(
        "0xlow_vol",
        start_date="2025-12-01T00:00:00Z",
        end_date="2026-01-15T12:00:00Z",
        volume="999999999",
    )
    # 4) v2 returns 404 -> falls back to CLOB price, trades <= t_i sum $600 -> ok!
    raw_ok = _make_gamma_closed_payload(
        "0xok_404_clob",
        start_date="2025-12-01T00:00:00Z",
        end_date="2026-01-20T12:00:00Z",
    )

    snaps: dict[str, int] = {}
    for raw in (raw_low_vol, raw_ok):
        res = parse_gamma_resolved_market(raw)
        assert res is not None
        ts = sample_variable_tau_snapshot_ts(res, perimeter)
        assert ts is not None
        snaps[res.metadata.market_id] = int(ts.timestamp())

    price_calls: list[str] = []
    trade_calls: list[str] = []
    gamma_windows: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if request.url.host == "gamma-api.polymarket.com":
            win = (params["end_date_min"], params["end_date_max"])
            gamma_windows.append(win)
            if win[0].startswith("2025-12"):
                return httpx.Response(200, json=[raw_short_life, raw_no_price])
            return httpx.Response(200, json=[raw_low_vol, raw_ok])

        if (
            request.url.host == "data-api.polymarket.com"
            and request.url.path == "/v2/prices-history"
        ):
            price_calls.append(params["token_id"])
            return httpx.Response(404, json={"error": "not found"})

        if request.url.host == "clob.polymarket.com" and request.url.path == "/prices-history":
            tid = params["market"]
            end_s = int(params["endTs"])
            if tid == "tok_yes_0xno_price":
                return httpx.Response(200, json={"history": []})
            return httpx.Response(200, json={"history": [{"t": end_s - 300, "p": 0.48}]})

        if request.url.host == "data-api.polymarket.com" and request.url.path == "/v2/trades":
            conds = params["condition"].split(",")
            trade_calls.extend(conds)
            assert "0xno_price" not in conds
            assert "0xshort_life" not in conds
            data: list[dict[str, Any]] = []
            for cid in conds:
                amt = "100" if cid == "0xlow_vol" else "600"
                data.append(
                    {
                        "condition_id": cid,
                        "token_id": f"tok_yes_{cid}",
                        "price": "0.50",
                        "size": amt,
                        "usdc_size": amt,
                        "timestamp": snaps[cid] - 60,
                        "side": "BUY",
                    }
                )
            return httpx.Response(
                200,
                json={"data": data, "pagination": {"has_more": False, "next_cursor": None}},
            )

        raise AssertionError(f"Unexpected call: {request.url}")

    transport = httpx.MockTransport(handler)
    async with PolymarketClient(transport=transport, max_retries=0) as client:
        run_1 = await backfill_polymarket(perimeter, client=client, checkpoint_path=ckpt)
        assert len(run_1) == 1
        assert run_1[0].market_id == "0xok_404_clob"
        assert run_1[0].price_source == "poly_clob"
        assert run_1[0].vol_usd_le_t == Decimal("600")

        # Year-crossing monthly windows covered Dec 2025 and Jan 2026 cleanly.
        assert sorted(gamma_windows) == [
            ("2025-12-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            ("2026-01-01T00:00:00Z", "2026-01-31T23:59:59Z"),
        ]

        # All 3 permanent skips are recorded in checkpoint with skip_reason.
        entries = [json.loads(line) for line in ckpt.read_text(encoding="utf-8").splitlines()]
        by_mid = {e["market_id"]: e for e in entries}
        assert by_mid["0xshort_life"]["status"] == "skipped"
        assert by_mid["0xshort_life"]["skip_reason"] == "no_valid_snapshot_ts"
        assert by_mid["0xno_price"]["skip_reason"] == "no_price_history"
        assert by_mid["0xlow_vol"]["skip_reason"] == "vol_below_min"
        assert by_mid["0xok_404_clob"]["status"] == "ok"

        # Second run skips ALL 3 previously skipped markets and the ok market (0 price/trade calls).
        price_calls.clear()
        trade_calls.clear()
        run_2 = await backfill_polymarket(perimeter, client=client, checkpoint_path=ckpt)
        assert run_2 == run_1
        assert price_calls == []
        assert trade_calls == []


def test_extract_asof_price_timestamp_formats_and_validation() -> None:
    """Supports ms epoch, ISO-8601 strings, aware datetimes, and rejects invalid inputs."""
    snap = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)
    snap_s = int(snap.timestamp())

    points: list[dict[str, Any]] = [
        {"timestamp": (snap_s - 120) * 1000, "price": "0.31"},  # ms epoch
        {"timestamp": "2026-06-15T11:59:00Z", "price": "0.42"},  # ISO-8601 string
        {"timestamp": snap - timedelta(seconds=10), "price": 0.55},  # aware datetime
        {"timestamp": snap - timedelta(seconds=5), "price": "not-a-number"},  # invalid price
        {"timestamp": snap - timedelta(seconds=2), "price": "1.50"},  # out-of-range price
        {"timestamp": "bad-ts", "price": "0.99"},  # invalid timestamp
    ]
    assert extract_asof_price(points, snap) == Decimal("0.55")

    with pytest.raises(ValueError, match="non-empty"):
        compute_point_in_time_volume_usd([], "   ", snap)


@pytest.mark.asyncio
async def test_backfill_polymarket_argument_validation() -> None:
    """Validates orchestrator and stratification bounds upfront."""
    perimeter = BackfillPerimeter(
        start_date_utc=datetime(2026, 6, 1, tzinfo=UTC),
        end_date_utc=datetime(2026, 6, 30, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="max_markets"):
        stratify_resolved_markets([], perimeter, max_markets=0)
    with pytest.raises(ValueError, match="max_gamma_pages"):
        await backfill_polymarket(perimeter, max_gamma_pages=-1)
    with pytest.raises(ValueError, match="max_trade_pages_per_batch"):
        await backfill_polymarket(perimeter, max_trade_pages_per_batch=0)
    with pytest.raises(ValueError, match="max_concurrency"):
        await backfill_polymarket(perimeter, max_concurrency=0)
    with pytest.raises(ValueError, match="trade_batch_size"):
        await backfill_polymarket(perimeter, trade_batch_size=21)
    with pytest.raises(ValueError, match="oversample_factor"):
        await backfill_polymarket(perimeter, oversample_factor=0.5)
