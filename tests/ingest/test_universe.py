"""Offline unit tests for deterministic L2 universe selection and live screening helpers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import httpx
import pytest

from pm_quant.ingest import (
    DEFAULT_MACRO_SERIES_PREFIXES,
    DEFAULT_WEATHER_SERIES_PREFIXES,
    DEFAULT_WEEKLY_SERIES_PREFIXES,
    KalshiClient,
    LiveMarketCandidate,
    MarketMetadata,
    PolymarketClient,
    SelectedMarket,
    UniversePolicy,
    is_binary_non_combo,
    is_short_term_crypto,
    is_target_series,
    parse_gamma_live_candidate,
    parse_gamma_market,
    parse_kalshi_live_candidate,
    parse_kalshi_market,
    select_universe,
)

NOW_UTC = datetime(2026, 11, 23, 12, 0, 0, tzinfo=UTC)


def _poly_meta(
    market_id: str = "0xpoly-1",
    *,
    event_id: str | None = "evt-poly-1",
    title: str = "Will candidate A win the election?",
    category: str | None = "Politics",
    outcomes: tuple[str, ...] = ("Yes", "No"),
    is_yes_no: bool | None = None,
    end_time_utc: datetime | None = NOW_UTC + timedelta(days=10),
    status: str = "active",
    enable_order_book: Any = True,
    accepting_orders: Any = True,
    clob_token_ids: Any = ("tok-yes", "tok-no"),
    extra_overrides: dict[str, Any] | None = None,
) -> MarketMetadata:
    extra: dict[str, Any] = {
        "clob_token_ids": clob_token_ids,
        "neg_risk": False,
        "enable_order_book": enable_order_book,
    }
    if accepting_orders is not None:
        extra["accepting_orders"] = accepting_orders
    if extra_overrides:
        extra.update(extra_overrides)

    resolved_yes_no = (
        is_yes_no
        if is_yes_no is not None
        else (len(outcomes) == 2 and {o.strip().lower() for o in outcomes} == {"yes", "no"})
    )
    return MarketMetadata(
        venue="polymarket",
        market_id=market_id,
        event_id=event_id,
        title=title,
        category=category,
        outcomes=outcomes,
        is_yes_no=resolved_yes_no,
        end_time_utc=end_time_utc,
        status=status,
        extra=extra,
    )


def _kalshi_meta(
    market_id: str = "KXPOL-26DEC-A",
    *,
    event_id: str | None = "KXPOL-26DEC",
    title: str = "Will bill X pass Congress?",
    category: str | None = "Politics",
    outcomes: tuple[str, ...] = ("Yes", "No"),
    is_yes_no: bool = True,
    end_time_utc: datetime | None = NOW_UTC + timedelta(days=10),
    status: str = "open",
    extra_overrides: dict[str, Any] | None = None,
) -> MarketMetadata:
    extra: dict[str, Any] = {"market_type": "binary"}
    if extra_overrides:
        extra.update(extra_overrides)
    return MarketMetadata(
        venue="kalshi",
        market_id=market_id,
        event_id=event_id,
        title=title,
        category=category,
        outcomes=outcomes,
        is_yes_no=is_yes_no,
        end_time_utc=end_time_utc,
        status=status,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# 1. Immutable dataclasses & UniversePolicy validation
# ---------------------------------------------------------------------------


def test_dataclasses_are_frozen_slotted_and_validated() -> None:
    """LiveMarketCandidate, SelectedMarket, and UniversePolicy are immutable and validated."""
    policy = UniversePolicy()
    assert policy.min_volume_24h_polymarket_usd == Decimal("5000")
    assert policy.min_volume_24h_kalshi_contracts == Decimal("1000")
    assert policy.max_core_per_venue == 200
    assert policy.max_target_series_per_venue == 200
    assert policy.target_series_max_horizon_days == 45
    assert set(DEFAULT_WEATHER_SERIES_PREFIXES).issubset(policy.target_series_prefixes)
    assert set(DEFAULT_MACRO_SERIES_PREFIXES).issubset(policy.target_series_prefixes)
    assert set(DEFAULT_WEEKLY_SERIES_PREFIXES).issubset(policy.target_series_prefixes)
    assert set(policy.target_categories) == {"weather", "economics"}

    cand = LiveMarketCandidate(
        metadata=_poly_meta(),
        volume_24h=Decimal("7500"),
        open_interest=Decimal("12000"),
        volume_24h_unit="usd",
        tag_ids=(" 84 ", "84", ""),
    )
    assert cand.tag_ids == ("84",)
    selected = SelectedMarket(candidate=cand, selection_reason="volume_core")

    with pytest.raises(FrozenInstanceError):
        cand.volume_24h = Decimal("0")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        selected.selection_reason = "target_series"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        policy.max_core_per_venue = 10  # type: ignore[misc]

    # Invalid screening values raise ValueError.
    with pytest.raises(ValueError, match="volume_24h"):
        LiveMarketCandidate(metadata=_poly_meta(), volume_24h=Decimal("-1"))
    with pytest.raises(ValueError, match="volume_24h"):
        LiveMarketCandidate(metadata=_poly_meta(), volume_24h=Decimal("NaN"))
    with pytest.raises(ValueError, match="open_interest"):
        LiveMarketCandidate(metadata=_poly_meta(), open_interest=Decimal("-0.01"))
    with pytest.raises(ValueError, match="volume_24h_unit"):
        LiveMarketCandidate(metadata=_poly_meta(), volume_24h_unit="shares")  # type: ignore[arg-type]

    # Invalid policy parameters raise ValueError.
    with pytest.raises(ValueError, match="min_volume_24h_polymarket_usd"):
        UniversePolicy(min_volume_24h_polymarket_usd=Decimal("-1"))
    with pytest.raises(ValueError, match="min_volume_24h_kalshi_contracts"):
        UniversePolicy(min_volume_24h_kalshi_contracts=Decimal("Infinity"))
    with pytest.raises(ValueError, match="max_core_per_venue"):
        UniversePolicy(max_core_per_venue=-1)
    with pytest.raises(ValueError, match="max_target_series_per_venue"):
        UniversePolicy(max_target_series_per_venue=True)
    with pytest.raises(ValueError, match="target_series_max_horizon_days"):
        UniversePolicy(target_series_max_horizon_days=-5)
    with pytest.raises(ValueError, match="invalid regex"):
        UniversePolicy(target_series_patterns=("[unclosed",))


# ---------------------------------------------------------------------------
# 2. Binary non-combo filter & MVE/combo exclusion
# ---------------------------------------------------------------------------


def test_is_binary_non_combo_accepts_valid_polymarket_and_kalshi_markets() -> None:
    """Active/open binary markets (Yes/No and non-Yes/No head-to-head) pass cleanly."""
    assert is_binary_non_combo(_poly_meta()) is True
    # Non-Yes/No binary market (e.g., Lakers vs Celtics) with accepting_orders omitted.
    assert (
        is_binary_non_combo(
            _poly_meta(
                outcomes=("Lakers", "Celtics"),
                is_yes_no=False,
                accepting_orders=None,
            )
        )
        is True
    )
    assert is_binary_non_combo(_kalshi_meta(status="open")) is True
    assert is_binary_non_combo(_kalshi_meta(status="active")) is True
    # Ensure ISO country code SGP (Singapore) does not trigger parlay false positive.
    assert (
        is_binary_non_combo(
            _kalshi_meta(market_id="KXSGP-26Q4", title="Will Singapore GDP exceed 3%?")
        )
        is True
    )


@pytest.mark.parametrize(
    ("label", "meta"),
    [
        ("poly_closed_status", _poly_meta(status="closed")),
        ("poly_archived_status", _poly_meta(status="archived")),
        ("poly_unknown_status", _poly_meta(status="unknown")),
        ("poly_extra_closed", _poly_meta(extra_overrides={"closed": True})),
        ("poly_extra_archived", _poly_meta(extra_overrides={"archived": True})),
        ("poly_extra_inactive", _poly_meta(extra_overrides={"active": False})),
        ("kalshi_settled", _kalshi_meta(status="settled")),
        ("kalshi_finalized", _kalshi_meta(status="finalized")),
        ("multi_outcome_3", _poly_meta(outcomes=("A", "B", "C"))),
        ("single_outcome", _poly_meta(outcomes=("Yes",))),
        ("empty_outcomes", _poly_meta(outcomes=())),
        ("blank_outcome_label", _poly_meta(outcomes=("Yes", "   "))),
        ("kalshi_scalar_empty_outcomes", _kalshi_meta(outcomes=(), is_yes_no=False)),
        (
            "kalshi_scalar_market_type",
            _kalshi_meta(extra_overrides={"market_type": "scalar"}),
        ),
        ("poly_orderbook_disabled", _poly_meta(enable_order_book=False)),
        ("poly_orderbook_missing", _poly_meta(enable_order_book=None)),
        ("poly_not_accepting_orders", _poly_meta(accepting_orders=False)),
        ("poly_only_1_token", _poly_meta(clob_token_ids=("tok-1",))),
        ("poly_no_tokens", _poly_meta(clob_token_ids=())),
        ("poly_3_tokens", _poly_meta(clob_token_ids=("t1", "t2", "t3"))),
        ("poly_blank_token", _poly_meta(clob_token_ids=("tok-1", "   "))),
    ],
)
def test_is_binary_non_combo_rejects_inactive_non_binary_and_untradable(
    label: str,
    meta: MarketMetadata,
) -> None:
    """Inactive, scalar, multi-outcome, and untradable Polymarket markets are excluded."""
    assert is_binary_non_combo(meta) is False, f"expected rejection for {label}"


@pytest.mark.parametrize(
    ("label", "meta"),
    [
        (
            "kalshi_mve_collection_ticker",
            _kalshi_meta(extra_overrides={"mve_collection_ticker": "KXMVECROSSCATEGORY-1"}),
        ),
        (
            "kalshi_mve_selected_legs",
            _kalshi_meta(extra_overrides={"mve_selected_legs": [{"market_ticker": "A"}]}),
        ),
        (
            "kalshi_kxmve_market_id",
            _kalshi_meta(market_id="KXMVECROSSCATEGORY-S2026ABC"),
        ),
        (
            "kalshi_kxmulti_market_id",
            _kalshi_meta(market_id="KXMULTI-26NOV-XYZ"),
        ),
        (
            "kalshi_kxmve_event_id",
            _kalshi_meta(event_id="KXMVESPORTS-26NOV"),
        ),
        (
            "kalshi_kxmulti_series_ticker",
            _kalshi_meta(extra_overrides={"series_ticker": "KXMULTIGAME"}),
        ),
        (
            "poly_title_combo",
            _poly_meta(title="NBA Combo: Lakers win AND Over 220.5"),
        ),
        (
            "poly_title_parlay",
            _poly_meta(title="NFL Week 12 Same-Game Parlay"),
        ),
        (
            "poly_slug_combo",
            _poly_meta(extra_overrides={"slug": "sports-combo-lakers-celtics"}),
        ),
        (
            "poly_slug_parlay",
            _poly_meta(extra_overrides={"slug": "nfl-parlay-sunday"}),
        ),
        (
            "poly_fee_type_combos",
            _poly_meta(extra_overrides={"fee_type": "combos"}),
        ),
        (
            "poly_category_combos",
            _poly_meta(category="Combos"),
        ),
        (
            "kalshi_title_multi_leg",
            _kalshi_meta(title="Multi-leg sports bundle"),
        ),
    ],
)
def test_is_binary_non_combo_excludes_mve_and_combos(label: str, meta: MarketMetadata) -> None:
    """MVE collections, KXMVE/KXMULTI prefixes, and combo/parlay markets are excluded."""
    assert is_binary_non_combo(meta) is False, f"expected combo/MVE rejection for {label}"


# ---------------------------------------------------------------------------
# 3. Short-term crypto 5m/15m exclusion + zero false positives on longer crypto
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "meta"),
    [
        ("btc_5m_title", _poly_meta(title="BTC 5m price higher?", category="Crypto")),
        ("eth_15m_title", _poly_meta(title="ETH 15m candle green?", category="Crypto")),
        ("sol_5_min_title", _poly_meta(title="Solana 5-min price up?", category="Crypto")),
        ("xrp_15_min_title", _poly_meta(title="XRP 15-min above $1.50?", category="Crypto")),
        (
            "btc_5_minute_title",
            _poly_meta(title="Bitcoin 5 minute price higher?", category="Crypto"),
        ),
        (
            "eth_15_minute_title",
            _poly_meta(title="Ethereum 15 minute up or down", category="Crypto"),
        ),
        (
            "btc_15m_upper_context",
            _poly_meta(title="BTC 15M Up or Down", category="Crypto"),
        ),
        (
            "kalshi_kxbtc15m_ticker",
            _kalshi_meta(
                market_id="KXBTC15M-26NOV231415-T98500",
                event_id="KXBTC15M-26NOV231415",
                title="Bitcoin price above $98,500?",
                category="Crypto",
            ),
        ),
        (
            "kalshi_kxeth15m_ticker",
            _kalshi_meta(
                market_id="KXETH15M-26NOV231415-T3400",
                event_id="KXETH15M-26NOV231415",
                title="Ethereum price above $3,400?",
                category="Crypto",
            ),
        ),
        (
            "kalshi_kxbtc5m_series",
            _kalshi_meta(
                market_id="KXCRYPTO-1",
                title="Bitcoin short interval",
                extra_overrides={"series_ticker": "KXBTC5M"},
            ),
        ),
        (
            "kalshi_kxsol15m_ticker",
            _kalshi_meta(market_id="KXSOL15M-26NOV231415-T250", title="Solana price"),
        ),
        (
            "kalshi_kxxrp15m_ticker",
            _kalshi_meta(market_id="KXXRP15M-26NOV231415-T150", title="XRP price"),
        ),
        (
            "poly_btc_up_or_down_clock_range",
            _poly_meta(
                title="Bitcoin Up or Down - November 23, 2:00PM-2:15PM ET",
                outcomes=("Up", "Down"),
                category="Crypto",
            ),
        ),
        (
            "poly_eth_up_or_down_utc",
            _poly_meta(
                title="Ethereum Up or Down - 14:00-14:05 UTC",
                outcomes=("Up", "Down"),
                category="Crypto",
            ),
        ),
        (
            "poly_sol_intraday_up_or_down",
            _poly_meta(
                title="Solana Intraday Up or Down",
                outcomes=("Up", "Down"),
                category="Crypto",
            ),
        ),
        (
            "poly_xrp_bare_up_or_down",
            _poly_meta(
                title="XRP Up or Down",
                outcomes=("Up", "Down"),
                category="Crypto",
            ),
        ),
        (
            "poly_updown_15m_slug",
            _poly_meta(
                title="Bitcoin direction",
                extra_overrides={"slug": "btc-updown-15m-1763890500"},
            ),
        ),
    ],
)
def test_is_short_term_crypto_excludes_5m_15m_and_intraday_up_down(
    label: str,
    meta: MarketMetadata,
) -> None:
    """5m/15m crypto tickers, minute patterns, and intraday Up/Down markets are excluded."""
    assert is_short_term_crypto(meta) is True, f"expected short-term crypto match for {label}"


@pytest.mark.parametrize(
    ("label", "meta"),
    [
        (
            "btc_long_term_price_target",
            _poly_meta(
                title="Will Bitcoin exceed $150k by Dec 2026?",
                category="Crypto",
                extra_overrides={"slug": "will-bitcoin-exceed-150k-by-dec-2026"},
            ),
        ),
        (
            "eth_long_term_price_target",
            _poly_meta(title="Will Ethereum reach $6,000 before 2027?", category="Crypto"),
        ),
        (
            "sol_long_term_flippening",
            _poly_meta(title="Will Solana flip Ethereum market cap in 2026?", category="Crypto"),
        ),
        (
            "xrp_etf_approval",
            _poly_meta(title="Will an XRP spot ETF be approved in 2026?", category="Crypto"),
        ),
        (
            "kalshi_btc_weekly_series",
            _kalshi_meta(
                market_id="KXBTCW-26NOV27-T100000",
                event_id="KXBTCW-26NOV27",
                title="Bitcoin weekly close above $100,000?",
                category="Crypto",
                extra_overrides={"series_ticker": "KXBTCW"},
            ),
        ),
        (
            "kalshi_eth_weekly_series",
            _kalshi_meta(
                market_id="KXETHW-26NOV27-T3500",
                event_id="KXETHW-26NOV27",
                title="Ethereum weekly close above $3,500?",
                category="Crypto",
                extra_overrides={"series_ticker": "KXETHW"},
            ),
        ),
        (
            "dollar_15m_inflows_not_minutes",
            _poly_meta(
                title="Will Bitcoin ETF inflows exceed $15M on November 24?",
                category="Crypto",
            ),
        ),
        (
            "dollar_5m_tvl_not_minutes",
            _poly_meta(
                title="Will Ethereum L2 TVL grow by $5M this quarter?",
                category="Crypto",
            ),
        ),
        (
            "15m_active_addresses_not_minutes",
            _poly_meta(
                title="Will Solana exceed 15M daily active addresses in 2026?",
                category="Crypto",
            ),
        ),
        (
            "crypto_yearly_up_or_down",
            _poly_meta(
                title="Will Bitcoin finish 2026 Up or Down compared to 2025?",
                category="Crypto",
            ),
        ),
        (
            "crypto_weekly_up_or_down",
            _poly_meta(
                title="Will Ethereum be Up or Down this week?",
                category="Crypto",
            ),
        ),
        (
            "non_crypto_resolution_substring_sol",
            _poly_meta(
                title="Will the UN resolution pass within 15 minutes of debate?",
                category="Politics",
            ),
        ),
        (
            "non_crypto_whether_substring_eth",
            _poly_meta(
                title="Whether the Fed mentions inflation in 15m remarks?",
                category="Economics",
            ),
        ),
        (
            "non_crypto_netherlands_substring_eth",
            _poly_meta(
                title="Will Netherlands win the 5m diving event?",
                category="Sports",
            ),
        ),
        (
            "kalshi_btc_25m_not_5m",
            _kalshi_meta(
                market_id="KXBTC25M-26DEC",
                event_id="KXBTC25M-26DEC",
                title="Bitcoin monthly milestone",
                category="Crypto",
            ),
        ),
        (
            "kalshi_btc_2025m_not_5m",
            _kalshi_meta(
                market_id="KXBTC2025M-DEC",
                event_id="KXBTC2025M-DEC",
                title="Bitcoin 2025 monthly close",
                category="Crypto",
            ),
        ),
        (
            "kalshi_eth_45m_not_5m",
            _kalshi_meta(
                market_id="KXETH45M-26NOV",
                event_id="KXETH45M-26NOV",
                title="Ethereum 45M active addresses",
                category="Crypto",
            ),
        ),
        (
            "kalshi_btc_115m_not_15m",
            _kalshi_meta(
                market_id="KXBTC115M-26NOV",
                event_id="KXBTC115M-26NOV",
                title="Bitcoin 115M milestone",
                category="Crypto",
            ),
        ),
    ],
)
def test_is_short_term_crypto_avoids_false_positives(label: str, meta: MarketMetadata) -> None:
    """Medium/long-term crypto, $15M magnitude strings, and non-crypto markets are retained."""
    assert is_short_term_crypto(meta) is False, f"unexpected false positive for {label}"


# ---------------------------------------------------------------------------
# 4. Target series forced inclusion (Weather, Macro, Weekly, Polymarket tags)
# ---------------------------------------------------------------------------


def test_target_series_forced_inclusion_at_zero_and_none_volume() -> None:
    """Weather, macro, weekly series, and configured Polymarket tags enter even at 0/None volume."""
    policy = UniversePolicy(
        polymarket_target_tag_ids=("84", "100328"),
        target_series_patterns=(r"^KXCUSTOMWEEKLY-",),
    )

    candidates = [
        # Kalshi Weather prefixes (volume 0 and None)
        LiveMarketCandidate(
            _kalshi_meta("KXHIGHNY-26NOV24-T55", title="NYC high temp > 55F", category="Climate"),
            volume_24h=Decimal("0"),
            volume_24h_unit="contracts",
        ),
        LiveMarketCandidate(
            _kalshi_meta("KXLOWCHI-26NOV24-T30", title="Chicago low temp", category=None),
            volume_24h=None,
            volume_24h_unit="contracts",
        ),
        LiveMarketCandidate(
            _kalshi_meta("KXTEMP-26NOV24-LA", title="LA temperature"),
            volume_24h=Decimal("5"),
            volume_24h_unit="contracts",
        ),
        LiveMarketCandidate(
            _kalshi_meta("KXRAINNYC-26NOV24", title="NYC rain"),
            volume_24h=None,
            volume_24h_unit="contracts",
        ),
        # Kalshi Macro prefixes (all configured defaults sampled)
        LiveMarketCandidate(
            _kalshi_meta("KXCPI-26DEC-T30", title="US CPI YoY"),
            volume_24h=Decimal("0"),
            volume_24h_unit="contracts",
        ),
        LiveMarketCandidate(
            _kalshi_meta("KXFED-26DEC-T425", title="Fed Funds rate"),
            volume_24h=None,
            volume_24h_unit="contracts",
        ),
        LiveMarketCandidate(
            _kalshi_meta("KXFOMC-26DEC-CUT", title="FOMC cut"),
            volume_24h=Decimal("10"),
            volume_24h_unit="contracts",
        ),
        LiveMarketCandidate(
            _kalshi_meta("KXPAYROLLS-26DEC-T180", title="Nonfarm payrolls"),
            volume_24h=Decimal("0"),
            volume_24h_unit="contracts",
        ),
        LiveMarketCandidate(
            _kalshi_meta("KXGDP-26Q4-T25", title="US GDP"),
            volume_24h=None,
            volume_24h_unit="contracts",
        ),
        LiveMarketCandidate(
            _kalshi_meta("KXUNRATE-26DEC-T42", title="Unemployment rate"),
            volume_24h=Decimal("0"),
            volume_24h_unit="contracts",
        ),
        LiveMarketCandidate(
            _kalshi_meta("KXPCE-26DEC-T25", title="Core PCE"),
            volume_24h=None,
            volume_24h_unit="contracts",
        ),
        LiveMarketCandidate(
            _kalshi_meta("KXJOBLESS-26NOV28-T220", title="Initial jobless claims"),
            volume_24h=Decimal("0"),
            volume_24h_unit="contracts",
        ),
        # Kalshi Weekly target series (including weekly crypto KXBTCW)
        LiveMarketCandidate(
            _kalshi_meta("KXINXW-26NOV27-T6000", title="S&P 500 weekly close"),
            volume_24h=Decimal("0"),
            volume_24h_unit="contracts",
        ),
        LiveMarketCandidate(
            _kalshi_meta("KXBTCW-26NOV27-T100000", title="Bitcoin weekly close", category="Crypto"),
            volume_24h=None,
            volume_24h_unit="contracts",
        ),
        # Custom target pattern
        LiveMarketCandidate(
            _kalshi_meta("KXCUSTOMWEEKLY-26NOV27", title="Custom weekly"),
            volume_24h=Decimal("0"),
            volume_24h_unit="contracts",
        ),
        # Polymarket Weather & Economics categories (volume 0 / None)
        LiveMarketCandidate(
            _poly_meta("0xpoly-weather", title="NYC high > 55F", category="Weather"),
            volume_24h=Decimal("0"),
            volume_24h_unit="usd",
        ),
        LiveMarketCandidate(
            _poly_meta("0xpoly-econ", title="Fed decision in December", category="Economics"),
            volume_24h=None,
            volume_24h_unit="usd",
        ),
        # Polymarket target tag_id with category=None
        LiveMarketCandidate(
            _poly_meta("0xpoly-tag-weather", title="Rain in NYC tomorrow?", category=None),
            volume_24h=Decimal("0"),
            volume_24h_unit="usd",
            tag_ids=("84",),
        ),
        LiveMarketCandidate(
            _poly_meta(
                "0xpoly-extra-tag-econ",
                title="CPI above 2.8%?",
                category=None,
                extra_overrides={"tag_ids": ("100328",)},
            ),
            volume_24h=None,
            volume_24h_unit="usd",
        ),
        # Non-target low-volume controls (must be excluded)
        LiveMarketCandidate(
            _poly_meta("0xpoly-low-vol", title="Will team X win?", category="Sports"),
            volume_24h=Decimal("100"),
            volume_24h_unit="usd",
            tag_ids=("999",),
        ),
        LiveMarketCandidate(
            _kalshi_meta("KXSPORTS-1", title="Will team Y win?", category="Sports"),
            volume_24h=Decimal("50"),
            volume_24h_unit="contracts",
        ),
        # Closed target market must still be excluded by is_binary_non_combo
        LiveMarketCandidate(
            _kalshi_meta("KXHIGHNY-CLOSED", title="Closed weather", status="closed"),
            volume_24h=Decimal("50000"),
            volume_24h_unit="contracts",
        ),
    ]

    selected = select_universe(candidates, policy, now_utc=NOW_UTC)
    selected_ids = {item.candidate.metadata.market_id for item in selected}

    assert all(item.selection_reason == "target_series" for item in selected)
    assert len(selected) == 19
    assert "0xpoly-low-vol" not in selected_ids
    assert "KXSPORTS-1" not in selected_ids
    assert "KXHIGHNY-CLOSED" not in selected_ids
    assert {
        "KXHIGHNY-26NOV24-T55",
        "KXLOWCHI-26NOV24-T30",
        "KXTEMP-26NOV24-LA",
        "KXRAINNYC-26NOV24",
        "KXCPI-26DEC-T30",
        "KXFED-26DEC-T425",
        "KXFOMC-26DEC-CUT",
        "KXPAYROLLS-26DEC-T180",
        "KXGDP-26Q4-T25",
        "KXUNRATE-26DEC-T42",
        "KXPCE-26DEC-T25",
        "KXJOBLESS-26NOV28-T220",
        "KXINXW-26NOV27-T6000",
        "KXBTCW-26NOV27-T100000",
        "KXCUSTOMWEEKLY-26NOV27",
        "0xpoly-weather",
        "0xpoly-econ",
        "0xpoly-tag-weather",
        "0xpoly-extra-tag-econ",
    } == selected_ids

    # is_target_series also works directly on MarketMetadata.
    assert is_target_series(_kalshi_meta("KXCPI-26DEC"), policy) is True
    assert is_target_series(_poly_meta("0xother", category="Sports"), policy) is False


# ---------------------------------------------------------------------------
# 5. Exact volume threshold cut-offs, 45d horizon filter, separate caps,
#    deduplication, and deterministic ordering
# ---------------------------------------------------------------------------


def test_exact_volume_threshold_cutoffs_per_venue() -> None:
    """Exact threshold boundary is inclusive (>=): 5000 USD (Poly), 1000 contracts (Kalshi)."""
    policy = UniversePolicy(
        min_volume_24h_polymarket_usd=Decimal("5000"),
        min_volume_24h_kalshi_contracts=Decimal("1000"),
    )
    candidates = [
        LiveMarketCandidate(_poly_meta("0xp-none"), volume_24h=None, volume_24h_unit="usd"),
        LiveMarketCandidate(
            _poly_meta("0xp-below"), volume_24h=Decimal("4999.99"), volume_24h_unit="usd"
        ),
        LiveMarketCandidate(
            _poly_meta("0xp-exact"), volume_24h=Decimal("5000"), volume_24h_unit="usd"
        ),
        LiveMarketCandidate(
            _poly_meta("0xp-above"), volume_24h=Decimal("5000.01"), volume_24h_unit="usd"
        ),
        LiveMarketCandidate(_kalshi_meta("KXK-NONE"), volume_24h=None, volume_24h_unit="contracts"),
        LiveMarketCandidate(
            _kalshi_meta("KXK-BELOW"), volume_24h=Decimal("999.99"), volume_24h_unit="contracts"
        ),
        LiveMarketCandidate(
            _kalshi_meta("KXK-EXACT"), volume_24h=Decimal("1000"), volume_24h_unit="contracts"
        ),
        LiveMarketCandidate(
            _kalshi_meta("KXK-ABOVE"), volume_24h=Decimal("1000.01"), volume_24h_unit="contracts"
        ),
    ]

    selected = select_universe(candidates, policy, now_utc=NOW_UTC)
    assert [(s.candidate.metadata.market_id, s.selection_reason) for s in selected] == [
        ("0xp-above", "volume_core"),
        ("0xp-exact", "volume_core"),
        ("KXK-ABOVE", "volume_core"),
        ("KXK-EXACT", "volume_core"),
    ]


def test_target_series_45_day_horizon_filter_and_core_fallthrough() -> None:
    """Target series > 45d are excluded from target_series; high-volume ones can enter core."""
    policy = UniversePolicy(target_series_max_horizon_days=45)

    candidates = [
        # Exactly 45d -> included in target_series even at volume 0.
        LiveMarketCandidate(
            _kalshi_meta("KXCPI-EXACT-45D", end_time_utc=NOW_UTC + timedelta(days=45)),
            volume_24h=Decimal("0"),
            volume_24h_unit="contracts",
        ),
        # 45d + 1s with low volume -> excluded from target_series and fails volume_core.
        LiveMarketCandidate(
            _kalshi_meta(
                "KXCPI-OVER-45D-LOW", end_time_utc=NOW_UTC + timedelta(days=45, seconds=1)
            ),
            volume_24h=Decimal("200"),
            volume_24h_unit="contracts",
        ),
        # 90d with high volume -> excluded from target_series, falls through into volume_core.
        LiveMarketCandidate(
            _kalshi_meta("KXFED-90D-HIGH", end_time_utc=NOW_UTC + timedelta(days=90)),
            volume_24h=Decimal("15000"),
            volume_24h_unit="contracts",
        ),
        # Missing end_time_utc -> still eligible for target_series.
        LiveMarketCandidate(
            _kalshi_meta("KXHIGHNY-NO-END", end_time_utc=None),
            volume_24h=Decimal("0"),
            volume_24h_unit="contracts",
        ),
        # Non-target core market > 45d is NOT restricted by target_series_max_horizon_days.
        LiveMarketCandidate(
            _poly_meta(
                "0xpoly-long-core",
                title="Will Bitcoin exceed $150k by Dec 2026?",
                category="Crypto",
                end_time_utc=NOW_UTC + timedelta(days=180),
            ),
            volume_24h=Decimal("25000"),
            volume_24h_unit="usd",
        ),
    ]

    # Verify non-UTC timezone-aware now_utc produces identical cutoff behavior.
    now_plus_2 = NOW_UTC.astimezone(timezone(timedelta(hours=2)))
    selected = select_universe(candidates, policy, now_utc=now_plus_2)

    by_id = {s.candidate.metadata.market_id: s.selection_reason for s in selected}
    assert by_id == {
        "KXCPI-EXACT-45D": "target_series",
        "KXHIGHNY-NO-END": "target_series",
        "KXFED-90D-HIGH": "volume_core",
        "0xpoly-long-core": "volume_core",
    }
    assert "KXCPI-OVER-45D-LOW" not in by_id

    # Naive now_utc must raise ValueError.
    with pytest.raises(ValueError, match="timezone-aware"):
        select_universe(candidates, policy, now_utc=datetime(2026, 11, 23, 12, 0, 0))


def test_separate_per_venue_caps_prevent_target_series_saturating_core() -> None:
    """max_target_series_per_venue and max_core_per_venue are independent per-venue budgets."""
    policy = UniversePolicy(
        max_target_series_per_venue=2,
        max_core_per_venue=3,
    )

    candidates: list[LiveMarketCandidate] = []
    # 5 low-volume Kalshi weather targets -> must cap at 2 and NEVER consume core budget.
    for i in range(5):
        candidates.append(
            LiveMarketCandidate(
                _kalshi_meta(f"KXHIGHNY-26NOV24-T{50 + i}", title=f"Temp {i}"),
                volume_24h=Decimal(str(i * 10)),
                volume_24h_unit="contracts",
            )
        )
    # 5 high-volume Kalshi core markets -> must cap at 3.
    for i in range(5):
        candidates.append(
            LiveMarketCandidate(
                _kalshi_meta(f"KXCORE-{i}", title=f"Kalshi core {i}"),
                volume_24h=Decimal(str(2000 + i * 500)),
                volume_24h_unit="contracts",
            )
        )
    # 4 low-volume Polymarket Weather targets -> must cap at 2 independently of Kalshi.
    for i in range(4):
        candidates.append(
            LiveMarketCandidate(
                _poly_meta(f"0xpm-weather-{i}", title=f"PM weather {i}", category="Weather"),
                volume_24h=Decimal(str(i * 100)),
                volume_24h_unit="usd",
            )
        )
    # 4 high-volume Polymarket core markets -> must cap at 3 independently of Kalshi.
    for i in range(4):
        candidates.append(
            LiveMarketCandidate(
                _poly_meta(f"0xpm-core-{i}", title=f"PM core {i}"),
                volume_24h=Decimal(str(10000 + i * 1000)),
                volume_24h_unit="usd",
            )
        )

    selected = select_universe(candidates, policy, now_utc=NOW_UTC)

    poly_targets = [
        s.candidate.metadata.market_id
        for s in selected
        if s.candidate.metadata.venue == "polymarket" and s.selection_reason == "target_series"
    ]
    poly_core = [
        s.candidate.metadata.market_id
        for s in selected
        if s.candidate.metadata.venue == "polymarket" and s.selection_reason == "volume_core"
    ]
    kalshi_targets = [
        s.candidate.metadata.market_id
        for s in selected
        if s.candidate.metadata.venue == "kalshi" and s.selection_reason == "target_series"
    ]
    kalshi_core = [
        s.candidate.metadata.market_id
        for s in selected
        if s.candidate.metadata.venue == "kalshi" and s.selection_reason == "volume_core"
    ]

    assert poly_targets == ["0xpm-weather-3", "0xpm-weather-2"]
    assert poly_core == ["0xpm-core-3", "0xpm-core-2", "0xpm-core-1"]
    assert kalshi_targets == ["KXHIGHNY-26NOV24-T54", "KXHIGHNY-26NOV24-T53"]
    assert kalshi_core == ["KXCORE-4", "KXCORE-3", "KXCORE-2"]
    assert len(selected) == 10


def test_deduplication_tag_merge_and_deterministic_tie_break_ordering() -> None:
    """Dedup merges tag_ids across general+tag passes and output is permutation-invariant."""
    policy = UniversePolicy(polymarket_target_tag_ids=("84",))

    # Same market fetched in general pass (no tags) and tag pass (tag 84), plus tied volumes.
    c_general = LiveMarketCandidate(
        _poly_meta("0xpm-shared-tag", title="Rain tomorrow?", category=None),
        volume_24h=None,
        volume_24h_unit="usd",
    )
    c_tagged = LiveMarketCandidate(
        _poly_meta("0xpm-shared-tag", title="Rain tomorrow?", category=None),
        volume_24h=Decimal("25"),
        open_interest=Decimal("100"),
        volume_24h_unit="usd",
        tag_ids=("84",),
    )
    c_tie_b = LiveMarketCandidate(
        _poly_meta("0xpm-tie-b"), volume_24h=Decimal("8000"), volume_24h_unit="usd"
    )
    c_tie_a = LiveMarketCandidate(
        _poly_meta("0xpm-tie-a"), volume_24h=Decimal("8000"), volume_24h_unit="usd"
    )
    c_high = LiveMarketCandidate(
        _poly_meta("0xpm-high"), volume_24h=Decimal("20000"), volume_24h_unit="usd"
    )
    # Same market_id across different venues stays distinct (keyed by (venue, market_id)).
    c_same_id_poly = LiveMarketCandidate(
        _poly_meta("SHARED-ID"), volume_24h=Decimal("6000"), volume_24h_unit="usd"
    )
    c_same_id_kalshi = LiveMarketCandidate(
        _kalshi_meta("SHARED-ID"), volume_24h=Decimal("2000"), volume_24h_unit="contracts"
    )
    # Target tie-break: 0 before None, and alphabetical market_id on ties.
    c_t_none_b = LiveMarketCandidate(
        _kalshi_meta("KXHIGH-NONE-B"), volume_24h=None, volume_24h_unit="contracts"
    )
    c_t_none_a = LiveMarketCandidate(
        _kalshi_meta("KXHIGH-NONE-A"), volume_24h=None, volume_24h_unit="contracts"
    )
    c_t_zero = LiveMarketCandidate(
        _kalshi_meta("KXHIGH-ZERO"), volume_24h=Decimal("0"), volume_24h_unit="contracts"
    )

    order_1 = [
        c_general,
        c_tie_b,
        c_t_none_b,
        c_same_id_kalshi,
        c_high,
        c_tagged,
        c_t_zero,
        c_tie_a,
        c_t_none_a,
        c_same_id_poly,
    ]
    order_2 = list(reversed(order_1))

    res_1 = select_universe(order_1, policy, now_utc=NOW_UTC)
    res_2 = select_universe(order_2, policy, now_utc=NOW_UTC)

    expected = [
        ("polymarket", "0xpm-shared-tag", "target_series"),
        ("kalshi", "KXHIGH-ZERO", "target_series"),
        ("kalshi", "KXHIGH-NONE-A", "target_series"),
        ("kalshi", "KXHIGH-NONE-B", "target_series"),
        ("polymarket", "0xpm-high", "volume_core"),
        ("polymarket", "0xpm-tie-a", "volume_core"),
        ("polymarket", "0xpm-tie-b", "volume_core"),
        ("polymarket", "SHARED-ID", "volume_core"),
        ("kalshi", "SHARED-ID", "volume_core"),
    ]
    assert [
        (s.candidate.metadata.venue, s.candidate.metadata.market_id, s.selection_reason)
        for s in res_1
    ] == expected
    assert [
        (s.candidate.metadata.venue, s.candidate.metadata.market_id, s.selection_reason)
        for s in res_2
    ] == expected
    # Merged candidate preserved tag_ids, volume_24h, and open_interest.
    assert res_1[0].candidate.tag_ids == ("84",)
    assert res_1[0].candidate.volume_24h == Decimal("25")
    assert res_1[0].candidate.open_interest == Decimal("100")


# ---------------------------------------------------------------------------
# 6. Live screening parsers & client methods (offline MockTransport)
# ---------------------------------------------------------------------------


def test_parse_gamma_live_candidate_extracts_volume_without_leaking_into_metadata() -> None:
    """parse_gamma_live_candidate extracts volume24hr (usd) while MarketMetadata stays leak-free."""
    raw: dict[str, Any] = {
        "id": "99001",
        "conditionId": "0xgamma-live-1",
        "question": "Will US CPI exceed 3.0% in December?",
        "slug": "us-cpi-dec-2026",
        "category": "Economics",
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": '["1001", "1002"]',
        "endDate": "2026-12-15T13:30:00Z",
        "active": True,
        "closed": False,
        "enableOrderBook": True,
        "acceptingOrders": True,
        "volume": "950000.00",
        "volume24hr": 18450.25,
        "liquidity": "125000.00",
        "outcomePrices": '["0.42", "0.58"]',
        "bestBid": 0.41,
        "bestAsk": 0.43,
        "lastTradePrice": 0.42,
        "tags": [{"id": "100328", "label": "Economics"}],
        "events": [{"id": "evt-cpi", "tags": [{"id": 55, "label": "Macro"}]}],
    }

    candidate = parse_gamma_live_candidate(raw, tag_ids=("84",))

    assert candidate.metadata == parse_gamma_market(raw)
    assert candidate.volume_24h == Decimal("18450.25")
    assert candidate.volume_24h_unit == "usd"
    # Liquidity must NEVER be coerced into open_interest.
    assert candidate.open_interest is None
    assert candidate.tag_ids == ("100328", "55", "84")

    # Critical anti-leakage invariant on MarketMetadata.extra.
    forbidden_keys = {
        "volume",
        "volumeNum",
        "volumeClob",
        "volume24hr",
        "liquidity",
        "outcomePrices",
        "bestBid",
        "bestAsk",
        "lastTradePrice",
    }
    assert forbidden_keys.isdisjoint(candidate.metadata.extra.keys())

    # Fallback to volume24hrClob and sanitization of negative/missing values.
    clob_vol_cand = parse_gamma_live_candidate(
        {
            "conditionId": "0xgamma-live-2",
            "volume24hr": "-10",
            "volume24hrClob": "6200.50",
            "openInterest": "12000.0",
        }
    )
    assert clob_vol_cand.volume_24h == Decimal("6200.50")
    assert clob_vol_cand.open_interest == Decimal("12000.0")

    zero_cand = parse_gamma_live_candidate({"conditionId": "0xgamma-zero", "volume24hr": 0})
    assert zero_cand.volume_24h == Decimal("0")


def test_parse_kalshi_live_candidate_extracts_fp_and_legacy_contracts() -> None:
    """parse_kalshi_live_candidate extracts volume_24h(_fp) (contracts) & open_interest_fp."""
    raw_fp: dict[str, Any] = {
        "ticker": "KXFED-26DEC-T425",
        "event_ticker": "KXFED-26DEC",
        "series_ticker": "KXFED",
        "title": "Fed rate above 4.25%?",
        "status": "active",
        "market_type": "binary",
        "close_time": "2026-12-18T19:00:00Z",
        "volume_24h": 1500,
        "volume_24h_fp": "1500.75",
        "open_interest": 4200,
        "open_interest_fp": "4200.50",
    }
    cand_fp = parse_kalshi_live_candidate(raw_fp)
    assert cand_fp.metadata == parse_kalshi_market(raw_fp)
    assert cand_fp.volume_24h == Decimal("1500.75")
    assert cand_fp.open_interest == Decimal("4200.50")
    assert cand_fp.volume_24h_unit == "contracts"
    assert "volume_24h" not in cand_fp.metadata.extra
    assert "volume_24h_fp" not in cand_fp.metadata.extra
    assert "open_interest_fp" not in cand_fp.metadata.extra

    # Fallback to legacy integer fields when *_fp are absent or invalid.
    raw_legacy: dict[str, Any] = {
        "ticker": "KXCPI-26DEC-T30",
        "status": "open",
        "volume_24h_fp": "",
        "volume_24h": 1250,
        "open_interest_fp": "-5",
        "open_interest": 3100,
    }
    cand_legacy = parse_kalshi_live_candidate(raw_legacy)
    assert cand_legacy.volume_24h == Decimal("1250")
    assert cand_legacy.open_interest == Decimal("3100")
    assert cand_legacy.volume_24h_unit == "contracts"

    # Missing/negative fields become None without raising.
    cand_empty = parse_kalshi_live_candidate(
        {"ticker": "KXEMPTY-1", "status": "open", "volume_24h": -10, "open_interest_fp": "NaN"}
    )
    assert cand_empty.volume_24h is None
    assert cand_empty.open_interest is None


async def test_clients_get_live_market_candidates_and_end_to_end_screening() -> None:
    """Client screening helpers send expected query params and feed cleanly into select_universe."""
    pm_requests: list[dict[str, str]] = []
    kalshi_requests: list[dict[str, str]] = []

    def pm_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/markets"
        params = dict(request.url.params)
        pm_requests.append(params)
        if params.get("tag_id") == "84":
            return httpx.Response(
                200,
                json=[
                    {
                        "conditionId": "0xpm-weather-tag",
                        "question": "NYC high temp > 55F?",
                        "outcomes": '["Yes", "No"]',
                        "clobTokenIds": '["w1", "w2"]',
                        "endDate": "2026-11-26T23:59:00Z",
                        "active": True,
                        "closed": False,
                        "enableOrderBook": True,
                        "acceptingOrders": True,
                        "volume24hr": "15.50",
                    }
                ],
                request=request,
            )
        return httpx.Response(
            200,
            json=[
                {
                    "conditionId": "0xpm-core-btc-long",
                    "question": "Will Bitcoin exceed $150k by Dec 2026?",
                    "category": "Crypto",
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["b1", "b2"]',
                    "endDate": "2026-12-31T23:59:00Z",
                    "active": True,
                    "closed": False,
                    "enableOrderBook": True,
                    "acceptingOrders": True,
                    "volume24hr": 45000,
                },
                {
                    "conditionId": "0xpm-btc-15m",
                    "question": "Bitcoin Up or Down - November 23, 2:00PM-2:15PM ET",
                    "category": "Crypto",
                    "outcomes": '["Up", "Down"]',
                    "clobTokenIds": '["u1", "u2"]',
                    "endDate": "2026-11-23T19:15:00Z",
                    "active": True,
                    "closed": False,
                    "enableOrderBook": True,
                    "acceptingOrders": True,
                    "volume24hr": 999999,
                },
            ],
            request=request,
        )

    def kalshi_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/trade-api/v2/markets"
        params = dict(request.url.params)
        kalshi_requests.append(params)
        return httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "ticker": "KXHIGHNY-26NOV24-T55",
                        "event_ticker": "KXHIGHNY-26NOV24",
                        "series_ticker": "KXHIGHNY",
                        "title": "NYC high temp > 55F?",
                        "status": "active",
                        "market_type": "binary",
                        "close_time": "2026-11-24T23:59:00Z",
                        "volume_24h_fp": "0.00",
                        "open_interest_fp": "120.00",
                    },
                    {
                        "ticker": "KXPOL-26DEC-BILL",
                        "event_ticker": "KXPOL-26DEC",
                        "title": "Will bill pass?",
                        "status": "active",
                        "market_type": "binary",
                        "close_time": "2026-12-20T23:59:00Z",
                        "volume_24h_fp": "2500.00",
                        "open_interest_fp": "8000.00",
                    },
                    {
                        "ticker": "KXBTC15M-26NOV231415-T98500",
                        "event_ticker": "KXBTC15M-26NOV231415",
                        "title": "BTC 15m above $98,500?",
                        "status": "active",
                        "market_type": "binary",
                        "close_time": "2026-11-23T19:15:00Z",
                        "volume_24h_fp": "50000.00",
                    },
                ],
                "cursor": "",
            },
            request=request,
        )

    async with (
        PolymarketClient(transport=httpx.MockTransport(pm_handler)) as pm_client,
        KalshiClient(transport=httpx.MockTransport(kalshi_handler)) as kalshi_client,
    ):
        pm_general = await pm_client.get_live_market_candidates()
        pm_weather = await pm_client.get_live_market_candidates(tag_id="84")
        kalshi_live = await kalshi_client.get_live_market_candidates()

    # Verify default Gamma params (active=true, closed=false) and optional tag_id.
    assert pm_requests[0]["active"] == "true"
    assert pm_requests[0]["closed"] == "false"
    assert "tag_id" not in pm_requests[0]
    assert pm_requests[1]["tag_id"] == "84"
    assert pm_weather[0].tag_ids == ("84",)
    assert pm_weather[0].volume_24h == Decimal("15.50")
    assert pm_weather[0].volume_24h_unit == "usd"

    # Verify default Kalshi params (status=open, mve_filter=exclude).
    assert kalshi_requests[0]["status"] == "open"
    assert kalshi_requests[0]["mve_filter"] == "exclude"
    assert kalshi_live[1].volume_24h == Decimal("2500.00")
    assert kalshi_live[1].open_interest == Decimal("8000.00")
    assert kalshi_live[1].volume_24h_unit == "contracts"

    # End-to-end deterministic universe selection.
    policy = UniversePolicy(polymarket_target_tag_ids=("84",))
    selected = select_universe([*pm_general, *pm_weather, *kalshi_live], policy, now_utc=NOW_UTC)

    assert [
        (s.candidate.metadata.venue, s.candidate.metadata.market_id, s.selection_reason)
        for s in selected
    ] == [
        ("polymarket", "0xpm-weather-tag", "target_series"),
        ("kalshi", "KXHIGHNY-26NOV24-T55", "target_series"),
        ("polymarket", "0xpm-core-btc-long", "volume_core"),
        ("kalshi", "KXPOL-26DEC-BILL", "volume_core"),
    ]
