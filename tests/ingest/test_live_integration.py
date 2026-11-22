"""Opt-in live integration tests against public Kalshi and Polymarket APIs.

These tests are marked with ``@pytest.mark.live`` and deselected by default
(``addopts = "-m 'not live'"`` in ``pyproject.toml``). Run explicitly with::

    uv run pytest -m live
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from pm_quant.ingest import (
    SCHEMA_VERSION,
    KalshiClient,
    MarketMetadata,
    NormalizedOrderBook,
    PolymarketClient,
    TradeRecord,
)

_ZERO = Decimal("0")
_ONE = Decimal("1")


def _assert_valid_orderbook(book: NormalizedOrderBook, *, max_depth: int) -> None:
    """Assert structural, ordering, UTC, and schema invariants on a parsed orderbook."""
    assert isinstance(book, NormalizedOrderBook)
    assert book.schema_version == SCHEMA_VERSION
    assert book.ts_utc.tzinfo is not None
    assert book.ts_utc.utcoffset() == timedelta(0)
    assert len(book.bids) <= max_depth
    assert len(book.asks) <= max_depth

    for level in (*book.bids, *book.asks):
        assert _ZERO < level.price < _ONE
        assert level.size > _ZERO

    # Bids must be sorted descending (best bid first).
    for idx in range(len(book.bids) - 1):
        assert book.bids[idx].price >= book.bids[idx + 1].price

    # Asks must be sorted ascending (best ask first).
    for idx in range(len(book.asks) - 1):
        assert book.asks[idx].price <= book.asks[idx + 1].price


@pytest.mark.live
async def test_live_kalshi_client_parses_real_orderbook() -> None:
    """Fetch 2 real open Kalshi markets, historical cutoff, and batch orderbooks."""
    max_depth = 20

    async with KalshiClient() as client:
        cutoff = await client.get_historical_cutoff()
        assert isinstance(cutoff, datetime)
        assert cutoff.tzinfo is not None
        assert cutoff.utcoffset() == timedelta(0)
        assert client.cached_historical_cutoff == cutoff

        markets = await client.get_markets(status="open", limit=10, max_pages=1)
        assert len(markets) >= 2

        selected = markets[:2]
        for market in selected:
            assert isinstance(market, MarketMetadata)
            assert market.venue == "kalshi"
            assert market.market_id
            assert market.schema_version == SCHEMA_VERSION

        tickers = [market.market_id for market in selected]
        books = await client.get_orderbooks_batch(tickers, depth=max_depth)
        assert isinstance(books, dict)
        # Kalshi /markets/orderbooks returns one entry per requested ticker.
        assert set(books) == set(tickers)

        for ticker, (yes_book, no_book) in books.items():
            assert yes_book.venue == "kalshi"
            assert yes_book.market_id == ticker
            assert yes_book.outcome_side == "yes"
            _assert_valid_orderbook(yes_book, max_depth=max_depth)

            assert no_book.venue == "kalshi"
            assert no_book.market_id == ticker
            assert no_book.outcome_side == "no"
            _assert_valid_orderbook(no_book, max_depth=max_depth)

            # Exact Decimal reciprocal symmetry: YES bids <-> NO asks, NO bids <-> YES asks.
            assert len(yes_book.bids) == len(no_book.asks)
            for y_bid, n_ask in zip(yes_book.bids, no_book.asks, strict=True):
                assert y_bid.price + n_ask.price == _ONE
                assert y_bid.size == n_ask.size

            assert len(no_book.bids) == len(yes_book.asks)
            for n_bid, y_ask in zip(no_book.bids, yes_book.asks, strict=True):
                assert n_bid.price + y_ask.price == _ONE
                assert n_bid.size == y_ask.size


@pytest.mark.live
async def test_live_polymarket_client_parses_real_market_and_trades() -> None:
    """End-to-end live check: Gamma active market -> CLOB book -> Data API v2 trades."""
    max_depth = 20

    async with PolymarketClient() as client:
        markets = await client.get_markets(active=True, closed=False, limit=10, max_pages=1)
        assert markets

        selected: MarketMetadata | None = None
        selected_token_id: str | None = None
        for prefer_yes_no in (True, False):
            for market in markets:
                if prefer_yes_no and not market.is_yes_no:
                    continue
                raw_token_ids = market.extra.get("clob_token_ids")
                if (
                    isinstance(raw_token_ids, tuple)
                    and raw_token_ids
                    and market.extra.get("enable_order_book", True) is not False
                ):
                    token_idx = 0
                    if market.is_yes_no and len(market.outcomes) == len(raw_token_ids):
                        for idx, label in enumerate(market.outcomes):
                            if label.strip().lower() == "yes":
                                token_idx = idx
                                break
                    selected = market
                    selected_token_id = str(raw_token_ids[token_idx])
                    break
            if selected is not None:
                break

        assert selected is not None
        assert selected_token_id is not None
        assert selected.venue == "polymarket"
        assert selected.market_id
        assert selected.schema_version == SCHEMA_VERSION

        book = await client.get_orderbook(
            selected_token_id,
            max_depth=max_depth,
            market_id=selected.market_id,
        )
        assert book.venue == "polymarket"
        assert book.market_id == selected.market_id
        assert book.token_id == selected_token_id
        _assert_valid_orderbook(book, max_depth=max_depth)

        # PolymarketClient.get_trades enforces condition= and taker_only=true.
        trades = await client.get_trades([selected.market_id], max_pages=1, limit=20)
        assert isinstance(trades, list)
        for trade in trades:
            assert isinstance(trade, TradeRecord)
            assert trade.venue == "polymarket"
            assert trade.market_id == selected.market_id
            assert trade.ts_utc.tzinfo is not None
            assert trade.ts_utc.utcoffset() == timedelta(0)
            assert _ZERO <= trade.price <= _ONE
            assert trade.size >= _ZERO
            assert trade.notional_usd >= _ZERO
            assert trade.schema_version == SCHEMA_VERSION
