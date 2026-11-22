# Prediction Market APIs & Taker Fee Schedule (November 2026)

This document records the live API verification results for **Polymarket** and **Kalshi** executed on GitHub Actions, highlights key schema differences across endpoints, and specifies the official **November 2026 taker fee schedules** used as counterfactual transaction costs in `pm-quant` historical backtests.

---

## 1. GitHub Actions API Smoke Test Summary

- **Run timestamp (UTC)**: `2026-11-22T17:59:40Z`
- **Script**: `scripts/smoke_test_apis.py` (workflow: `.github/workflows/smoke_test_apis.yml`)
- **Overall result**: **PASS (9/9 endpoints returned HTTP 200 with valid schema)**
- **WAF / Rate-limit outcome**: **Zero HTTP 403 or 429 responses** across all 9 checks from the GitHub Actions runner IP using a standard `User-Agent: pm-quant-smoke-test/0.1` header. Neither Cloudflare nor venue edge firewalls blocked read-only public API traffic. Bulk ingestion must still enforce bounded concurrency and adaptive retry/backoff (`pm_quant.ingest.http.AsyncRateLimitedClient`).

### Verified Hosts & Endpoint Results

| Endpoint | Verified Host / Path | HTTP | Latency (ms) | Schema | Observed Notes |
| --- | --- | ---: | ---: | :---: | --- |
| Polymarket Gamma `/markets` | `https://gamma-api.polymarket.com/markets` | 200 | 204.1 | PASS | `markets=5`, valid `conditionId` (`0xa467b14d5...`) and `clobTokenIds` (`32338220190...`) |
| Polymarket CLOB `/book` | `https://clob.polymarket.com/book` | 200 | 165.6 | PASS | `bids=15`, `asks=147` |
| Polymarket CLOB `/prices-history` | `https://clob.polymarket.com/prices-history` | 200 | 172.7 | PASS | `key=history`, `points=1418`, `fields=['p', 't']` |
| Polymarket Data v2 `/trades` | `https://data-api.polymarket.com/v2/trades` | 200 | 204.6 | PASS | `trades=5`, `next_cursor=eyJkYXRhI...`, `snake_case=ok` (`condition=`, `taker_only=true`) |
| Polymarket Data v2 `/prices-history` | `https://data-api.polymarket.com/v2/prices-history` | 200 | 309.3 | PASS | `key=data`, `points=1437` (`+19` vs CLOB), `fields=['price', 'resolution_seconds', 'timestamp']` |
| Kalshi `/historical/cutoff` (elections) | `https://api.elections.kalshi.com/trade-api/v2/historical/cutoff` | 200 | 19.8 | PASS | `market_settled_ts=2026-09-22T00:00:00Z` |
| Kalshi `/historical/cutoff` (external) | `https://external-api.kalshi.com/trade-api/v2/historical/cutoff` | 200 | 252.1 | PASS | `market_settled_ts=2026-09-22T00:00:00Z` (matches primary) |
| Kalshi `/markets` (open) | `https://api.elections.kalshi.com/trade-api/v2/markets` | 200 | 64.3 | PASS | `markets=5`, returned open tickers (`KXMVECROSSCATEGORY-...`) |
| Kalshi `/markets/orderbooks` | `https://api.elections.kalshi.com/trade-api/v2/markets/orderbooks` | 200 | 59.8 | PASS | `books=1`, `yes_levels=0`, `no_levels=0`, `orderbook_fp` ascending check passed |

### Polymarket CLOB vs. Data API v2 `/prices-history` Comparison

Both endpoints were queried for the same active token (`32338220190...`) with `interval=1d`:

| Attribute | CLOB `/prices-history` | Data API v2 `/prices-history` |
| --- | --- | --- |
| Base URL | `https://clob.polymarket.com/prices-history` | `https://data-api.polymarket.com/v2/prices-history` |
| Token query parameter | `market=<token_id>` | `token_id=<token_id>` |
| Top-level array key | `history` | `data` |
| Point schema fields | `['p', 't']` | `['price', 'resolution_seconds', 'timestamp']` |
| Returned points (`interval=1d`) | `1418` | `1437` (`+19` vs. CLOB) |
| Observed latency | `172.7 ms` | `309.3 ms` |

**Ingestion implications**:
- Data API v2 `/prices-history` provides richer metadata (`resolution_seconds`, explicit `price`/`timestamp` names) and slightly denser coverage (`+19` points on the sampled 1-day window) at higher latency (`309.3 ms` vs. `172.7 ms`).
- Data API v2 `/trades` must always be queried with `condition=<condition_id>` and `taker_only=true`, paginating via `pagination.next_cursor` to avoid maker/taker double-counting.

### Kalshi Endpoint Notes

- Both `api.elections.kalshi.com` (`19.8 ms`, primary) and `external-api.kalshi.com` (`252.1 ms`, external mirror) report the exact same `market_settled_ts=2026-09-22T00:00:00Z` cutoff boundary separating active and historical archives. `api.elections.kalshi.com` is the preferred primary host due to significantly lower latency.
- Batch `/markets/orderbooks` returns `orderbook_fp` objects containing bid-only fixed-point dollar arrays (`yes_dollars`, `no_dollars`) sorted in **ascending** price order (best bid is the last element; complementary asks are implied at `1 - opposite_bid`).
- On sparse open contracts (such as newly listed cross-category combo tickers), batch orderbook responses may omit empty books or return `0` levels on one or both sides (`books=1`, `yes_levels=0`, `no_levels=0`). Normalized book schemas therefore treat empty and one-sided orderbooks as valid first-class states.

---

## 2. Official Taker Fee Schedule (November 2026 — Backtest Counterfactual)

Historical trade and orderbook archives prior to late November 2026 reflect legacy fee regimes (including prolonged zero-fee windows on Polymarket). To evaluate forward-looking net alpha without optimistic bias, `pm-quant` applies the **November 2026 live taker fee schedule** as a unified counterfactual cost model across historical backtests.

### Summary Comparison

| Venue | Effective Regime | Taker Fee Formula | Maker Fee | Max Fee per 100 Contracts / Shares (at $p = 0.50$) |
| --- | --- | --- | --- | --- |
| **Polymarket International** | Effective `2026-11-20 23:59 UTC` | $\text{fee} = \text{shares} \times c \times p \times (1 - p)$ | `0` | $\$0.00$ ($c=0.00$), $\$0.25$ ($c=0.01$), $\$1.25$ ($c=0.05$) |
| **Kalshi** | Active Nov 2026 | $\text{fee} = \lceil 100 \times 0.07 \times \text{contracts} \times p \times (1 - p) \rceil / 100$ | `0` (general markets) | $\$1.75$ |

*(In both formulas, $p \in [0, 1]$ denotes the contract execution price in USD, so $p(1 - p)$ peaks at $0.25$ when $p = 0.50$.)*

### Polymarket International Category Coefficients

Effective **November 20, 2026 at 23:59 UTC**, Polymarket International charges zero maker fees (`maker = 0`) and parabolic taker fees:

$$\text{fee}_{\text{poly}} = \text{shares} \cdot c_{\text{category}} \cdot p \cdot (1 - p)$$

| Coefficient $c$ | Covered Market Categories | Max Taker Fee / Share ($p=0.50$) | Max Taker Fee / 100 Shares ($p=0.50$) |
| ---: | --- | ---: | ---: |
| `0.00` | Geopolitics | `$0.0000` | `$0.00` |
| `0.01` | Economics, Weather, Culture, Mentions, non-election Politics, non-short-term Crypto | `$0.0025` | `$0.25` |
| `0.05` | Elections, Finance, Tech, Sports, Combos, BTC 5m/15m Up/Down | `$0.0125` | `$1.25` |

### Kalshi Taker Fee Formula

Kalshi charges zero maker fees across the general market universe (`maker = 0`) and rounds taker fees **up to the nearest cent** (`ceil_to_cent`):

$$\text{fee}_{\text{kalshi}} = \operatorname{ceil\_to\_cent}\!\left(0.07 \times \text{contracts} \times p \times (1 - p)\right)$$

| Trade Size (Contracts) | Execution Price $p$ | Raw Fee $0.07 \cdot N \cdot p(1-p)$ | Charged Taker Fee (`ceil_to_cent`) |
| ---: | ---: | ---: | ---: |
| `1` | `0.50` | `$0.0175` | `$0.02` |
| `10` | `0.50` | `$0.1750` | `$0.18` |
| `100` | `0.50` | `$1.7500` | `$1.75` |
| `100` | `0.10` or `0.90` | `$0.6300` | `$0.63` |

> **Implementation note**: All orderbook prices, spreads, and fee calculations must use `decimal.Decimal` (never binary floating-point) to preserve exact cent-rounding (`ROUND_CEILING` to `Decimal("0.01")`) and avoid representation drift on tick boundaries.
