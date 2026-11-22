"""Read-only network smoke test for Polymarket and Kalshi public APIs.

Run with:
    uv run python scripts/smoke_test_apis.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com/markets"
POLYMARKET_CLOB_BOOK_URL = "https://clob.polymarket.com/book"
POLYMARKET_CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"
POLYMARKET_DATA_TRADES_URL = "https://data-api.polymarket.com/v2/trades"
POLYMARKET_DATA_HISTORY_URL = "https://data-api.polymarket.com/v2/prices-history"

KALSHI_PRIMARY_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_EXTERNAL_BASE = "https://external-api.kalshi.com/trade-api/v2"

WAF_DIAGNOSTIC_HEADERS: tuple[str, ...] = (
    "server",
    "cf-ray",
    "cf-mitigated",
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "x-cache",
    "via",
    "content-type",
)

SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
REQUEST_TIMEOUT_SECONDS = 20.0
USER_AGENT = "pm-quant-smoke-test/0.1"


@dataclass
class CheckResult:
    """Result of a single API endpoint smoke check."""

    endpoint: str
    url: str
    status_code: int | None
    latency_ms: float | None
    schema_ok: bool
    notes: str
    waf_headers: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status_code == 200 and self.schema_ok


def _extract_waf_headers(headers: httpx.Headers) -> dict[str, str]:
    extracted: dict[str, str] = {}
    for name in WAF_DIAGNOSTIC_HEADERS:
        value = headers.get(name)
        if value is not None:
            extracted[name] = value
    return extracted


def _request_json(
    client: httpx.Client,
    endpoint: str,
    url: str,
    params: dict[str, str | int] | None = None,
) -> tuple[CheckResult | None, Any, float]:
    """Send a GET request and decode JSON, returning a failed CheckResult on HTTP/network error."""
    start = time.perf_counter()
    try:
        response = client.get(url, params=params)
    except httpx.HTTPError as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return (
            CheckResult(
                endpoint=endpoint,
                url=url,
                status_code=None,
                latency_ms=latency_ms,
                schema_ok=False,
                notes=f"network error: {type(exc).__name__}: {exc}",
            ),
            None,
            latency_ms,
        )

    latency_ms = (time.perf_counter() - start) * 1000.0
    waf_headers = _extract_waf_headers(response.headers)

    if response.status_code != 200:
        header_summary = ", ".join(f"{k}={v}" for k, v in waf_headers.items()) or "no waf headers"
        note = f"unexpected HTTP {response.status_code}"
        if response.status_code in (403, 429):
            note = f"WAF/rate-limit HTTP {response.status_code} ({header_summary})"
        return (
            CheckResult(
                endpoint=endpoint,
                url=str(response.url),
                status_code=response.status_code,
                latency_ms=latency_ms,
                schema_ok=False,
                notes=note,
                waf_headers=waf_headers,
            ),
            None,
            latency_ms,
        )

    try:
        payload = response.json()
    except ValueError as exc:
        return (
            CheckResult(
                endpoint=endpoint,
                url=str(response.url),
                status_code=response.status_code,
                latency_ms=latency_ms,
                schema_ok=False,
                notes=f"invalid JSON: {exc}",
                waf_headers=waf_headers,
            ),
            None,
            latency_ms,
        )

    return None, payload, latency_ms


def _parse_clob_token_ids(raw: Any) -> list[str]:
    """Parse Polymarket Gamma clobTokenIds (either a JSON-encoded string or a list)."""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return []
        raw = parsed
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, str) and item.strip()]
    return []


def _shorten(value: str, max_len: int = 14) -> str:
    if len(value) <= max_len:
        return value
    return f"{value[: max_len - 3]}..."


def check_polymarket_gamma(
    client: httpx.Client,
) -> tuple[CheckResult, str | None, str | None]:
    endpoint = "Polymarket Gamma /markets"
    params: dict[str, str | int] = {"limit": 5, "active": "true", "closed": "false"}
    err, payload, latency_ms = _request_json(client, endpoint, POLYMARKET_GAMMA_URL, params=params)
    if err is not None:
        return err, None, None

    if not isinstance(payload, list) or not payload:
        return (
            CheckResult(
                endpoint=endpoint,
                url=POLYMARKET_GAMMA_URL,
                status_code=200,
                latency_ms=latency_ms,
                schema_ok=False,
                notes="expected non-empty list of markets",
            ),
            None,
            None,
        )

    chosen_condition: str | None = None
    chosen_token: str | None = None

    for prefer_orderbook in (True, False):
        for market in payload:
            if not isinstance(market, dict):
                continue
            if prefer_orderbook and not market.get("enableOrderBook", False):
                continue
            condition_id = market.get("conditionId")
            token_ids = _parse_clob_token_ids(market.get("clobTokenIds"))
            if isinstance(condition_id, str) and condition_id.strip() and token_ids:
                chosen_condition = condition_id.strip()
                chosen_token = token_ids[0].strip()
                break
        if chosen_condition and chosen_token:
            break

    if not chosen_condition or not chosen_token:
        return (
            CheckResult(
                endpoint=endpoint,
                url=POLYMARKET_GAMMA_URL,
                status_code=200,
                latency_ms=latency_ms,
                schema_ok=False,
                notes=f"markets={len(payload)}, missing valid conditionId/clobTokenIds",
            ),
            None,
            None,
        )

    return (
        CheckResult(
            endpoint=endpoint,
            url=POLYMARKET_GAMMA_URL,
            status_code=200,
            latency_ms=latency_ms,
            schema_ok=True,
            notes=(
                f"markets={len(payload)}, condition={_shorten(chosen_condition)}, "
                f"token={_shorten(chosen_token)}"
            ),
        ),
        chosen_condition,
        chosen_token,
    )


def check_polymarket_clob_book(client: httpx.Client, token_id: str | None) -> CheckResult:
    endpoint = "Polymarket CLOB /book"
    if not token_id:
        return CheckResult(
            endpoint=endpoint,
            url=POLYMARKET_CLOB_BOOK_URL,
            status_code=None,
            latency_ms=None,
            schema_ok=False,
            notes="skipped: no token_id from Gamma",
        )

    err, payload, latency_ms = _request_json(
        client, endpoint, POLYMARKET_CLOB_BOOK_URL, params={"token_id": token_id}
    )
    if err is not None:
        return err

    if not isinstance(payload, dict):
        return CheckResult(
            endpoint=endpoint,
            url=POLYMARKET_CLOB_BOOK_URL,
            status_code=200,
            latency_ms=latency_ms,
            schema_ok=False,
            notes="expected JSON object",
        )

    bids = payload.get("bids")
    asks = payload.get("asks")
    schema_ok = isinstance(bids, list) and isinstance(asks, list)
    notes = (
        f"bids={len(bids)}, asks={len(asks)}"
        if isinstance(bids, list) and isinstance(asks, list)
        else f"missing bids/asks lists (keys={sorted(payload.keys())})"
    )
    return CheckResult(
        endpoint=endpoint,
        url=POLYMARKET_CLOB_BOOK_URL,
        status_code=200,
        latency_ms=latency_ms,
        schema_ok=schema_ok,
        notes=notes,
    )


def _extract_history_points(payload: Any) -> tuple[str | None, list[Any] | None]:
    if isinstance(payload, dict):
        for key in ("history", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return key, value
    return None, None


def _sample_point_fields(points: list[Any]) -> list[str]:
    for item in points:
        if isinstance(item, dict):
            return sorted(str(k) for k in item.keys())
    return []


def check_polymarket_clob_history(
    client: httpx.Client, token_id: str | None
) -> tuple[CheckResult, int | None, list[str]]:
    endpoint = "Polymarket CLOB /prices-history"
    if not token_id:
        return (
            CheckResult(
                endpoint=endpoint,
                url=POLYMARKET_CLOB_HISTORY_URL,
                status_code=None,
                latency_ms=None,
                schema_ok=False,
                notes="skipped: no token_id from Gamma",
            ),
            None,
            [],
        )

    err, payload, latency_ms = _request_json(
        client,
        endpoint,
        POLYMARKET_CLOB_HISTORY_URL,
        params={"market": token_id, "interval": "1d"},
    )
    if err is not None:
        return err, None, []

    container_key, points = _extract_history_points(payload)
    if points is None:
        return (
            CheckResult(
                endpoint=endpoint,
                url=POLYMARKET_CLOB_HISTORY_URL,
                status_code=200,
                latency_ms=latency_ms,
                schema_ok=False,
                notes="expected object with 'history' list",
            ),
            None,
            [],
        )

    fields = _sample_point_fields(points)
    return (
        CheckResult(
            endpoint=endpoint,
            url=POLYMARKET_CLOB_HISTORY_URL,
            status_code=200,
            latency_ms=latency_ms,
            schema_ok=True,
            notes=f"key={container_key}, points={len(points)}, fields={fields}",
        ),
        len(points),
        fields,
    )


def check_polymarket_data_trades(client: httpx.Client, condition_id: str | None) -> CheckResult:
    endpoint = "Polymarket Data v2 /trades"
    if not condition_id:
        return CheckResult(
            endpoint=endpoint,
            url=POLYMARKET_DATA_TRADES_URL,
            status_code=None,
            latency_ms=None,
            schema_ok=False,
            notes="skipped: no condition_id from Gamma",
        )

    err, payload, latency_ms = _request_json(
        client,
        endpoint,
        POLYMARKET_DATA_TRADES_URL,
        params={"condition": condition_id, "taker_only": "true", "limit": 5},
    )
    if err is not None:
        return err

    if not isinstance(payload, dict):
        return CheckResult(
            endpoint=endpoint,
            url=POLYMARKET_DATA_TRADES_URL,
            status_code=200,
            latency_ms=latency_ms,
            schema_ok=False,
            notes="expected JSON object envelope",
        )

    data = payload.get("data")
    pagination = payload.get("pagination")
    has_data = isinstance(data, list)
    has_pagination = isinstance(pagination, dict) and "next_cursor" in pagination
    next_cursor = pagination.get("next_cursor") if isinstance(pagination, dict) else None
    cursor_valid = next_cursor is None or isinstance(next_cursor, str)

    non_snake_keys: list[str] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                non_snake_keys.append("<non-dict-item>")
                break
            for key in item:
                if not isinstance(key, str) or not SNAKE_CASE_RE.match(key):
                    non_snake_keys.append(str(key))

    schema_ok = has_data and has_pagination and cursor_valid and not non_snake_keys
    cursor_state = "null" if next_cursor is None else _shorten(str(next_cursor), 12)
    snake_status = "ok" if not non_snake_keys else f"non-snake={sorted(set(non_snake_keys))[:4]}"
    notes = (
        f"trades={len(data) if isinstance(data, list) else 'n/a'}, "
        f"next_cursor={cursor_state}, snake_case={snake_status}"
    )
    if not (has_data and has_pagination and cursor_valid):
        notes = f"envelope invalid (keys={sorted(payload.keys())}); {notes}"

    return CheckResult(
        endpoint=endpoint,
        url=POLYMARKET_DATA_TRADES_URL,
        status_code=200,
        latency_ms=latency_ms,
        schema_ok=schema_ok,
        notes=notes,
    )


def check_polymarket_data_history(
    client: httpx.Client,
    token_id: str | None,
    clob_count: int | None,
    clob_fields: list[str],
) -> CheckResult:
    endpoint = "Polymarket Data v2 /prices-history"
    if not token_id:
        return CheckResult(
            endpoint=endpoint,
            url=POLYMARKET_DATA_HISTORY_URL,
            status_code=None,
            latency_ms=None,
            schema_ok=False,
            notes="skipped: no token_id from Gamma",
        )

    err, payload, latency_ms = _request_json(
        client,
        endpoint,
        POLYMARKET_DATA_HISTORY_URL,
        params={"token_id": token_id, "interval": "1d"},
    )
    if err is not None:
        return err

    container_key, points = _extract_history_points(payload)
    if points is None:
        keys = sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
        return CheckResult(
            endpoint=endpoint,
            url=POLYMARKET_DATA_HISTORY_URL,
            status_code=200,
            latency_ms=latency_ms,
            schema_ok=False,
            notes=f"expected object with 'history'/'data' list (got {keys})",
        )

    fields = _sample_point_fields(points)
    count_cmp = (
        f"delta_vs_clob={len(points) - clob_count:+d}" if clob_count is not None else "clob=n/a"
    )
    fields_cmp = (
        "same_fields" if fields == clob_fields else f"fields={fields} vs clob={clob_fields}"
    )
    return CheckResult(
        endpoint=endpoint,
        url=POLYMARKET_DATA_HISTORY_URL,
        status_code=200,
        latency_ms=latency_ms,
        schema_ok=True,
        notes=f"key={container_key}, points={len(points)} ({count_cmp}), {fields_cmp}",
    )


def check_kalshi_cutoff(
    client: httpx.Client,
    label: str,
    base_url: str,
    reference_ts: str | None = None,
) -> tuple[CheckResult, str | None]:
    url = f"{base_url}/historical/cutoff"
    err, payload, latency_ms = _request_json(client, label, url)
    if err is not None:
        return err, None

    if not isinstance(payload, dict):
        return (
            CheckResult(
                endpoint=label,
                url=url,
                status_code=200,
                latency_ms=latency_ms,
                schema_ok=False,
                notes="expected JSON object",
            ),
            None,
        )

    market_settled_ts = payload.get("market_settled_ts")
    schema_ok = isinstance(market_settled_ts, str) and bool(market_settled_ts.strip())
    if not schema_ok:
        return (
            CheckResult(
                endpoint=label,
                url=url,
                status_code=200,
                latency_ms=latency_ms,
                schema_ok=False,
                notes=f"missing market_settled_ts (keys={sorted(payload.keys())})",
            ),
            None,
        )

    assert isinstance(market_settled_ts, str)
    notes = f"market_settled_ts={market_settled_ts}"
    if reference_ts is not None:
        match_state = (
            "matches primary" if market_settled_ts == reference_ts else "differs from primary"
        )
        notes = f"{notes} ({match_state})"

    return (
        CheckResult(
            endpoint=label,
            url=url,
            status_code=200,
            latency_ms=latency_ms,
            schema_ok=True,
            notes=notes,
        ),
        market_settled_ts,
    )


def check_kalshi_open_markets(client: httpx.Client) -> tuple[CheckResult, list[str]]:
    endpoint = "Kalshi /markets (open)"
    url = f"{KALSHI_PRIMARY_BASE}/markets"
    err, payload, latency_ms = _request_json(
        client, endpoint, url, params={"status": "open", "limit": 5}
    )
    if err is not None:
        return err, []

    if not isinstance(payload, dict) or not isinstance(payload.get("markets"), list):
        return (
            CheckResult(
                endpoint=endpoint,
                url=url,
                status_code=200,
                latency_ms=latency_ms,
                schema_ok=False,
                notes="expected object with 'markets' list",
            ),
            [],
        )

    markets = payload["markets"]
    tickers: list[str] = []
    for market in markets:
        if isinstance(market, dict):
            ticker = market.get("ticker")
            if isinstance(ticker, str) and ticker.strip():
                tickers.append(ticker.strip())
        if len(tickers) == 2:
            break

    schema_ok = len(tickers) == 2
    notes = (
        f"markets={len(markets)}, tickers={','.join(tickers)}"
        if schema_ok
        else f"markets={len(markets)}, found only {len(tickers)} valid tickers"
    )
    return (
        CheckResult(
            endpoint=endpoint,
            url=url,
            status_code=200,
            latency_ms=latency_ms,
            schema_ok=schema_ok,
            notes=notes,
        ),
        tickers,
    )


def _validate_kalshi_levels(raw_levels: Any) -> tuple[bool, int, bool, str]:
    """Validate a yes_dollars/no_dollars list and confirm ascending Decimal price order."""
    if raw_levels is None:
        return True, 0, True, ""
    if not isinstance(raw_levels, list):
        return False, 0, False, f"expected list or null, got {type(raw_levels).__name__}"

    prices: list[Decimal] = []
    for level in raw_levels:
        if not isinstance(level, list) or len(level) < 2:
            return False, len(raw_levels), False, "level is not a [price, size] pair"
        try:
            price = Decimal(str(level[0]))
        except InvalidOperation:
            return False, len(raw_levels), False, f"non-decimal price {level[0]!r}"
        prices.append(price)

    ascending = all(prices[idx] <= prices[idx + 1] for idx in range(len(prices) - 1))
    if not ascending:
        return False, len(prices), False, "prices not sorted ascending"
    return True, len(prices), True, ""


def check_kalshi_batch_orderbooks(client: httpx.Client, tickers: list[str]) -> CheckResult:
    endpoint = "Kalshi /markets/orderbooks"
    url = f"{KALSHI_PRIMARY_BASE}/markets/orderbooks"
    if len(tickers) < 2:
        return CheckResult(
            endpoint=endpoint,
            url=url,
            status_code=None,
            latency_ms=None,
            schema_ok=False,
            notes="skipped: fewer than 2 open tickers",
        )

    err, payload, latency_ms = _request_json(
        client,
        endpoint,
        url,
        params={"tickers": ",".join(tickers[:2]), "depth": 20},
    )
    if err is not None:
        return err

    if not isinstance(payload, dict) or not isinstance(payload.get("orderbooks"), list):
        keys = sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
        return CheckResult(
            endpoint=endpoint,
            url=url,
            status_code=200,
            latency_ms=latency_ms,
            schema_ok=False,
            notes=f"expected object with 'orderbooks' list (got {keys})",
        )

    orderbooks = payload["orderbooks"]
    if not orderbooks:
        return CheckResult(
            endpoint=endpoint,
            url=url,
            status_code=200,
            latency_ms=latency_ms,
            schema_ok=False,
            notes="empty orderbooks list",
        )

    total_yes = 0
    total_no = 0
    for idx, entry in enumerate(orderbooks):
        if not isinstance(entry, dict):
            return CheckResult(
                endpoint=endpoint,
                url=url,
                status_code=200,
                latency_ms=latency_ms,
                schema_ok=False,
                notes=f"orderbooks[{idx}] is not an object",
            )
        ob_fp = entry.get("orderbook_fp")
        if not isinstance(ob_fp, dict):
            return CheckResult(
                endpoint=endpoint,
                url=url,
                status_code=200,
                latency_ms=latency_ms,
                schema_ok=False,
                notes=f"orderbooks[{idx}] missing orderbook_fp (keys={sorted(entry.keys())})",
            )
        if "yes_dollars" not in ob_fp or "no_dollars" not in ob_fp:
            return CheckResult(
                endpoint=endpoint,
                url=url,
                status_code=200,
                latency_ms=latency_ms,
                schema_ok=False,
                notes=(
                    f"orderbooks[{idx}].orderbook_fp missing yes_dollars/no_dollars "
                    f"(keys={sorted(ob_fp.keys())})"
                ),
            )

        yes_ok, yes_cnt, _, yes_err = _validate_kalshi_levels(ob_fp["yes_dollars"])
        if not yes_ok:
            return CheckResult(
                endpoint=endpoint,
                url=url,
                status_code=200,
                latency_ms=latency_ms,
                schema_ok=False,
                notes=f"orderbooks[{idx}].yes_dollars invalid: {yes_err}",
            )
        no_ok, no_cnt, _, no_err = _validate_kalshi_levels(ob_fp["no_dollars"])
        if not no_ok:
            return CheckResult(
                endpoint=endpoint,
                url=url,
                status_code=200,
                latency_ms=latency_ms,
                schema_ok=False,
                notes=f"orderbooks[{idx}].no_dollars invalid: {no_err}",
            )
        total_yes += yes_cnt
        total_no += no_cnt

    return CheckResult(
        endpoint=endpoint,
        url=url,
        status_code=200,
        latency_ms=latency_ms,
        schema_ok=True,
        notes=(
            f"books={len(orderbooks)}, yes_levels={total_yes}, no_levels={total_no}, "
            "orderbook_fp ascending=ok"
        ),
    )


def format_table(results: list[CheckResult]) -> str:
    """Format check results as a compact aligned ASCII table."""
    headers = ("Endpoint", "HTTP", "Latency (ms)", "Schema", "Notes")
    rows: list[tuple[str, str, str, str, str]] = []
    for res in results:
        http_str = str(res.status_code) if res.status_code is not None else "ERR/SKIP"
        lat_str = f"{res.latency_ms:.1f}" if res.latency_ms is not None else "-"
        schema_str = "PASS" if res.passed else "FAIL"
        rows.append((res.endpoint, http_str, lat_str, schema_str, res.notes))

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows)) for col in range(len(headers))
    ]

    def _fmt_row(row: tuple[str, str, str, str, str]) -> str:
        return " | ".join(
            row[idx].rjust(widths[idx]) if idx in (1, 2) else row[idx].ljust(widths[idx])
            for idx in range(len(row))
        )

    sep = "-+-".join("-" * width for width in widths)
    lines = [_fmt_row(headers), sep, *(_fmt_row(row) for row in rows)]
    return "\n".join(lines)


def run_smoke_tests() -> list[CheckResult]:
    """Run all Polymarket and Kalshi read-only API smoke checks sequentially."""
    results: list[CheckResult] = []
    with httpx.Client(
        timeout=REQUEST_TIMEOUT_SECONDS,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    ) as client:
        gamma_res, condition_id, token_id = check_polymarket_gamma(client)
        results.append(gamma_res)

        results.append(check_polymarket_clob_book(client, token_id))

        clob_hist_res, clob_count, clob_fields = check_polymarket_clob_history(client, token_id)
        results.append(clob_hist_res)

        results.append(check_polymarket_data_trades(client, condition_id))
        results.append(check_polymarket_data_history(client, token_id, clob_count, clob_fields))

        kalshi_primary_res, primary_ts = check_kalshi_cutoff(
            client,
            "Kalshi /historical/cutoff (elections)",
            KALSHI_PRIMARY_BASE,
        )
        results.append(kalshi_primary_res)

        kalshi_ext_res, _ = check_kalshi_cutoff(
            client,
            "Kalshi /historical/cutoff (external)",
            KALSHI_EXTERNAL_BASE,
            reference_ts=primary_ts,
        )
        results.append(kalshi_ext_res)

        kalshi_markets_res, tickers = check_kalshi_open_markets(client)
        results.append(kalshi_markets_res)

        results.append(check_kalshi_batch_orderbooks(client, tickers))

    return results


def main() -> int:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"pm-quant API Smoke Test Report ({timestamp})\n")

    results = run_smoke_tests()
    print(format_table(results))

    waf_failures = [res for res in results if res.status_code in (403, 429)]
    if waf_failures:
        print("\nWAF / Rate-Limit Diagnostics (403/429):")
        for res in waf_failures:
            print(f"  - {res.endpoint} -> HTTP {res.status_code} ({res.url})")
            if res.waf_headers:
                for name, value in res.waf_headers.items():
                    print(f"      {name}: {value}")
            else:
                print("      (no diagnostic headers present)")

    passed = sum(1 for res in results if res.passed)
    total = len(results)
    verdict = "PASS" if passed == total else "FAIL"
    print(f"\nSummary: {verdict} ({passed}/{total} endpoints returned HTTP 200 with valid schema)")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
