"""Rate-limited asynchronous HTTP client with exponential backoff and retry."""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any, Final

import httpx

DEFAULT_MAX_CONCURRENCY: Final[int] = 5
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_BACKOFF_BASE_SECONDS: Final[float] = 0.5
DEFAULT_BACKOFF_MAX_SECONDS: Final[float] = 30.0
DEFAULT_MAX_RETRY_AFTER_SECONDS: Final[float] = 60.0
DEFAULT_USER_AGENT: Final[str] = "pm-quant/0.1"
DEFAULT_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(
    connect=10.0,
    read=20.0,
    write=10.0,
    pool=10.0,
)

SleepFn = Callable[[float], Awaitable[None]]
NowFn = Callable[[], datetime]


def parse_retry_after(
    header_value: str | None,
    *,
    now: datetime | None = None,
) -> float | None:
    """Parse an HTTP ``Retry-After`` header (delta-seconds or HTTP-date) into seconds.

    Returns ``None`` if the header is missing or malformed.
    """
    if header_value is None:
        return None
    raw = header_value.strip()
    if not raw:
        return None

    try:
        seconds = float(raw)
    except ValueError:
        seconds = math.nan

    if math.isfinite(seconds):
        return max(0.0, seconds)

    try:
        retry_dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None

    if retry_dt.tzinfo is None:
        retry_dt = retry_dt.replace(tzinfo=UTC)
    reference = now if now is not None else datetime.now(UTC)
    return max(0.0, (retry_dt - reference).total_seconds())


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


class AsyncRateLimitedClient:
    """Async HTTP client wrapping ``httpx.AsyncClient`` with concurrency limits and retries.

    Retry policy:
    - Retries HTTP ``429`` and ``5xx`` (honoring ``Retry-After`` on ``429`` and ``503`` when
      present), plus ``httpx.TransportError``, with exponential backoff and equal jitter.
    - Fails fast without retrying on non-429 ``4xx`` responses (e.g. ``400``, ``404``, ``410``).
    - Bounds concurrent in-flight requests using an ``asyncio.Semaphore`` (released while sleeping).
    """

    def __init__(
        self,
        *,
        base_url: str = "",
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        backoff_base: float = DEFAULT_BACKOFF_BASE_SECONDS,
        backoff_max: float = DEFAULT_BACKOFF_MAX_SECONDS,
        max_retry_after: float | None = DEFAULT_MAX_RETRY_AFTER_SECONDS,
        headers: Mapping[str, str] | None = None,
        follow_redirects: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: SleepFn = asyncio.sleep,
        rng: random.Random | None = None,
        now_fn: NowFn | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if backoff_base <= 0:
            raise ValueError(f"backoff_base must be > 0, got {backoff_base}")
        if backoff_max < backoff_base:
            raise ValueError(
                f"backoff_max ({backoff_max}) must be >= backoff_base ({backoff_base})"
            )
        if max_retry_after is not None and max_retry_after < 0:
            raise ValueError(f"max_retry_after must be >= 0 or None, got {max_retry_after}")

        merged_headers: dict[str, str] = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
        }
        if headers is not None:
            merged_headers.update(headers)

        resolved_timeout = httpx.Timeout(timeout) if isinstance(timeout, int | float) else timeout

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_concurrency = max_concurrency
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._max_retry_after = max_retry_after
        self._sleep = sleep
        self._rng = rng if rng is not None else random.Random()
        self._now_fn: NowFn = now_fn if now_fn is not None else (lambda: datetime.now(UTC))
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=resolved_timeout,
            headers=merged_headers,
            follow_redirects=follow_redirects,
            transport=transport,
        )

    @property
    def max_concurrency(self) -> int:
        """Maximum number of concurrent in-flight requests."""
        return self._max_concurrency

    @property
    def max_retries(self) -> int:
        """Maximum number of retry attempts after the initial request."""
        return self._max_retries

    async def __aenter__(self) -> AsyncRateLimitedClient:
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def aclose(self) -> None:
        """Close the underlying ``httpx.AsyncClient``."""
        await self._client.aclose()

    def _compute_backoff_delay(self, attempt: int) -> float:
        """Compute exponential backoff with equal jitter for a 0-indexed failed attempt."""
        cap = min(self._backoff_max, self._backoff_base * (2.0**attempt))
        half = cap / 2.0
        return half + self._rng.uniform(0.0, half)

    def _compute_retry_delay(self, attempt: int, response: httpx.Response | None) -> float:
        """Compute delay before retrying, honoring ``Retry-After`` on HTTP 429 and 503."""
        if response is not None and response.status_code in (429, 503):
            retry_after = parse_retry_after(
                response.headers.get("Retry-After"),
                now=self._now_fn(),
            )
            if retry_after is not None:
                if self._max_retry_after is not None:
                    return min(retry_after, self._max_retry_after)
                return retry_after
        return self._compute_backoff_delay(attempt)

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        json: Any = None,
    ) -> httpx.Response:
        """Send an HTTP request with concurrency limiting and automatic retry/backoff."""
        for attempt in range(self._max_retries + 1):
            try:
                async with self._semaphore:
                    response = await self._client.request(
                        method,
                        url,
                        params=params,
                        headers=headers,
                        json=json,
                    )
            except httpx.TransportError:
                if attempt >= self._max_retries:
                    raise
                delay = self._compute_retry_delay(attempt, response=None)
                await self._sleep(delay)
                continue

            if _is_retryable_status(response.status_code):
                if attempt >= self._max_retries:
                    response.raise_for_status()
                delay = self._compute_retry_delay(attempt, response=response)
                await response.aclose()
                await self._sleep(delay)
                continue

            # Non-retryable 4xx (e.g. 400/404/410) fails fast immediately here.
            response.raise_for_status()
            return response

        raise AssertionError("unreachable")  # pragma: no cover

    async def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """Send a rate-limited HTTP GET request."""
        return await self.request("GET", url, params=params, headers=headers)

    async def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Send a rate-limited HTTP GET request and decode the JSON body."""
        response = await self.get(url, params=params, headers=headers)
        return response.json()
