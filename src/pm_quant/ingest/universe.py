"""Deterministic universe selection policy for the daily L2 orderbook collector.

This module implements pure, offline-testable screening rules that filter live
Polymarket and Kalshi market candidates into a bounded, reproducible daily L2
snapshot universe:

- **Anti-leakage separation**: live 24h volume and open-interest screening
  metrics live exclusively on :class:`LiveMarketCandidate` (never on
  :class:`pm_quant.ingest.schemas.MarketMetadata`) so post-hoc lifetime volume
  snapshots cannot leak into point-in-time statistical features.
- **Eligibility filters**: keep only active/open binary non-combo markets
  (:func:`is_binary_non_combo`) and exclude high-churn 5m/15m and intraday
  crypto "Up or Down" contracts (:func:`is_short_term_crypto`) without false
  positives on medium/long-term crypto markets.
- **Separate venue budgets**: force-include near-term target calibration series
  (Weather, Economics/Macro, recurring weekly series, and configurable
  Polymarket tag IDs) within ``target_series_max_horizon_days`` up to
  ``max_target_series_per_venue``, and allocate a separate
  ``max_core_per_venue`` budget to high-volume core markets sorted
  deterministically by ``volume_24h`` descending with ``market_id`` tie-break.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Final, Literal

from pm_quant.ingest.schemas import MarketMetadata, Venue

VolumeUnit = Literal["usd", "contracts"]
SelectionReason = Literal["target_series", "volume_core"]

_ZERO: Final[Decimal] = Decimal("0")
_VENUE_ORDER: Final[tuple[Venue, ...]] = ("polymarket", "kalshi")
_ACTIVE_STATUSES: Final[frozenset[str]] = frozenset({"active", "open"})
_COMBO_TICKER_PREFIXES: Final[tuple[str, ...]] = ("KXMVE", "KXMULTI")
_VALID_VOLUME_UNITS: Final[frozenset[str]] = frozenset({"usd", "contracts"})

DEFAULT_WEATHER_SERIES_PREFIXES: Final[tuple[str, ...]] = (
    "KXHIGH",
    "KXLOW",
    "KXTEMP",
    "KXRAIN",
)
DEFAULT_MACRO_SERIES_PREFIXES: Final[tuple[str, ...]] = (
    "KXCPI",
    "KXFED",
    "KXFOMC",
    "KXPAYROLLS",
    "KXGDP",
    "KXUNRATE",
    "KXPCE",
    "KXJOBLESS",
)
DEFAULT_WEEKLY_SERIES_PREFIXES: Final[tuple[str, ...]] = (
    "KXINXW",
    "KXNASDAQ100W",
    "KXWTIW",
    "KXAAAGASW",
    "KXTNOTEW",
    "KXBTCW",
    "KXETHW",
    "KXGOLDW",
    "KXEURUSDW",
)
DEFAULT_TARGET_SERIES_PREFIXES: Final[tuple[str, ...]] = (
    DEFAULT_WEATHER_SERIES_PREFIXES + DEFAULT_MACRO_SERIES_PREFIXES + DEFAULT_WEEKLY_SERIES_PREFIXES
)
DEFAULT_TARGET_CATEGORIES: Final[tuple[str, ...]] = (
    "weather",
    "economics",
)

# ---------------------------------------------------------------------------
# Compiled patterns for combo/MVE and short-term crypto screening
# ---------------------------------------------------------------------------

_COMBO_TEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[^a-z0-9])"
    r"(?:combo|combos|parlay|parlays|same[-_ ]?game[-_ ]?parlay|multi[-_ ]?leg|multileg)"
    r"(?:$|[^a-z0-9])",
    re.IGNORECASE,
)

# Kalshi / compact ticker identifiers like KXBTC15M-26NOV23, KXETH5M, btc15m-updown.
_CRYPTO_TICKER_SHORT_TERM_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[^a-z0-9])"
    r"(?:kx)?(?:btc|eth|sol|xrp|doge|crypto)[a-z0-9]*?(?<!\d)(?:5|15)m(?:in)?"
    r"(?:$|[^a-z0-9])",
    re.IGNORECASE,
)

# Word-bounded crypto asset mentions (avoids substrings such as "resolution" or "whether").
_CRYPTO_ASSET_TEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[^a-z0-9])"
    r"(?:btc|bitcoin|eth|ethereum|sol|solana|xrp|ripple|doge|dogecoin)"
    r"(?:$|[^a-z0-9])",
    re.IGNORECASE,
)
_CRYPTO_TICKER_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"^kx(?:btc|eth|sol|xrp|doge|crypto)",
    re.IGNORECASE,
)

# Explicit 5/15 minute phrases ("5-min", "15-min", "5 minute", "15 minutes", "5min").
_EXPLICIT_SHORT_MINUTE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[^a-z0-9.])(?:5|15)[-\s]*(?:min|mins|minute|minutes)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)

# Compact "5m" / "15m" tokens in machine identifiers (slugs, tickers, event IDs).
_IDENTIFIER_SHORT_M_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[^a-z0-9$€£.])(?:5|15)m(?:$|[^a-z0-9])",
    re.IGNORECASE,
)

# Magnitude words that indicate "5M"/"15M" means millions rather than minutes.
_MILLION_UNIT_LOOKAHEAD: Final[str] = (
    r"(?!\s*(?:usd|usdc|usdt|dollars?|users?|addresses?|inflows?|outflows?|"
    r"tvl|volume|market\s*cap|cap|daily|weekly|monthly|active|holders?|"
    r"transactions?|tx|tokens?|coins?|shares?|btc|eth|sol|xrp|million))"
)
_TITLE_LOWER_SHORT_M_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[^a-zA-Z0-9$€£.])(?:5|15)m(?![a-zA-Z0-9])" + _MILLION_UNIT_LOOKAHEAD
)
_TITLE_UPPER_SHORT_M_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[^a-zA-Z0-9$€£.])(?:5|15)M(?![a-zA-Z0-9])"
    r"(?=\s*(?:$|[\)\]\-:,/]|(?i:up\b|down\b|up/down\b|candle|chart|window|"
    r"interval|expir|close|price|binary|contract|intraday)))"
)

# Intraday "Up or Down" detection for BTC/ETH/SOL/XRP/DOGE contracts.
_UP_OR_DOWN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[^a-z0-9])(?:up\s+or\s+down|up/down|up-or-down|updown)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)
_INTRADAY_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[^a-z0-9])(?:intraday|hourly|updown|1h|1-?hour|30m|30-?min(?:ute)?s?)(?:$|[^a-z0-9])"
    r"|\b\d{1,2}:\d{2}\b"
    r"|\b\d{1,2}\s*(?:am|pm)\b",
    re.IGNORECASE,
)
_LONG_HORIZON_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[^a-z0-9])"
    r"(?:weekly|this\s+week|next\s+week|monthly|this\s+month|next\s+month|"
    r"quarterly|yearly|annual|this\s+year|next\s+year|q[1-4]|20\d{2})"
    r"(?:$|[^a-z0-9])",
    re.IGNORECASE,
)


def _validate_non_negative_decimal(value: Decimal, field_name: str) -> None:
    if not value.is_finite() or value < _ZERO:
        raise ValueError(f"{field_name} must be a finite non-negative Decimal, got {value!r}")


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative int, got {value!r}")


def _normalize_utc(dt: datetime, field_name: str) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware (UTC), got naive datetime {dt!r}")
    return dt.astimezone(UTC)


def _normalize_string_tuple(
    values: Iterable[str],
    *,
    transform: Literal["none", "upper", "lower"] = "none",
) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in values:
        text = raw.strip()
        if not text:
            continue
        if transform == "upper":
            text = text.upper()
        elif transform == "lower":
            text = text.lower()
        if text not in seen:
            seen.add(text)
            normalized.append(text)
    return tuple(normalized)


def _extra_str(meta: MarketMetadata, key: str) -> str | None:
    value = meta.extra.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _extract_meta_tag_ids(meta: MarketMetadata) -> tuple[str, ...]:
    tags: list[str] = []
    raw_single = meta.extra.get("tag_id")
    if isinstance(raw_single, str) and raw_single.strip():
        tags.append(raw_single.strip())
    elif isinstance(raw_single, int) and not isinstance(raw_single, bool):
        tags.append(str(raw_single))

    raw_multi = meta.extra.get("tag_ids")
    if isinstance(raw_multi, tuple | list | set | frozenset):
        for item in raw_multi:
            if isinstance(item, str) and item.strip():
                tags.append(item.strip())
            elif isinstance(item, int) and not isinstance(item, bool):
                tags.append(str(item))
    return _normalize_string_tuple(tags)


# ---------------------------------------------------------------------------
# Immutable screening & policy dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LiveMarketCandidate:
    """Live screening record pairing ``MarketMetadata`` with ``t_now`` volume metrics.

    Used **only** for live universe screening at ``t_now`` (Polymarket 24h
    volume in USD notional; Kalshi 24h volume in contracts/face-value). Live
    volume and open-interest metrics are intentionally kept separate from
    ``MarketMetadata`` to prevent lookahead leakage into historical features.
    """

    metadata: MarketMetadata
    volume_24h: Decimal | None = None
    open_interest: Decimal | None = None
    volume_24h_unit: VolumeUnit = "usd"
    tag_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.volume_24h is not None:
            _validate_non_negative_decimal(self.volume_24h, "volume_24h")
        if self.open_interest is not None:
            _validate_non_negative_decimal(self.open_interest, "open_interest")
        if self.volume_24h_unit not in _VALID_VOLUME_UNITS:
            raise ValueError(
                f"volume_24h_unit must be 'usd' or 'contracts', got {self.volume_24h_unit!r}"
            )
        object.__setattr__(self, "tag_ids", _normalize_string_tuple(self.tag_ids))


@dataclass(frozen=True, slots=True)
class SelectedMarket:
    """Market retained by :func:`select_universe` together with its inclusion reason."""

    candidate: LiveMarketCandidate
    selection_reason: SelectionReason


@dataclass(frozen=True, slots=True)
class UniversePolicy:
    """Configurable deterministic policy governing daily L2 universe selection.

    Separate per-venue caps (``max_target_series_per_venue`` and
    ``max_core_per_venue``) ensure recurring Kalshi weather/macro ladders do not
    crowd out high-volume core contracts, while ``target_series_max_horizon_days``
    prevents far-dated zero-volume target strikes from saturating the target
    budget.
    """

    min_volume_24h_polymarket_usd: Decimal = Decimal("5000")
    min_volume_24h_kalshi_contracts: Decimal = Decimal("1000")
    max_core_per_venue: int = 200
    max_target_series_per_venue: int = 200
    target_series_max_horizon_days: int = 45
    target_series_prefixes: tuple[str, ...] = DEFAULT_TARGET_SERIES_PREFIXES
    target_categories: tuple[str, ...] = DEFAULT_TARGET_CATEGORIES
    target_series_patterns: tuple[str, ...] = ()
    polymarket_target_tag_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_non_negative_decimal(
            self.min_volume_24h_polymarket_usd,
            "min_volume_24h_polymarket_usd",
        )
        _validate_non_negative_decimal(
            self.min_volume_24h_kalshi_contracts,
            "min_volume_24h_kalshi_contracts",
        )
        _validate_non_negative_int(self.max_core_per_venue, "max_core_per_venue")
        _validate_non_negative_int(
            self.max_target_series_per_venue,
            "max_target_series_per_venue",
        )
        _validate_non_negative_int(
            self.target_series_max_horizon_days,
            "target_series_max_horizon_days",
        )
        object.__setattr__(
            self,
            "target_series_prefixes",
            _normalize_string_tuple(self.target_series_prefixes, transform="upper"),
        )
        object.__setattr__(
            self,
            "target_categories",
            _normalize_string_tuple(self.target_categories, transform="lower"),
        )
        object.__setattr__(
            self,
            "polymarket_target_tag_ids",
            _normalize_string_tuple(self.polymarket_target_tag_ids),
        )
        cleaned_patterns = _normalize_string_tuple(self.target_series_patterns)
        for pattern in cleaned_patterns:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"invalid regex in target_series_patterns: {pattern!r}") from exc
        object.__setattr__(self, "target_series_patterns", cleaned_patterns)

    def min_volume_for_venue(self, venue: Venue) -> Decimal:
        """Return the configured 24h volume threshold for ``venue``."""
        if venue == "polymarket":
            return self.min_volume_24h_polymarket_usd
        return self.min_volume_24h_kalshi_contracts


# ---------------------------------------------------------------------------
# Pure eligibility, classification, and selection functions
# ---------------------------------------------------------------------------


def is_short_term_crypto(meta: MarketMetadata) -> bool:
    """Return ``True`` for 5m/15m and intraday crypto 'Up or Down' contracts.

    Excludes high-churn 5-minute / 15-minute crypto markets (e.g., ``5m``,
    ``15m``, ``5-min``, ``15-min``, ``5 minute``, ``15 minute``,
    ``KXBTC15M`` / ``KXETH15M``, and intraday ``Up or Down`` contracts on
    BTC/ETH/SOL/XRP) while avoiding false positives on medium/long-term crypto
    markets (e.g., ``"Will Bitcoin exceed $150k by Dec 2026?"``) and on
    dollar-magnitude strings such as ``"$15M"``.
    """
    series_ticker = _extra_str(meta, "series_ticker")
    slug = _extra_str(meta, "slug")
    identifiers = tuple(
        text for text in (meta.market_id, meta.event_id, series_ticker, slug) if text is not None
    )

    # 1. Explicit short-term crypto ticker/slug prefixes (KXBTC15M, KXETH15M, btc15m, etc.).
    if any(_CRYPTO_TICKER_SHORT_TERM_RE.search(ident) for ident in identifiers):
        return True

    # 2. Determine whether the market references a crypto asset.
    category_is_crypto = meta.category is not None and meta.category.strip().lower() == "crypto"
    has_crypto_asset = any(_CRYPTO_TICKER_PREFIX_RE.search(ident) for ident in identifiers) or any(
        _CRYPTO_ASSET_TEXT_RE.search(text) for text in (*identifiers, meta.title)
    )
    if not (has_crypto_asset or category_is_crypto):
        return False

    # 3. 5m / 15m duration patterns on crypto markets.
    if any(_EXPLICIT_SHORT_MINUTE_RE.search(text) for text in (*identifiers, meta.title)):
        return True
    if any(_IDENTIFIER_SHORT_M_RE.search(ident) for ident in identifiers):
        return True
    if _TITLE_LOWER_SHORT_M_RE.search(meta.title):
        return True
    if _TITLE_UPPER_SHORT_M_RE.search(meta.title):
        return True

    # 4. Intraday "Up or Down" contracts on major crypto assets (BTC/ETH/SOL/XRP/DOGE).
    if has_crypto_asset:
        up_down_texts = (*identifiers, meta.title)
        if any(_UP_OR_DOWN_RE.search(text) for text in up_down_texts):
            if any(_INTRADAY_MARKER_RE.search(text) for text in up_down_texts):
                return True
            if not any(_LONG_HORIZON_MARKER_RE.search(text) for text in up_down_texts):
                return True

    return False


def is_binary_non_combo(meta: MarketMetadata) -> bool:
    """Return ``True`` if ``meta`` is an active/open binary non-combo market.

    Requirements:
    - Status is active/open (and not marked closed/archived/inactive in ``extra``).
    - Exactly two non-empty outcomes (excludes scalar and multi-outcome markets).
    - Not a combo/MVE/parlay market (no ``mve_collection_ticker``, no
      ``KXMVE``/``KXMULTI`` prefix, and no combo/parlay indicators).
    - For Polymarket: ``enable_order_book is True``, ``accepting_orders is not False``,
      and ``len(clob_token_ids) == 2`` with non-empty string token IDs.
    """
    if meta.status.strip().lower() not in _ACTIVE_STATUSES:
        return False
    if meta.extra.get("closed") is True or meta.extra.get("archived") is True:
        return False
    if meta.extra.get("active") is False:
        return False

    if len(meta.outcomes) != 2 or any(not outcome.strip() for outcome in meta.outcomes):
        return False

    market_type = _extra_str(meta, "market_type")
    if market_type is not None and market_type.lower() != "binary":
        return False

    # Exclude Kalshi MVE / multi-leg combo contracts.
    mve_collection = meta.extra.get("mve_collection_ticker")
    if isinstance(mve_collection, str) and mve_collection.strip():
        return False
    if mve_collection is not None and not isinstance(mve_collection, str) and bool(mve_collection):
        return False

    mve_legs = meta.extra.get("mve_selected_legs")
    if isinstance(mve_legs, tuple | list) and len(mve_legs) > 0:
        return False

    series_ticker = _extra_str(meta, "series_ticker")
    for ident in (meta.market_id, meta.event_id, series_ticker):
        if ident is not None and ident.upper().startswith(_COMBO_TICKER_PREFIXES):
            return False

    slug = _extra_str(meta, "slug")
    fee_type = _extra_str(meta, "fee_type")
    combo_fields = (
        meta.market_id,
        meta.event_id,
        series_ticker,
        slug,
        meta.category,
        fee_type,
        meta.title,
    )
    if any(field is not None and _COMBO_TEXT_RE.search(field) for field in combo_fields):
        return False

    if meta.venue == "polymarket":
        if meta.extra.get("enable_order_book") is not True:
            return False
        if meta.extra.get("accepting_orders") is False:
            return False
        clob_token_ids = meta.extra.get("clob_token_ids")
        if not isinstance(clob_token_ids, tuple | list) or len(clob_token_ids) != 2:
            return False
        if any(
            not isinstance(token_id, str) or not token_id.strip() for token_id in clob_token_ids
        ):
            return False

    return True


def is_target_series(
    candidate_or_meta: LiveMarketCandidate | MarketMetadata,
    policy: UniversePolicy,
) -> bool:
    """Return ``True`` if a market matches any configured target calibration series."""
    if isinstance(candidate_or_meta, LiveMarketCandidate):
        meta = candidate_or_meta.metadata
        candidate_tag_ids = candidate_or_meta.tag_ids
    else:
        meta = candidate_or_meta
        candidate_tag_ids = ()

    series_ticker = _extra_str(meta, "series_ticker")
    slug = _extra_str(meta, "slug")

    if policy.target_series_prefixes:
        for ident in (meta.market_id, meta.event_id, series_ticker):
            if ident is not None and ident.upper().startswith(policy.target_series_prefixes):
                return True

    if policy.target_categories and meta.category is not None:
        if meta.category.strip().lower() in policy.target_categories:
            return True

    if meta.venue == "polymarket" and policy.polymarket_target_tag_ids:
        all_tags = set(candidate_tag_ids) | set(_extract_meta_tag_ids(meta))
        if all_tags.intersection(policy.polymarket_target_tag_ids):
            return True

    if policy.target_series_patterns:
        searchable = tuple(
            text
            for text in (meta.market_id, meta.event_id, series_ticker, slug, meta.title)
            if text is not None
        )
        for pattern in policy.target_series_patterns:
            compiled = re.compile(pattern, re.IGNORECASE)
            if any(compiled.search(text) for text in searchable):
                return True

    return False


def _merge_duplicate_candidate(
    existing: LiveMarketCandidate,
    incoming: LiveMarketCandidate,
) -> LiveMarketCandidate:
    """Merge duplicate ``(venue, market_id)`` entries without losing tags or screening metrics."""
    merged_tags = _normalize_string_tuple((*existing.tag_ids, *incoming.tag_ids))
    merged_vol = existing.volume_24h if existing.volume_24h is not None else incoming.volume_24h
    merged_oi = (
        existing.open_interest if existing.open_interest is not None else incoming.open_interest
    )
    if (
        merged_tags == existing.tag_ids
        and merged_vol == existing.volume_24h
        and merged_oi == existing.open_interest
    ):
        return existing
    return LiveMarketCandidate(
        metadata=existing.metadata,
        volume_24h=merged_vol,
        open_interest=merged_oi,
        volume_24h_unit=existing.volume_24h_unit,
        tag_ids=merged_tags,
    )


def _within_target_horizon(
    meta: MarketMetadata,
    *,
    reference_utc: datetime,
    max_horizon_days: int,
) -> bool:
    if meta.end_time_utc is None:
        return True
    horizon_cutoff = reference_utc + timedelta(days=max_horizon_days)
    return meta.end_time_utc <= horizon_cutoff


def _volume_sort_key(candidate: LiveMarketCandidate) -> tuple[int, Decimal, str]:
    """Sort key: ``volume_24h`` descending (``None`` last), tie-break by ``market_id`` ascending."""
    if candidate.volume_24h is None:
        return (1, _ZERO, candidate.metadata.market_id)
    return (0, -candidate.volume_24h, candidate.metadata.market_id)


def select_universe(
    candidates: Iterable[LiveMarketCandidate],
    policy: UniversePolicy | None = None,
    *,
    now_utc: datetime | None = None,
) -> list[SelectedMarket]:
    """Deterministically select the daily L2 snapshot universe under ``policy``.

    Pipeline:
    1. Deduplicate candidates by ``(venue, market_id)`` (preserving merged
       ``tag_ids`` and first non-``None`` live screening metrics).
    2. Discard markets failing :func:`is_binary_non_combo` or matching
       :func:`is_short_term_crypto`.
    3. Per venue, select ``target_series`` first (even when ``volume_24h`` is
       ``0`` or ``None``), requiring expiration within
       ``policy.target_series_max_horizon_days`` of ``now_utc``, sorted by
       ``volume_24h`` descending (``None`` last) with ascending ``market_id``
       tie-break and capped at ``policy.max_target_series_per_venue``.
    4. Per venue, select remaining eligible markets with
       ``volume_24h >= venue_threshold`` as ``volume_core``, sorted by
       ``volume_24h`` descending with ascending ``market_id`` tie-break and
       capped separately at ``policy.max_core_per_venue``.
    """
    active_policy = policy if policy is not None else UniversePolicy()
    reference_utc = _normalize_utc(now_utc, "now_utc") if now_utc is not None else datetime.now(UTC)

    deduped: dict[tuple[Venue, str], LiveMarketCandidate] = {}
    for candidate in candidates:
        key = (candidate.metadata.venue, candidate.metadata.market_id)
        if key not in deduped:
            deduped[key] = candidate
        else:
            deduped[key] = _merge_duplicate_candidate(deduped[key], candidate)

    eligible_by_venue: dict[Venue, list[LiveMarketCandidate]] = {
        venue: [] for venue in _VENUE_ORDER
    }
    for candidate in deduped.values():
        meta = candidate.metadata
        if not is_binary_non_combo(meta):
            continue
        if is_short_term_crypto(meta):
            continue
        eligible_by_venue[meta.venue].append(candidate)

    target_selected: list[SelectedMarket] = []
    core_selected: list[SelectedMarket] = []

    for venue in _VENUE_ORDER:
        venue_candidates = eligible_by_venue[venue]

        target_pool = [
            candidate
            for candidate in venue_candidates
            if is_target_series(candidate, active_policy)
            and _within_target_horizon(
                candidate.metadata,
                reference_utc=reference_utc,
                max_horizon_days=active_policy.target_series_max_horizon_days,
            )
        ]
        target_pool.sort(key=_volume_sort_key)
        chosen_targets = target_pool[: active_policy.max_target_series_per_venue]
        chosen_target_ids = {candidate.metadata.market_id for candidate in chosen_targets}
        target_selected.extend(
            SelectedMarket(candidate=candidate, selection_reason="target_series")
            for candidate in chosen_targets
        )

        threshold = active_policy.min_volume_for_venue(venue)
        core_pool = [
            candidate
            for candidate in venue_candidates
            if candidate.metadata.market_id not in chosen_target_ids
            and candidate.volume_24h is not None
            and candidate.volume_24h >= threshold
        ]
        core_pool.sort(key=_volume_sort_key)
        chosen_core = core_pool[: active_policy.max_core_per_venue]
        core_selected.extend(
            SelectedMarket(candidate=candidate, selection_reason="volume_core")
            for candidate in chosen_core
        )

    return target_selected + core_selected
