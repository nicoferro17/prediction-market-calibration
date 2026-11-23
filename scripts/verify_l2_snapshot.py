"""DuckDB-backed invariant verifier for daily L2 orderbook Parquet snapshots.

Reusable across both daily CI verification (Week 2) and multi-day historical
dataset audits (Week 6+). Reads one or more Parquet files via DuckDB
``read_parquet(..., union_by_name=true)`` and checks six core invariants:

  (a) ``schema_version == 'v1'`` on 100% of rows (with at least 1 row).
  (b) ``max(len(bid_prices)) <= 20`` and ``max(len(ask_prices)) <= 20``.
  (c) All non-null ``best_bid`` / ``best_ask`` values lie strictly in ``(0, 1)``,
      with ``bid_prices`` sorted descending and ``ask_prices`` sorted ascending.
  (d) Zero duplicates on ``(snapshot_date, venue, market_id, outcome_side)``.
  (e) Zero ``KXMVE*`` combo tickers and zero short-term 5m/15m crypto markets.
  (f) Expected venue presence and valid ``target_series`` / ``volume_core`` breakdown.

Run with:
    uv run python scripts/verify_l2_snapshot.py "data/l2_snapshots/*.parquet"
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import duckdb

from pm_quant.ingest.schemas import SCHEMA_VERSION, MarketMetadata, Venue
from pm_quant.ingest.storage import MAX_L2_MAX_DEPTH
from pm_quant.ingest.universe import is_short_term_crypto

DEFAULT_PARQUET_GLOB: Final[str] = "data/l2_snapshots/*.parquet"
DEFAULT_EXPECTED_VENUES: Final[tuple[str, ...]] = ("polymarket", "kalshi")
_VALID_VENUES: Final[frozenset[str]] = frozenset({"polymarket", "kalshi"})
_VALID_SELECTION_REASONS: Final[frozenset[str]] = frozenset({"target_series", "volume_core"})

_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "snapshot_date",
    "venue",
    "market_id",
    "event_id",
    "title",
    "category",
    "outcome_side",
    "selection_reason",
    "best_bid",
    "best_ask",
    "bid_prices",
    "ask_prices",
)


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Structured result of verifying one or more L2 snapshot Parquet files."""

    matched_files: tuple[Path, ...]
    total_size_bytes: int
    total_rows: int
    rows_by_venue: dict[str, int]
    rows_by_selection_reason: dict[str, int]
    max_bid_depth: int
    max_ask_depth: int
    checks: dict[str, bool]
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return ``True`` if all invariants passed with zero failures."""
        return len(self.failures) == 0


def resolve_parquet_inputs(patterns: Sequence[str | Path]) -> tuple[Path, ...]:
    """Expand one or more file paths or glob patterns into sorted unique Parquet paths."""
    resolved: dict[Path, None] = {}
    for raw in patterns:
        text = str(raw)
        if glob.has_magic(text):
            for match in sorted(glob.glob(text, recursive=True)):
                path = Path(match)
                if path.is_file():
                    resolved[path] = None
        else:
            path = Path(text)
            if path.is_file():
                resolved[path] = None
    return tuple(sorted(resolved.keys()))


def _is_crypto_short_term_row(
    *,
    venue: str | None,
    market_id: str | None,
    event_id: str | None,
    title: str | None,
    category: str | None,
) -> bool:
    """Check whether a snapshot row represents an excluded 5m/15m crypto market."""
    normalized_venue: Venue = cast(
        Venue,
        venue if venue in _VALID_VENUES else "polymarket",
    )
    meta = MarketMetadata(
        venue=normalized_venue,
        market_id=market_id or "",
        event_id=event_id,
        title=title or "",
        category=category,
        outcomes=("Yes", "No"),
        is_yes_no=True,
        end_time_utc=None,
        status="active",
    )
    return is_short_term_crypto(meta)


def verify_l2_snapshots(
    parquet_inputs: str | Path | Sequence[str | Path],
    *,
    expected_venues: Sequence[str] = DEFAULT_EXPECTED_VENUES,
    max_depth: int = MAX_L2_MAX_DEPTH,
    expected_schema_version: str = SCHEMA_VERSION,
) -> VerificationReport:
    """Verify the 6 L2 snapshot invariants via DuckDB ``union_by_name=true``."""
    patterns: tuple[str | Path, ...] = (
        (parquet_inputs,) if isinstance(parquet_inputs, str | Path) else tuple(parquet_inputs)
    )
    matched_files = resolve_parquet_inputs(patterns)
    if not matched_files:
        raise FileNotFoundError(
            f"No Parquet files matched input(s): {[str(item) for item in patterns]}"
        )

    total_size_bytes = sum(path.stat().st_size for path in matched_files)
    read_arg: str | list[str] = (
        str(patterns[0])
        if len(patterns) == 1 and glob.has_magic(str(patterns[0]))
        else [str(path) for path in matched_files]
    )

    failures: list[str] = []
    checks: dict[str, bool] = {
        "schema_version_v1": False,
        "max_depth_le_20": False,
        "price_range_and_ordering": False,
        "no_duplicates": False,
        "no_excluded_markets": False,
        "venue_and_reason_breakdown": False,
    }

    with duckdb.connect(":memory:") as con:
        con.execute("SET TimeZone = 'UTC'")
        rel = con.sql(
            "SELECT * FROM read_parquet(?, union_by_name=true)",
            params=[read_arg],
        )
        columns = set(rel.columns)
        missing_cols = [col for col in _REQUIRED_COLUMNS if col not in columns]
        if missing_cols:
            failures.append(f"Missing required column(s): {missing_cols}")
            return VerificationReport(
                matched_files=matched_files,
                total_size_bytes=total_size_bytes,
                total_rows=0,
                rows_by_venue={},
                rows_by_selection_reason={},
                max_bid_depth=0,
                max_ask_depth=0,
                checks=checks,
                failures=tuple(failures),
            )

        con.register("snapshots", rel)

        total_rows_row = con.execute("SELECT COUNT(*) FROM snapshots").fetchone()
        total_rows = int(total_rows_row[0]) if total_rows_row is not None else 0

        # (a) schema_version == expected_schema_version on 100% of rows (and non-empty).
        bad_schema_row = con.execute(
            "SELECT COUNT(*) FROM snapshots WHERE schema_version IS NULL OR schema_version <> ?",
            [expected_schema_version],
        ).fetchone()
        bad_schema_count = int(bad_schema_row[0]) if bad_schema_row is not None else 0
        if total_rows == 0:
            failures.append("(a) Snapshot dataset is empty (0 rows).")
        elif bad_schema_count > 0:
            failures.append(
                f"(a) Found {bad_schema_count} row(s) with schema_version != "
                f"{expected_schema_version!r} (or NULL)."
            )
        else:
            checks["schema_version_v1"] = True

        # (b) max(len(bid_prices)) <= max_depth and max(len(ask_prices)) <= max_depth.
        depth_row = con.execute(
            "SELECT "
            "  COALESCE(MAX(len(bid_prices)), 0), "
            "  COALESCE(MAX(len(ask_prices)), 0), "
            "  COUNT(*) FILTER (WHERE bid_prices IS NULL OR ask_prices IS NULL) "
            "FROM snapshots"
        ).fetchone()
        max_bid_depth = int(depth_row[0]) if depth_row is not None else 0
        max_ask_depth = int(depth_row[1]) if depth_row is not None else 0
        null_price_lists = int(depth_row[2]) if depth_row is not None else 0
        if null_price_lists > 0:
            failures.append(
                f"(b) Found {null_price_lists} row(s) with NULL bid_prices or ask_prices."
            )
        elif max_bid_depth > max_depth or max_ask_depth > max_depth:
            failures.append(
                f"(b) Max L2 depth exceeds {max_depth}: "
                f"max_bid_depth={max_bid_depth}, max_ask_depth={max_ask_depth}."
            )
        else:
            checks["max_depth_le_20"] = True

        # (c) Non-null best_bid/best_ask and all inner levels in (0, 1), top-of-book
        #     consistency, and bids desc / asks asc ordering.
        price_row = con.execute(
            "SELECT "
            "  COUNT(*) FILTER ("
            "    WHERE (best_bid IS NOT NULL AND (best_bid <= 0 OR best_bid >= 1)) "
            "       OR (best_ask IS NOT NULL AND (best_ask <= 0 OR best_ask >= 1)) "
            "       OR (bid_prices IS NOT NULL AND COALESCE("
            "            list_bool_or(list_transform(bid_prices, "
            "              lambda x: x IS NULL OR x <= 0 OR x >= 1)), FALSE)) "
            "       OR (ask_prices IS NOT NULL AND COALESCE("
            "            list_bool_or(list_transform(ask_prices, "
            "              lambda x: x IS NULL OR x <= 0 OR x >= 1)), FALSE))"
            "  ), "
            "  COUNT(*) FILTER ("
            "    WHERE (bid_prices IS NOT NULL AND ("
            "            (len(bid_prices) = 0 AND best_bid IS NOT NULL) "
            "            OR (len(bid_prices) > 0 AND ("
            "                 best_bid IS NULL OR bid_prices[1] IS NULL "
            "                 OR best_bid <> bid_prices[1]))"
            "          )) "
            "       OR (ask_prices IS NOT NULL AND ("
            "            (len(ask_prices) = 0 AND best_ask IS NOT NULL) "
            "            OR (len(ask_prices) > 0 AND ("
            "                 best_ask IS NULL OR ask_prices[1] IS NULL "
            "                 OR best_ask <> ask_prices[1]))"
            "          ))"
            "  ), "
            "  COUNT(*) FILTER ("
            "    WHERE bid_prices IS NOT NULL AND list_reverse_sort(bid_prices) <> bid_prices"
            "  ), "
            "  COUNT(*) FILTER ("
            "    WHERE ask_prices IS NOT NULL AND list_sort(ask_prices) <> ask_prices"
            "  ) "
            "FROM snapshots"
        ).fetchone()
        out_of_range_count = int(price_row[0]) if price_row is not None else 0
        top_mismatch_count = int(price_row[1]) if price_row is not None else 0
        unsorted_bids_count = int(price_row[2]) if price_row is not None else 0
        unsorted_asks_count = int(price_row[3]) if price_row is not None else 0
        price_ok = True
        if out_of_range_count > 0:
            failures.append(
                f"(c) Found {out_of_range_count} row(s) with price level(s) outside (0, 1) or NULL."
            )
            price_ok = False
        if top_mismatch_count > 0:
            failures.append(
                f"(c) Found {top_mismatch_count} row(s) with best_bid/best_ask inconsistent "
                "with bid_prices/ask_prices."
            )
            price_ok = False
        if unsorted_bids_count > 0:
            failures.append(
                f"(c) Found {unsorted_bids_count} row(s) with bid_prices not sorted descending."
            )
            price_ok = False
        if unsorted_asks_count > 0:
            failures.append(
                f"(c) Found {unsorted_asks_count} row(s) with ask_prices not sorted ascending."
            )
            price_ok = False
        checks["price_range_and_ordering"] = price_ok

        # (d) Zero duplicates on (snapshot_date, venue, market_id, outcome_side).
        dup_row = con.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT snapshot_date, venue, market_id, outcome_side "
            "  FROM snapshots "
            "  GROUP BY snapshot_date, venue, market_id, outcome_side "
            "  HAVING COUNT(*) > 1"
            ")"
        ).fetchone()
        duplicate_keys = int(dup_row[0]) if dup_row is not None else 0
        if duplicate_keys > 0:
            failures.append(
                f"(d) Found {duplicate_keys} duplicate key(s) on "
                "(snapshot_date, venue, market_id, outcome_side)."
            )
        else:
            checks["no_duplicates"] = True

        # (e) Zero KXMVE*/KXMULTI* combo tickers and zero short-term 5m/15m crypto markets.
        combo_row = con.execute(
            "SELECT COUNT(*) FROM snapshots "
            "WHERE UPPER(COALESCE(market_id, '')) LIKE 'KXMVE%' "
            "   OR UPPER(COALESCE(event_id, '')) LIKE 'KXMVE%' "
            "   OR UPPER(COALESCE(market_id, '')) LIKE 'KXMULTI%' "
            "   OR UPPER(COALESCE(event_id, '')) LIKE 'KXMULTI%'"
        ).fetchone()
        combo_count = int(combo_row[0]) if combo_row is not None else 0

        distinct_markets = con.execute(
            "SELECT DISTINCT venue, market_id, event_id, title, category FROM snapshots"
        ).fetchall()
        crypto_short_term_ids = sorted(
            str(market_id)
            for venue, market_id, event_id, title, category in distinct_markets
            if _is_crypto_short_term_row(
                venue=venue,
                market_id=market_id,
                event_id=event_id,
                title=title,
                category=category,
            )
        )
        exclusions_ok = True
        if combo_count > 0:
            failures.append(
                f"(e) Found {combo_count} row(s) matching excluded KXMVE*/KXMULTI* prefix."
            )
            exclusions_ok = False
        if crypto_short_term_ids:
            sample = ", ".join(crypto_short_term_ids[:5])
            failures.append(
                f"(e) Found {len(crypto_short_term_ids)} excluded 5m/15m crypto market(s) "
                f"(sample: {sample})."
            )
            exclusions_ok = False
        checks["no_excluded_markets"] = exclusions_ok

        # (f) Venue presence + target_series vs volume_core breakdown.
        venue_rows = con.execute(
            "SELECT COALESCE(venue, '<NULL>'), COUNT(*) FROM snapshots GROUP BY 1 ORDER BY 1"
        ).fetchall()
        rows_by_venue = {str(venue): int(count) for venue, count in venue_rows}

        reason_rows = con.execute(
            "SELECT COALESCE(selection_reason, '<NULL>'), COUNT(*) "
            "FROM snapshots GROUP BY 1 ORDER BY 1"
        ).fetchall()
        rows_by_selection_reason = {str(reason): int(count) for reason, count in reason_rows}

        breakdown_ok = True
        unexpected_venues = sorted(v for v in rows_by_venue if v not in _VALID_VENUES)
        if unexpected_venues:
            failures.append(f"(f) Found unexpected venue value(s): {unexpected_venues}.")
            breakdown_ok = False

        missing_venues = [v for v in expected_venues if rows_by_venue.get(v, 0) <= 0]
        if missing_venues:
            failures.append(f"(f) Missing expected venue(s) in snapshot: {missing_venues}.")
            breakdown_ok = False

        unexpected_reasons = sorted(
            r for r in rows_by_selection_reason if r not in _VALID_SELECTION_REASONS
        )
        if unexpected_reasons:
            failures.append(
                f"(f) Found unexpected selection_reason value(s): {unexpected_reasons}."
            )
            breakdown_ok = False
        checks["venue_and_reason_breakdown"] = breakdown_ok

    return VerificationReport(
        matched_files=matched_files,
        total_size_bytes=total_size_bytes,
        total_rows=total_rows,
        rows_by_venue=rows_by_venue,
        rows_by_selection_reason=rows_by_selection_reason,
        max_bid_depth=max_bid_depth,
        max_ask_depth=max_ask_depth,
        checks=checks,
        failures=tuple(failures),
    )


def format_markdown_summary(report: VerificationReport) -> str:
    """Render a compact Markdown verification summary for CI step summaries and artifacts."""
    size_kb = report.total_size_bytes / 1024.0
    status = "PASS" if report.ok else "FAIL"
    check_labels: tuple[tuple[str, str], ...] = (
        ("schema_version_v1", "(a) `schema_version == 'v1'` on 100% of rows"),
        ("max_depth_le_20", "(b) `max(len(bid_prices)) <= 20` and `ask_prices <= 20`"),
        (
            "price_range_and_ordering",
            "(c) Non-null `best_bid`/`best_ask` in `(0, 1)` & bids desc / asks asc",
        ),
        (
            "no_duplicates",
            "(d) Zero duplicates on `(snapshot_date, venue, market_id, outcome_side)`",
        ),
        ("no_excluded_markets", "(e) Zero `KXMVE*`/`KXMULTI*` and zero 5m/15m crypto markets"),
        (
            "venue_and_reason_breakdown",
            "(f) Venue presence & `target_series`/`volume_core` breakdown",
        ),
    )

    target_rows = report.rows_by_selection_reason.get("target_series", 0)
    core_rows = report.rows_by_selection_reason.get("volume_core", 0)
    lines: list[str] = [
        f"## L2 Snapshot Verification ({status})",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Files verified | {len(report.matched_files)} |",
        f"| Total size | {size_kb:.2f} KB ({report.total_size_bytes} B) |",
        f"| Total rows | {report.total_rows} |",
        f"| Rows (`polymarket`) | {report.rows_by_venue.get('polymarket', 0)} |",
        f"| Rows (`kalshi`) | {report.rows_by_venue.get('kalshi', 0)} |",
        f"| Breakdown (`target_series`) | {target_rows} |",
        f"| Breakdown (`volume_core`) | {core_rows} |",
        f"| Max bid depth | {report.max_bid_depth} |",
        f"| Max ask depth | {report.max_ask_depth} |",
        "",
        "| Invariant | Status |",
        "| --- | --- |",
    ]
    for key, label in check_labels:
        check_status = "PASS" if report.checks.get(key, False) else "FAIL"
        lines.append(f"| {label} | {check_status} |")

    if report.failures:
        lines.extend(["", "### Failures"])
        for failure in report.failures:
            lines.append(f"- {failure}")

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``verify_l2_snapshot.py``."""
    parser = argparse.ArgumentParser(
        prog="verify_l2_snapshot.py",
        description=(
            "Verify L2 orderbook Parquet snapshots against the 6 core DuckDB invariants "
            "and emit a compact Markdown summary."
        ),
    )
    parser.add_argument(
        "parquet_inputs",
        nargs="*",
        default=[DEFAULT_PARQUET_GLOB],
        help=f"Parquet file path(s) or glob pattern(s) (default: {DEFAULT_PARQUET_GLOB}).",
    )
    parser.add_argument(
        "--expected-venues",
        type=str,
        default=",".join(DEFAULT_EXPECTED_VENUES),
        help=(
            "Comma-separated list of required venues (default: "
            f"{','.join(DEFAULT_EXPECTED_VENUES)})."
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=MAX_L2_MAX_DEPTH,
        help=f"Maximum allowed L2 depth per side (default: {MAX_L2_MAX_DEPTH}).",
    )
    parser.add_argument(
        "--schema-version",
        type=str,
        default=SCHEMA_VERSION,
        help=f"Expected schema_version value on 100%% of rows (default: {SCHEMA_VERSION}).",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=None,
        help="Optional file path where the Markdown verification summary is written.",
    )
    parser.add_argument(
        "--write-step-summary",
        action="store_true",
        default=False,
        help="Append the Markdown summary to $GITHUB_STEP_SUMMARY when the env var is set.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run L2 snapshot verification CLI and return 0 on pass or 1 on failure."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    expected_venues = tuple(
        part.strip() for part in args.expected_venues.split(",") if part.strip()
    )

    try:
        report = verify_l2_snapshots(
            args.parquet_inputs,
            expected_venues=expected_venues,
            max_depth=args.max_depth,
            expected_schema_version=args.schema_version,
        )
    except Exception as exc:
        print(f"ERROR: L2 snapshot verification failed: {exc}", file=sys.stderr)
        return 1

    summary_md = format_markdown_summary(report)
    print(summary_md, end="")

    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(summary_md, encoding="utf-8")

    if args.write_step_summary:
        step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary_path:
            with Path(step_summary_path).open("a", encoding="utf-8") as handle:
                handle.write(summary_md)

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
