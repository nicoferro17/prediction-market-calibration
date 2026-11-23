"""CLI entrypoint for collecting daily L2 orderbook snapshots from Polymarket and Kalshi.

Run with:
    uv run python scripts/collect_daily.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path

from pm_quant.ingest import (
    DEFAULT_L2_MAX_DEPTH,
    MAX_L2_MAX_DEPTH,
    MIN_L2_MAX_DEPTH,
    CollectionSummary,
    UniversePolicy,
    collect_daily_l2,
    upload_daily_parquet,
)

DEFAULT_OUTPUT_DIR = Path("data/l2_snapshots")
logger = logging.getLogger(__name__)


def _parse_iso_date(raw: str) -> date:
    """Parse a strict ``YYYY-MM-DD`` UTC calendar date."""
    text = raw.strip()
    if len(text) != 10:
        raise argparse.ArgumentTypeError(f"invalid --date {raw!r}: expected YYYY-MM-DD")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid --date {raw!r}: expected YYYY-MM-DD") from exc


def _parse_max_depth(raw: str) -> int:
    """Parse and validate ``--max-depth`` within ``[MIN_L2_MAX_DEPTH, MAX_L2_MAX_DEPTH]``."""
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid --max-depth {raw!r}: must be an integer"
        ) from exc
    if not (MIN_L2_MAX_DEPTH <= value <= MAX_L2_MAX_DEPTH):
        raise argparse.ArgumentTypeError(
            f"--max-depth must be between {MIN_L2_MAX_DEPTH} and {MAX_L2_MAX_DEPTH}, got {value}"
        )
    return value


def _parse_non_negative_int(raw: str) -> int:
    """Parse a non-negative integer CLI override."""
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer value {raw!r}") from exc
    if value < 0:
        raise argparse.ArgumentTypeError(f"value must be >= 0, got {value}")
    return value


def _parse_csv_tokens(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated CLI list into deduplicated non-empty tokens."""
    seen: set[str] = set()
    tokens: list[str] = []
    for part in raw.split(","):
        cleaned = part.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            tokens.append(cleaned)
    return tuple(tokens)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for daily L2 snapshot collection."""
    parser = argparse.ArgumentParser(
        prog="collect_daily.py",
        description=(
            "Collect daily L2 orderbook snapshots from Polymarket and Kalshi "
            "into a Zstandard-compressed Parquet file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for daily Parquet snapshots (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--date",
        dest="snapshot_date",
        type=_parse_iso_date,
        default=None,
        help="Snapshot UTC date in YYYY-MM-DD format (default: today UTC).",
    )
    parser.add_argument(
        "--max-depth",
        type=_parse_max_depth,
        default=DEFAULT_L2_MAX_DEPTH,
        help=(
            f"Maximum L2 depth per side ({MIN_L2_MAX_DEPTH}..{MAX_L2_MAX_DEPTH}, "
            f"default: {DEFAULT_L2_MAX_DEPTH})."
        ),
    )
    parser.add_argument(
        "--max-core-per-venue",
        type=_parse_non_negative_int,
        default=None,
        help="Optional override for UniversePolicy.max_core_per_venue.",
    )
    parser.add_argument(
        "--max-target-series-per-venue",
        type=_parse_non_negative_int,
        default=None,
        help="Optional override for UniversePolicy.max_target_series_per_venue.",
    )
    parser.add_argument(
        "--polymarket-target-tag-ids",
        type=_parse_csv_tokens,
        default=None,
        help=(
            "Optional comma-separated Polymarket Gamma tag_id values for targeted "
            "target_series discovery."
        ),
    )
    parser.add_argument(
        "--kalshi-target-series",
        type=_parse_csv_tokens,
        default=None,
        help=(
            "Optional comma-separated exact Kalshi series_ticker values to query "
            "instead of the default policy expansion."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Write the local Parquet snapshot only without uploading to remote storage.",
    )
    parser.add_argument(
        "--upload-hf",
        action="store_true",
        default=False,
        help=(
            "Upload the written Parquet snapshot to Hugging Face Datasets "
            "(always skipped when --dry-run is active)."
        ),
    )
    parser.add_argument(
        "--upload-only",
        type=Path,
        default=None,
        help=(
            "Skip API collection and upload an already-verified local YYYY-MM-DD.parquet "
            "snapshot to Hugging Face Datasets (skipped when --dry-run is active)."
        ),
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="Hugging Face dataset repository ID (overrides HF_DATASET_REPO env var).",
    )
    parser.add_argument(
        "--fail-on-partial",
        action="store_true",
        default=False,
        help=(
            "After writing (and optionally uploading) the Parquet snapshot, exit with a "
            "non-zero code if any venue failed (for CI alerting)."
        ),
    )
    return parser


def build_policy(
    *,
    max_core_per_venue: int | None = None,
    max_target_series_per_venue: int | None = None,
    polymarket_target_tag_ids: Sequence[str] | None = None,
) -> UniversePolicy:
    """Construct a ``UniversePolicy`` applying any CLI budget or tag overrides."""
    base = UniversePolicy()
    return replace(
        base,
        max_core_per_venue=(
            max_core_per_venue if max_core_per_venue is not None else base.max_core_per_venue
        ),
        max_target_series_per_venue=(
            max_target_series_per_venue
            if max_target_series_per_venue is not None
            else base.max_target_series_per_venue
        ),
        polymarket_target_tag_ids=(
            tuple(polymarket_target_tag_ids)
            if polymarket_target_tag_ids is not None
            else base.polymarket_target_tag_ids
        ),
    )


def format_summary_table(
    summary: CollectionSummary,
    *,
    dry_run: bool = False,
    upload_hf: bool = False,
    upload_ref: str | None = None,
) -> str:
    """Format ``CollectionSummary`` as a compact aligned ASCII table."""
    size_kb = summary.file_size_bytes / 1024.0
    failed_display = ", ".join(summary.failed_venues) if summary.failed_venues else "none"
    if dry_run and upload_hf:
        mode_display = "dry-run (HF upload skipped)"
    elif dry_run:
        mode_display = "dry-run (local Parquet only)"
    elif upload_ref is not None:
        mode_display = "local Parquet + Hugging Face upload"
    else:
        mode_display = "local Parquet (no upload configured)"

    rows: list[tuple[str, str]] = [
        ("Snapshot date (UTC)", summary.snapshot_date.isoformat()),
        ("Timestamp (UTC)", summary.ts_utc.isoformat().replace("+00:00", "Z")),
        ("Mode", mode_display),
        ("Total rows", str(summary.total_rows)),
        ("Rows (polymarket)", str(summary.rows_by_venue.get("polymarket", 0))),
        ("Rows (kalshi)", str(summary.rows_by_venue.get("kalshi", 0))),
        ("Selected (target_series)", str(summary.selected_by_reason.get("target_series", 0))),
        ("Selected (volume_core)", str(summary.selected_by_reason.get("volume_core", 0))),
        ("Empty books", str(summary.empty_books_count)),
        ("One-sided books", str(summary.one_sided_books_count)),
        ("Failed venues", failed_display),
        ("Parquet path", str(summary.parquet_path)),
        ("Parquet size", f"{size_kb:.2f} KB ({summary.file_size_bytes} B)"),
    ]
    if dry_run and upload_hf:
        rows.append(("HF upload", "skipped (--dry-run)"))
    elif upload_ref is not None:
        rows.append(("HF upload", upload_ref))

    metric_header = "Metric"
    value_header = "Value"
    metric_width = max(len(metric_header), *(len(metric) for metric, _ in rows))
    value_width = max(len(value_header), *(len(value) for _, value in rows))

    header_line = f"{metric_header.ljust(metric_width)} | {value_header.ljust(value_width)}"
    sep_line = f"{'-' * metric_width}-+-{'-' * value_width}"
    body_lines = [
        f"{metric.ljust(metric_width)} | {value.ljust(value_width)}" for metric, value in rows
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def main(argv: Sequence[str] | None = None) -> int:
    """Run the daily L2 snapshot collection CLI and return an exit status code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.upload_only is not None:
        if args.dry_run:
            msg = "Skipping Hugging Face upload because --dry-run is active."
            logger.info(msg)
            print(f"INFO: {msg}")
            return 0
        try:
            upload_only_ref = upload_daily_parquet(
                args.upload_only,
                repo_id=args.hf_repo,
            )
        except Exception as exc:
            print(f"ERROR: Hugging Face upload failed: {exc}", file=sys.stderr)
            return 1
        print(f"HF upload: {upload_only_ref}")
        return 0

    policy = build_policy(
        max_core_per_venue=args.max_core_per_venue,
        max_target_series_per_venue=args.max_target_series_per_venue,
        polymarket_target_tag_ids=args.polymarket_target_tag_ids,
    )

    try:
        summary = asyncio.run(
            collect_daily_l2(
                output_dir=args.output_dir,
                policy=policy,
                max_depth=args.max_depth,
                snapshot_date=args.snapshot_date,
                kalshi_target_series=args.kalshi_target_series,
            )
        )
    except Exception as exc:
        print(f"ERROR: Daily L2 snapshot collection failed: {exc}", file=sys.stderr)
        return 1

    upload_ref: str | None = None
    if args.upload_hf and args.dry_run:
        msg = "Skipping Hugging Face upload because --dry-run is active."
        logger.info(msg)
        print(f"INFO: {msg}")
    elif args.upload_hf:
        try:
            upload_ref = upload_daily_parquet(
                summary.parquet_path,
                repo_id=args.hf_repo,
            )
        except Exception as exc:
            print(
                format_summary_table(
                    summary,
                    dry_run=args.dry_run,
                    upload_hf=args.upload_hf,
                )
            )
            print(f"ERROR: Hugging Face upload failed: {exc}", file=sys.stderr)
            return 1

    print(
        format_summary_table(
            summary,
            dry_run=args.dry_run,
            upload_hf=args.upload_hf,
            upload_ref=upload_ref,
        )
    )

    if args.fail_on_partial and summary.failed_venues:
        print(
            "ERROR: Partial collection failure (--fail-on-partial): "
            f"failed_venues={list(summary.failed_venues)}",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
