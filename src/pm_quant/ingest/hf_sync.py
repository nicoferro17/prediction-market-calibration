"""Idempotent daily Parquet snapshot upload to Hugging Face Datasets.

Validates daily L2 orderbook Parquet files locally via DuckDB before making any
network request, then uploads them to a deterministic ``<path_prefix>/YYYY-MM-DD.parquet``
path inside a Hugging Face Dataset repository (1 idempotent commit per day).
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date
from pathlib import Path
from typing import Final, Protocol

import duckdb
from huggingface_hub import HfApi

from pm_quant.ingest.storage import read_snapshots_duckdb

logger = logging.getLogger(__name__)

HF_DATASET_REPO_ENV: Final[str] = "HF_DATASET_REPO"
HF_TOKEN_ENV: Final[str] = "HF_TOKEN"
DEFAULT_HF_PATH_PREFIX: Final[str] = "l2_snapshots"

_DAILY_PARQUET_FILENAME_RE: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}\.parquet$")


class HfUploadApi(Protocol):
    """Minimal Hugging Face Hub upload interface for dependency injection and offline tests."""

    def upload_file(
        self,
        *,
        path_or_fileobj: str,
        path_in_repo: str,
        repo_id: str,
        repo_type: str | None = None,
        commit_message: str | None = None,
    ) -> object:
        """Upload a single local file to a Hugging Face Hub repository."""


def _sanitize_secret(message: str, secret: str) -> str:
    """Redact ``secret`` from ``message`` if present."""
    if secret and secret in message:
        return message.replace(secret, "[REDACTED]")
    return message


def validate_daily_parquet_for_upload(local_path: Path) -> int:
    """Validate a daily L2 Parquet snapshot before uploading to Hugging Face.

    Verifies:
    - The file exists and is a regular file.
    - The filename strictly matches ``YYYY-MM-DD.parquet`` with a valid ISO calendar date.
    - The file size is strictly greater than 0 bytes.
    - DuckDB can read the Parquet file cleanly.
    - The ``schema_version`` column is present and contains no null values.
    - The dataset contains at least 1 row.

    Returns:
        The validated row count.
    """
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot Parquet file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Snapshot path is not a regular file: {path}")

    if _DAILY_PARQUET_FILENAME_RE.match(path.name) is None:
        raise ValueError(
            f"Snapshot filename must strictly match YYYY-MM-DD.parquet, got {path.name!r}"
        )
    try:
        date.fromisoformat(path.stem)
    except ValueError as exc:
        raise ValueError(f"Invalid calendar date in snapshot filename {path.name!r}") from exc

    if path.stat().st_size <= 0:
        raise ValueError(f"Snapshot Parquet file is empty (0 bytes): {path.name}")

    try:
        frame = read_snapshots_duckdb(path)
    except duckdb.Error as exc:
        raise ValueError(
            f"Failed to read snapshot Parquet via DuckDB ({path.name}): {exc}"
        ) from exc

    if "schema_version" not in frame.columns:
        raise ValueError(f"Snapshot {path.name} is missing required column 'schema_version'")

    row_count = frame.height
    if row_count < 1:
        raise ValueError(f"Snapshot {path.name} contains 0 rows")

    null_count = frame["schema_version"].null_count()
    if null_count > 0:
        raise ValueError(
            f"Snapshot {path.name} contains {null_count} null value(s) in column 'schema_version'"
        )

    return row_count


def upload_daily_parquet(
    local_path: Path,
    *,
    repo_id: str | None = None,
    token: str | None = None,
    path_prefix: str = DEFAULT_HF_PATH_PREFIX,
    api: HfUploadApi | None = None,
) -> str:
    """Validate and upload a daily L2 Parquet snapshot to a Hugging Face Dataset repo.

    Re-running for the same ``YYYY-MM-DD.parquet`` file overwrites the same remote path
    in a single idempotent commit. Never logs or exposes the authentication token.

    Returns:
        The remote commit URL (or ``path_in_repo`` if the API does not return a URL).
    """
    raw_repo_id = repo_id if repo_id is not None else os.environ.get(HF_DATASET_REPO_ENV)
    resolved_repo_id = raw_repo_id.strip() if raw_repo_id is not None else ""
    if not resolved_repo_id:
        raise ValueError(
            f"Missing Hugging Face dataset repository ID: pass repo_id or set {HF_DATASET_REPO_ENV}"
        )

    raw_token = token if token is not None else os.environ.get(HF_TOKEN_ENV)
    resolved_token = raw_token.strip() if raw_token is not None else ""
    if not resolved_token:
        raise ValueError(
            f"Missing Hugging Face authentication token: pass token or set {HF_TOKEN_ENV}"
        )

    path = Path(local_path)
    row_count = validate_daily_parquet_for_upload(path)

    clean_prefix = path_prefix.strip().strip("/")
    path_in_repo = f"{clean_prefix}/{path.name}" if clean_prefix else path.name
    commit_message = f"chore(data): upload L2 snapshot {path.stem}"

    client: HfUploadApi = api if api is not None else HfApi(token=resolved_token)

    logger.info(
        "Uploading L2 snapshot %s (%d rows) to Hugging Face dataset %s at %s",
        path.name,
        row_count,
        resolved_repo_id,
        path_in_repo,
    )
    try:
        result = client.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=resolved_repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )
    except Exception as exc:
        sanitized = _sanitize_secret(str(exc), resolved_token)
        logger.error(
            "Hugging Face upload failed for %s -> %s (%s): %s",
            path.name,
            resolved_repo_id,
            path_in_repo,
            sanitized,
        )
        if sanitized != str(exc):
            raise RuntimeError(
                f"Hugging Face upload failed for {path_in_repo}: {sanitized}"
            ) from None
        raise

    commit_url = getattr(result, "commit_url", None)
    if isinstance(commit_url, str) and commit_url:
        upload_ref = _sanitize_secret(commit_url, resolved_token)
    elif isinstance(result, str) and result:
        upload_ref = _sanitize_secret(result, resolved_token)
    else:
        upload_ref = path_in_repo

    logger.info(
        "Uploaded L2 snapshot %s to Hugging Face dataset %s (%s)",
        path.name,
        resolved_repo_id,
        upload_ref,
    )
    return upload_ref
