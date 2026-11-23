"""100% offline tests for Hugging Face Datasets sync and CLI upload integration."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl
import pytest

from pm_quant.ingest import (
    HF_DATASET_REPO_ENV,
    HF_TOKEN_ENV,
    SCHEMA_VERSION,
    CollectionSummary,
    upload_daily_parquet,
    validate_daily_parquet_for_upload,
)

SENTINEL_TOKEN = "hf_SECRET_TEST_TOKEN"
TEST_REPO_ID = "pm-quant-org/l2-snapshots"


@dataclass(frozen=True)
class _FakeCommitInfo:
    commit_url: str


class _FakeHfApi:
    """Offline fake matching ``HfUploadApi``."""

    def __init__(
        self,
        *,
        return_value: object = _FakeCommitInfo(
            commit_url="https://huggingface.co/datasets/pm-quant-org/l2-snapshots/commit/abc123"
        ),
        error: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._return_value = return_value
        self._error = error

    def upload_file(
        self,
        *,
        path_or_fileobj: str,
        path_in_repo: str,
        repo_id: str,
        repo_type: str | None = None,
        commit_message: str | None = None,
    ) -> object:
        self.calls.append(
            {
                "path_or_fileobj": path_or_fileobj,
                "path_in_repo": path_in_repo,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "commit_message": commit_message,
            }
        )
        if self._error is not None:
            raise self._error
        return self._return_value


def _write_valid_snapshot(path: Path, *, rows: int = 2) -> Path:
    df = pl.DataFrame(
        {
            "schema_version": [SCHEMA_VERSION] * rows,
            "venue": ["polymarket"] * rows,
            "market_id": [f"mkt-{i}" for i in range(rows)],
        }
    )
    df.write_parquet(path, compression="zstd")
    return path


def _load_cli_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "collect_daily.py"
    spec = importlib.util.spec_from_file_location("collect_daily_cli_hf", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. Valid upload & idempotency
# ---------------------------------------------------------------------------


def test_upload_daily_parquet_valid_single_call(tmp_path: Path) -> None:
    """Valid daily Parquet triggers a single idempotent dataset upload with expected metadata."""
    parquet_path = _write_valid_snapshot(tmp_path / "2026-12-01.parquet", rows=3)
    fake_api = _FakeHfApi()

    assert validate_daily_parquet_for_upload(parquet_path) == 3

    url = upload_daily_parquet(
        parquet_path,
        repo_id=TEST_REPO_ID,
        token=SENTINEL_TOKEN,
        api=fake_api,
    )

    assert url == "https://huggingface.co/datasets/pm-quant-org/l2-snapshots/commit/abc123"
    assert len(fake_api.calls) == 1
    assert fake_api.calls[0] == {
        "path_or_fileobj": str(parquet_path),
        "path_in_repo": "l2_snapshots/2026-12-01.parquet",
        "repo_id": TEST_REPO_ID,
        "repo_type": "dataset",
        "commit_message": "chore(data): upload L2 snapshot 2026-12-01",
    }


def test_upload_daily_parquet_reads_env_vars_and_custom_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falls back to HF_DATASET_REPO / HF_TOKEN env vars and normalizes path_prefix."""
    parquet_path = _write_valid_snapshot(tmp_path / "2026-12-01.parquet", rows=1)
    monkeypatch.setenv(HF_DATASET_REPO_ENV, f"  {TEST_REPO_ID}  ")
    monkeypatch.setenv(HF_TOKEN_ENV, f"  {SENTINEL_TOKEN}  ")
    fake_api = _FakeHfApi(return_value=None)

    result = upload_daily_parquet(
        parquet_path,
        path_prefix="/custom/prefix/",
        api=fake_api,
    )

    assert result == "custom/prefix/2026-12-01.parquet"
    assert len(fake_api.calls) == 1
    assert fake_api.calls[0]["path_in_repo"] == "custom/prefix/2026-12-01.parquet"
    assert fake_api.calls[0]["repo_id"] == TEST_REPO_ID


# ---------------------------------------------------------------------------
# 2. Pre-upload validation rejections (fake API must never be called)
# ---------------------------------------------------------------------------


def test_reject_nonexistent_file_never_calls_api(tmp_path: Path) -> None:
    missing = tmp_path / "2026-12-01.parquet"
    fake_api = _FakeHfApi()

    with pytest.raises(FileNotFoundError, match="does not exist"):
        upload_daily_parquet(
            missing,
            repo_id=TEST_REPO_ID,
            token=SENTINEL_TOKEN,
            api=fake_api,
        )

    assert fake_api.calls == []


@pytest.mark.parametrize(
    "bad_name",
    [
        "latest.parquet",
        "2026-12-1.parquet",
        "2026_12_01.parquet",
        "2026-13-45.parquet",
        "2026-12-01.csv",
    ],
)
def test_reject_nonconforming_filename_never_calls_api(tmp_path: Path, bad_name: str) -> None:
    parquet_path = _write_valid_snapshot(tmp_path / bad_name, rows=1)
    fake_api = _FakeHfApi()

    with pytest.raises(ValueError, match="YYYY-MM-DD|Invalid calendar date"):
        upload_daily_parquet(
            parquet_path,
            repo_id=TEST_REPO_ID,
            token=SENTINEL_TOKEN,
            api=fake_api,
        )

    assert fake_api.calls == []


def test_reject_zero_byte_and_corrupted_file_never_calls_api(tmp_path: Path) -> None:
    empty_file = tmp_path / "2026-12-01.parquet"
    empty_file.write_bytes(b"")
    fake_api = _FakeHfApi()

    with pytest.raises(ValueError, match="0 bytes"):
        upload_daily_parquet(
            empty_file,
            repo_id=TEST_REPO_ID,
            token=SENTINEL_TOKEN,
            api=fake_api,
        )

    corrupted = tmp_path / "2026-12-02.parquet"
    corrupted.write_bytes(b"not-a-valid-parquet-file")
    with pytest.raises(ValueError, match="DuckDB"):
        upload_daily_parquet(
            corrupted,
            repo_id=TEST_REPO_ID,
            token=SENTINEL_TOKEN,
            api=fake_api,
        )

    assert fake_api.calls == []


def test_reject_zero_row_parquet_never_calls_api(tmp_path: Path) -> None:
    parquet_path = tmp_path / "2026-12-01.parquet"
    pl.DataFrame(schema={"schema_version": pl.String(), "venue": pl.String()}).write_parquet(
        parquet_path
    )
    fake_api = _FakeHfApi()

    with pytest.raises(ValueError, match="0 rows"):
        upload_daily_parquet(
            parquet_path,
            repo_id=TEST_REPO_ID,
            token=SENTINEL_TOKEN,
            api=fake_api,
        )

    assert fake_api.calls == []


def test_reject_missing_or_null_schema_version_never_calls_api(tmp_path: Path) -> None:
    no_col_path = tmp_path / "2026-12-01.parquet"
    pl.DataFrame({"venue": ["polymarket"]}).write_parquet(no_col_path)
    fake_api = _FakeHfApi()

    with pytest.raises(ValueError, match="schema_version"):
        upload_daily_parquet(
            no_col_path,
            repo_id=TEST_REPO_ID,
            token=SENTINEL_TOKEN,
            api=fake_api,
        )

    null_col_path = tmp_path / "2026-12-02.parquet"
    pl.DataFrame(
        {"schema_version": [SCHEMA_VERSION, None], "venue": ["polymarket", "kalshi"]}
    ).write_parquet(null_col_path)
    with pytest.raises(ValueError, match="null"):
        upload_daily_parquet(
            null_col_path,
            repo_id=TEST_REPO_ID,
            token=SENTINEL_TOKEN,
            api=fake_api,
        )

    assert fake_api.calls == []


# ---------------------------------------------------------------------------
# 3. Fail-fast credential checks
# ---------------------------------------------------------------------------


def test_fail_fast_missing_repo_or_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_path = _write_valid_snapshot(tmp_path / "2026-12-01.parquet", rows=1)
    monkeypatch.delenv(HF_DATASET_REPO_ENV, raising=False)
    monkeypatch.delenv(HF_TOKEN_ENV, raising=False)
    fake_api = _FakeHfApi()

    with pytest.raises(ValueError, match=HF_DATASET_REPO_ENV):
        upload_daily_parquet(parquet_path, token=SENTINEL_TOKEN, api=fake_api)

    with pytest.raises(ValueError, match=HF_TOKEN_ENV):
        upload_daily_parquet(parquet_path, repo_id=TEST_REPO_ID, api=fake_api)

    assert fake_api.calls == []


# ---------------------------------------------------------------------------
# 4. Security: sentinel token must never appear in logs or exception messages
# ---------------------------------------------------------------------------


def test_sentinel_token_never_leaks_in_logs_or_exceptions(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sentinel token must never appear in caplog output or exception messages."""
    parquet_path = _write_valid_snapshot(tmp_path / "2026-12-01.parquet", rows=2)
    monkeypatch.delenv(HF_DATASET_REPO_ENV, raising=False)
    monkeypatch.delenv(HF_TOKEN_ENV, raising=False)

    with caplog.at_level(logging.DEBUG):
        # 1) Normal success path
        ok_api = _FakeHfApi()
        result = upload_daily_parquet(
            parquet_path,
            repo_id=TEST_REPO_ID,
            token=SENTINEL_TOKEN,
            api=ok_api,
        )
        assert SENTINEL_TOKEN not in result

        # 2) Missing repo_id while token is set
        with pytest.raises(ValueError) as missing_repo_exc:
            upload_daily_parquet(parquet_path, token=SENTINEL_TOKEN, api=ok_api)
        assert SENTINEL_TOKEN not in str(missing_repo_exc.value)

        # 3) Upstream API error that tries to echo the token
        leaking_api = _FakeHfApi(error=RuntimeError(f"401 Unauthorized for {SENTINEL_TOKEN}"))
        with pytest.raises(RuntimeError) as upload_exc:
            upload_daily_parquet(
                parquet_path,
                repo_id=TEST_REPO_ID,
                token=SENTINEL_TOKEN,
                api=leaking_api,
            )
        assert SENTINEL_TOKEN not in str(upload_exc.value)
        assert "[REDACTED]" in str(upload_exc.value)

    assert SENTINEL_TOKEN not in caplog.text


# ---------------------------------------------------------------------------
# 5. CLI integration (--upload-hf, --hf-repo, --dry-run, --fail-on-partial)
# ---------------------------------------------------------------------------


def test_cli_upload_hf_flags_and_behaviors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli_module()
    parquet_path = _write_valid_snapshot(tmp_path / "2026-12-01.parquet", rows=2)

    def _make_summary(*, failed_venues: tuple[str, ...] = ()) -> CollectionSummary:
        return CollectionSummary(
            snapshot_date=date(2026, 12, 1),
            ts_utc=datetime(2026, 12, 1, 14, 0, tzinfo=UTC),
            total_rows=2,
            rows_by_venue={"polymarket": 2, "kalshi": 0},
            selected_by_reason={"target_series": 0, "volume_core": 1},
            empty_books_count=0,
            one_sided_books_count=0,
            failed_venues=failed_venues,
            parquet_path=parquet_path,
            file_size_bytes=parquet_path.stat().st_size,
        )

    upload_calls: list[dict[str, Any]] = []

    def fake_upload(local_path: Path, *, repo_id: str | None = None) -> str:
        upload_calls.append({"local_path": local_path, "repo_id": repo_id})
        return "https://huggingface.co/datasets/pm-quant-org/l2-snapshots/commit/xyz"

    async def fake_collect_ok(**kwargs: Any) -> CollectionSummary:
        return _make_summary()

    monkeypatch.setattr(cli, "collect_daily_l2", fake_collect_ok)
    monkeypatch.setattr(cli, "upload_daily_parquet", fake_upload)

    # 1) --dry-run + --upload-hf -> upload skipped with explicit message
    rc_dry = cli.main(["--output-dir", str(tmp_path), "--dry-run", "--upload-hf"])
    io_dry = capsys.readouterr()
    assert rc_dry == 0
    assert upload_calls == []
    assert "Skipping Hugging Face upload" in io_dry.out
    assert "--dry-run" in io_dry.out

    # 2) --upload-hf + --hf-repo -> invokes upload_daily_parquet
    rc_up = cli.main(["--output-dir", str(tmp_path), "--upload-hf", "--hf-repo", TEST_REPO_ID])
    io_up = capsys.readouterr()
    assert rc_up == 0
    assert upload_calls == [{"local_path": parquet_path, "repo_id": TEST_REPO_ID}]
    assert "HF upload" in io_up.out

    # 3) --upload-hf + --fail-on-partial -> uploads partial Parquet BEFORE exiting != 0
    upload_calls.clear()

    async def fake_collect_partial(**kwargs: Any) -> CollectionSummary:
        return _make_summary(failed_venues=("kalshi",))

    monkeypatch.setattr(cli, "collect_daily_l2", fake_collect_partial)
    rc_partial = cli.main(
        [
            "--output-dir",
            str(tmp_path),
            "--upload-hf",
            "--hf-repo",
            TEST_REPO_ID,
            "--fail-on-partial",
        ]
    )
    io_partial = capsys.readouterr()
    assert rc_partial != 0
    assert upload_calls == [{"local_path": parquet_path, "repo_id": TEST_REPO_ID}]
    assert "--fail-on-partial" in io_partial.err

    # 4) Upload failure -> non-zero exit code with stderr error
    def fake_upload_fail(local_path: Path, *, repo_id: str | None = None) -> str:
        raise ValueError("Missing Hugging Face authentication token")

    monkeypatch.setattr(cli, "collect_daily_l2", fake_collect_ok)
    monkeypatch.setattr(cli, "upload_daily_parquet", fake_upload_fail)
    rc_fail = cli.main(["--output-dir", str(tmp_path), "--upload-hf"])
    io_fail = capsys.readouterr()
    assert rc_fail != 0
    assert "ERROR: Hugging Face upload failed:" in io_fail.err
