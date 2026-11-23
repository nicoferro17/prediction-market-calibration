"""Offline tests validating GitHub Actions workflow definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[1]
DAILY_PIPELINE_PATH = REPO_ROOT / ".github" / "workflows" / "daily_pipeline.yml"


def _load_workflow(path: Path) -> dict[Any, Any]:
    raw = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw)
    assert isinstance(loaded, dict)
    return loaded


def test_daily_pipeline_workflow_schedule_concurrency_and_secrets() -> None:
    workflow = _load_workflow(DAILY_PIPELINE_PATH)
    raw_text = DAILY_PIPELINE_PATH.read_text(encoding="utf-8")

    # PyYAML 1.1 parses unquoted top-level key `on:` as boolean True.
    triggers = workflow.get("on") or workflow.get(True)
    assert isinstance(triggers, dict)

    # Exact UTC schedule cron: 15:17 UTC daily.
    schedule = triggers.get("schedule")
    assert isinstance(schedule, list)
    assert any(isinstance(entry, dict) and entry.get("cron") == "17 15 * * *" for entry in schedule)

    # Manual workflow_dispatch with dry_run boolean input defaulting to false.
    dispatch = triggers.get("workflow_dispatch")
    assert isinstance(dispatch, dict)
    dry_run_input = dispatch["inputs"]["dry_run"]
    assert dry_run_input["type"] == "boolean"
    assert dry_run_input["default"] is False

    # Concurrency guard and minimal permissions.
    concurrency = workflow.get("concurrency")
    assert concurrency == {
        "group": "daily-pipeline",
        "cancel-in-progress": False,
    }
    assert workflow.get("permissions") == {"contents": "read"}

    # Single sequential job with expected runner, timeout, action versions, and steps.
    jobs = workflow["jobs"]
    assert list(jobs.keys()) == ["daily-pipeline"]
    job = jobs["daily-pipeline"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 15

    steps: list[dict[str, Any]] = job["steps"]
    uses_values = [step.get("uses") for step in steps if "uses" in step]
    assert "actions/checkout@v7" in uses_values
    assert "astral-sh/setup-uv@v10.4.0" in uses_values
    assert "actions/upload-artifact@v7" in uses_values

    # Collection step writes local Parquet only (never uploads before DuckDB verification).
    step_names = [step.get("name") for step in steps]
    collect_idx = step_names.index("Collect daily L2 snapshots")
    verify_idx = step_names.index("Verify generated Parquet with DuckDB")
    hf_upload_idx = step_names.index("Upload verified Parquet to Hugging Face")
    assert collect_idx < verify_idx < hf_upload_idx

    collect_step = steps[collect_idx]
    collect_env = collect_step["env"]
    assert "HF_TOKEN" not in collect_env
    assert "--upload-hf" not in collect_step["run"]
    assert "--fail-on-partial" in collect_step["run"]
    assert "--dry-run" in collect_step["run"]

    # DuckDB verification step runs with if: always() BEFORE HF upload and writes summary.
    verify_step = steps[verify_idx]
    assert verify_step.get("if") == "always()"
    assert "scripts/verify_l2_snapshot.py" in verify_step["run"]
    assert "--write-step-summary" in verify_step["run"]

    # HF upload runs strictly AFTER DuckDB verification passes and never on dry_run=true.
    hf_upload_step = steps[hf_upload_idx]
    assert hf_upload_step.get("if") == "${{ success() && inputs.dry_run != true }}"
    hf_env = hf_upload_step["env"]
    assert hf_env["HF_TOKEN"] == "${{ secrets.HF_TOKEN }}"
    assert hf_env["HF_DATASET_REPO"] == "${{ vars.HF_DATASET_REPO }}"
    assert "--upload-only" in hf_upload_step["run"]

    # Artifact upload step has retention-days: 7 and never uploads parquet files.
    upload_step = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v7")
    assert upload_step.get("if") == "always()"
    assert upload_step["with"]["retention-days"] == 7
    assert ".parquet" not in str(upload_step["with"]["path"])
    assert "secrets.HF_TOKEN" in raw_text
