"""Smoke tests verifying package import and metadata."""

import pm_quant


def test_package_version() -> None:
    """Verify pm_quant imports cleanly and exposes the expected version."""
    assert pm_quant.__version__ == "0.1.0"
