"""Smoke tests: each entrypoint runs `--help` without crashing."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ENTRYPOINTS = ["Distribution.py", "Augmentation.py", "Transformation.py", "train.py", "predict.py"]


@pytest.mark.parametrize("script", ENTRYPOINTS)
def test_entrypoint_help_exits_zero(script: str) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / script
    if not path.exists():
        pytest.skip(f"{script} not yet implemented")
    result = subprocess.run(
        ["uv", "run", "python", str(path), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"{script} --help failed: {result.stderr}"
