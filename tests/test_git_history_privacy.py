"""Fabricated repositories verify that old data cannot hide behind a clean tip."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

GUARD = Path(__file__).resolve().parents[1] / "scripts/check_git_privacy.py"
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="Git is not installed")


def git(directory, *args):
    return subprocess.check_output(
        ["git", "-c", "commit.gpgsign=false", *args], cwd=directory, stderr=subprocess.PIPE
    ).decode("utf-8")


def initialize(directory):
    git(directory, "init", "-q")
    git(directory, "config", "user.name", "Synthetic privacy test")
    git(directory, "config", "user.email", "test@example.invalid")


def guard(directory, *args, input=None):
    return subprocess.run(
        [sys.executable, str(GUARD), *args],
        cwd=directory,
        input=input,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_deleted_data_in_ancestor_still_blocks_push(tmp_path):
    initialize(tmp_path)
    (tmp_path / "data").mkdir()
    private = tmp_path / "data/fabricated.csv"
    private.write_text("fictional,value\nmade_up,1\n", encoding="utf-8")
    git(tmp_path, "add", "data/fabricated.csv")
    git(tmp_path, "commit", "-qm", "Fabricated historical data")
    private.unlink()
    (tmp_path / "README.md").write_text("Only artificial test material.\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "Remove fabricated data from tip")
    assert guard(tmp_path, "--all-tracked").returncode == 0
    head = git(tmp_path, "rev-parse", "HEAD").strip()
    proposed = f"refs/heads/test {head} refs/heads/test {'0' * 40}\n"
    result = guard(tmp_path, "--pre-push", input=proposed)
    assert result.returncode == 1 and "Private or unexpected path" in result.stderr
    assert "made_up,1" not in result.stderr


def test_clean_initial_history_passes_push_check(tmp_path):
    initialize(tmp_path)
    (tmp_path / "README.md").write_text("Synthetic code-only repository.\n", encoding="utf-8")
    git(tmp_path, "add", "README.md")
    git(tmp_path, "commit", "-qm", "Initial source commit")
    head = git(tmp_path, "rev-parse", "HEAD").strip()
    assert guard(tmp_path, "--history", "HEAD").returncode == 0
    proposed = f"refs/heads/test {head} refs/heads/test {'0' * 40}\n"
    assert guard(tmp_path, "--pre-push", input=proposed).returncode == 0
    internal = f"refs/codex/local {head} refs/codex/local {'0' * 40}\n"
    assert guard(tmp_path, "--pre-push", input=internal).returncode == 1


def test_credential_detection_never_echoes_value(tmp_path):
    initialize(tmp_path)
    fake_token = "ghp_" + "A" * 40
    (tmp_path / "README.md").write_text(fake_token, encoding="utf-8")
    git(tmp_path, "add", "README.md")
    result = guard(tmp_path, "--all-tracked")
    assert result.returncode == 1 and "Possible credential" in result.stderr
    assert fake_token not in result.stdout + result.stderr


def test_unknown_and_binary_file_types_fail_closed():
    spec = importlib.util.spec_from_file_location("privacy_guard_for_test", GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        "src/checkpoint.ckpt",
        "DAT/src/model.safetensors",
        "configs/derived.json",
        "tests/clinical.feather",
        "docs/unknown.bin",
        "src/results/metrics.yaml",
        "src/../data/private.py",
        "data/processed/data.csv",
    ):
        assert not module.allowed(name)
