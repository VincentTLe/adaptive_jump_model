"""End-to-end checks for the repository handoff audit trail."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RENDER = ROOT / ".agent" / "render_log.py"
HANDOFF = ROOT / ".agent" / "handoff.sh"


def canonical_entry() -> dict[str, object]:
    return {
        "ts": "2026-07-11T12:00Z",
        "agent": "codex",
        "model": "test-model",
        "goal": "Exercise handoff validation",
        "files": ["example.py"],
        "verification": ["pytest: passed"],
        "commit": "uncommitted",
        "next": "Review the result",
        "notes": "Temporary test entry",
    }


def run_renderer(agent_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RENDER), str(agent_dir), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_validate_entry_requires_canonical_fields_and_types(tmp_path: Path) -> None:
    agent_dir = tmp_path / ".agent"
    valid = json.dumps(canonical_entry())
    assert run_renderer(agent_dir, "--validate-entry", valid).returncode == 0

    missing = canonical_entry()
    del missing["model"]
    result = run_renderer(agent_dir, "--validate-entry", json.dumps(missing))
    assert result.returncode != 0
    assert "missing required field: model" in result.stderr

    wrong_type = canonical_entry()
    wrong_type["verification"] = "pytest: passed"
    result = run_renderer(agent_dir, "--validate-entry", json.dumps(wrong_type))
    assert result.returncode != 0
    assert "verification must be a list of strings" in result.stderr


def test_renderer_fails_loudly_on_corrupt_jsonl(tmp_path: Path) -> None:
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir()
    log = agent_dir / "session-log.jsonl"
    log.write_text(json.dumps(canonical_entry()) + "\n{broken\n", encoding="utf-8")

    result = run_renderer(agent_dir, "--check-log")

    assert result.returncode != 0
    assert f"{log}:2: invalid JSON" in result.stderr
    assert not (agent_dir / "session-log.html").exists()


def test_renderer_supports_legacy_and_canonical_entries(tmp_path: Path) -> None:
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir()
    canonical = canonical_entry()
    legacy = canonical_entry()
    del legacy["model"]
    legacy["next_step"] = legacy.pop("next")
    legacy["verification"] = "legacy verification"
    log = agent_dir / "session-log.jsonl"
    log.write_text(
        "\n".join((json.dumps(legacy), json.dumps(canonical))) + "\n",
        encoding="utf-8",
    )

    result = run_renderer(agent_dir)

    assert result.returncode == 0, result.stderr
    rendered = (agent_dir / "session-log.html").read_text(encoding="utf-8")
    assert "Review the result" in rendered
    assert "legacy verification" in rendered
    assert "<li>pytest: passed</li>" in rendered
    assert "['pytest: passed']" not in rendered


def test_handoff_rejects_bad_input_before_append(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir()
    log = agent_dir / "session-log.jsonl"

    bad = canonical_entry()
    bad["files"] = [1]
    result = subprocess.run(
        ["bash", str(HANDOFF), json.dumps(bad)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "files must be a list of strings" in result.stderr
    assert not log.exists() or log.read_text(encoding="utf-8") == ""

    valid = json.dumps(canonical_entry())
    result = subprocess.run(
        ["bash", str(HANDOFF), valid],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [valid]
