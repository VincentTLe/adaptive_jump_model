import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from adaptive_jump.config import SourceConfig, load_config
from adaptive_jump.data import (
    AcquisitionError,
    HttpResult,
    acquire,
    canonical_bytes,
    fetch_source,
    quality,
    research_git_sha,
)

CONFIG = load_config(
    Path(__file__).resolve().parents[1] / "configs/baselines/legacy/research.toml"
)
START = date(1970, 1, 1)
CUTOFF = date(2023, 12, 31)


def _localfile_source(path: str, payload: bytes) -> SourceConfig:
    return SourceConfig(
        provider="localfile",
        source_id="fixture-localfile",
        frequency="daily",
        value_field="value",
        classification="test fixture",
        settings={
            "file_path": path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "construction": "test fixture",
        },
    )


def test_localfile_resolves_from_repo_root_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"date,value\n2023-01-03,100.0\n"
    root = tmp_path / "repo"
    target = root / "data/external/fixture.csv"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    source = _localfile_source("data/external/fixture.csv", payload)
    config = replace(
        CONFIG,
        path=root / "research.toml",
        markets=tuple(
            replace(market, equity=source, cash=source) for market in CONFIG.markets
        ),
    )

    manifest_path = acquire(
        config,
        repo_root=root,
        run_id="localfile-cwd-regression",
        created_at=datetime(2024, 1, 2, tzinfo=UTC),
        git_sha="abc123",
    )
    manifest = json.loads(manifest_path.read_text())
    canonical_path = root / manifest["sources"][0]["canonical"]["path"]

    assert {source["payload_type"] for source in manifest["sources"]} == {"local_file"}
    assert pd.read_csv(canonical_path).to_dict("records") == [
        {"date": "2023-01-03", "value": 100.0}
    ]


def test_localfile_requires_repo_root() -> None:
    payload = b"date,value\n2023-01-03,100.0\n"

    with pytest.raises(AcquisitionError, match="localfile requires repo_root"):
        fetch_source(
            _localfile_source("data/external/fixture.csv", payload),
            START,
            CUTOFF,
        )


@pytest.mark.parametrize("path", ["/tmp/outside.csv", "../outside.csv"])
def test_localfile_rejects_absolute_and_traversal_paths(
    tmp_path: Path, path: str
) -> None:
    payload = b"date,value\n2023-01-03,100.0\n"

    with pytest.raises(AcquisitionError, match="unsafe localfile path"):
        fetch_source(
            _localfile_source(path, payload),
            START,
            CUTOFF,
            repo_root=tmp_path,
        )


def test_localfile_rejects_symlink_escape(tmp_path: Path) -> None:
    payload = b"date,value\n2023-01-03,100.0\n"
    root = tmp_path / "repo"
    local_dir = root / "data/external"
    local_dir.mkdir(parents=True)
    outside = tmp_path / "outside.csv"
    outside.write_bytes(payload)
    (local_dir / "fixture.csv").symlink_to(outside)

    with pytest.raises(AcquisitionError, match="unsafe localfile path"):
        fetch_source(
            _localfile_source("data/external/fixture.csv", payload),
            START,
            CUTOFF,
            repo_root=root,
        )


@pytest.mark.parametrize("provider", ["yahoo", "boj"])
def test_retired_live_providers_refuse_to_acquire(provider: str) -> None:
    source = replace(CONFIG.markets[0].equity, provider=provider)

    with pytest.raises(AcquisitionError, match="is retired"):
        fetch_source(source, START, CUTOFF)


def test_fred_adapter_sends_frozen_bounds_and_preserves_missing() -> None:
    source = CONFIG.markets[0].cash

    def getter(url, params):
        assert params == {"cosd": "1970-01-01", "coed": "2023-12-31"}
        return HttpResult(
            b"observation_date,DTB3\n1970-01-02,7.08\n1970-01-05,.\n",
            f"{url}&cosd=1970-01-01&coed=2023-12-31",
            200,
            "text/csv",
        )

    payload = fetch_source(source, START, CUTOFF, http_get=getter)

    assert payload.payload_type == "provider_response"
    assert payload.raw.startswith(b"observation_date,DTB3")
    assert payload.canonical["value"].iloc[0] == 7.08
    assert pd.isna(payload.canonical["value"].iloc[1])


@pytest.mark.parametrize(
    ("dates", "values", "message"),
    [
        (["2023-01-03", "2023-01-03"], ["1.0", "2.0"], "duplicate dates"),
        (["2023-01-03", "2024-01-02"], ["1.0", "2.0"], "outside frozen interval"),
        (["2023-01-03"], ["bad"], "non-numeric value"),
    ],
)
def test_canonical_funnel_rejects_invalid_observations(dates, values, message) -> None:
    source = CONFIG.markets[0].cash
    rows = "".join(f"{day},{value}\n" for day, value in zip(dates, values, strict=True))

    def getter(url, _params):
        return HttpResult(
            f"observation_date,{source.value_field}\n{rows}".encode(),
            url,
            200,
            "text/csv",
        )

    with pytest.raises(AcquisitionError, match=message):
        fetch_source(source, START, CUTOFF, http_get=getter)


def test_quality_reports_the_facts_the_manifest_and_report_read() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2023-01-03", "2023-01-04", "2023-01-05"],
            "value": [1.0, None, 3.0],
        }
    )

    assert quality(frame) == {
        "rows": 3,
        "valid_rows": 2,
        "missing_values": 1,
        "first_valid_date": "2023-01-03",
        "last_valid_date": "2023-01-05",
    }


def test_canonical_serialization_is_deterministic() -> None:
    frame = pd.DataFrame({"date": ["2023-01-02", "2023-01-03"], "value": [1.0, None]})

    assert canonical_bytes(frame) == b"date,value\n2023-01-02,1.0\n2023-01-03,\n"


def test_acquire_writes_six_hashed_sources_and_manifest_last(tmp_path: Path) -> None:
    manifest_path = _fixture_run(tmp_path)
    manifest = json.loads(manifest_path.read_text())

    assert manifest_path == (tmp_path / "data/raw/fixture-run/manifest.json")
    assert manifest["config_sha256"] == CONFIG.sha256
    assert manifest["git_sha"] == "abc123"
    assert manifest["replication_cutoff"] == "2023-12-31"
    assert len(manifest["sources"]) == 6
    payload_types = [row["payload_type"] for row in manifest["sources"]]
    assert payload_types.count("local_file") == 4
    assert payload_types.count("provider_response") == 2
    for source in manifest["sources"]:
        for key in ("raw", "canonical"):
            record = source[key]
            payload = (tmp_path / record["path"]).read_bytes()
            assert len(payload) == record["bytes"]
            assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        canonical = pd.read_csv(tmp_path / source["canonical"]["path"])
        assert canonical.columns.tolist() == ["date", "value"]
        assert canonical["date"].max() <= "2023-12-31"


def test_fixture_runs_have_identical_canonical_hashes(tmp_path: Path) -> None:
    first = json.loads(_fixture_run(tmp_path / "first").read_text())
    second = json.loads(_fixture_run(tmp_path / "second").read_text())

    assert [row["canonical"]["sha256"] for row in first["sources"]] == [
        row["canonical"]["sha256"] for row in second["sources"]
    ]


def test_acquire_rejects_existing_run(tmp_path: Path) -> None:
    _fixture_run(tmp_path)

    with pytest.raises(AcquisitionError, match="already exists"):
        _fixture_run(tmp_path)


def test_git_provenance_rejects_result_affecting_diff(tmp_path: Path) -> None:
    source = tmp_path / "src/example.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "add",
            ".",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    assert len(research_git_sha(tmp_path)) == 40

    source.write_text("VALUE = 2\n")
    with pytest.raises(AcquisitionError, match="tracked files are dirty"):
        research_git_sha(tmp_path)

    source.write_text("VALUE = 1\n")
    untracked = tmp_path / "tests/new_test.py"
    untracked.parent.mkdir()
    untracked.write_text("assert True\n")
    with pytest.raises(AcquisitionError, match="untracked files exist"):
        research_git_sha(tmp_path)


@pytest.mark.parametrize(
    "relative",
    [
        # The root, where a config could still reappear...
        "research.toml",
        "research-calibrated-v11.toml",
        # ...and configs/, where they live now. Both must stay guarded, or a
        # config edited mid-run would be silently absent from the recorded
        # provenance.
        "configs/research-calibrated-v11.toml",
        "configs/baselines/research-calibrated-v11.toml",
        "configs/baselines/legacy/research-calibrated-v11.toml",
    ],
)
def test_git_provenance_guards_configs_wherever_they_live(
    tmp_path: Path, relative: str
) -> None:
    config = tmp_path / relative
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("config_id = 'fixture'\n")
    _git_commit_all(tmp_path)
    assert len(research_git_sha(tmp_path)) == 40

    config.write_text("config_id = 'edited'\n")
    with pytest.raises(AcquisitionError, match="tracked files are dirty"):
        research_git_sha(tmp_path)

    config.write_text("config_id = 'fixture'\n")
    sibling = config.with_name("research-untracked.toml")
    sibling.write_text("config_id = 'untracked'\n")
    with pytest.raises(AcquisitionError, match="untracked files exist"):
        research_git_sha(tmp_path)


def test_git_provenance_ignores_files_outside_the_guarded_scope(
    tmp_path: Path,
) -> None:
    """The guard must fail on real config edits, not on everything."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src/example.py").write_text("VALUE = 1\n")
    _git_commit_all(tmp_path)

    (tmp_path / "notes.md").write_text("scratch\n")
    assert len(research_git_sha(tmp_path)) == 40


def _git_commit_all(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    identity = ["-c", "user.name=Test", "-c", "user.email=test@example.com"]
    subprocess.run(["git", *identity, "add", "."], cwd=root, check=True)
    subprocess.run(["git", *identity, "commit", "-qm", "fixture"], cwd=root, check=True)


def _fixture_run(root: Path) -> Path:
    """Acquire through the two live providers: pinned local files plus FRED."""
    payload = b"date,value\n2023-01-03,100.0\n2023-12-29,101.0\n"
    local = root / "data/external/fixture.csv"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(payload)
    source = _localfile_source("data/external/fixture.csv", payload)
    config = replace(
        CONFIG,
        path=root / "research.toml",
        markets=tuple(
            replace(
                market,
                equity=source,
                cash=source if market.cash.provider == "boj" else market.cash,
            )
            for market in CONFIG.markets
        ),
    )

    def http_get(url, _params):
        source_id = "DTB3" if "DTB3" in url else "IR3TIB01DEM156N"
        dates = (
            ["1970-01-02", "2023-12-29"]
            if source_id == "DTB3"
            else ["1970-01-01", "2023-12-01"]
        )
        content = (
            f"observation_date,{source_id}\n{dates[0]},1.0\n{dates[1]},2.0\n"
        ).encode()
        return HttpResult(content, url, 200, "text/csv")

    return acquire(
        config,
        repo_root=root,
        run_id="fixture-run",
        created_at=datetime(2024, 1, 2, tzinfo=UTC),
        git_sha="abc123",
        http_get=http_get,
    )
