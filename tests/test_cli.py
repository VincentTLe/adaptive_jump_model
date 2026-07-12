import json
from pathlib import Path

import pandas as pd

from adaptive_jump import data
from adaptive_jump.cli import main
from adaptive_jump.data import HttpResult

ROOT = Path(__file__).resolve().parents[1]


def test_fetch_cli_runs_complete_fixture_pipeline(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = tmp_path / "research.toml"
    config.write_bytes((ROOT / "research.toml").read_bytes())

    def yahoo_loader(_source, _start, _end):
        return pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.to_datetime(["2023-01-03", "2023-12-29"]),
        ), {"adapter": "fixture"}

    def http_get(url, _params):
        if "fredgraph" in url:
            source_id = "DTB3" if "DTB3" in url else "IR3TIB01DEM156N"
            content = (
                f"observation_date,{source_id}\n1970-01-02,1.0\n2023-12-01,2.0\n"
            ).encode()
        else:
            content = (
                b"STATUS,200\nNEXTPOSITION,\n"
                b"SERIES_CODE,NAME_OF_TIME_SERIES,UNIT,FREQUENCY,CATEGORY,"
                b"LAST_UPDATE,SURVEY_DATES,VALUES\n"
                b"STRACLUC3M,Call,percent,MONTHLY,Call,20240101,198901,1.0\n"
                b"STRACLUC3M,Call,percent,MONTHLY,Call,20240101,202312,2.0\n"
            )
        return HttpResult(content, url, 200, "text/csv")

    monkeypatch.setattr(data, "_download_yahoo", yahoo_loader)
    monkeypatch.setattr(data, "_get_http", http_get)
    monkeypatch.setattr(data, "research_git_sha", lambda _root: "abc123")

    assert main(["fetch", "--config", str(config)]) == 0

    manifest_path = Path(capsys.readouterr().out.strip())
    manifest = json.loads(manifest_path.read_text())
    assert manifest["config_sha256"] == (
        "77b60e4e57bc2356ee2b4bb5d177f22295b1cb0b58a5e047670ecdead80cad61"
    )
    assert len(manifest["sources"]) == 6
    assert manifest_path.parent.parent == tmp_path / "data/raw"


def test_fetch_cli_reports_missing_config(capsys) -> None:
    assert main(["fetch", "--config", "missing.toml"]) == 2
    assert "missing.toml" in capsys.readouterr().err
