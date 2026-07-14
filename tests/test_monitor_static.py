import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vendored_browser_dependencies_match_the_locked_manifest() -> None:
    manifest = json.loads((ROOT / "docs/monitor/browser-dependencies.json").read_text())

    assert manifest["schema_version"] == 1
    assert len(manifest["dependencies"]) == 1
    dependency = manifest["dependencies"][0]
    assert dependency["name"] == "Apache ECharts"
    assert dependency["version"] == "6.1.0"
    assert dependency["license"] == "Apache-2.0"
    for path_key, hash_key in (
        ("vendored_asset_path", "asset_sha256"),
        ("vendored_license_path", "license_sha256"),
        ("vendored_notice_path", "notice_sha256"),
    ):
        path = ROOT / dependency[path_key]
        assert path.is_file()
        assert _sha256(path) == dependency[hash_key]


def test_vendored_echarts_reports_the_locked_release() -> None:
    asset = ROOT / "src/adaptive_jump/monitor/static/vendor/echarts/echarts.min.js"
    content = asset.read_text(encoding="utf-8")

    assert 't.version="6.1.0"' in content
    assert "sourceMappingURL" not in content
