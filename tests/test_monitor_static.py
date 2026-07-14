import hashlib
import json
import re
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


def test_monitor_shell_is_packaged_accessible_and_csp_compatible() -> None:
    static = ROOT / "src/adaptive_jump/monitor/static"
    html = (static / "index.html").read_text(encoding="utf-8")
    css = (static / "app.css").read_text(encoding="utf-8")

    assert "Adaptive Jump Research Monitor" in html
    views = ("live", "queue", "replay", "compare", "evidence")
    assert all(f'data-view="{view}"' in html for view in views)
    scripts = re.findall(r"<script([^>]*)>", html)
    assert scripts and all("src=" in script for script in scripts)
    assert "<style" not in html
    assert "@media (max-width: 600px)" in css
    assert "linear-gradient" not in css


def test_monitor_browser_code_uses_server_contract_without_inline_data() -> None:
    script = (ROOT / "src/adaptive_jump/monitor/static/app.js").read_text(
        encoding="utf-8"
    )

    assert all(path in script for path in ("/api/session", "/api/studies", "/api/jobs"))
    assert "EventSource" in script and "research_event" in script
    assert "innerHTML" not in script and "localStorage" not in script
