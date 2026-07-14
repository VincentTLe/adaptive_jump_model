from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_systemd_unit_uses_canonical_loopback_monitor_and_external_secrets() -> None:
    unit = (ROOT / "deploy/adaptive-jump-monitor.service").read_text()

    assert "User=tle" in unit
    assert "EnvironmentFile=/home/tle/.config/adaptive-jump/monitor.env" in unit
    assert "adaptive-jump monitor --config" in unit
    assert "KillMode=control-group" in unit and "TimeoutStopSec=55s" in unit
    assert "UMask=0077" in unit
    assert "0.0.0.0" not in unit and "ADAPTIVE_JUMP_CSRF_SECRET=" not in unit


def test_deployment_templates_are_fail_closed_and_secret_free() -> None:
    tunnel = (ROOT / "deploy/cloudflared-config.yml.example").read_text()
    environment = (ROOT / "deploy/monitor.env.example").read_text()

    assert "service: http://127.0.0.1:8765" in tunnel
    assert tunnel.rstrip().endswith("service: http_status:404")
    required = (
        "ADAPTIVE_JUMP_MONITOR_ACCESS",
        "ADAPTIVE_JUMP_ACCESS_ISSUER",
        "ADAPTIVE_JUMP_ACCESS_AUDIENCE",
        "ADAPTIVE_JUMP_OWNER_EMAIL",
        "ADAPTIVE_JUMP_VIEWER_EMAILS",
        "ADAPTIVE_JUMP_MONITOR_ORIGIN",
        "ADAPTIVE_JUMP_CSRF_SECRET",
    )
    assert all(f"{name}=" in environment for name in required)
    assert "ADAPTIVE_JUMP_MONITOR_ACCESS=cloudflare" in environment
    assert "cloudflareaccess.com" in environment and "example.com" in environment


def test_deployment_guide_pins_and_checks_the_official_tunnel_binary() -> None:
    guide = (ROOT / "docs/monitor/deployment.md").read_text()

    assert "CLOUDFLARED_VERSION=2026.6.0" in guide
    assert (
        "CLOUDFLARED_SHA256="
        "08d27c4c5d3ed73ee3e98ef2ddceb4ad09fd4cfc28e243565a189538e8ccd706"
    ) in guide
    assert "sha256sum --check --strict" in guide
    assert "releases/latest" not in guide
    assert "Simple Browser: Show" in guide


def test_readme_documents_the_single_locked_monitor_stack() -> None:
    readme = (ROOT / "README.md").read_text()

    assert "uv sync --locked --extra data --extra monitor" in readme
    assert "adaptive-jump monitor --config research.toml" in readme
    assert "Local use requires no authentication environment variables" in readme
    assert "separately launched `adaptive-jump run`" in readme
    assert "docs/monitor/deployment.md" in readme
    assert "requirements.txt" in readme and "dependency source" in readme
