import base64

import pytest

from adaptive_jump.monitor.http_security import (
    SECURITY_HEADERS,
    HttpSecurityConfig,
    RequestSecurity,
    RequestSecurityError,
)

SECRET = bytes(range(32))


def _security(now: list[float] | None = None) -> RequestSecurity:
    current = now or [1000.0]
    return RequestSecurity(
        HttpSecurityConfig("https://monitor.example.com", SECRET, 60),
        clock=lambda: current[0],
        nonce_factory=lambda: "fixed-nonce",
    )


def test_csrf_round_trip_is_bound_to_email_and_expiration() -> None:
    now = [1000.0]
    security = _security(now)
    token = security.issue_csrf("owner@example.com")

    security.verify_csrf(token, "owner@example.com")
    with pytest.raises(RequestSecurityError, match="misbound"):
        security.verify_csrf(token, "advisor@example.com")
    now[0] = 1060.0
    with pytest.raises(RequestSecurityError, match="expired"):
        security.verify_csrf(token, "owner@example.com")


def test_csrf_rejects_tampering_and_malformed_values() -> None:
    security = _security()
    token = security.issue_csrf("owner@example.com")
    encoded, signature = token.split(".")
    tampered = ("A" if encoded[0] != "A" else "B") + encoded[1:]

    with pytest.raises(RequestSecurityError, match="signature"):
        security.verify_csrf(f"{tampered}.{signature}", "owner@example.com")
    for invalid in (None, "", "not-a-token", "a.b.c", "!invalid!.value"):
        with pytest.raises(RequestSecurityError, match="malformed"):
            security.verify_csrf(invalid, "owner@example.com")


def test_origin_must_match_the_configured_browser_origin_exactly() -> None:
    security = _security()
    security.require_origin("https://monitor.example.com")
    for origin in (
        None,
        "null",
        "https://evil.example.com",
        "https://monitor.example.com/",
        "http://monitor.example.com",
    ):
        with pytest.raises(RequestSecurityError, match="origin"):
            security.require_origin(origin)


def test_environment_loads_base64url_secret_and_allows_loopback_http() -> None:
    encoded = base64.urlsafe_b64encode(SECRET).rstrip(b"=").decode()
    config = HttpSecurityConfig.from_environment(
        {
            "ADAPTIVE_JUMP_MONITOR_ORIGIN": "http://127.0.0.1:8765",
            "ADAPTIVE_JUMP_CSRF_SECRET": encoded,
        }
    )

    assert config.public_origin == "http://127.0.0.1:8765"
    assert config.csrf_secret == SECRET


@pytest.mark.parametrize(
    ("origin", "secret"),
    [
        ("http://public.example.com", SECRET),
        ("https://user@monitor.example.com", SECRET),
        ("https://monitor.example.com/path", SECRET),
        ("https://monitor.example.com", b"short"),
    ],
)
def test_configuration_rejects_unsafe_origins_and_short_secrets(
    origin: str, secret: bytes
) -> None:
    with pytest.raises(RequestSecurityError):
        HttpSecurityConfig(origin, secret)


def test_security_headers_enforce_same_origin_without_cors() -> None:
    assert "default-src 'self'" in SECURITY_HEADERS["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in SECURITY_HEADERS["Content-Security-Policy"]
    assert "Access-Control-Allow-Origin" not in SECURITY_HEADERS
