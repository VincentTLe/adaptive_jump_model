"""Same-origin and signed-CSRF policy for authenticated monitor requests."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

CSRF_HEADER = "X-CSRF-Token"
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}
_SECRET_PATTERN = re.compile(r"[A-Za-z0-9_-]{43,}\Z")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\Z")


class RequestSecurityError(ValueError):
    """Raised when an HTTP origin or CSRF credential is invalid."""


@dataclass(frozen=True)
class HttpSecurityConfig:
    public_origin: str
    csrf_secret: bytes
    csrf_ttl_seconds: int = 3600

    def __post_init__(self) -> None:
        origin = self.public_origin.rstrip("/")
        parsed = urlsplit(origin)
        loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if (
            parsed.scheme not in ({"http", "https"} if loopback else {"https"})
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise RequestSecurityError("monitor origin must be one safe HTTP origin")
        object.__setattr__(self, "public_origin", origin)
        if not isinstance(self.csrf_secret, bytes) or len(self.csrf_secret) < 32:
            raise RequestSecurityError("CSRF secret must contain at least 32 bytes")
        if (
            isinstance(self.csrf_ttl_seconds, bool)
            or not isinstance(self.csrf_ttl_seconds, int)
            or not 60 <= self.csrf_ttl_seconds <= 86_400
        ):
            raise RequestSecurityError("CSRF lifetime must be 60 to 86400 seconds")

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> HttpSecurityConfig:
        """Load the browser origin and base64url secret from the environment."""
        values = os.environ if environ is None else environ
        origin = values.get("ADAPTIVE_JUMP_MONITOR_ORIGIN", "")
        encoded = values.get("ADAPTIVE_JUMP_CSRF_SECRET", "")
        if not origin or _SECRET_PATTERN.fullmatch(encoded) is None:
            raise RequestSecurityError("monitor origin or CSRF secret is missing")
        try:
            secret = _decode_base64url(encoded)
        except (ValueError, UnicodeError) as exc:
            raise RequestSecurityError("CSRF secret is not valid base64url") from exc
        return cls(origin, secret)


class RequestSecurity:
    """Issue and verify stateless CSRF tokens bound to one authenticated email."""

    def __init__(
        self,
        config: HttpSecurityConfig,
        *,
        clock: Callable[[], float] = time.time,
        nonce_factory: Callable[[], str] | None = None,
    ) -> None:
        self.config = config
        self._clock = clock
        self._nonce_factory = nonce_factory or (lambda: secrets.token_urlsafe(18))

    def require_origin(self, origin: str | None) -> None:
        """Reject missing, opaque, or cross-origin browser mutations."""
        if origin != self.config.public_origin:
            raise RequestSecurityError("request origin is not authorized")

    def issue_csrf(self, email: str) -> str:
        """Create a short-lived signed token for one authenticated principal."""
        if not isinstance(email, str) or not email:
            raise RequestSecurityError("CSRF principal is invalid")
        payload = {
            "email": email,
            "expires_at": int(self._clock()) + self.config.csrf_ttl_seconds,
            "nonce": self._nonce_factory(),
        }
        encoded = _encode_base64url(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        )
        signature = hmac.new(
            self.config.csrf_secret, encoded.encode(), hashlib.sha256
        ).digest()
        return f"{encoded}.{_encode_base64url(signature)}"

    def verify_csrf(self, token: str | None, email: str) -> None:
        """Verify schema, signature, principal binding, and strict expiration."""
        if (
            not isinstance(token, str)
            or len(token) > 1024
            or _TOKEN_PATTERN.fullmatch(token) is None
        ):
            raise RequestSecurityError("CSRF token is missing or malformed")
        encoded, supplied_signature = token.split(".")
        expected = hmac.new(
            self.config.csrf_secret, encoded.encode(), hashlib.sha256
        ).digest()
        try:
            signature = _decode_base64url(supplied_signature)
        except (UnicodeError, ValueError) as exc:
            raise RequestSecurityError("CSRF signature cannot be decoded") from exc
        if not hmac.compare_digest(signature, expected):
            raise RequestSecurityError("CSRF signature does not match")
        try:
            payload: Any = json.loads(_decode_base64url(encoded))
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise RequestSecurityError("CSRF payload cannot be decoded") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "email",
            "expires_at",
            "nonce",
        }:
            raise RequestSecurityError("CSRF payload schema is invalid")
        expires = payload["expires_at"]
        if (
            payload["email"] != email
            or not isinstance(payload["nonce"], str)
            or not payload["nonce"]
            or isinstance(expires, bool)
            or not isinstance(expires, int)
            or self._clock() >= expires
        ):
            raise RequestSecurityError("CSRF token is expired or misbound")


def _encode_base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_base64url(value: str) -> bytes:
    if not value or not value.isascii():
        raise ValueError("base64url value is empty")
    return base64.b64decode(
        value + "=" * (-len(value) % 4), altchars=b"-_", validate=True
    )
