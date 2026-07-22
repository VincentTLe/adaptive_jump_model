"""Local and Cloudflare authentication for the loopback monitor origin."""

from __future__ import annotations

import base64
import binascii
import hmac
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

import jwt

ACCESS_ASSERTION_HEADER = "Cf-Access-Jwt-Assertion"
AUTHORIZATION_HEADER = "Authorization"
ALGORITHM = "RS256"
_MAX_ASSERTION_BYTES = 16_384
_MAX_BASIC_BYTES = 1_024
KeyResolver = Callable[[str], Any]


class AuthenticationError(ValueError):
    """Raised when an Access assertion or role configuration fails closed."""


@dataclass(frozen=True)
class Principal:
    email: str
    role: str


class Authenticator(Protocol):
    credential_header: str
    challenge: str

    def authenticate(self, credential: str) -> Principal: ...


class LocalAuthenticator:
    """Authenticate one local owner through browser-native HTTP Basic auth."""

    credential_header = AUTHORIZATION_HEADER
    challenge = 'Basic realm="Adaptive Jump Monitor", charset="UTF-8"'
    username = "owner"
    principal = Principal("local-owner@localhost", "owner")

    def __init__(self, password: str) -> None:
        if (
            not _plain_value(password)
            or not password.isascii()
            or not 16 <= len(password) <= 256
        ):
            raise AuthenticationError(
                "local monitor password must contain 16 to 256 ASCII characters"
            )
        self._password = password

    def authenticate(self, credential: str) -> Principal:
        if (
            not isinstance(credential, str)
            or not credential
            or len(credential.encode("utf-8")) > _MAX_BASIC_BYTES
        ):
            raise AuthenticationError("local monitor credentials are missing")
        try:
            scheme, encoded = credential.split(" ", 1)
            decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
            username, separator, password = decoded.partition(":")
        except (ValueError, UnicodeError, binascii.Error) as exc:
            raise AuthenticationError("local monitor credentials are invalid") from exc
        valid = (
            scheme.lower() == "basic"
            and bool(separator)
            and username.isascii()
            and password.isascii()
            and hmac.compare_digest(username, self.username)
            and hmac.compare_digest(password, self._password)
        )
        if not valid:
            raise AuthenticationError("local monitor credentials are invalid")
        return self.principal


@dataclass(frozen=True)
class AccessConfig:
    issuer: str
    audience: str
    owner_email: str
    viewer_emails: tuple[str, ...]

    def __post_init__(self) -> None:
        issuer = self.issuer.rstrip("/")
        parsed = urlsplit(issuer)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise AuthenticationError("Access issuer must be one HTTPS origin")
        object.__setattr__(self, "issuer", issuer)
        if not _plain_value(self.audience):
            raise AuthenticationError("Access audience is required")
        emails = (self.owner_email, *self.viewer_emails)
        if any(not _valid_email(value) for value in emails):
            raise AuthenticationError("role emails must be exact non-empty addresses")
        if len(set(emails)) != len(emails):
            raise AuthenticationError("owner and viewer emails must be distinct")

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> AccessConfig:
        """Load secrets and role bindings without introducing a config file."""
        values = os.environ if environ is None else environ
        required = {
            name: values.get(name, "")
            for name in (
                "ADAPTIVE_JUMP_ACCESS_ISSUER",
                "ADAPTIVE_JUMP_ACCESS_AUDIENCE",
                "ADAPTIVE_JUMP_OWNER_EMAIL",
            )
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise AuthenticationError(
                "missing monitor authentication variables: " + ", ".join(missing)
            )
        viewers = tuple(
            value.strip()
            for value in values.get("ADAPTIVE_JUMP_VIEWER_EMAILS", "").split(",")
            if value.strip()
        )
        return cls(
            issuer=required["ADAPTIVE_JUMP_ACCESS_ISSUER"],
            audience=required["ADAPTIVE_JUMP_ACCESS_AUDIENCE"],
            owner_email=required["ADAPTIVE_JUMP_OWNER_EMAIL"],
            viewer_emails=viewers,
        )


class AccessAuthenticator:
    """Verify Cloudflare-signed assertions and resolve exact-email roles."""

    credential_header = ACCESS_ASSERTION_HEADER
    challenge = "Cloudflare-Access"

    def __init__(
        self, config: AccessConfig, key_resolver: KeyResolver | None = None
    ) -> None:
        self.config = config
        self._jwk_client = (
            None
            if key_resolver is not None
            else jwt.PyJWKClient(
                f"{config.issuer}/cdn-cgi/access/certs",
                cache_keys=True,
                lifespan=300,
                timeout=10,
            )
        )
        self._key_resolver = key_resolver or self._resolve_key

    def authenticate(self, assertion: str) -> Principal:
        """Return the authenticated role or reject every malformed condition."""
        if (
            not isinstance(assertion, str)
            or not assertion
            or len(assertion.encode("utf-8")) > _MAX_ASSERTION_BYTES
        ):
            raise AuthenticationError("Access assertion is missing or invalid")
        try:
            header = jwt.get_unverified_header(assertion)
            if header.get("alg") != ALGORITHM or not _plain_value(header.get("kid")):
                raise AuthenticationError("Access assertion algorithm is invalid")
            key = self._key_resolver(assertion)
            claims = jwt.decode(
                assertion,
                key,
                algorithms=[ALGORITHM],
                audience=self.config.audience,
                issuer=self.config.issuer,
                options={"require": ["exp", "email"]},
            )
        except AuthenticationError:
            raise
        except (jwt.PyJWTError, OSError, TypeError, ValueError) as exc:
            raise AuthenticationError("Access assertion verification failed") from exc
        email = claims.get("email")
        if not isinstance(email, str):
            raise AuthenticationError("Access assertion email is invalid")
        if email == self.config.owner_email:
            return Principal(email, "owner")
        if email in self.config.viewer_emails:
            return Principal(email, "viewer")
        raise AuthenticationError("Access email has no monitor role")

    def _resolve_key(self, assertion: str) -> Any:
        assert self._jwk_client is not None
        return self._jwk_client.get_signing_key_from_jwt(assertion).key


def _plain_value(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value.strip() == value


def _valid_email(value: object) -> bool:
    return (
        _plain_value(value)
        and "@" in value
        and not any(character.isspace() for character in value)
    )
