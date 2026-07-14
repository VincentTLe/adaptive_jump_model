from datetime import UTC, datetime, timedelta

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from adaptive_jump.monitor.security import (
    ALGORITHM,
    AccessAuthenticator,
    AccessConfig,
    AuthenticationError,
    Principal,
)

ISSUER = "https://research.cloudflareaccess.com"
AUDIENCE = "monitor-audience"
OWNER = "owner@example.com"
VIEWER = "advisor@example.com"


@pytest.fixture(scope="module")
def keys():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private, private.public_key(), other.public_key()


def _token(private, **updates) -> str:
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "email": OWNER,
        "exp": datetime.now(UTC) + timedelta(minutes=5),
    }
    claims.update(updates)
    return jwt.encode(claims, private, algorithm=ALGORITHM, headers={"kid": "test"})


def _authenticator(public) -> AccessAuthenticator:
    config = AccessConfig(ISSUER, AUDIENCE, OWNER, (VIEWER,))
    return AccessAuthenticator(config, lambda _assertion: public)


def test_valid_assertions_resolve_exact_owner_and_viewer_roles(keys) -> None:
    private, public, _other = keys
    authenticator = _authenticator(public)

    assert authenticator.authenticate(_token(private)) == Principal(OWNER, "owner")
    assert authenticator.authenticate(_token(private, email=VIEWER)) == Principal(
        VIEWER, "viewer"
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"iss": "https://wrong.cloudflareaccess.com"},
        {"aud": "wrong-audience"},
        {"exp": datetime.now(UTC) - timedelta(seconds=1)},
        {"email": "unknown@example.com"},
    ],
)
def test_claim_mismatches_fail_closed(keys, updates) -> None:
    private, public, _other = keys
    with pytest.raises(AuthenticationError):
        _authenticator(public).authenticate(_token(private, **updates))


def test_signature_algorithm_and_required_email_fail_closed(keys) -> None:
    private, public, other = keys
    authenticator = _authenticator(public)
    with pytest.raises(AuthenticationError):
        _authenticator(other).authenticate(_token(private))
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": datetime.now(UTC) + timedelta(minutes=5),
    }
    missing_email = jwt.encode(
        claims, private, algorithm=ALGORITHM, headers={"kid": "test"}
    )
    with pytest.raises(AuthenticationError):
        authenticator.authenticate(missing_email)
    wrong_algorithm = jwt.encode(
        {**claims, "email": OWNER},
        "local-secret-with-at-least-32-bytes",
        algorithm="HS256",
        headers={"kid": "test"},
    )
    with pytest.raises(AuthenticationError, match="algorithm"):
        authenticator.authenticate(wrong_algorithm)


@pytest.mark.parametrize(
    "values",
    [
        {"ADAPTIVE_JUMP_ACCESS_ISSUER": "http://not-https.example.com"},
        {
            "ADAPTIVE_JUMP_OWNER_EMAIL": "same@example.com",
            "viewers": "same@example.com",
        },
        {"ADAPTIVE_JUMP_OWNER_EMAIL": "not-an-email"},
    ],
)
def test_environment_configuration_rejects_unsafe_roles(values) -> None:
    environment = {
        "ADAPTIVE_JUMP_ACCESS_ISSUER": ISSUER,
        "ADAPTIVE_JUMP_ACCESS_AUDIENCE": AUDIENCE,
        "ADAPTIVE_JUMP_OWNER_EMAIL": OWNER,
        "ADAPTIVE_JUMP_VIEWER_EMAILS": VIEWER,
    }
    environment.update(
        {key: value for key, value in values.items() if key != "viewers"}
    )
    if "viewers" in values:
        environment["ADAPTIVE_JUMP_VIEWER_EMAILS"] = values["viewers"]
    with pytest.raises(AuthenticationError):
        AccessConfig.from_environment(environment)


def test_environment_requires_every_primary_authentication_value() -> None:
    with pytest.raises(AuthenticationError, match="missing"):
        AccessConfig.from_environment({})
