"""Unit tests for auth helpers — password hashing and JWT utilities."""
import pytest
from unittest.mock import patch, AsyncMock


def test_hash_password_returns_bcrypt_hash():
    from api.auth import hash_password
    h = hash_password("secret123")
    assert h.startswith("$2b$")
    assert h != "secret123"


def test_verify_password_correct():
    from api.auth import hash_password, verify_password
    h = hash_password("secret123")
    assert verify_password("secret123", h) is True


def test_verify_password_wrong():
    from api.auth import hash_password, verify_password
    h = hash_password("secret123")
    assert verify_password("wrong", h) is False


def test_verify_password_bad_hash_returns_false():
    from api.auth import verify_password
    assert verify_password("anything", "not-a-real-hash") is False


def test_hash_is_salted():
    from api.auth import hash_password
    assert hash_password("same") != hash_password("same")


def test_sanitize_username_strips_and_lowercases():
    from api.auth import sanitize_username
    assert sanitize_username("  Alice  ") == "alice"
    assert sanitize_username("BOB") == "bob"


def test_create_access_token_contains_sub():
    from api.auth import create_access_token
    token = create_access_token({"sub": "user-123"})
    import base64, json
    payload = json.loads(base64.b64decode(token.split(".")[1] + "=="))
    assert payload["sub"] == "user-123"


def test_access_token_has_exp():
    from api.auth import create_access_token
    import base64, json
    token = create_access_token({"sub": "u"})
    payload = json.loads(base64.b64decode(token.split(".")[1] + "=="))
    assert "exp" in payload
