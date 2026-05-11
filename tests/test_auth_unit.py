"""Unit tests for authentication helpers — password hashing and JWT logic."""
import pytest
from api.auth import (
    hash_password,
    verify_password,
    validate_password_strength,
    create_access_token,
    create_refresh_token_str,
)


class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        h = hash_password("SecurePass1")
        assert h != "SecurePass1"
        assert h.startswith("$2b$")

    def test_verify_correct_password(self):
        h = hash_password("MyPassword9")
        assert verify_password("MyPassword9", h) is True

    def test_verify_wrong_password(self):
        h = hash_password("MyPassword9")
        assert verify_password("WrongPass9", h) is False

    def test_hashes_are_unique(self):
        h1 = hash_password("SamePass1")
        h2 = hash_password("SamePass1")
        assert h1 != h2  # bcrypt uses random salt

    def test_verify_invalid_hash_returns_false(self):
        assert verify_password("anything", "not_a_bcrypt_hash") is False


class TestPasswordValidation:
    def test_valid_password_returns_no_errors(self):
        assert validate_password_strength("Secure123") == []

    def test_short_password_fails(self):
        errors = validate_password_strength("Ab1")
        assert any("8 characters" in e for e in errors)

    def test_no_digit_fails(self):
        errors = validate_password_strength("NoDigitHere")
        assert any("digit" in e for e in errors)

    def test_no_uppercase_fails(self):
        errors = validate_password_strength("nouppercase1")
        assert any("uppercase" in e for e in errors)

    def test_no_lowercase_fails(self):
        errors = validate_password_strength("NOLOWERCASE1")
        assert any("lowercase" in e for e in errors)

    def test_multiple_failures_reported(self):
        errors = validate_password_strength("a")
        assert len(errors) >= 2


class TestJWTTokens:
    def test_access_token_is_string(self):
        token = create_access_token("user-123")
        assert isinstance(token, str)
        assert len(token) > 20

    def test_refresh_token_is_string(self):
        token = create_refresh_token_str("user-123")
        assert isinstance(token, str)
        assert len(token) > 20

    def test_access_and_refresh_tokens_differ(self):
        uid = "user-456"
        assert create_access_token(uid) != create_refresh_token_str(uid)
