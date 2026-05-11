"""Unit tests for Pydantic request schema validation."""
import pytest
from pydantic import ValidationError
from api.schemas import RegisterRequest, QueryRequest


class TestRegisterRequest:
    def test_valid_registration(self):
        r = RegisterRequest(username="hridam", email="h@example.com", password="SecurePass1")
        assert r.username == "hridam"

    def test_username_too_short_fails(self):
        with pytest.raises(ValidationError):
            RegisterRequest(username="ab", email="h@example.com", password="SecurePass1")

    def test_username_invalid_chars_fails(self):
        with pytest.raises(ValidationError):
            RegisterRequest(username="bad user!", email="h@example.com", password="SecurePass1")

    def test_invalid_email_fails(self):
        with pytest.raises(ValidationError):
            RegisterRequest(username="hridam", email="not-an-email", password="SecurePass1")

    def test_short_password_fails(self):
        with pytest.raises(ValidationError):
            RegisterRequest(username="hridam", email="h@example.com", password="abc")


class TestQueryRequest:
    def test_valid_query(self):
        q = QueryRequest(query="What is machine learning?")
        assert q.query == "What is machine learning?"

    def test_empty_query_fails(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_control_chars_stripped(self):
        q = QueryRequest(query="hello\x00world")
        assert "\x00" not in q.query

    def test_query_too_long_fails(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="x" * 5000)
