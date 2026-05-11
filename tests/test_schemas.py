# tests/test_schemas.py — Unit tests for Pydantic schema validators
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import pytest
from pydantic import ValidationError


class TestRegisterRequestValidators:

    def test_valid_username_accepted(self):
        from api.schemas import RegisterRequest
        r = RegisterRequest(username="hridam_01", email="h@test.com", password="Pass1word")
        assert r.username == "hridam_01"

    def test_username_with_hyphen_accepted(self):
        from api.schemas import RegisterRequest
        r = RegisterRequest(username="my-user", email="h@test.com", password="Pass1word")
        assert r.username == "my-user"

    def test_username_with_space_rejected(self):
        from api.schemas import RegisterRequest
        with pytest.raises(ValidationError, match="Username may only contain"):
            RegisterRequest(username="bad user", email="h@test.com", password="Pass1word")

    def test_username_with_dot_rejected(self):
        from api.schemas import RegisterRequest
        with pytest.raises(ValidationError, match="Username may only contain"):
            RegisterRequest(username="bad.user", email="h@test.com", password="Pass1word")

    def test_username_with_angle_bracket_rejected(self):
        from api.schemas import RegisterRequest
        with pytest.raises(ValidationError, match="Username may only contain"):
            RegisterRequest(username="<script>", email="h@test.com", password="Pass1word")

    def test_username_too_short_rejected(self):
        from api.schemas import RegisterRequest
        with pytest.raises(ValidationError):
            RegisterRequest(username="ab", email="h@test.com", password="Pass1word")

    def test_username_too_long_rejected(self):
        from api.schemas import RegisterRequest
        with pytest.raises(ValidationError):
            RegisterRequest(username="a" * 33, email="h@test.com", password="Pass1word")

    def test_password_no_digit_rejected(self):
        from api.schemas import RegisterRequest
        with pytest.raises(ValidationError, match="at least one digit"):
            RegisterRequest(username="hridam", email="h@test.com", password="NoDigitsHere")

    def test_password_no_letter_rejected(self):
        from api.schemas import RegisterRequest
        with pytest.raises(ValidationError, match="at least one letter"):
            RegisterRequest(username="hridam", email="h@test.com", password="12345678")

    def test_password_too_short_rejected(self):
        from api.schemas import RegisterRequest
        with pytest.raises(ValidationError):
            RegisterRequest(username="hridam", email="h@test.com", password="P1")

    def test_valid_password_accepted(self):
        from api.schemas import RegisterRequest
        r = RegisterRequest(username="hridam", email="h@test.com", password="Secur3Pass")
        assert r.password == "Secur3Pass"

    def test_invalid_email_rejected(self):
        from api.schemas import RegisterRequest
        with pytest.raises(ValidationError):
            RegisterRequest(username="hridam", email="notanemail", password="Pass1word")


class TestQueryRequestValidators:

    def test_null_byte_stripped(self):
        from api.schemas import QueryRequest
        q = QueryRequest(query="hello\x00world")
        assert "\x00" not in q.query
        assert "helloworld" in q.query

    def test_other_control_chars_stripped(self):
        from api.schemas import QueryRequest
        q = QueryRequest(query="hello\x01\x02\x03world")
        assert q.query == "helloworld"

    def test_newline_and_tab_preserved(self):
        from api.schemas import QueryRequest
        text = "line1\nline2\ttabbed"
        q = QueryRequest(query=text)
        assert q.query == text

    def test_query_too_long_rejected(self):
        from api.schemas import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(query="x" * 4097)

    def test_empty_query_rejected(self):
        from api.schemas import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_valid_query_accepted(self):
        from api.schemas import QueryRequest
        q = QueryRequest(query="What is CARLE?")
        assert q.query == "What is CARLE?"


class TestHistoryMessage:

    def test_valid_history_attached_to_query(self):
        from api.schemas import QueryRequest, HistoryMessage
        q = QueryRequest(
            query="tell me more",
            history=[
                HistoryMessage(role="user", content="What is RAG?"),
                HistoryMessage(role="assistant", content="RAG stands for…"),
            ]
        )
        assert len(q.history) == 2
        assert q.history[0].role == "user"

    def test_history_capped_at_20(self):
        from api.schemas import QueryRequest, HistoryMessage
        with pytest.raises(ValidationError):
            QueryRequest(
                query="hi",
                history=[HistoryMessage(role="user", content="x")] * 21
            )

    def test_empty_history_defaults(self):
        from api.schemas import QueryRequest
        q = QueryRequest(query="hello")
        assert q.history == []


class TestConversationSchemas:

    def test_create_conversation_request_defaults(self):
        from api.schemas import CreateConversationRequest
        req = CreateConversationRequest()
        assert req.title == "New Chat"

    def test_append_message_role_validation(self):
        from api.schemas import AppendMessageRequest
        with pytest.raises(ValidationError):
            AppendMessageRequest(role="system", content="bad role")

    def test_append_message_valid_roles(self):
        from api.schemas import AppendMessageRequest
        for role in ("user", "assistant"):
            msg = AppendMessageRequest(role=role, content="hello")
            assert msg.role == role
