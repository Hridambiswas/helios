"""Unit tests for synthesizer parsing utilities."""
import pytest
from agents.synthesizer import _parse_follow_ups, _format_web_sources, _format_local_docs

SEPARATOR = "<<<FOLLOW_UPS>>>"


class TestParseFollowUps:
    def test_splits_answer_and_questions(self):
        text = f"The answer is here.\n\n{SEPARATOR}\n1. First question?\n2. Second question?"
        answer, questions = _parse_follow_ups(text)
        assert answer == "The answer is here."
        assert len(questions) == 2
        assert questions[0] == "First question?"
        assert questions[1] == "Second question?"

    def test_returns_empty_when_no_separator(self):
        text = "Just an answer with no follow-ups."
        answer, questions = _parse_follow_ups(text)
        assert answer == text
        assert questions == []

    def test_handles_numbered_list_with_dot(self):
        text = f"Answer.\n{SEPARATOR}\n1. Q one?\n2. Q two?"
        _, questions = _parse_follow_ups(text)
        assert questions == ["Q one?", "Q two?"]

    def test_caps_at_two_questions(self):
        text = f"Answer.\n{SEPARATOR}\n1. Q1?\n2. Q2?\n3. Q3?"
        _, questions = _parse_follow_ups(text)
        assert len(questions) == 2


class TestFormatWebSources:
    def test_empty_list_returns_empty_string(self):
        assert _format_web_sources([]) == ""

    def test_formats_sources_correctly(self):
        sources = [{"title": "PyTorch Docs", "url": "https://pytorch.org", "snippet": "Deep learning framework"}]
        result = _format_web_sources(sources)
        assert "[W1]" in result
        assert "PyTorch Docs" in result
        assert "pytorch.org" in result

    def test_limits_snippet_to_400_chars(self):
        sources = [{"title": "Test", "url": "http://test.com", "snippet": "x" * 1000}]
        result = _format_web_sources(sources)
        assert len(result) < 1000


class TestFormatLocalDocs:
    def test_empty_list_returns_empty_string(self):
        assert _format_local_docs([]) == ""

    def test_formats_doc_with_id(self):
        docs = [{"id": "abc123", "document": "This is content", "score": 0.9}]
        result = _format_local_docs(docs)
        assert "[D1]" in result
        assert "This is content" in result
