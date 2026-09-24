# tests/test_config_defaults.py — Guard against silent defaults drift
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations


def test_groq_default_model_is_current():
    """llama-3.3-70b-versatile was retired; make sure we don't ship it again."""
    from config import Settings
    s = Settings()
    assert s.groq_model == "openai/gpt-oss-120b"
    assert "llama-3.3" not in s.groq_model


def test_gemini_defaults_present():
    from config import Settings
    s = Settings()
    assert s.gemini_model == "gemini-2.5-flash"
    assert s.verifier_enabled is True
    assert 0.0 <= s.verifier_min_score <= 1.0
    assert s.verifier_timeout_seconds > 0
