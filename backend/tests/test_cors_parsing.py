# tests/test_cors_parsing.py — Verify Settings.cors_origins_list accepts both
# the JSON-list format written by the deploy scripts and the comma-separated
# legacy format still used in .env.production.example.
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations

import pytest


def _fresh_settings(monkeypatch, value: str):
    """Load config.Settings with a controlled CORS_ALLOWED_ORIGINS value.

    lru_cache-wrapped get_settings must be reset so each test sees fresh env.
    """
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", value)
    import config
    config.get_settings.cache_clear()
    from config import Settings
    return Settings()


def test_cors_accepts_json_list(monkeypatch):
    """This is the format both deploy scripts (digitalocean/first-deploy and
    vm/first-deploy) and .github/workflows/deploy-backend.yml write."""
    s = _fresh_settings(
        monkeypatch,
        '["https://helios-hridam.vercel.app","https://frontend-omega-blush-87.vercel.app"]',
    )
    assert s.cors_origins_list == [
        "https://helios-hridam.vercel.app",
        "https://frontend-omega-blush-87.vercel.app",
    ]


def test_cors_accepts_comma_separated(monkeypatch):
    """Legacy shape used in backend/.env.production.example and
    backend/docker-compose.prod.yml's inline environment block."""
    s = _fresh_settings(
        monkeypatch,
        "https://helios-hridam.vercel.app,https://frontend-omega-blush-87.vercel.app",
    )
    assert s.cors_origins_list == [
        "https://helios-hridam.vercel.app",
        "https://frontend-omega-blush-87.vercel.app",
    ]


def test_cors_accepts_single_origin(monkeypatch):
    s = _fresh_settings(monkeypatch, "https://helios-hridam.vercel.app")
    assert s.cors_origins_list == ["https://helios-hridam.vercel.app"]


def test_cors_empty_string_yields_empty_list(monkeypatch):
    s = _fresh_settings(monkeypatch, "")
    assert s.cors_origins_list == []


def test_cors_deploy_scripts_include_the_vercel_origin(monkeypatch):
    """Guard the exact origin the frontend runs on. If someone edits the
    deploy scripts and drops helios-hridam.vercel.app, browsers see a CORS
    error and re-experience the 'request failed' surface."""
    payload = (
        '["https://helios-hridam.vercel.app",'
        '"https://frontend-omega-blush-87.vercel.app"]'
    )
    s = _fresh_settings(monkeypatch, payload)
    assert "https://helios-hridam.vercel.app" in s.cors_origins_list


@pytest.fixture(autouse=True)
def _clear_cache_after():
    yield
    import config
    config.get_settings.cache_clear()
