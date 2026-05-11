# api/oauth.py — Google & GitHub OAuth2 authorization-code flow
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import hashlib
import hmac
import logging
import re
import secrets
import time
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy import select

from config import cfg
from api.auth import issue_tokens
from storage.database import get_session
from storage.models import User

logger = logging.getLogger("helios.api.oauth")

oauth_router = APIRouter(prefix="/api/v1/auth", tags=["oauth"])

# ── Provider definitions ──────────────────────────────────────────────────────

_PROVIDERS: dict[str, dict] = {
    "google": {
        "auth_url":     "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url":    "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://www.googleapis.com/oauth2/v3/userinfo",
        "scope":        "openid email profile",
    },
    "github": {
        "auth_url":     "https://github.com/login/oauth/authorize",
        "token_url":    "https://github.com/login/oauth/access_token",
        "userinfo_url": "https://api.github.com/user",
        "emails_url":   "https://api.github.com/user/emails",
        "scope":        "read:user user:email",
    },
}


def _client_id(provider: str) -> str:
    return cfg.google_client_id if provider == "google" else cfg.github_client_id


def _client_secret(provider: str) -> str:
    s = cfg.google_client_secret if provider == "google" else cfg.github_client_secret
    return s.get_secret_value()


def _callback_uri(provider: str) -> str:
    return f"{cfg.oauth_backend_url}/api/v1/auth/{provider}/callback"


# ── CSRF state helpers ────────────────────────────────────────────────────────

def _make_state(provider: str) -> str:
    ts = str(int(time.time()))
    nonce = secrets.token_hex(8)
    raw = f"{provider}:{ts}:{nonce}"
    sig = hmac.new(cfg.jwt_secret_key.encode(), raw.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{raw}:{sig}"


def _verify_state(state: str, provider: str) -> bool:
    try:
        parts = state.split(":")
        if len(parts) != 4:
            return False
        p, ts, nonce, sig = parts
        if p != provider:
            return False
        if abs(time.time() - int(ts)) > 600:
            return False
        raw = f"{provider}:{ts}:{nonce}"
        expected = hmac.new(cfg.jwt_secret_key.encode(), raw.encode(), hashlib.sha256).hexdigest()[:16]
        return hmac.compare_digest(sig, expected)
    except Exception:
        return False


# ── User upsert ───────────────────────────────────────────────────────────────

_SLUG_RE = re.compile(r"[^a-z0-9_]")


def _safe_username(raw: str) -> str:
    base = _SLUG_RE.sub("", raw.lower().replace(" ", "_"))[:28] or "user"
    return base


async def _get_or_create_oauth_user(
    provider: str,
    oauth_id: str,
    email: str,
    display_name: str,
) -> User:
    """Return an existing user matched by oauth_id or email, or create one."""
    async with get_session() as session:
        # Prefer exact oauth_id match (handles email changes)
        result = await session.execute(
            select(User).where(User.oauth_provider == provider, User.oauth_id == oauth_id)
        )
        user = result.scalar_one_or_none()
        if user:
            return user

        # Fall back to email match (link existing password account)
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        if user:
            user.oauth_provider = provider
            user.oauth_id = oauth_id
            await session.flush()
            await session.refresh(user)
            return user

        # Create fresh OAuth-only account
        base = _safe_username(display_name or email.split("@")[0])
        username = base
        suffix = 1
        while True:
            existing = await session.execute(select(User).where(User.username == username))
            if not existing.scalar_one_or_none():
                break
            username = f"{base}{suffix}"
            suffix += 1

        user = User(
            username=username,
            email=email,
            hashed_password=None,
            oauth_provider=provider,
            oauth_id=oauth_id,
        )
        session.add(user)
        await session.flush()
        await session.refresh(user)
        return user


# ── Generic OAuth helpers ─────────────────────────────────────────────────────

async def _exchange_code(provider: str, code: str) -> str:
    """Exchange authorization code for an access token."""
    p = _PROVIDERS[provider]
    payload = {
        "client_id":     _client_id(provider),
        "client_secret": _client_secret(provider),
        "code":          code,
        "redirect_uri":  _callback_uri(provider),
    }
    if provider == "google":
        payload["grant_type"] = "authorization_code"

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            p["token_url"],
            data=payload,
            headers={"Accept": "application/json"},
        )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise HTTPException(502, "OAuth provider did not return an access token")
    return token


async def _fetch_google_user(access_token: str) -> tuple[str, str, str]:
    """Return (oauth_id, email, display_name)."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            _PROVIDERS["google"]["userinfo_url"],
            headers={"Authorization": f"Bearer {access_token}"},
        )
    resp.raise_for_status()
    data = resp.json()
    return str(data["sub"]), data["email"], data.get("name", "")


async def _fetch_github_user(access_token: str) -> tuple[str, str, str]:
    """Return (oauth_id, email, display_name)."""
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=10) as client:
        profile = (await client.get(_PROVIDERS["github"]["userinfo_url"], headers=headers)).json()
        email = profile.get("email") or ""
        if not email:
            emails_resp = await client.get(_PROVIDERS["github"]["emails_url"], headers=headers)
            for e in emails_resp.json():
                if e.get("primary") and e.get("verified"):
                    email = e["email"]
                    break
    if not email:
        raise HTTPException(400, "GitHub account has no verified public email")
    return str(profile["id"]), email, profile.get("name") or profile.get("login", "")


# ── Routes ────────────────────────────────────────────────────────────────────

def _oauth_start(provider: str) -> RedirectResponse:
    if not _client_id(provider):
        raise HTTPException(501, f"{provider.title()} OAuth is not configured on this server")
    p = _PROVIDERS[provider]
    params = {
        "client_id":     _client_id(provider),
        "redirect_uri":  _callback_uri(provider),
        "scope":         p["scope"],
        "state":         _make_state(provider),
        "response_type": "code",
    }
    if provider == "google":
        params["access_type"] = "online"
    return RedirectResponse(f"{p['auth_url']}?{urlencode(params)}", status_code=302)


async def _oauth_callback(provider: str, code: str | None, state: str | None) -> RedirectResponse:
    error_url = f"{cfg.oauth_frontend_url}?oauth_error=true"
    if not code or not state:
        return RedirectResponse(error_url, status_code=302)
    if not _verify_state(state, provider):
        logger.warning("OAuth CSRF state mismatch for provider=%s", provider)
        return RedirectResponse(error_url, status_code=302)
    try:
        access_token = await _exchange_code(provider, code)
        if provider == "google":
            oauth_id, email, name = await _fetch_google_user(access_token)
        else:
            oauth_id, email, name = await _fetch_github_user(access_token)
        user = await _get_or_create_oauth_user(provider, oauth_id, email, name)
        tokens = await issue_tokens(user)
        redirect = (
            f"{cfg.oauth_frontend_url}"
            f"?access_token={tokens.access_token}"
            f"&refresh_token={tokens.refresh_token}"
        )
        return RedirectResponse(redirect, status_code=302)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("OAuth callback failed for provider=%s: %s", provider, exc)
        return RedirectResponse(error_url, status_code=302)


@oauth_router.get("/google")
async def google_start():
    return _oauth_start("google")


@oauth_router.get("/google/callback")
async def google_callback(code: str | None = None, state: str | None = None):
    return await _oauth_callback("google", code, state)


@oauth_router.get("/github")
async def github_start():
    return _oauth_start("github")


@oauth_router.get("/github/callback")
async def github_callback(code: str | None = None, state: str | None = None):
    return await _oauth_callback("github", code, state)
