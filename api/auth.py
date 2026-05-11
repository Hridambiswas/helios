# api/auth.py — Helios JWT authentication
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt as _bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select

from config import cfg
from storage.database import get_session
from storage.models import User, RefreshToken
from api.schemas import TokenResponse

logger = logging.getLogger("helios.api.auth")

_MAX_PASSWORD_BYTES = 72  # bcrypt hard limit

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# ── Password helpers ──────────────────────────────────────────────────────────

_MIN_PASSWORD_LENGTH = 8


def validate_password_strength(password: str) -> list[str]:
    """Return a list of unmet password requirements, empty if valid."""
    errors: list[str] = []
    if len(password) < _MIN_PASSWORD_LENGTH:
        errors.append(f"at least {_MIN_PASSWORD_LENGTH} characters")
    if not any(c.isdigit() for c in password):
        errors.append("at least one digit")
    if not any(c.isupper() for c in password):
        errors.append("at least one uppercase letter")
    if not any(c.islower() for c in password):
        errors.append("at least one lowercase letter")
    return errors


def hash_password(plain: str) -> str:
    return _bcrypt.hashpw(plain.encode(), _bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ── JWT helpers ───────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "type": "access",
        "iat": _utcnow(),
        "exp": _utcnow() + timedelta(minutes=cfg.jwt_expiry_minutes),
    }
    return jwt.encode(payload, cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm)


def create_refresh_token_str(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "type": "refresh",
        "iat": _utcnow(),
        "exp": _utcnow() + timedelta(days=cfg.jwt_refresh_expiry_days),
    }
    return jwt.encode(payload, cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, cfg.jwt_secret_key, algorithms=[cfg.jwt_algorithm])
    except JWTError as exc:
        logger.debug("JWT decode failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── DB helpers ────────────────────────────────────────────────────────────────

async def get_user_by_username(username: str) -> User | None:
    async with get_session() as session:
        result = await session.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()


async def get_user_by_id(user_id: str) -> User | None:
    async with get_session() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()


async def create_user(username: str, email: str, password: str) -> User:
    async with get_session() as session:
        user = User(
            username=username,
            email=email,
            hashed_password=hash_password(password),
        )
        session.add(user)
        await session.flush()
        await session.refresh(user)
        return user


async def issue_tokens(user: User) -> TokenResponse:
    access = create_access_token(user.id)
    refresh = create_refresh_token_str(user.id)
    token_hash = hashlib.sha256(refresh.encode()).hexdigest()

    expires = _utcnow() + timedelta(days=cfg.jwt_refresh_expiry_days)
    async with get_session() as session:
        session.add(RefreshToken(user_id=user.id, token_hash=token_hash, expires_at=expires))

    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=cfg.jwt_expiry_minutes * 60,
    )


# ── FastAPI dependencies ──────────────────────────────────────────────────────

async def get_current_user_from_token_str(token: str) -> User:
    """Used by WebSocket upgrade where token comes from query param."""
    payload = _decode_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Not an access token")
    user = await get_user_by_id(payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


async def get_current_user(token: Annotated[str, Depends(_oauth2_scheme)]) -> User:
    from observability.metrics import auth_failure_counter
    payload = _decode_token(token)
    if payload.get("type") != "access":
        auth_failure_counter.labels(reason="bad_token").inc()
        raise HTTPException(status_code=401, detail="Not an access token")

    user = await get_user_by_id(payload["sub"])
    if not user or not user.is_active:
        auth_failure_counter.labels(reason="inactive_user").inc()
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]

_oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


async def get_current_user_optional(
    token: Annotated[str | None, Depends(_oauth2_scheme_optional)],
) -> User | None:
    if not token:
        return None
    try:
        payload = _decode_token(token)
        if payload.get("type") != "access":
            return None
        return await get_user_by_id(payload["sub"])
    except Exception:
        return None


OptionalUser = Annotated[User | None, Depends(get_current_user_optional)]


async def refresh_access_token(refresh_token: str) -> TokenResponse:
    payload = _decode_token(refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Not a refresh token")

    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
    async with get_session() as session:
        result = await session.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == token_hash,
                RefreshToken.revoked == False,  # noqa: E712
                RefreshToken.expires_at > _utcnow(),
            )
        )
        db_token = result.scalar_one_or_none()
        if not db_token:
            raise HTTPException(status_code=401, detail="Refresh token invalid or expired")

        # Rotate: revoke old, issue new
        db_token.revoked = True

    user = await get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return await issue_tokens(user)


async def get_user_by_email(email: str) -> "User | None":
    async with get_session() as session:
        result = await session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()


def sanitize_username(username: str) -> str:
    """Lowercase and strip a username for case-insensitive comparison."""
    return username.strip().lower()
