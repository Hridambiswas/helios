# api/auth.py — Helios JWT authentication
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select

from config import cfg
from storage.database import get_session
from storage.models import User, RefreshToken
from api.schemas import TokenResponse

logger = logging.getLogger("helios.api.auth")

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
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

    from datetime import timezone
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
    payload = _decode_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Not an access token")

    user = await get_user_by_id(payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


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
