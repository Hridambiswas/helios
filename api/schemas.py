# api/schemas.py — Helios Pydantic request/response schemas
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, EmailStr


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int     # seconds


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    stream: bool = False    # if True, use WebSocket instead


class SubTask(BaseModel):
    id: int
    type: str
    description: str


class PlanResponse(BaseModel):
    query_type: str
    subtasks: list[SubTask]
    requires_retrieval: bool
    requires_code: bool


class RetrievedDoc(BaseModel):
    id: str
    document: str
    metadata: dict[str, Any]
    score: float
    source: str     # dense / clip / bm25


class ExecutionResult(BaseModel):
    stdout: str
    stderr: str
    error: str | None
    success: bool


class CriticScores(BaseModel):
    groundedness: float
    faithfulness: float
    completeness: float
    overall: float
    passed: bool = Field(alias="pass")
    reasoning: str
    suggestions: list[str]

    model_config = {"populate_by_name": True}


class QueryResponse(BaseModel):
    query_id: str
    query: str
    answer: str
    plan: PlanResponse | None
    retrieved_docs: list[RetrievedDoc]
    execution_result: ExecutionResult | None
    critic_scores: CriticScores | None
    critic_passed: bool | None
    latency_ms: float
    status: str


# ── Document ingest ───────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    size_bytes: int
    indexed: bool


# ── History ───────────────────────────────────────────────────────────────────

class QueryHistoryItem(BaseModel):
    id: str
    query_text: str
    answer: str | None
    critic_scores: dict | None
    latency_ms: float | None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str           # ok / degraded / down
    postgres: bool
    redis: bool
    minio: bool
    chroma: bool
    version: str = "1.0.0"
