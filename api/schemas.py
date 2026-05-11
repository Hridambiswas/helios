# api/schemas.py — Helios Pydantic request/response schemas
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, EmailStr, field_validator


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def username_safe_chars(cls, v: str) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", v):
            raise ValueError("Username may only contain letters, digits, underscores, and hyphens")
        return v

    @field_validator("password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        if not re.search(r"[A-Za-z]", v):
            raise ValueError("Password must contain at least one letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        return v


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


class LogoutRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Query ─────────────────────────────────────────────────────────────────────

class HistoryMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str = Field(..., max_length=4096)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    stream: bool = False    # if True, use WebSocket instead
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)

    @field_validator("query")
    @classmethod
    def strip_control_chars(cls, v: str) -> str:
        # Remove null bytes and other ASCII control chars (except tab/newline/CR)
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v)


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


class WebSource(BaseModel):
    title: str
    url: str
    snippet: str


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
    web_sources: list[WebSource] = []
    execution_result: ExecutionResult | None
    critic_scores: CriticScores | None
    critic_passed: bool | None
    follow_up_questions: list[str] = []
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


# ── Task status (async pipeline polling) ─────────────────────────────────────

class TaskStatusResponse(BaseModel):
    task_id: str
    query_id: str | None = None
    celery_state: str           # PENDING / STARTED / SUCCESS / FAILURE / RETRY
    status: str                 # queued / running / done / failed
    result: dict | None = None  # present when celery_state == SUCCESS


# ── Document chunk preview ────────────────────────────────────────────────────

class ChunkPreview(BaseModel):
    chunk_index: int
    text: str
    char_count: int


class DocumentChunksResponse(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    chunks: list[ChunkPreview]


class TestRetrievalResult(BaseModel):
    chunk_index: int
    text: str
    score: float
    source: str


class TestRetrievalResponse(BaseModel):
    query: str
    results: list[TestRetrievalResult]


# ── Conversations ─────────────────────────────────────────────────────────────

class ConversationMessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    model_config = {"from_attributes": True}


class ConversationDetailOut(ConversationOut):
    messages: list[ConversationMessageOut] = []


class CreateConversationRequest(BaseModel):
    title: str = Field("New Chat", max_length=255)


class AppendMessageRequest(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=32768)


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str           # ok / degraded / down
    postgres: bool
    redis: bool
    minio: bool
    chroma: bool
    version: str = "1.0.0"
