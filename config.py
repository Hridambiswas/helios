# config.py — Helios Central Configuration
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
from functools import lru_cache
from typing import Any
import json
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    openai_api_key: SecretStr = SecretStr("sk-placeholder")
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── JWT ───────────────────────────────────────────────────────────────────
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    jwt_refresh_expiry_days: int = 7

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "helios"
    postgres_user: str = "helios"
    postgres_password: str = "helios_pass"
    database_url: str = "postgresql+asyncpg://helios:helios_pass@localhost:5432/helios"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── MinIO ─────────────────────────────────────────────────────────────────
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "helios-docs"
    minio_secure: bool = False

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "helios"

    # ── OpenTelemetry ─────────────────────────────────────────────────────────
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "helios"

    # ── App ───────────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"
    log_level: str = "INFO"
    log_format: str = "json"

    # ── Agent Tuning ──────────────────────────────────────────────────────────
    planner_max_subtasks: int = 5
    retriever_top_k: int = 10
    retriever_dense_weight: float = 0.6
    retriever_clip_weight: float = 0.3
    retriever_bm25_weight: float = 0.1
    executor_timeout_seconds: int = 15
    critic_min_score: float = 0.7






    # ── Read Replica (CQRS) ───────────────────────────────────────────────────
    postgres_read_url: str = ""  # Falls back to primary when empty

    # ── API Gateway ───────────────────────────────────────────────────────────
    canary_percentage: int = 0  # % of traffic to canary pipeline (0 = off)

    # ── Backpressure ──────────────────────────────────────────────────────────
    backpressure_queue_depth_threshold: int = 100
    backpressure_active_pipelines_threshold: int = 20

    # ── Bulkhead ──────────────────────────────────────────────────────────────
    bulkhead_executor_limit: int = 3
    bulkhead_retriever_limit: int = 8
    bulkhead_default_limit: int = 10

    # ── Circuit Breaker ───────────────────────────────────────────────────────
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout_seconds: float = 30.0

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_per_user: int = 7
    rate_limit_window_seconds: int = 60

    # ── Security ──────────────────────────────────────────────────────────────
    cors_allowed_origins: list[str] = []
    max_upload_bytes: int = 52_428_800  # 50 MB
    trusted_proxy_count: int = 0
    allowed_upload_extensions: list[str] = [".txt", ".md", ".pdf", ".csv", ".json", ".rst"]

    @field_validator("cors_allowed_origins", "allowed_upload_extensions", mode="before")
    @classmethod
    def parse_str_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                return json.loads(v)
            return [i.strip() for i in v.split(",") if i.strip()]
        return v
    ws_max_message_bytes: int = 65_536      # 64 KB per WebSocket message
    ws_max_connections_per_user: int = 3    # concurrent WS sessions per user

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Convenience alias so callers can do: from config import cfg
cfg = get_settings()
