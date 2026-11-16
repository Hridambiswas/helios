# main.py — Helios FastAPI application entry point
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from config import cfg
from observability.logging_config import setup_logging
from observability.tracing import setup_tracing
from api.routes import router
from api.oauth import oauth_router
from api.websocket import ws_router
from api.middleware import RequestIDMiddleware, RateLimitMiddleware
from api.security import SecurityHeadersMiddleware, AuthBruteForceMiddleware
from gateway.router import GatewayMiddleware
from storage.database import create_tables, close_engine
from storage.object_store import ensure_bucket
from storage.read_replica import close_read_engine

logger = logging.getLogger("helios.main")


# ── Lifespan ──────────────────────────────────────────────────────────────────

async def _warmup_embeddings() -> None:
    """Preload BGE (dense) and CLIP (multimodal) encoders in a worker thread."""
    def _work() -> None:
        from langchain_community.embeddings import FastEmbedEmbeddings
        FastEmbedEmbeddings(model_name=cfg.embedding_model).embed_query("warmup")
        from retrieval.clip_encoder import encode_text
        encode_text("warmup")

    try:
        logger.info("Warming up embedding models (BGE + CLIP) …")
        await asyncio.to_thread(_work)
        logger.info("Embedding models ready")
    except Exception as exc:  # noqa: BLE001 — never crash startup on warm-up
        logger.warning("Embedding warm-up failed, falling back to lazy load: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    setup_logging()
    setup_tracing()
    cfg.validate_secrets()
    logger.info(
        "Helios v1.2.0 starting up (env=%s, host=%s:%s, verifier=%s, groq_model=%s)",
        cfg.app_env, cfg.app_host, cfg.app_port,
        "on" if cfg.verifier_enabled else "off", cfg.groq_model,
    )

    await create_tables()
    ensure_bucket()
    logger.info("Storage layer ready")

    # Warm up embedding models so the first query doesn't pay the
    # ~30s one-shot load cost for BGE + CLIP. Runs off the event loop
    # in a worker thread and never blocks readiness — if it fails the
    # models will simply load lazily on first use as before.
    asyncio.create_task(_warmup_embeddings())

    yield  # ← app runs here

    await close_engine()
    await close_read_engine()
    logger.info("Helios shut down cleanly")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Helios",
        description="Distributed Multi-Modal Agentic GenAI Platform",
        version="1.2.0",
        docs_url="/docs" if cfg.is_development else None,
        redoc_url="/redoc" if cfg.is_development else None,
        openapi_url="/openapi.json" if cfg.is_development else None,
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    origins = ["*"] if cfg.is_development else cfg.cors_origins_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    )

    # ── Middleware (outermost first — executed in reverse order) ─────────────
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(AuthBruteForceMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(GatewayMiddleware)

    # ── Prometheus ────────────────────────────────────────────────────────────
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/health", "/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics")

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")
    app.include_router(oauth_router)
    app.include_router(ws_router)

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s: %s", request.url, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=cfg.app_host,
        port=cfg.app_port,
        reload=cfg.is_development,
        log_level=cfg.log_level.lower(),
    )
