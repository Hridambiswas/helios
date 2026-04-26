# main.py — Helios FastAPI application entry point
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
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
from api.websocket import ws_router
from api.middleware import RequestIDMiddleware, RateLimitMiddleware
from storage.database import create_tables, close_engine
from storage.object_store import ensure_bucket

logger = logging.getLogger("helios.main")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    setup_logging()
    setup_tracing()
    logger.info("Helios starting up (env=%s)", cfg.app_env)

    await create_tables()
    ensure_bucket()
    logger.info("Storage layer ready")

    yield  # ← app runs here

    await close_engine()
    logger.info("Helios shut down cleanly")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Helios",
        description="Distributed Multi-Modal Agentic GenAI Platform",
        version="1.0.0",
        docs_url="/docs" if cfg.is_development else None,
        redoc_url="/redoc" if cfg.is_development else None,
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if cfg.is_development else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # ── Prometheus ────────────────────────────────────────────────────────────
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/health", "/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics")

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")
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
