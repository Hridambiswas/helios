# observability/logging_config.py — Helios structured JSON logging
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
import sys

import structlog

from config import cfg


def setup_logging() -> None:
    """
    Configure structlog for structured JSON output in production and
    pretty console output in development. Standard library logging
    is piped through structlog so all loggers get the same format.
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if cfg.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
