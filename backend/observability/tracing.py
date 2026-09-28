# observability/tracing.py — Helios OpenTelemetry distributed tracing
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Span, Status, StatusCode

from config import cfg

logger = logging.getLogger("helios.observability.tracing")

_tracer: trace.Tracer | None = None


def setup_tracing() -> None:
    """Initialise the OTEL tracer and wire it to the OTLP gRPC exporter.

    When OTEL_EXPORTER_OTLP_ENDPOINT is empty, install a no-op tracer:
    no BatchSpanProcessor is registered, so spans are dropped locally
    without any background thread trying (and failing) to connect to a
    collector. Required for the HF Space deploy where there is no OTel
    collector reachable inside the container.
    """
    global _tracer

    resource = Resource(attributes={SERVICE_NAME: cfg.otel_service_name})
    provider = TracerProvider(resource=resource)

    if cfg.otel_exporter_otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=cfg.otel_exporter_otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(cfg.otel_service_name)
        logger.info("OpenTelemetry tracing initialised → %s", cfg.otel_exporter_otlp_endpoint)
    else:
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(cfg.otel_service_name)
        logger.info("OpenTelemetry tracing in no-op mode (OTEL_EXPORTER_OTLP_ENDPOINT empty)")


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        if cfg.is_production:
            setup_tracing()
        else:
            _tracer = trace.get_tracer(cfg.otel_service_name)
    return _tracer


@contextmanager
def span(name: str, attributes: dict | None = None) -> Generator[Span, None, None]:
    """
    Convenience context manager creating a child span.

    Usage:
        with span("retriever.bm25", {"query": q}) as s:
            results = bm25.search(q)
            s.set_attribute("num_results", len(results))
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as s:
        if attributes:
            for k, v in attributes.items():
                s.set_attribute(k, str(v))
        try:
            yield s
        except Exception as exc:
            s.set_status(Status(StatusCode.ERROR, str(exc)))
            s.record_exception(exc)
            raise


def inject_celery_context(headers: dict) -> dict:
    """
    Inject current trace context into Celery task headers so spans
    propagate across worker boundaries.
    """
    from opentelemetry.propagate import inject
    inject(headers)
    return headers


def extract_celery_context(headers: dict) -> None:
    """Restore trace context on the Celery worker side."""
    from opentelemetry.propagate import extract
    from opentelemetry import context as otel_context
    ctx = extract(headers)
    otel_context.attach(ctx)
