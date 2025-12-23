"""
OpenTelemetry instrumentation for observability.

Provides tracing, metrics, and logging integration
for production monitoring.
"""

from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, TypeVar

from genai_project.core.logging import get_logger
from genai_project.core.settings import settings

logger = get_logger(__name__)

# Type variable for generic function decoration
F = TypeVar("F", bound=Callable[..., Any])

# Global tracer instance (lazy initialized)
_tracer = None


def _get_tracer():
    """Get or create the OpenTelemetry tracer."""
    global _tracer

    if _tracer is not None:
        return _tracer

    if not settings.otel_exporter_otlp_endpoint:
        logger.debug("OpenTelemetry not configured, using no-op tracer")
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                "service.name": settings.otel_service_name,
                "deployment.environment": settings.environment,
            }
        )

        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(
            OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
        )
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        _tracer = trace.get_tracer(settings.otel_service_name)
        logger.info("OpenTelemetry initialized", endpoint=settings.otel_exporter_otlp_endpoint)

    except ImportError:
        logger.warning("OpenTelemetry packages not installed, tracing disabled")
        _tracer = None
    except Exception as e:
        logger.error("Failed to initialize OpenTelemetry", error=str(e))
        _tracer = None

    return _tracer


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """
    Context manager for creating trace spans.

    Args:
        name: Span name
        attributes: Optional span attributes

    Example:
        with trace_span("llm_call", {"model": "gpt-4"}):
            response = await client.generate(...)
    """
    tracer = _get_tracer()

    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        try:
            yield span
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            raise


def traced(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """
    Decorator for tracing function execution.

    Args:
        name: Optional span name (defaults to function name)
        attributes: Optional span attributes

    Example:
        @traced("generate_response")
        async def generate(self, prompt: str) -> str:
            ...
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with trace_span(span_name, attributes):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with trace_span(span_name, attributes):
                return func(*args, **kwargs)

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def record_llm_metrics(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
) -> None:
    """
    Record LLM usage metrics.

    Args:
        provider: LLM provider name
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        latency_ms: Request latency in milliseconds
    """
    logger.info(
        "llm_request",
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        latency_ms=round(latency_ms, 2),
    )
