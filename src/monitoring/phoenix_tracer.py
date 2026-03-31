"""Arize Phoenix integration for RAG pipeline observability.

Provides OpenTelemetry-based tracing that sends spans to a local
Phoenix server for visualization. Falls back gracefully when Phoenix
is not installed or the server is not running.

Usage:
    from src.monitoring.phoenix_tracer import init_phoenix_tracing

    # Call once at app startup
    init_phoenix_tracing(config)
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("rag.monitoring.phoenix")

_PHOENIX_INITIALIZED = False


def is_phoenix_available() -> bool:
    """Check whether the phoenix and opentelemetry packages are installed.

    Returns:
        True if all required packages are importable.
    """
    try:
        import phoenix  # noqa: F401
        from opentelemetry import trace  # noqa: F401
        return True
    except ImportError:
        return False


def init_phoenix_tracing(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize Phoenix tracing with OpenTelemetry instrumentation.

    This sets up an OTLP exporter that sends spans to a local Phoenix
    collector endpoint. It also instruments LangChain if the
    ``openinference-instrumentation-langchain`` package is installed.

    Args:
        config: Application config dict. Reads ``monitoring.phoenix``
                section for ``enabled``, ``endpoint``, and ``project_name``.

    Returns:
        True if Phoenix tracing was successfully initialized.
    """
    global _PHOENIX_INITIALIZED
    if _PHOENIX_INITIALIZED:
        return True

    if config is None:
        from src.config import load_config
        config = load_config()

    mon_cfg = config.get("monitoring", {}).get("phoenix", {})
    enabled = mon_cfg.get("enabled", False)

    if not enabled:
        logger.info("Phoenix tracing is disabled in config")
        return False

    if not is_phoenix_available():
        logger.warning(
            "Phoenix tracing enabled in config but packages not installed. "
            "Run: pip install arize-phoenix opentelemetry-sdk "
            "opentelemetry-exporter-otlp openinference-instrumentation-langchain"
        )
        return False

    endpoint = mon_cfg.get("endpoint", "http://localhost:6006/v1/traces")
    project_name = mon_cfg.get("project_name", "rag-knowledge-base")

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": project_name})
        tracer_provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint)
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(tracer_provider)

        # Instrument LangChain automatically if available
        try:
            from openinference.instrumentation.langchain import (
                LangChainInstrumentor,
            )
            LangChainInstrumentor().instrument()
            logger.info("LangChain auto-instrumentation enabled")
        except ImportError:
            logger.info(
                "openinference-instrumentation-langchain not installed, "
                "LangChain auto-instrumentation skipped"
            )

        _PHOENIX_INITIALIZED = True
        logger.info(
            f"Phoenix tracing initialized: endpoint={endpoint}, "
            f"project={project_name}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Phoenix tracing: {e}")
        return False


def shutdown_phoenix_tracing() -> None:
    """Flush and shut down the OpenTelemetry tracer provider."""
    global _PHOENIX_INITIALIZED
    if not _PHOENIX_INITIALIZED:
        return

    try:
        from opentelemetry import trace
        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
        _PHOENIX_INITIALIZED = False
        logger.info("Phoenix tracing shut down")
    except Exception as e:
        logger.error(f"Error shutting down Phoenix tracing: {e}")
