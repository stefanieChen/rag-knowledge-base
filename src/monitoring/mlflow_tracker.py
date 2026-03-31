"""MLflow integration for RAG evaluation and pipeline metrics tracking.

Provides helpers to log evaluation scores, pipeline parameters, and
query-level metrics to a local MLflow tracking server.

Usage:
    from src.monitoring.mlflow_tracker import init_mlflow, log_evaluation_run

    init_mlflow(config)
    log_evaluation_run(framework="ragas", scores={"faithfulness": 0.85}, params={...})
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("rag.monitoring.mlflow")

_MLFLOW_INITIALIZED = False


def is_mlflow_available() -> bool:
    """Check whether the mlflow package is installed.

    Returns:
        True if mlflow is importable.
    """
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


def init_mlflow(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize MLflow tracking with local configuration.

    Args:
        config: Application config dict. Reads ``monitoring.mlflow``
                section for ``enabled``, ``tracking_uri``, and
                ``experiment_name``.

    Returns:
        True if MLflow was successfully initialized.
    """
    global _MLFLOW_INITIALIZED
    if _MLFLOW_INITIALIZED:
        return True

    if config is None:
        from src.config import load_config
        config = load_config()

    mon_cfg = config.get("monitoring", {}).get("mlflow", {})
    enabled = mon_cfg.get("enabled", False)

    if not enabled:
        logger.info("MLflow tracking is disabled in config")
        return False

    if not is_mlflow_available():
        logger.warning(
            "MLflow enabled in config but not installed. "
            "Run: pip install mlflow"
        )
        return False

    tracking_uri = mon_cfg.get("tracking_uri", "mlruns")
    experiment_name = mon_cfg.get("experiment_name", "rag-knowledge-base")

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        _MLFLOW_INITIALIZED = True
        logger.info(
            f"MLflow initialized: uri={tracking_uri}, "
            f"experiment={experiment_name}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {e}")
        return False


def log_evaluation_run(
    framework: str,
    scores: Dict[str, float],
    params: Optional[Dict[str, Any]] = None,
    num_samples: int = 0,
    evaluation_time_s: float = 0.0,
    tags: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Log an evaluation run to MLflow.

    Args:
        framework: Evaluation framework name (e.g., 'ragas', 'deepeval').
        scores: Metric name → score mapping.
        params: Pipeline parameters to log (e.g., model name, top_k).
        num_samples: Number of test samples evaluated.
        evaluation_time_s: Total evaluation time in seconds.
        tags: Additional tags to attach to the run.

    Returns:
        MLflow run ID if successful, None otherwise.
    """
    if not _MLFLOW_INITIALIZED:
        return None

    try:
        import mlflow

        with mlflow.start_run(run_name=f"eval_{framework}") as run:
            # Log parameters
            mlflow.log_param("framework", framework)
            mlflow.log_param("num_samples", num_samples)
            if params:
                for key, value in params.items():
                    mlflow.log_param(key, str(value)[:250])

            # Log metrics with framework prefix
            for metric_name, score in scores.items():
                if isinstance(score, (int, float)):
                    mlflow.log_metric(f"{framework}/{metric_name}", score)

            mlflow.log_metric("evaluation_time_s", evaluation_time_s)

            # Log tags
            mlflow.set_tag("evaluation.framework", framework)
            if tags:
                for tag_key, tag_val in tags.items():
                    mlflow.set_tag(tag_key, tag_val)

            logger.info(
                f"MLflow evaluation run logged: {run.info.run_id} "
                f"({framework}, {len(scores)} metrics)"
            )
            return run.info.run_id

    except Exception as e:
        logger.error(f"Failed to log MLflow evaluation run: {e}")
        return None


def log_query_metrics(
    trace_id: str,
    retrieval_latency_ms: int,
    generation_latency_ms: int,
    total_latency_ms: int,
    num_context_chunks: int,
    retrieval_mode: str,
) -> None:
    """Log per-query metrics to the current MLflow run (if active).

    Args:
        trace_id: RAG trace identifier.
        retrieval_latency_ms: Retrieval stage latency.
        generation_latency_ms: Generation stage latency.
        total_latency_ms: Total query latency.
        num_context_chunks: Number of context chunks used.
        retrieval_mode: 'hybrid' or 'dense'.
    """
    if not _MLFLOW_INITIALIZED:
        return

    try:
        import mlflow

        # Log as step metrics (step = hash of trace_id for ordering)
        step = abs(hash(trace_id)) % (10**9)
        mlflow.log_metric("query/retrieval_latency_ms", retrieval_latency_ms, step=step)
        mlflow.log_metric("query/generation_latency_ms", generation_latency_ms, step=step)
        mlflow.log_metric("query/total_latency_ms", total_latency_ms, step=step)
        mlflow.log_metric("query/num_context_chunks", num_context_chunks, step=step)

    except Exception as e:
        logger.debug(f"Failed to log query metrics: {e}")
