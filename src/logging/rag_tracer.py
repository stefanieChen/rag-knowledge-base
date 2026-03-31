"""RAG full-pipeline tracer for effect retrospection.

Each query produces a structured JSON trace file containing:
- Original and rewritten query
- Retrieval results from each stage (dense, BM25, fused, reranked)
- Generation metadata (model, token counts, latency)
- Total latency and optional user feedback
"""

import json
import os
import time
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import load_config
from src.logging.logger import get_logger

logger = get_logger("tracer")


class RAGTracer:
    """Context manager that collects RAG pipeline data and saves a JSON trace.

    Usage:
        with RAGTracer() as tracer:
            tracer.log_query(raw_query, rewritten_query, language)
            tracer.log_retrieval_dense(results)
            tracer.log_retrieval_bm25(results)
            tracer.log_retrieval_fused(results)
            tracer.log_retrieval_reranked(results, final_chunk_ids)
            tracer.log_generation(model, prompt_template, context_tokens, answer)
        # trace is auto-saved on exit
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()

        log_cfg = config.get("logging", {})
        self._enabled = log_cfg.get("enable_rag_trace", True)
        self._trace_dir = log_cfg.get("trace_dir", "./logs/rag_traces")

        self._trace_id = str(uuid.uuid4())[:12]
        self._timestamp = datetime.now().isoformat()
        self._start_time: float = 0.0
        self._retrieval_start: float = 0.0

        self._data: Dict[str, Any] = {
            "trace_id": self._trace_id,
            "timestamp": self._timestamp,
            "query": {},
            "retrieval": {},
            "generation": {},
            "latency_total_ms": 0,
            "user_feedback": None,
        }

    def __enter__(self) -> "RAGTracer":
        self._start_time = time.perf_counter()
        logger.info(f"RAG trace started: {self._trace_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed_ms = int((time.perf_counter() - self._start_time) * 1000)
        self._data["latency_total_ms"] = elapsed_ms

        if exc_type is not None:
            self._data["error"] = f"{exc_type.__name__}: {exc_val}"
            logger.error(f"RAG trace {self._trace_id} ended with error: {exc_val}")

        self._save()
        logger.info(
            f"RAG trace completed: {self._trace_id} "
            f"(total={elapsed_ms}ms)"
        )

    # ---- Query logging ----

    def log_query(
        self,
        raw_query: str,
        rewritten_query: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        """Log the user query and optional rewrite.

        Args:
            raw_query: Original user question.
            rewritten_query: Query after rewriting (HyDE, multi-query, etc.).
            language: Detected language code (e.g., 'zh', 'en').
        """
        self._data["query"] = {
            "raw": raw_query,
            "rewritten": rewritten_query,
            "language": language,
        }
        logger.info(f"[{self._trace_id}] Query: {raw_query[:100]}")

    # ---- Retrieval logging ----

    def log_retrieval_dense(self, results: List[Dict[str, Any]]) -> None:
        """Log dense retrieval results.

        Args:
            results: List of dicts with keys: chunk_id, score, source, page (optional).
        """
        self._data["retrieval"]["dense_results"] = results
        logger.info(
            f"[{self._trace_id}] Dense retrieval: {len(results)} results"
        )

    def log_retrieval_bm25(self, results: List[Dict[str, Any]]) -> None:
        """Log BM25 sparse retrieval results.

        Args:
            results: List of dicts with keys: chunk_id, score.
        """
        self._data["retrieval"]["bm25_results"] = results
        logger.info(
            f"[{self._trace_id}] BM25 retrieval: {len(results)} results"
        )

    def log_retrieval_fused(self, results: List[Dict[str, Any]]) -> None:
        """Log RRF-fused retrieval results.

        Args:
            results: List of dicts with keys: chunk_id, rrf_score.
        """
        self._data["retrieval"]["fused_results"] = results
        logger.info(
            f"[{self._trace_id}] Fused retrieval: {len(results)} results"
        )

    def log_retrieval_reranked(
        self,
        results: List[Dict[str, Any]],
        final_chunk_ids: Optional[List[str]] = None,
        retrieval_latency_ms: Optional[int] = None,
    ) -> None:
        """Log reranked retrieval results and final context selection.

        Args:
            results: List of dicts with keys: chunk_id, rerank_score.
            final_chunk_ids: IDs of chunks selected for the LLM context.
            retrieval_latency_ms: Total retrieval stage latency in ms.
        """
        self._data["retrieval"]["reranked_results"] = results
        if final_chunk_ids is not None:
            self._data["retrieval"]["final_context_chunks"] = final_chunk_ids
        if retrieval_latency_ms is not None:
            self._data["retrieval"]["retrieval_latency_ms"] = retrieval_latency_ms
        logger.info(
            f"[{self._trace_id}] Reranked: {len(results)} results, "
            f"final context: {len(final_chunk_ids or [])} chunks"
        )

    def log_retrieval_simple(
        self,
        results: List[Dict[str, Any]],
        retrieval_latency_ms: Optional[int] = None,
    ) -> None:
        """Log simple single-stage retrieval (no rerank).

        Args:
            results: List of dicts with keys: chunk_id, score, source, etc.
            retrieval_latency_ms: Total retrieval latency in ms.
        """
        self._data["retrieval"]["dense_results"] = results
        self._data["retrieval"]["final_context_chunks"] = [
            r.get("chunk_id", "") for r in results
        ]
        if retrieval_latency_ms is not None:
            self._data["retrieval"]["retrieval_latency_ms"] = retrieval_latency_ms
        logger.info(
            f"[{self._trace_id}] Retrieval: {len(results)} results "
            f"(latency={retrieval_latency_ms}ms)"
        )

    # ---- Generation logging ----

    def log_generation(
        self,
        model: str,
        prompt_template: str,
        context_token_count: int,
        answer: str,
        generation_latency_ms: Optional[int] = None,
    ) -> None:
        """Log LLM generation details.

        Args:
            model: Model name used for generation.
            prompt_template: Name/version of the prompt template.
            context_token_count: Approximate token count of context fed to LLM.
            answer: Generated answer text.
            generation_latency_ms: Generation latency in ms.
        """
        self._data["generation"] = {
            "model": model,
            "prompt_template": prompt_template,
            "context_token_count": context_token_count,
            "answer": answer,
            "generation_latency_ms": generation_latency_ms,
        }
        logger.info(
            f"[{self._trace_id}] Generation: model={model}, "
            f"ctx_tokens={context_token_count}, "
            f"latency={generation_latency_ms}ms"
        )

    # ---- User feedback ----

    def log_feedback(self, feedback: str) -> None:
        """Log user feedback (e.g., 'thumbs_up', 'thumbs_down').

        Args:
            feedback: User feedback string.
        """
        self._data["user_feedback"] = feedback
        logger.info(f"[{self._trace_id}] Feedback: {feedback}")

    # ---- Persistence ----

    def _save(self) -> None:
        """Save trace data to a JSON file organized by date."""
        if not self._enabled:
            return

        today = date.today().isoformat()
        trace_day_dir = os.path.join(self._trace_dir, today)
        Path(trace_day_dir).mkdir(parents=True, exist_ok=True)

        trace_file = os.path.join(
            trace_day_dir, f"trace_{self._trace_id}.json"
        )
        try:
            with open(trace_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Trace saved: {trace_file}")
        except Exception as e:
            logger.error(f"Failed to save trace {self._trace_id}: {e}")

    @property
    def trace_id(self) -> str:
        """Return the trace ID for this session."""
        return self._trace_id

    @property
    def data(self) -> Dict[str, Any]:
        """Return the raw trace data dict."""
        return self._data
