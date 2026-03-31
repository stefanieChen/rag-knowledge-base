"""Cross-encoder reranker for retrieval refinement.

Uses sentence-transformers CrossEncoder with a reranking model
(default: BAAI/bge-reranker-v2-m3) for high-quality relevance scoring.
"""

import os
import time
import warnings
from typing import Any, Dict, List, Optional

import requests
from sentence_transformers import CrossEncoder

from src.config import load_config
from src.logging.logger import get_logger

logger = get_logger("retrieval.reranker")

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Set environment variable for huggingface hub timeout
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"  # 30 seconds timeout
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Reduce transformers logging


def _resolve_model_path(rerank_cfg: dict) -> str:
    """Resolve the model path: prefer local_model_path, fallback to HF model ID.

    Args:
        rerank_cfg: Reranker config dict.

    Returns:
        Local directory path if it exists and contains weights, else HF model ID.
    """
    local_path = rerank_cfg.get("local_model_path", "")
    if local_path:
        from pathlib import Path
        p = Path(local_path)
        weight_files = ["model.safetensors", "pytorch_model.bin"]
        if p.is_dir() and any((p / w).exists() for w in weight_files):
            return str(p)
        logger.warning(
            f"local_model_path '{local_path}' missing or has no weights, "
            f"falling back to HuggingFace model ID"
        )
    return rerank_cfg.get("model", DEFAULT_RERANKER_MODEL)


class Reranker:
    """Cross-encoder reranker for improving retrieval precision.

    Scores each (query, document) pair independently with a cross-encoder,
    then re-sorts results by relevance score.

    Supports loading from a local model directory (offline) or from HuggingFace.
    Set ``reranker.local_model_path`` in settings.yaml to use a pre-downloaded model.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()

        rerank_cfg = config.get("reranker", {})
        self._top_n = rerank_cfg.get("top_n", 5)
        timeout = rerank_cfg.get("timeout", 30)
        max_retries = rerank_cfg.get("max_retries", 3)
        fallback_enabled = rerank_cfg.get("fallback_to_dense", True)

        model_path = _resolve_model_path(rerank_cfg)
        is_local = os.path.isdir(model_path)

        start = time.perf_counter()

        if is_local:
            # Local model: load directly, no network needed
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                logger.info(f"Loading reranker from local path: {model_path}")
                self._model = CrossEncoder(model_path)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                logger.info(f"Reranker initialized (local): {model_path} ({elapsed_ms}ms)")
                return
            except Exception as e:
                if fallback_enabled:
                    logger.error(f"Failed to load local reranker: {e}")
                    logger.warning("Reranker will be disabled.")
                    self._model = None
                    return
                raise

        # Remote model: retry with network
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout)
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing reranker (attempt {attempt + 1}/{max_retries}): {model_path}")
                self._model = CrossEncoder(model_path)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                logger.info(f"Reranker initialized: model={model_path} ({elapsed_ms}ms)")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    if fallback_enabled:
                        logger.error(f"Failed to initialize reranker after {max_retries} attempts: {e}")
                        logger.warning("Reranker will be disabled. System will use dense-only retrieval.")
                        self._model = None
                    else:
                        raise
                    return
                logger.warning(f"Reranker init failed (attempt {attempt + 1}), retrying in {retry_delay}s: {e}")
                time.sleep(retry_delay)
                retry_delay *= 2

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank retrieval results using cross-encoder scoring.

        Args:
            query: The original query string.
            results: List of retrieval result dicts (must have 'content' key).
            top_n: Number of top results to return after reranking.
                   Defaults to config reranker.top_n.

        Returns:
            Reranked list of result dicts, sorted by reranker score descending,
            with an added 'rerank_score' key. If reranker failed to initialize,
            returns original results unchanged.
        """
        if not results:
            return []
        
        # If reranker failed to initialize, return original results
        if self._model is None:
            logger.warning("Reranker not available, returning original results")
            return results[:top_n] if top_n else results

        if top_n is None:
            top_n = self._top_n

        start = time.perf_counter()

        # Build (query, document) pairs
        pairs = [(query, r["content"]) for r in results]

        # Score all pairs
        scores = self._model.predict(pairs)

        # Attach scores and sort
        scored_results = []
        for i, result in enumerate(results):
            result_copy = dict(result)
            result_copy["rerank_score"] = round(float(scores[i]), 4)
            scored_results.append(result_copy)

        scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_results = scored_results[:top_n]

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"Reranked {len(results)} → {len(top_results)} results ({elapsed_ms}ms)"
        )

        return top_results
