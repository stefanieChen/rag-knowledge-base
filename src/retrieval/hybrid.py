"""Hybrid retrieval with Reciprocal Rank Fusion (RRF).

Combines dense (vector) and sparse (BM25) retrieval results
using RRF for improved retrieval quality. Optionally applies
cross-encoder reranking on the fused results.
"""

import time
from typing import Any, Dict, List, Optional

from src.config import load_config
from src.logging.logger import get_logger
from src.retrieval.bm25_store import BM25Store
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore

logger = get_logger("retrieval.hybrid")


def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for document d = sum(1 / (k + rank_i)) across all lists
    where rank_i is the 1-based rank of d in list i.

    Args:
        result_lists: List of ranked result lists. Each result must
                      have a 'chunk_id' key.
        k: RRF constant (default 60, as per original paper).

    Returns:
        Merged list of result dicts sorted by RRF score descending,
        with an added 'rrf_score' key.
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    for results in result_lists:
        for rank, result in enumerate(results, 1):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in doc_map:
                doc_map[chunk_id] = result

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    merged = []
    for chunk_id in sorted_ids:
        result = dict(doc_map[chunk_id])
        result["rrf_score"] = round(rrf_scores[chunk_id], 6)
        merged.append(result)

    return merged


class HybridRetriever:
    """Combines dense and sparse retrieval with RRF fusion.

    Flow: Query → Dense (VectorStore) + Sparse (BM25) → RRF → optional Rerank → top_n
    Supports separate document and code collections with search scope filtering.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        config: Optional[dict] = None,
        enable_reranker: bool = True,
        code_store: Optional[VectorStore] = None,
    ) -> None:
        if config is None:
            config = load_config()

        self._config = config
        self._doc_store = vector_store
        self._code_store = code_store
        self._bm25_doc = BM25Store()
        self._bm25_code = BM25Store()
        self._enable_reranker = enable_reranker
        self._reranker: Optional[Reranker] = None

        ret_cfg = config.get("retrieval", {})
        self._dense_top_k = ret_cfg.get("top_k", 20)
        self._sparse_top_k = ret_cfg.get("bm25_top_k", 20)
        self._rrf_k = ret_cfg.get("rrf_k", 60)
        self._top_n = ret_cfg.get("top_n", 5)

        # Build BM25 indexes from existing stores
        self._bm25_doc.build_from_vector_store(vector_store)
        if code_store and code_store.count > 0:
            self._bm25_code.build_from_vector_store(code_store)

        # Lazy-load reranker (large model download on first use)
        if enable_reranker:
            try:
                self._reranker = Reranker(config)
            except Exception as e:
                logger.warning(f"Reranker init failed, continuing without: {e}")
                self._reranker = None

        logger.info(
            f"HybridRetriever initialized: "
            f"dense_top_k={self._dense_top_k}, "
            f"sparse_top_k={self._sparse_top_k}, "
            f"reranker={'enabled' if self._reranker else 'disabled'}, "
            f"code_store={'yes' if code_store else 'no'}"
        )

    def search(
        self,
        query: str,
        top_n: Optional[int] = None,
        search_scope: str = "all",
    ) -> List[Dict[str, Any]]:
        """Run hybrid retrieval: dense + sparse → RRF → optional rerank.

        Args:
            query: User query string.
            top_n: Number of final results to return.
            search_scope: Search scope — "all", "docs", or "code".

        Returns:
            List of result dicts with keys: chunk_id, content, score,
            metadata, rrf_score, and optionally rerank_score.
        """
        if top_n is None:
            top_n = self._top_n

        start = time.perf_counter()

        # 1. Dense retrieval (vector similarity) — scope-aware
        # Use configured similarity_threshold (not 0.0) to filter irrelevant results
        dense_doc_results = []
        dense_code_results = []
        if search_scope in ("all", "docs"):
            dense_doc_results = self._doc_store.search(
                query=query,
                top_k=self._dense_top_k,
            )
        if search_scope in ("all", "code") and self._code_store:
            dense_code_results = self._code_store.search(
                query=query,
                top_k=self._dense_top_k,
            )

        # 2. Sparse retrieval (BM25) — scope-aware
        sparse_doc_results = []
        sparse_code_results = []
        if search_scope in ("all", "docs"):
            sparse_doc_results = self._bm25_doc.search(
                query=query,
                top_k=self._sparse_top_k,
            )
        if search_scope in ("all", "code") and self._bm25_code.document_count > 0:
            sparse_code_results = self._bm25_code.search(
                query=query,
                top_k=self._sparse_top_k,
            )

        # 3. RRF fusion — use separate lists per source so doc and code
        #    get equal weight instead of code dominating both merged lists
        rrf_lists = []
        if dense_doc_results:
            rrf_lists.append(dense_doc_results)
        if dense_code_results:
            rrf_lists.append(dense_code_results)
        if sparse_doc_results:
            rrf_lists.append(sparse_doc_results)
        if sparse_code_results:
            rrf_lists.append(sparse_code_results)

        fused = reciprocal_rank_fusion(rrf_lists, k=self._rrf_k) if rrf_lists else []

        dense_total = len(dense_doc_results) + len(dense_code_results)
        sparse_total = len(sparse_doc_results) + len(sparse_code_results)
        logger.info(
            f"Hybrid fusion ({search_scope}): {dense_total} dense + "
            f"{sparse_total} sparse → {len(fused)} fused "
            f"(RRF lists: {len(rrf_lists)})"
        )

        # 4. Optional reranking
        if self._reranker and fused:
            candidates = fused[:top_n * 2]
            results = self._reranker.rerank(query, candidates, top_n=top_n)
        else:
            results = fused[:top_n]

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"Hybrid search complete: '{query[:50]}' → "
            f"{len(results)} results ({elapsed_ms}ms)"
        )

        return results
