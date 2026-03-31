"""BM25 sparse retrieval index for keyword-based search.

Provides keyword-based retrieval to complement dense vector search.
Uses rank-bm25 for scoring and maintains an in-memory index
that is rebuilt from the vector store's documents.
"""

import re
import time
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from src.logging.logger import get_logger

logger = get_logger("retrieval.bm25_store")


def _split_identifier(token: str) -> List[str]:
    """Split a code identifier into sub-tokens.

    Handles snake_case, camelCase, and PascalCase.

    Args:
        token: A single token string (already lowercased).

    Returns:
        List of sub-tokens. Returns [token] if no split is needed.
    """
    # Split on underscores (snake_case)
    if "_" in token:
        parts = [p for p in token.split("_") if p]
        return parts

    # Split on camelCase / PascalCase boundaries
    # Insert split before uppercase letters: "camelCase" → ["camel", "case"]
    sub = re.sub(r"([a-z])([A-Z])", r"\1 \2", token)
    sub = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", sub)
    parts = sub.lower().split()
    if len(parts) > 1:
        return parts

    return [token]


def _tokenize(text: str) -> List[str]:
    """Tokenizer for BM25 that handles both natural language and code.

    Handles mixed Chinese/English text by splitting on whitespace
    and CJK character boundaries. Additionally splits code identifiers
    (snake_case, camelCase, PascalCase) into sub-tokens.

    Args:
        text: Input text string.

    Returns:
        List of lowercase token strings.
    """
    # Split CJK characters into individual tokens, keep latin words together
    raw_tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text)
    tokens = []
    for raw in raw_tokens:
        low = raw.lower()
        # CJK single char — add directly
        if len(raw) == 1 and "\u4e00" <= raw <= "\u9fff":
            tokens.append(raw)
            continue

        # Try splitting code identifiers
        parts = _split_identifier(raw)
        if len(parts) > 1:
            # Add both sub-tokens and the joined form for exact match
            tokens.extend([p.lower() for p in parts])
            joined = "".join(parts).lower()
            if joined != low:
                tokens.append(joined)
        else:
            tokens.append(low)

    return tokens


class BM25Store:
    """BM25 sparse retrieval index for keyword matching.

    Maintains an in-memory BM25 index over a corpus of documents.
    Must be built (or rebuilt) before searching.
    """

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[Dict[str, Any]] = []
        self._corpus_tokens: List[List[str]] = []
        logger.info("BM25Store initialized (empty, call build() to populate)")

    def build(self, documents: List[Dict[str, Any]]) -> None:
        """Build the BM25 index from a list of document dicts.

        Args:
            documents: List of dicts with at least 'chunk_id' and 'content' keys.
                       May also include 'metadata'.
        """
        start = time.perf_counter()

        self._documents = documents
        self._corpus_tokens = [_tokenize(doc["content"]) for doc in documents]
        self._bm25 = BM25Okapi(self._corpus_tokens)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(f"BM25 index built: {len(documents)} documents ({elapsed_ms}ms)")

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search the BM25 index for relevant documents.

        Args:
            query: Query string.
            top_k: Maximum number of results to return.

        Returns:
            List of result dicts with keys: chunk_id, content, score, metadata.
            Sorted by BM25 score descending.
        """
        if self._bm25 is None or not self._documents:
            logger.warning("BM25 index is empty, call build() first")
            return []

        start = time.perf_counter()
        query_tokens = _tokenize(query)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score descending
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for idx, score in scored_indices:
            if score <= 0:
                continue
            doc = self._documents[idx]
            results.append({
                "chunk_id": doc.get("chunk_id", ""),
                "content": doc["content"],
                "score": round(float(score), 4),
                "metadata": doc.get("metadata", {}),
            })

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"BM25 search: '{query[:50]}' → {len(results)} results ({elapsed_ms}ms)"
        )
        return results

    @property
    def document_count(self) -> int:
        """Return the number of documents in the BM25 index."""
        return len(self._documents)

    def build_from_vector_store(self, vector_store) -> None:
        """Convenience: rebuild BM25 index from all documents in a VectorStore.

        Args:
            vector_store: VectorStore instance with a ChromaDB collection.
        """
        start = time.perf_counter()

        collection = vector_store._collection
        total = collection.count()

        if total == 0:
            logger.warning("Vector store is empty, BM25 index will be empty")
            self._documents = []
            self._corpus_tokens = []
            self._bm25 = None
            return

        # Fetch all documents from ChromaDB in batches
        batch_size = 1000
        all_docs = []
        for offset in range(0, total, batch_size):
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            for i, doc_id in enumerate(result["ids"]):
                all_docs.append({
                    "chunk_id": doc_id,
                    "content": result["documents"][i],
                    "metadata": result["metadatas"][i] if result["metadatas"] else {},
                })

        self.build(all_docs)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"BM25 index rebuilt from vector store: "
            f"{len(all_docs)} documents ({elapsed_ms}ms)"
        )
