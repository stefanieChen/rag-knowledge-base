"""Query cache for reducing latency on repeated or similar queries.

Caches RAG query results keyed by embedding similarity. When a new query
arrives, the cache checks if a sufficiently similar query has been answered
recently and returns the cached result instead of running the full pipeline.

Uses an in-memory LRU-style cache with configurable TTL and similarity threshold.
"""

import hashlib
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from src.config import load_config
from src.logging.logger import get_logger

logger = get_logger("retrieval.query_cache")


class QueryCache:
    """In-memory query result cache with similarity-based lookup.

    Stores (query_text, embedding, result, timestamp) entries and matches
    new queries by cosine similarity against cached embeddings.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        max_size: int = 100,
        ttl_seconds: int = 3600,
        similarity_threshold: float = 0.95,
    ) -> None:
        """Initialize the query cache.

        Args:
            config: Optional config dict. Overrides defaults from config cache section.
            max_size: Maximum number of cached entries.
            ttl_seconds: Time-to-live in seconds for cache entries.
            similarity_threshold: Minimum cosine similarity to consider a cache hit.
        """
        if config is None:
            config = load_config()

        cache_cfg = config.get("cache", {})
        self._max_size = cache_cfg.get("max_size", max_size)
        self._ttl = cache_cfg.get("ttl_seconds", ttl_seconds)
        self._threshold = cache_cfg.get("similarity_threshold", similarity_threshold)
        self._enabled = cache_cfg.get("enabled", True)

        # OrderedDict for LRU eviction: key → (query_text, embedding, result, timestamp)
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._embedder = None  # Lazy-loaded

        self._hits = 0
        self._misses = 0

        logger.info(
            f"QueryCache initialized: max_size={self._max_size}, "
            f"ttl={self._ttl}s, threshold={self._threshold}, "
            f"enabled={self._enabled}"
        )

    def _get_embedder(self):
        """Lazy-load the embedder instance.

        Returns:
            Embedder instance for computing query embeddings.
        """
        if self._embedder is None:
            from src.embedding.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First embedding vector.
            b: Second embedding vector.

        Returns:
            Cosine similarity score between 0 and 1.
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _make_key(self, query: str) -> str:
        """Generate a cache key from query text.

        Args:
            query: Query string.

        Returns:
            SHA-256 hex digest of the normalized query.
        """
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _evict_expired(self) -> int:
        """Remove expired entries from the cache.

        Returns:
            Number of entries evicted.
        """
        now = time.time()
        expired_keys = [
            k for k, v in self._cache.items()
            if now - v["timestamp"] > self._ttl
        ]
        for k in expired_keys:
            del self._cache[k]
        if expired_keys:
            logger.debug(f"Evicted {len(expired_keys)} expired cache entries")
        return len(expired_keys)

    def get(
        self,
        query: str,
        embedder=None,
    ) -> Optional[Dict[str, Any]]:
        """Look up a cached result for a similar query.

        First checks for an exact match (by hash), then falls back to
        embedding similarity search across all cached entries.

        Args:
            query: The new query string.
            embedder: Optional Embedder instance to reuse.

        Returns:
            Cached result dict if found, None otherwise.
        """
        if not self._enabled:
            return None

        self._evict_expired()

        # 1. Exact match (fast path)
        key = self._make_key(query)
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] <= self._ttl:
                self._cache.move_to_end(key)
                self._hits += 1
                logger.info(f"Cache HIT (exact): '{query[:50]}...'")
                return entry["result"]
            else:
                del self._cache[key]

        # 2. Similarity match (embedding-based)
        if not self._cache:
            self._misses += 1
            return None

        emb = embedder or self._get_embedder()
        query_embedding = emb.embed_query(query)

        best_score = 0.0
        best_entry = None

        for cached_key, entry in self._cache.items():
            if time.time() - entry["timestamp"] > self._ttl:
                continue
            score = self._cosine_similarity(query_embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry
                best_key = cached_key

        if best_entry and best_score >= self._threshold:
            self._cache.move_to_end(best_key)
            self._hits += 1
            logger.info(
                f"Cache HIT (similarity={best_score:.4f}): "
                f"'{query[:50]}...' ≈ '{best_entry['query'][:50]}...'"
            )
            return best_entry["result"]

        self._misses += 1
        return None

    def put(
        self,
        query: str,
        result: Dict[str, Any],
        embedder=None,
    ) -> None:
        """Store a query result in the cache.

        Args:
            query: The query string.
            result: The full pipeline result dict.
            embedder: Optional Embedder instance to reuse.
        """
        if not self._enabled:
            return

        # Evict oldest if at capacity
        while len(self._cache) >= self._max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug(f"Cache LRU eviction: {evicted_key}")

        emb = embedder or self._get_embedder()
        query_embedding = emb.embed_query(query)

        key = self._make_key(query)
        self._cache[key] = {
            "query": query,
            "embedding": query_embedding,
            "result": result,
            "timestamp": time.time(),
        }
        self._cache.move_to_end(key)

        logger.debug(f"Cache PUT: '{query[:50]}...' (size={len(self._cache)})")

    def invalidate(self) -> int:
        """Clear all cached entries.

        Returns:
            Number of entries cleared.
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache invalidated: {count} entries cleared")
        return count

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        return len(self._cache)

    @property
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dict with hit/miss counts and hit rate.
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_queries": total,
            "hit_rate": round(hit_rate, 4),
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "similarity_threshold": self._threshold,
        }
