"""Embedding wrapper using Ollama for vector generation."""

import time
from typing import Dict, List, Optional

from langchain_ollama import OllamaEmbeddings

from src.config import load_config
from src.logging.logger import get_logger

logger = get_logger("embedding")


class Embedder:
    """Wraps Ollama embedding model for document and query embedding.

    Uses bge-m3 by default for Chinese/English bilingual support.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()

        emb_cfg = config.get("embedding", {})
        self._model = emb_cfg.get("model", "bge-m3")
        self._base_url = emb_cfg.get("base_url", "http://localhost:11434")
        self._batch_size = emb_cfg.get("batch_size", 32)

        self._embeddings = OllamaEmbeddings(
            model=self._model,
            base_url=self._base_url,
        )

        logger.info(
            f"Embedder initialized: model={self._model}, "
            f"base_url={self._base_url}"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts in batches.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (list of floats).
        """
        start = time.perf_counter()
        all_embeddings = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            batch_embeddings = self._embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)

            logger.debug(
                f"Embedded batch {i // self._batch_size + 1}: "
                f"{len(batch)} texts"
            )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"Embedded {len(texts)} documents "
            f"(dim={len(all_embeddings[0]) if all_embeddings else 0}, "
            f"{elapsed_ms}ms)"
        )

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Query string to embed.

        Returns:
            Embedding vector (list of floats).
        """
        start = time.perf_counter()
        embedding = self._embeddings.embed_query(text)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        logger.debug(
            f"Embedded query ({len(text)} chars, "
            f"dim={len(embedding)}, {elapsed_ms}ms)"
        )

        return embedding

    @property
    def model_name(self) -> str:
        """Return the embedding model name."""
        return self._model

    @property
    def langchain_embeddings(self) -> OllamaEmbeddings:
        """Return the underlying LangChain embeddings object for direct use."""
        return self._embeddings
