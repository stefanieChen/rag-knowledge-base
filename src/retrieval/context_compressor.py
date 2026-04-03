"""Context compression utilities for retrieved chunks.

Provides an optional LangChain-based contextual compression stage that trims
retrieved chunks before they are passed to the generator.  This keeps the
prompt focused on the most relevant spans of text while still operating fully
offline via the local Ollama deployment.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from src.config import load_config
from src.logging.logger import get_logger

logger = get_logger("retrieval.context_compressor")


class ContextCompressor:
    """Wraps LangChain contextual compression for retrieved chunks."""

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()

        retrieval_cfg = config.get("retrieval", {})
        compression_cfg = retrieval_cfg.get("context_compression", {})

        self.enabled: bool = bool(compression_cfg.get("enabled", False))
        compressor_llm_cfg = compression_cfg.get("llm", {})
        self._max_chunks: int = int(compressor_llm_cfg.get("max_chunks", 5))

        if not self.enabled:
            self._compressor = None
            logger.info("Context compression disabled")
            return

        llm_cfg = config.get("llm", {})

        model = compressor_llm_cfg.get("model", llm_cfg.get("model", "qwen2.5:7b"))
        base_url = compressor_llm_cfg.get("base_url", llm_cfg.get("base_url", "http://localhost:11434"))
        temperature = float(compressor_llm_cfg.get("temperature", 0.2))
        num_ctx = int(compressor_llm_cfg.get("num_ctx", llm_cfg.get("num_ctx", 4096)))

        logger.info(
            "Initializing contextual compressor: model=%s, base_url=%s, max_chunks=%s",
            model,
            base_url,
            self._max_chunks,
        )

        llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_ctx=num_ctx,
        )
        self._compressor = LLMChainExtractor.from_llm(llm)

    def compress(self, question: str, chunks: List[Dict]) -> List[Dict]:
        """Compress retrieved chunks using the configured LLM extractor.

        Args:
            question: Original user question.
            chunks: Retrieved chunk dicts that will be fed to the generator.

        Returns:
            Potentially reduced list of chunk dicts.
        """
        if not self.enabled or not chunks:
            return chunks

        if not self._compressor:
            return self._truncate(chunks)

        doc_lookup: Dict[str, Dict] = {}
        documents: List[Document] = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                # Skip malformed chunk records but keep them in the output as-is.
                continue
            doc_lookup[chunk_id] = chunk
            metadata = {**chunk.get("metadata", {}), "__chunk_id": chunk_id}
            documents.append(Document(page_content=chunk.get("content", ""), metadata=metadata))

        if not documents:
            return self._truncate(chunks)

        try:
            compressed_docs = self._compressor.compress_documents(
                documents,
                {"question": question},
            )
            if not compressed_docs:
                return self._truncate(chunks)

            compressed_chunks: List[Dict] = []
            for doc in compressed_docs:
                chunk_id = doc.metadata.get("__chunk_id")
                if not chunk_id:
                    continue
                original = doc_lookup.get(chunk_id)
                if not original:
                    continue
                compressed_chunk = {
                    **original,
                    "content": doc.page_content,
                }
                compressed_chunks.append(compressed_chunk)

            return self._truncate(compressed_chunks)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Context compression failed, falling back to truncation: %s", exc)
            return self._truncate(chunks)

    def _truncate(self, chunks: List[Dict]) -> List[Dict]:
        if not chunks or self._max_chunks <= 0:
            return chunks
        if len(chunks) <= self._max_chunks:
            return chunks
        # Preserve ordering by score
        truncated = chunks[: self._max_chunks]
        logger.debug(
            "Context compression truncation applied: %s -> %s chunks",
            len(chunks),
            len(truncated),
        )
        return truncated


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Utility cosine similarity helper used by other modules."""

    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
