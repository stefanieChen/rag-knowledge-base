"""Document chunking strategies for the RAG pipeline."""

import hashlib
import time
from typing import Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import load_config
from src.logging.logger import get_logger

logger = get_logger("ingestion.chunker")


class DocumentChunker:
    """Chunks parsed documents using recursive character splitting.

    Generates unique chunk IDs and preserves source metadata.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()

        chunk_cfg = config.get("chunking", {})
        self._chunk_size = chunk_cfg.get("chunk_size", 512)
        self._chunk_overlap = chunk_cfg.get("chunk_overlap", 64)
        separators = chunk_cfg.get(
            "separators", ["\n\n", "\n", "。", ".", " "]
        )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=separators,
            length_function=len,
        )

        logger.info(
            f"Chunker initialized: size={self._chunk_size}, "
            f"overlap={self._chunk_overlap}"
        )

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Split documents into smaller chunks with metadata.

        Args:
            documents: List of dicts with 'content' and 'metadata' keys,
                       as produced by the parsers.

        Returns:
            List of chunk dicts with 'chunk_id', 'content', 'metadata'.
        """
        start = time.perf_counter()
        all_chunks = []

        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            if not content.strip():
                continue

            texts = self._splitter.split_text(content)

            for idx, chunk_text in enumerate(texts):
                chunk_id = _generate_chunk_id(
                    metadata.get("source_file", ""), chunk_text
                )
                chunk_metadata = {
                    **metadata,
                    "chunk_index": idx,
                    "chunk_total": len(texts),
                }
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "content": chunk_text,
                    "metadata": chunk_metadata,
                })

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"Chunked {len(documents)} documents → {len(all_chunks)} chunks "
            f"({elapsed_ms}ms)"
        )

        return all_chunks


def _generate_chunk_id(source_file: str, content: str) -> str:
    """Generate a deterministic chunk ID from source file + content hash.

    Args:
        source_file: Path of the source file.
        content: Chunk text content.

    Returns:
        A short hex hash string as chunk ID.
    """
    raw = f"{source_file}::{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
