"""ChromaDB vector store operations for document storage and retrieval."""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

from src.config import load_config
from src.embedding.embedder import Embedder
from src.logging.logger import get_logger

logger = get_logger("retrieval.vector_store")


class VectorStore:
    """Manages ChromaDB collection for storing and querying document embeddings.

    Supports adding documents with metadata, similarity search,
    and incremental updates via chunk ID deduplication.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        config: Optional[dict] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """Initialize VectorStore with a ChromaDB collection.

        Args:
            embedder: Optional Embedder instance to reuse.
            config: Optional config dict.
            collection_name: Override collection name. If None, uses config default.
        """
        if config is None:
            config = load_config()

        vs_cfg = config.get("vector_store", {})
        persist_dir = vs_cfg.get("persist_directory", "./data/chromadb")
        if collection_name is None:
            collection_name = vs_cfg.get("collection_name", "rag_knowledge_base")

        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection_name = collection_name
        self._embedder = embedder or Embedder(config)

        logger.info(
            f"VectorStore initialized: collection='{collection_name}', "
            f"persist='{persist_dir}', "
            f"existing_count={self._collection.count()}"
        )

    def add_chunks(self, chunks: List[Dict]) -> int:
        """Add document chunks to the vector store with deduplication.

        Args:
            chunks: List of dicts with 'chunk_id', 'content', 'metadata'.

        Returns:
            Number of new chunks actually added.
        """
        start = time.perf_counter()

        # First deduplicate within the current batch
        unique_chunks = {}
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            if chunk_id not in unique_chunks:
                unique_chunks[chunk_id] = chunk
        chunks = list(unique_chunks.values())
        
        # Deduplicate against existing IDs
        chunk_ids = [c["chunk_id"] for c in chunks]
        existing = set()
        try:
            result = self._collection.get(ids=chunk_ids)
            existing = set(result["ids"]) if result["ids"] else set()
        except Exception:
            pass

        new_chunks = [c for c in chunks if c["chunk_id"] not in existing]

        if not new_chunks:
            logger.info("No new chunks to add (all deduplicated)")
            return 0

        texts = [c["content"] for c in new_chunks]
        ids = [c["chunk_id"] for c in new_chunks]
        metadatas = [_flatten_metadata(c["metadata"]) for c in new_chunks]

        # Embed in batches
        embeddings = self._embedder.embed_documents(texts)

        # Add to ChromaDB
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"Added {len(new_chunks)} chunks to vector store "
            f"(skipped {len(existing)} existing duplicates, {elapsed_ms}ms). "
            f"Total: {self._collection.count()}"
        )

        return len(new_chunks)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents by query text.

        Args:
            query: Query string.
            top_k: Number of results to return. Defaults to config value.
            similarity_threshold: Minimum similarity score. Defaults to config value.

        Returns:
            List of result dicts with keys: chunk_id, content, score, metadata.
        """
        config = load_config()
        ret_cfg = config.get("retrieval", {})
        if top_k is None:
            top_k = ret_cfg.get("top_k", 20)
        if similarity_threshold is None:
            similarity_threshold = ret_cfg.get("similarity_threshold", 0.35)

        start = time.perf_counter()
        query_embedding = self._embedder.embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns cosine distance; convert to similarity
                distance = results["distances"][0][i]
                score = 1.0 - distance

                if score < similarity_threshold:
                    continue

                output.append({
                    "chunk_id": chunk_id,
                    "content": results["documents"][0][i],
                    "score": round(score, 4),
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"Search: '{query[:50]}...' → {len(output)} results "
            f"(top_k={top_k}, threshold={similarity_threshold}, "
            f"{elapsed_ms}ms)"
        )

        if not output:
            logger.warning(
                f"No results above threshold for query: '{query[:80]}'"
            )

        return output

    def get_file_hashes(self) -> Dict[str, str]:
        """Get a mapping of source_file → file_hash from the collection.

        Scans all stored chunks and returns the file_hash for each unique source_file.
        Used for incremental indexing to detect which files have changed.

        Returns:
            Dict mapping source_file path to its content hash string.
        """
        total = self._collection.count()
        if total == 0:
            return {}

        # Fetch all metadata (ChromaDB get without IDs returns all)
        result = self._collection.get(
            include=["metadatas"],
            limit=total,
        )

        file_hashes: Dict[str, str] = {}
        if result and result["metadatas"]:
            for meta in result["metadatas"]:
                src = meta.get("source_file", "")
                fh = meta.get("file_hash", "")
                if src and fh:
                    file_hashes[src] = fh

        logger.debug(f"Retrieved file hashes for {len(file_hashes)} files")
        return file_hashes

    def delete_by_source_file(self, source_file: str) -> int:
        """Delete all chunks belonging to a specific source file.

        Args:
            source_file: The source_file metadata value to match.

        Returns:
            Number of chunks deleted.
        """
        start = time.perf_counter()

        # Find all chunk IDs with this source_file
        result = self._collection.get(
            where={"source_file": source_file},
            include=[],
        )

        if not result or not result["ids"]:
            return 0

        ids_to_delete = result["ids"]
        self._collection.delete(ids=ids_to_delete)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"Deleted {len(ids_to_delete)} chunks for '{source_file}' ({elapsed_ms}ms). "
            f"Remaining: {self._collection.count()}"
        )
        return len(ids_to_delete)

    def delete_by_metadata(self, where: Dict) -> int:
        """Delete all chunks matching a ChromaDB where filter.

        Args:
            where: ChromaDB where clause dict, e.g. {"repo_name": "my_repo"}.

        Returns:
            Number of chunks deleted.
        """
        start = time.perf_counter()

        result = self._collection.get(
            where=where,
            include=[],
        )

        if not result or not result["ids"]:
            return 0

        ids_to_delete = result["ids"]
        self._collection.delete(ids=ids_to_delete)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"Deleted {len(ids_to_delete)} chunks matching {where} ({elapsed_ms}ms). "
            f"Remaining: {self._collection.count()}"
        )
        return len(ids_to_delete)

    @property
    def count(self) -> int:
        """Return total number of chunks in the collection."""
        return self._collection.count()

    @property
    def collection_name(self) -> str:
        """Return the name of the ChromaDB collection."""
        return self._collection_name


def _flatten_metadata(metadata: Dict) -> Dict:
    """Flatten metadata values to ChromaDB-compatible types (str, int, float, bool).

    Args:
        metadata: Original metadata dict.

    Returns:
        Flattened metadata dict.
    """
    flat = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif value is None:
            flat[key] = ""
        else:
            flat[key] = str(value)
    return flat
