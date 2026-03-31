"""Code-aware chunking using ASTChunk (tree-sitter) with LangChain fallback.

Primary: ASTChunk for AST-based chunking with chunk expansion
(auto-injects filepath → class → function hierarchy headers).
Fallback: LangChain RecursiveCharacterTextSplitter.from_language()
for languages not supported by ASTChunk.
"""

import hashlib
import time
from typing import Dict, List, Optional

from src.config import load_config
from src.logging.logger import get_logger

logger = get_logger("ingestion.code_chunker")

# Languages supported by ASTChunk for AST-based chunking
ASTCHUNK_LANGUAGES = {"python", "java", "csharp", "typescript"}

# Mapping from our language names to LangChain Language enum values
_LANGCHAIN_LANGUAGE_MAP = {
    "python": "PYTHON",
    "javascript": "JS",
    "typescript": "TS",
    "java": "JAVA",
    "go": "GO",
    "rust": "RUST",
    "cpp": "CPP",
    "c": "C",
    "ruby": "RUBY",
    "php": "PHP",
    "sql": "SQL",
    "csharp": "CSHARP",
    "css": "CSS",
    "xml": "HTML",  # LangChain uses HTML for XML-like
}


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


def _chunk_with_astchunk(
    content: str,
    language: str,
    metadata: Dict,
    config: dict,
) -> List[Dict]:
    """Chunk code using ASTChunk (tree-sitter AST-based splitting).

    Args:
        content: Source code text.
        language: Programming language (must be in ASTCHUNK_LANGUAGES).
        metadata: Base metadata dict from the parser.
        config: Config dict with code_ingestion settings.

    Returns:
        List of chunk dicts with 'chunk_id', 'content', 'metadata'.
    """
    from astchunk import ASTChunkBuilder

    code_cfg = config.get("code_ingestion", {})
    max_chunk_size = code_cfg.get("chunk_size", 1500)
    chunk_overlap = code_cfg.get("chunk_overlap", 1)
    chunk_expansion = code_cfg.get("chunk_expansion", True)

    # Map our language name to ASTChunk's expected name
    ast_lang = language
    if language == "typescript":
        ast_lang = "typescript"

    builder = ASTChunkBuilder(
        max_chunk_size=max_chunk_size,
        language=ast_lang,
        metadata_template="default",
    )

    # Build per-call configs for optional features
    call_configs = {
        "chunk_overlap": chunk_overlap,
        "chunk_expansion": chunk_expansion,
    }

    # Add repo-level metadata for chunk expansion headers
    if metadata.get("relative_path"):
        call_configs["repo_level_metadata"] = {
            "filepath": metadata["relative_path"],
        }

    chunks_raw = builder.chunkify(content, **call_configs)

    chunks = []
    source_file = metadata.get("source_file", "")
    for idx, chunk_data in enumerate(chunks_raw):
        chunk_content = chunk_data["content"]
        chunk_id = _generate_chunk_id(source_file, chunk_content)

        chunk_metadata = {
            **metadata,
            "chunk_index": idx,
            "chunk_total": len(chunks_raw),
            "chunker": "astchunk",
        }

        # Merge any AST metadata from ASTChunk
        ast_meta = chunk_data.get("metadata", {})
        if ast_meta:
            if "node_type" in ast_meta:
                chunk_metadata["ast_node_type"] = str(ast_meta["node_type"])

        chunks.append({
            "chunk_id": chunk_id,
            "content": chunk_content,
            "metadata": chunk_metadata,
        })

    return chunks


def _chunk_with_langchain(
    content: str,
    language: str,
    metadata: Dict,
    config: dict,
) -> List[Dict]:
    """Chunk code using LangChain language-aware splitter (fallback).

    Args:
        content: Source code text.
        language: Programming language name.
        metadata: Base metadata dict from the parser.
        config: Config dict with code_ingestion settings.

    Returns:
        List of chunk dicts with 'chunk_id', 'content', 'metadata'.
    """
    from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

    code_cfg = config.get("code_ingestion", {})
    chunk_size = code_cfg.get("chunk_size", 1500)
    chunk_overlap = code_cfg.get("chunk_overlap", 1)
    # LangChain uses character overlap, not AST node overlap
    char_overlap = min(chunk_overlap * 100, chunk_size // 4)

    # Try to use language-specific splitter
    lc_lang_name = _LANGCHAIN_LANGUAGE_MAP.get(language)
    if lc_lang_name and hasattr(Language, lc_lang_name):
        lc_lang = getattr(Language, lc_lang_name)
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lc_lang,
            chunk_size=chunk_size,
            chunk_overlap=char_overlap,
        )
        chunker_name = f"langchain_{lc_lang_name.lower()}"
    else:
        # Pure text fallback with code-friendly separators
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=char_overlap,
            separators=["\nclass ", "\ndef ", "\n\n", "\n", " "],
            length_function=len,
        )
        chunker_name = "langchain_text"

    texts = splitter.split_text(content)

    # Prepend file path header to each chunk for context
    relative_path = metadata.get("relative_path", metadata.get("file_name", ""))
    header = f"# File: {relative_path}\n\n" if relative_path else ""

    chunks = []
    source_file = metadata.get("source_file", "")
    for idx, chunk_text in enumerate(texts):
        full_content = header + chunk_text
        chunk_id = _generate_chunk_id(source_file, full_content)

        chunk_metadata = {
            **metadata,
            "chunk_index": idx,
            "chunk_total": len(texts),
            "chunker": chunker_name,
        }

        chunks.append({
            "chunk_id": chunk_id,
            "content": full_content,
            "metadata": chunk_metadata,
        })

    return chunks


class CodeChunker:
    """Chunks source code using ASTChunk (primary) or LangChain (fallback).

    For languages supported by ASTChunk (python, java, csharp, typescript),
    uses AST-based chunking with chunk expansion headers.
    For other languages, falls back to LangChain language-aware splitter.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()

        self._config = config
        code_cfg = config.get("code_ingestion", {})
        self._ast_languages = set(
            code_cfg.get("ast_languages", list(ASTCHUNK_LANGUAGES))
        )

        logger.info(
            f"CodeChunker initialized: AST languages={sorted(self._ast_languages)}"
        )

    def chunk_code(self, documents: List[Dict]) -> List[Dict]:
        """Chunk a list of parsed code documents.

        Args:
            documents: List of dicts with 'content' and 'metadata' keys,
                       as produced by code_parser.parse_code().

        Returns:
            List of chunk dicts with 'chunk_id', 'content', 'metadata'.
        """
        start = time.perf_counter()
        all_chunks = []

        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            language = metadata.get("language", "unknown")

            if not content.strip():
                continue

            try:
                if language in self._ast_languages:
                    chunks = _chunk_with_astchunk(
                        content, language, metadata, self._config
                    )
                else:
                    chunks = _chunk_with_langchain(
                        content, language, metadata, self._config
                    )
                all_chunks.extend(chunks)
            except Exception as e:
                # If AST chunking fails, try LangChain fallback
                file_name = metadata.get("file_name", "unknown")
                if language in self._ast_languages:
                    logger.warning(
                        f"ASTChunk failed for {file_name} ({language}), "
                        f"falling back to LangChain: {e}"
                    )
                    try:
                        chunks = _chunk_with_langchain(
                            content, language, metadata, self._config
                        )
                        all_chunks.extend(chunks)
                    except Exception as e2:
                        logger.error(
                            f"All chunking failed for {file_name}: {e2}"
                        )
                else:
                    logger.error(f"Chunking failed for {file_name}: {e}")

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"Chunked {len(documents)} code files → {len(all_chunks)} chunks "
            f"({elapsed_ms}ms)"
        )

        return all_chunks
