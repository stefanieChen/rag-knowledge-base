"""Source code file parser for the RAG pipeline.

Reads source code files, detects programming language from extension,
and produces document dicts with code-specific metadata.
"""

import hashlib
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.logging.logger import get_logger

logger = get_logger("ingestion.code_parser")

# Extension → (language name for ASTChunk/LangChain, display format)
EXTENSION_LANGUAGE_MAP: Dict[str, Dict[str, str]] = {
    ".py": {"language": "python", "format": "python"},
    ".js": {"language": "javascript", "format": "javascript"},
    ".ts": {"language": "typescript", "format": "typescript"},
    ".tsx": {"language": "typescript", "format": "typescript"},
    ".jsx": {"language": "javascript", "format": "javascript"},
    ".java": {"language": "java", "format": "java"},
    ".cs": {"language": "csharp", "format": "csharp"},
    ".go": {"language": "go", "format": "go"},
    ".rs": {"language": "rust", "format": "rust"},
    ".cpp": {"language": "cpp", "format": "cpp"},
    ".c": {"language": "c", "format": "c"},
    ".h": {"language": "c", "format": "c_header"},
    ".hpp": {"language": "cpp", "format": "cpp_header"},
    ".rb": {"language": "ruby", "format": "ruby"},
    ".php": {"language": "php", "format": "php"},
    ".sh": {"language": "bash", "format": "shell"},
    ".sql": {"language": "sql", "format": "sql"},
    ".yaml": {"language": "yaml", "format": "yaml"},
    ".yml": {"language": "yaml", "format": "yaml"},
    ".json": {"language": "json", "format": "json"},
    ".toml": {"language": "toml", "format": "toml"},
    ".xml": {"language": "xml", "format": "xml"},
    ".css": {"language": "css", "format": "css"},
    ".scss": {"language": "scss", "format": "scss"},
}


def detect_language(file_path: str) -> Dict[str, str]:
    """Detect programming language from file extension.

    Args:
        file_path: Path to the source code file.

    Returns:
        Dict with 'language' and 'format' keys.
        Returns language='unknown' for unrecognised extensions.
    """
    ext = Path(file_path).suffix.lower()
    return EXTENSION_LANGUAGE_MAP.get(ext, {"language": "unknown", "format": "unknown"})


def parse_code(
    file_path: str,
    relative_path: Optional[str] = None,
    repo_name: Optional[str] = None,
) -> List[Dict]:
    """Parse a source code file into a document dict with metadata.

    Args:
        file_path: Absolute path to the source code file.
        relative_path: Relative path within the repository (for metadata).
        repo_name: Name of the repository (for metadata).

    Returns:
        List containing a single dict with 'content' and 'metadata' keys.
        Returns empty list on error.
    """
    start = time.perf_counter()
    file_path = str(Path(file_path).resolve())

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    # Read file content with encoding fallback
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="gbk") as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Cannot read file (encoding error): {file_path}: {e}")
            return []

    if not content.strip():
        logger.debug(f"Empty file skipped: {file_path}")
        return []

    # Compute file content hash for incremental indexing
    file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    # Detect language
    lang_info = detect_language(file_path)

    # Build metadata
    metadata = {
        "source_file": file_path,
        "file_name": os.path.basename(file_path),
        "file_hash": file_hash,
        "content_type": "code",
        "format": lang_info["format"],
        "language": lang_info["language"],
    }

    if relative_path:
        metadata["relative_path"] = relative_path
    if repo_name:
        metadata["repo_name"] = repo_name

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.debug(
        f"Parsed code: {os.path.basename(file_path)} "
        f"({lang_info['language']}, {len(content)} chars, {elapsed_ms}ms)"
    )

    return [{"content": content, "metadata": metadata}]
