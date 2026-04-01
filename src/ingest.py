"""Document ingestion script: parse files from data/raw/ and add to vector store."""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import load_config
from src.ingestion.chunker import DocumentChunker
from src.ingestion.code_chunker import CodeChunker
from src.ingestion.code_parser import parse_code
from src.ingestion.markdown_parser import parse_markdown
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.pptx_parser import parse_pptx
from src.ingestion.onenote_parser import parse_onenote
from src.ingestion.repo_loader import load_repo
from src.ingestion.txt_parser import parse_txt
from src.logging.logger import get_logger, setup_logging
from src.retrieval.vector_store import VectorStore

logger = get_logger("ingestion")

# Parser registry: file extension → parser function
# PDF and PPTX parsers accept (file_path, extract_images, config) but
# ingest_file wraps them to match the common signature.
PARSERS = {
    ".txt": parse_txt,
    ".md": parse_markdown,
    ".pdf": parse_pdf,
    ".pptx": parse_pptx,
    ".htm": parse_onenote,
    ".html": parse_onenote,
}


def discover_files(raw_dir: str) -> List[str]:
    """Discover all supported files in the raw data directory.

    Args:
        raw_dir: Path to the raw documents directory.

    Returns:
        List of absolute file paths.
    """
    supported_extensions = set(PARSERS.keys())
    files = []

    for root, _, filenames in os.walk(raw_dir):
        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext in supported_extensions:
                files.append(os.path.join(root, filename))

    logger.info(
        f"Discovered {len(files)} supported files in {raw_dir}"
    )
    return files


def ingest_file(
    file_path: str,
    chunker: DocumentChunker,
    config: Optional[dict] = None,
    extract_images: bool = True,
) -> List[Dict]:
    """Parse and chunk a single file.

    Args:
        file_path: Absolute path to the file.
        chunker: DocumentChunker instance.
        config: Optional config dict (passed to PDF/PPTX parsers).
        extract_images: If True, extract and describe images from PDF/PPTX.

    Returns:
        List of chunk dicts ready for vector store insertion.
    """
    ext = Path(file_path).suffix.lower()
    parser = PARSERS.get(ext)

    if parser is None:
        logger.warning(f"No parser for extension '{ext}': {file_path}")
        return []

    try:
        # PDF, PPTX, and OneNote parsers accept extra kwargs
        if ext in (".pdf", ".pptx", ".htm", ".html", ".md"):
            documents = parser(
                file_path,
                extract_images=extract_images,
                config=config,
            )
        else:
            documents = parser(file_path)

        chunks = chunker.chunk_documents(documents)
        return chunks
    except Exception as e:
        logger.error(f"Failed to ingest {file_path}: {e}")
        return []


def run_ingestion(raw_dir: str = None) -> None:
    """Run the full ingestion pipeline.

    Args:
        raw_dir: Optional path to raw documents directory.
                 Defaults to config data.raw_dir.
    """
    setup_logging()
    config = load_config()

    if raw_dir is None:
        raw_dir = config.get("data", {}).get("raw_dir", "./data/raw")

    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    logger.info(f"=== Ingestion started: {raw_dir} ===")

    files = discover_files(raw_dir)
    if not files:
        logger.warning(
            f"No supported files found in {raw_dir}. "
            f"Supported formats: {list(PARSERS.keys())}"
        )
        return

    chunker = DocumentChunker(config)
    vector_store = VectorStore(config=config)

    total_chunks = 0
    total_added = 0

    for file_path in files:
        chunks = ingest_file(file_path, chunker, config=config)
        if chunks:
            added = vector_store.add_chunks(chunks)
            total_chunks += len(chunks)
            total_added += added

    elapsed = time.perf_counter() - start
    logger.info(
        f"=== Ingestion completed: {len(files)} files, "
        f"{total_chunks} chunks processed, {total_added} new chunks added, "
        f"{elapsed:.1f}s ==="
    )


def _compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file's content (first 16 hex chars).

    Args:
        file_path: Absolute path to the file.

    Returns:
        Short hex hash string, or empty string on error or empty file.
    """
    import hashlib
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="gbk") as f:
                content = f.read()
        except Exception:
            return ""
    except Exception:
        return ""
    if not content.strip():
        return ""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def run_code_ingestion(
    path: str,
    repo_name: Optional[str] = None,
    config: Optional[dict] = None,
    vector_store: Optional[VectorStore] = None,
) -> Tuple[int, int, int]:
    """Run incremental code ingestion from a local folder or GitHub URL.

    Compares file content hashes with existing data in the vector store.
    - **Unchanged** files are skipped entirely.
    - **Modified** files have old chunks deleted, then re-ingested.
    - **Deleted** files (present in store but not on disk) have their chunks removed.

    Args:
        path: Local folder path or Git repository URL.
        repo_name: Optional name to identify the repo. Auto-detected if None.
        config: Optional config dict.
        vector_store: Optional existing VectorStore to reuse.

    Returns:
        Tuple of (file_count, total_chunks, total_added).
    """
    setup_logging()
    if config is None:
        config = load_config()

    start = time.perf_counter()
    logger.info(f"=== Code ingestion started (incremental): {path} ===")

    # Load repo (clone if URL, or resolve local path)
    repo_name, folder, files = load_repo(path, repo_name, config)

    if not files:
        logger.warning(f"No supported code files found in {path}")
        return 0, 0, 0

    logger.info(f"Repository '{repo_name}': {len(files)} code files discovered")

    code_chunker = CodeChunker(config)
    if vector_store is None:
        vs_cfg = config.get("vector_store", {})
        code_collection = vs_cfg.get("code_collection_name", "rag_code_base")
        vector_store = VectorStore(
            config=config, collection_name=code_collection
        )

    # --- Incremental indexing: compare file hashes ---
    existing_hashes = vector_store.get_file_hashes()
    current_files_set = set()

    files_to_ingest = []
    skipped = 0
    updated = 0

    for file_info in files:
        abs_path = file_info["absolute_path"]
        current_files_set.add(abs_path)
        new_hash = _compute_file_hash(abs_path)

        if not new_hash:
            continue

        old_hash = existing_hashes.get(abs_path, "")
        if old_hash == new_hash:
            skipped += 1
            continue

        # File is new or changed — queue for ingestion
        if old_hash:
            # Modified: delete old chunks first
            vector_store.delete_by_source_file(abs_path)
            updated += 1

        files_to_ingest.append(file_info)

    # --- Clean up deleted files ---
    deleted_count = 0
    for existing_file in existing_hashes:
        if existing_file not in current_files_set:
            vector_store.delete_by_source_file(existing_file)
            deleted_count += 1

    logger.info(
        f"Incremental analysis: {skipped} unchanged, "
        f"{len(files_to_ingest)} to ingest ({updated} updated), "
        f"{deleted_count} deleted files cleaned"
    )

    if not files_to_ingest:
        elapsed = time.perf_counter() - start
        logger.info(
            f"=== Code ingestion completed (no changes): repo='{repo_name}', "
            f"{elapsed:.1f}s ==="
        )
        return len(files), 0, 0

    # --- Ingest new/changed files ---
    total_chunks = 0
    total_added = 0

    for file_info in files_to_ingest:
        abs_path = file_info["absolute_path"]
        rel_path = file_info["relative_path"]

        documents = parse_code(
            abs_path,
            relative_path=rel_path,
            repo_name=repo_name,
        )

        if documents:
            chunks = code_chunker.chunk_code(documents)
            if chunks:
                added = vector_store.add_chunks(chunks)
                total_chunks += len(chunks)
                total_added += added

    elapsed = time.perf_counter() - start
    logger.info(
        f"=== Code ingestion completed: repo='{repo_name}', "
        f"{len(files)} files scanned, {len(files_to_ingest)} ingested, "
        f"{skipped} skipped, {total_chunks} chunks, "
        f"{total_added} new, {elapsed:.1f}s ==="
    )

    return len(files_to_ingest), total_chunks, total_added


if __name__ == "__main__":
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run_ingestion(raw_dir)
