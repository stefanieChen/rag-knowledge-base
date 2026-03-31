"""Plain text file parser."""

import os
import time
from pathlib import Path
from typing import Dict, List

from src.logging.logger import get_logger

logger = get_logger("ingestion.txt_parser")


def parse_txt(file_path: str) -> List[Dict]:
    """Parse a plain text file into document chunks with metadata.

    Args:
        file_path: Absolute path to the .txt file.

    Returns:
        List of dicts, each with keys: 'content', 'metadata'.
    """
    start = time.perf_counter()
    file_path = str(Path(file_path).resolve())

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gbk") as f:
            text = f.read()

    if not text.strip():
        logger.warning(f"Empty file: {file_path}")
        return []

    metadata = {
        "source_file": file_path,
        "file_name": os.path.basename(file_path),
        "content_type": "text",
        "format": "txt",
    }

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        f"Parsed TXT: {os.path.basename(file_path)} "
        f"({len(text)} chars, {elapsed_ms}ms)"
    )

    return [{"content": text, "metadata": metadata}]
