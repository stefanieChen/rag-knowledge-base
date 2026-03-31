"""Markdown file parser with heading-aware structure preservation."""

import os
import re
import time
from pathlib import Path
from typing import Dict, List

from src.logging.logger import get_logger

logger = get_logger("ingestion.markdown_parser")


def parse_markdown(file_path: str) -> List[Dict]:
    """Parse a Markdown file into structured document sections.

    Splits on heading boundaries (# through ####) to preserve
    document structure. Each section becomes a separate document
    with its heading hierarchy in metadata.

    Args:
        file_path: Absolute path to the .md file.

    Returns:
        List of dicts, each with keys: 'content', 'metadata'.
    """
    start = time.perf_counter()
    file_path = str(Path(file_path).resolve())

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        logger.warning(f"Empty file: {file_path}")
        return []

    sections = _split_by_headings(text)
    file_name = os.path.basename(file_path)

    documents = []
    for section in sections:
        content = section["content"].strip()
        if not content:
            continue

        metadata = {
            "source_file": file_path,
            "file_name": file_name,
            "content_type": "text",
            "format": "markdown",
            "heading": section.get("heading", ""),
            "heading_level": section.get("heading_level", 0),
        }
        documents.append({"content": content, "metadata": metadata})

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        f"Parsed Markdown: {file_name} "
        f"({len(documents)} sections, {elapsed_ms}ms)"
    )

    return documents


def _split_by_headings(text: str) -> List[Dict]:
    """Split markdown text by heading boundaries.

    Args:
        text: Raw markdown text.

    Returns:
        List of dicts with 'heading', 'heading_level', 'content'.
    """
    heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(text))

    if not matches:
        return [{"heading": "", "heading_level": 0, "content": text}]

    sections = []

    # Content before first heading
    if matches[0].start() > 0:
        pre_content = text[: matches[0].start()]
        if pre_content.strip():
            sections.append({
                "heading": "",
                "heading_level": 0,
                "content": pre_content,
            })

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        content = text[start:end]
        sections.append({
            "heading": heading,
            "heading_level": level,
            "content": content,
        })

    return sections
