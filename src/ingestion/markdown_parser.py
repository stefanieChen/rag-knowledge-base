"""Markdown file parser with heading-aware structure preservation."""

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.logging.logger import get_logger

logger = get_logger("ingestion.markdown_parser")


def parse_markdown(
    file_path: str,
    extract_images: bool = False,
    config: Optional[dict] = None,
) -> List[Dict]:
    """Parse a Markdown file into structured document sections.

    Splits on heading boundaries (# through ####) to preserve
    document structure. Each section becomes a separate document
    with its heading hierarchy in metadata. Optionally extracts
    and describes image links (local and remote) via multimodal LLM.

    Args:
        file_path: Absolute path to the .md file.
        extract_images: If True, extract and describe image links.
        config: Optional config dict (used for image LLM settings).

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

    # Optionally extract and describe image links
    if extract_images:
        image_docs = _extract_image_links(text, file_path, file_name, config)
        documents.extend(image_docs)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        f"Parsed Markdown: {file_name} "
        f"({len(documents)} sections, {elapsed_ms}ms)"
    )

    return documents


def _extract_image_links(
    text: str,
    file_path: str,
    file_name: str,
    config: Optional[dict] = None,
) -> List[Dict]:
    """Extract image links from Markdown and describe them via LLM.

    Handles both local file references and remote URLs.
    Markdown image syntax: ![alt](path_or_url)

    Args:
        text: Raw markdown text.
        file_path: Absolute path to the source .md file.
        file_name: Base name of the source file.
        config: Optional config dict for LLM settings.

    Returns:
        List of document dicts with image descriptions as content.
    """
    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    matches = list(image_pattern.finditer(text))

    if not matches:
        return []

    documents = []
    file_dir = os.path.dirname(file_path)

    for i, match in enumerate(matches):
        alt_text = match.group(1)
        img_ref = match.group(2).strip()

        description = None

        if img_ref.startswith(("http://", "https://")):
            # Remote image: download and describe
            description = _describe_remote_image(img_ref, config)
        else:
            # Local image: resolve relative to the .md file
            img_path = os.path.normpath(os.path.join(file_dir, img_ref))
            if os.path.exists(img_path):
                try:
                    from src.ingestion.image_handler import describe_image
                    description = describe_image(img_path, config=config)
                except Exception as e:
                    logger.warning(f"Failed to describe local image {img_path}: {e}")
            else:
                logger.debug(f"Local image not found: {img_path}")

        if description:
            content = f"[Image: {alt_text}]\n{description}" if alt_text else f"[Image {i + 1}]\n{description}"
            metadata = {
                "source_file": file_path,
                "file_name": file_name,
                "content_type": "image_description",
                "format": "markdown",
                "image_index": i,
                "image_ref": img_ref,
            }
            documents.append({"content": content, "metadata": metadata})

    logger.info(f"Extracted {len(documents)} image descriptions from {file_name}")
    return documents


def _describe_remote_image(
    url: str,
    config: Optional[dict] = None,
) -> Optional[str]:
    """Download a remote image and describe it via multimodal LLM.

    Args:
        url: URL of the remote image.
        config: Optional config dict for LLM settings.

    Returns:
        Text description string, or None on failure.
    """
    try:
        import requests
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "image/png")
        # Ensure it's actually an image
        if not content_type.startswith("image/"):
            logger.debug(f"URL is not an image (Content-Type: {content_type}): {url}")
            return None

        from src.ingestion.image_handler import describe_image_bytes
        return describe_image_bytes(
            response.content, mime_type=content_type, config=config
        )

    except Exception as e:
        logger.warning(f"Failed to fetch/describe remote image {url}: {e}")
        return None


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
