"""OneNote document parser.

Parses OneNote pages exported as HTML files. OneNote does not have
a direct Python library, so the workflow is:
  1. Export pages from OneNote as .htm/.html (File → Export → HTML)
  2. Run this parser to extract structured text sections

Handles OneNote-specific HTML quirks:
- Nested div/span structures for outline groups
- Inline styles instead of semantic tags
- Embedded images (base64 data URIs) with optional LLM description
- Tables with OneNote-specific styling
"""

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup, Tag

from src.logging.logger import get_logger

logger = get_logger("ingestion.onenote_parser")


def parse_onenote(
    file_path: str,
    extract_images: bool = False,
    config: Optional[dict] = None,
) -> List[Dict]:
    """Parse a OneNote HTML export into document sections.

    Args:
        file_path: Absolute path to the exported .htm/.html file.
        extract_images: If True, describe embedded images via multimodal LLM.
        config: Optional config dict for LLM settings.

    Returns:
        List of dicts, each with keys: 'content', 'metadata'.
    """
    start = time.perf_counter()
    file_path = str(Path(file_path).resolve())

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        html_content = f.read()

    if not html_content.strip():
        logger.warning(f"Empty file: {file_path}")
        return []

    soup = BeautifulSoup(html_content, "lxml")
    file_name = os.path.basename(file_path)

    # Extract page title
    page_title = _extract_title(soup, file_name)

    # Extract sections from the HTML body
    sections = _extract_sections(soup)

    # Extract tables
    tables = _extract_tables(soup)

    # Optionally extract image descriptions
    image_docs = []
    if extract_images:
        image_docs = _extract_images(soup, file_path, config)

    # Build document list
    documents = []

    for i, section in enumerate(sections):
        content = section["content"].strip()
        if not content or len(content) < 10:
            continue

        metadata = {
            "source_file": file_path,
            "file_name": file_name,
            "content_type": "text",
            "format": "onenote_html",
            "page_title": page_title,
            "section_index": i,
            "heading": section.get("heading", ""),
        }
        documents.append({"content": content, "metadata": metadata})

    for i, table in enumerate(tables):
        metadata = {
            "source_file": file_path,
            "file_name": file_name,
            "content_type": "table",
            "format": "onenote_html",
            "page_title": page_title,
            "table_index": i,
        }
        documents.append({"content": table, "metadata": metadata})

    documents.extend(image_docs)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        f"Parsed OneNote HTML: {file_name} "
        f"({len(documents)} sections, title='{page_title}', {elapsed_ms}ms)"
    )

    return documents


def _extract_title(soup: BeautifulSoup, fallback: str) -> str:
    """Extract page title from OneNote HTML.

    Args:
        soup: Parsed HTML.
        fallback: Fallback title (usually the filename).

    Returns:
        Page title string.
    """
    # OneNote exports often have <title> tag
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        return title_tag.get_text(strip=True)

    # Fallback: first h1 or large-font element
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    return Path(fallback).stem


def _extract_sections(soup: BeautifulSoup) -> List[Dict]:
    """Extract text sections from OneNote HTML body.

    OneNote HTML uses nested divs with outline groups.
    This function extracts meaningful text blocks.

    Args:
        soup: Parsed HTML.

    Returns:
        List of dicts with 'heading' and 'content' keys.
    """
    sections = []
    current_heading = ""
    current_content = []

    body = soup.find("body")
    if body is None:
        # Fallback: treat entire soup as body
        body = soup

    # Walk through top-level block elements
    for element in body.children:
        if not isinstance(element, Tag):
            continue

        # Check for heading-like elements
        if element.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            # Flush previous section
            if current_content:
                sections.append({
                    "heading": current_heading,
                    "content": "\n".join(current_content),
                })
                current_content = []
            current_heading = element.get_text(strip=True)
            continue

        # Check for OneNote outline group divs with bold/large text as headings
        if element.name == "div":
            text = _clean_text(element.get_text(separator="\n"))
            if text:
                # Check if this div looks like a heading (short, bold)
                bold = element.find(["b", "strong"])
                if bold and len(text) < 100 and "\n" not in text.strip():
                    if current_content:
                        sections.append({
                            "heading": current_heading,
                            "content": "\n".join(current_content),
                        })
                        current_content = []
                    current_heading = text.strip()
                else:
                    current_content.append(text)
            continue

        # Paragraphs and other block elements
        if element.name in ("p", "span", "pre", "blockquote", "ul", "ol"):
            text = _clean_text(element.get_text(separator="\n"))
            if text:
                current_content.append(text)

    # Flush remaining content
    if current_content:
        sections.append({
            "heading": current_heading,
            "content": "\n".join(current_content),
        })

    # If no sections found, extract all text as a single section
    if not sections:
        all_text = _clean_text(body.get_text(separator="\n"))
        if all_text:
            sections.append({"heading": "", "content": all_text})

    return sections


def _extract_tables(soup: BeautifulSoup) -> List[str]:
    """Extract tables from OneNote HTML and convert to Markdown.

    Args:
        soup: Parsed HTML.

    Returns:
        List of Markdown-formatted table strings.
    """
    tables = []

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue

        md_rows = []
        for row in rows:
            cells = row.find_all(["td", "th"])
            cell_texts = [
                cell.get_text(strip=True).replace("|", "\\|")
                for cell in cells
            ]
            md_rows.append("| " + " | ".join(cell_texts) + " |")

        if len(md_rows) >= 1:
            # Insert header separator after first row
            num_cols = md_rows[0].count("|") - 1
            separator = "| " + " | ".join(["---"] * max(num_cols, 1)) + " |"
            md_rows.insert(1, separator)

        md_table = "\n".join(md_rows)
        if md_table.strip():
            tables.append(md_table)

    return tables


def _extract_images(
    soup: BeautifulSoup,
    file_path: str,
    config: Optional[dict] = None,
) -> List[Dict]:
    """Extract and describe embedded images from OneNote HTML.

    Args:
        soup: Parsed HTML.
        file_path: Source file path for metadata.
        config: Optional config dict.

    Returns:
        List of document dicts for image descriptions.
    """
    documents = []
    file_name = os.path.basename(file_path)

    try:
        from src.ingestion.image_handler import describe_image_bytes
    except ImportError:
        logger.debug("Image handler not available, skipping image extraction")
        return []

    for i, img in enumerate(soup.find_all("img")):
        src = img.get("src", "")

        # Handle base64 data URIs (common in OneNote exports)
        if src.startswith("data:image"):
            try:
                import base64
                # Extract base64 data after the comma
                b64_data = src.split(",", 1)[1]
                image_bytes = base64.b64decode(b64_data)

                description = describe_image_bytes(
                    image_bytes, config=config
                )
                if description:
                    documents.append({
                        "content": f"[Image {i + 1}]: {description}",
                        "metadata": {
                            "source_file": file_path,
                            "file_name": file_name,
                            "content_type": "image_description",
                            "format": "onenote_html",
                            "image_index": i,
                        },
                    })
            except Exception as e:
                logger.warning(f"Failed to process embedded image {i}: {e}")

        # Handle file references
        elif src and not src.startswith("http"):
            img_path = os.path.join(os.path.dirname(file_path), src)
            if os.path.exists(img_path):
                try:
                    from src.ingestion.image_handler import describe_image
                    description = describe_image(img_path, config=config)
                    if description:
                        documents.append({
                            "content": f"[Image {i + 1}]: {description}",
                            "metadata": {
                                "source_file": file_path,
                                "file_name": file_name,
                                "content_type": "image_description",
                                "format": "onenote_html",
                                "image_index": i,
                            },
                        })
                except Exception as e:
                    logger.warning(f"Failed to describe image {img_path}: {e}")

    return documents


def _clean_text(text: str) -> str:
    """Clean extracted text by removing excess whitespace.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text string.
    """
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()
