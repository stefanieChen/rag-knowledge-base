"""PDF document parser using PyMuPDF4LLM.

Extracts text per page in Markdown format, preserving tables and layout.
Images are optionally described via multimodal LLM.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf4llm
import pymupdf

from src.logging.logger import get_logger

logger = get_logger("ingestion.pdf_parser")


def parse_pdf(
    file_path: str,
    extract_images: bool = False,
    config: Optional[dict] = None,
) -> List[Dict]:
    """Parse a PDF file into per-page document dicts with metadata.

    Uses pymupdf4llm to convert each page to Markdown, preserving
    table structure and layout. Optionally extracts and describes
    embedded images using multimodal LLM.

    Args:
        file_path: Absolute path to the .pdf file.
        extract_images: If True, extract images and generate text
                        descriptions via image_handler.
        config: Optional config dict (used for image LLM settings).

    Returns:
        List of dicts, each with keys: 'content', 'metadata'.
    """
    start = time.perf_counter()
    file_path = str(Path(file_path).resolve())

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    file_name = os.path.basename(file_path)

    try:
        # pymupdf4llm returns a list of dicts, one per page
        pages = pymupdf4llm.to_markdown(
            file_path,
            page_chunks=True,
            show_progress=False,
        )
    except Exception as e:
        logger.error(f"PDF parsing failed for {file_name}: {e}")
        return []

    documents = []
    for page_data in pages:
        page_num = page_data.get("metadata", {}).get("page_number", 0)
        text = page_data.get("text", "").strip()

        if not text:
            logger.debug(f"Skipping empty page {page_num} in {file_name}")
            continue

        metadata = {
            "source_file": file_path,
            "file_name": file_name,
            "content_type": "text",
            "format": "pdf",
            "page_number": page_num,
        }

        documents.append({"content": text, "metadata": metadata})

    # Optionally extract and describe images
    if extract_images:
        image_docs = _extract_images_from_pdf(file_path, file_name, config)
        documents.extend(image_docs)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        f"Parsed PDF: {file_name} "
        f"({len(documents)} docs from {len(pages)} pages, {elapsed_ms}ms)"
    )

    return documents


def _extract_images_from_pdf(
    file_path: str,
    file_name: str,
    config: Optional[dict] = None,
) -> List[Dict]:
    """Extract images from PDF and generate text descriptions.

    Args:
        file_path: Absolute path to the PDF file.
        file_name: Base name of the PDF file.
        config: Optional config dict for LLM settings.

    Returns:
        List of document dicts with image descriptions as content.
    """
    # Lazy import to avoid circular dependency and allow graceful fallback
    from src.ingestion.image_handler import describe_image_bytes

    documents = []

    try:
        doc = pymupdf.open(file_path)
    except Exception as e:
        logger.warning(f"Cannot open PDF for image extraction: {e}")
        return []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                mime_ext = base_image.get("ext", "png")
                mime_type = f"image/{mime_ext}"

                description = describe_image_bytes(
                    image_bytes, mime_type=mime_type, config=config
                )

                if description:
                    metadata = {
                        "source_file": file_path,
                        "file_name": file_name,
                        "content_type": "image_description",
                        "format": "pdf",
                        "page_number": page_idx + 1,
                        "image_index": img_idx,
                    }
                    documents.append({
                        "content": f"[Image on page {page_idx + 1}]\n{description}",
                        "metadata": metadata,
                    })

            except Exception as e:
                logger.debug(
                    f"Failed to extract image {img_idx} from page "
                    f"{page_idx + 1} of {file_name}: {e}"
                )

    doc.close()
    logger.info(
        f"Extracted {len(documents)} image descriptions from {file_name}"
    )
    return documents
