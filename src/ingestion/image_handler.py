"""Image description generator and OCR text extractor for embedded images.

Uses multimodal LLM (llava via Ollama) to generate text descriptions
of images, and Tesseract OCR to extract text content from text-heavy images.
"""

import base64
import io
import os
import time
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.logging.logger import get_logger

logger = get_logger("ingestion.image_handler")

# Track OCR availability at import time
_TESSERACT_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image
    # Quick check that tesseract binary is accessible
    pytesseract.get_tesseract_version()
    _TESSERACT_AVAILABLE = True
except Exception:
    logger.info("Tesseract OCR not available — install pytesseract and Tesseract binary for OCR support")

# Default multimodal model for image description
DEFAULT_VISION_MODEL = "llava"

IMAGE_DESCRIBE_PROMPT = (
    "Describe this image in detail for a knowledge base. "
    "Include all visible text, data, labels, and visual elements. "
    "If it contains a chart or diagram, describe the structure and key data points. "
    "Answer in the same language as any text visible in the image. "
    "If no text is visible, describe in English."
)


def _encode_image_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string.

    Args:
        image_path: Absolute path to the image file.

    Returns:
        Base64-encoded string of the image.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_image_mime_type(image_path: str) -> str:
    """Infer MIME type from file extension.

    Args:
        image_path: Path to the image file.

    Returns:
        MIME type string (e.g., 'image/png').
    """
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    return mime_map.get(ext, "image/png")


def describe_image(
    image_path: str,
    config: Optional[dict] = None,
    prompt: Optional[str] = None,
) -> Optional[str]:
    """Generate a text description of an image using a multimodal LLM.

    Args:
        image_path: Absolute path to the image file.
        config: Optional config dict for LLM settings.
        prompt: Optional custom prompt for image description.

    Returns:
        Text description string, or None if extraction fails.
    """
    if not os.path.exists(image_path):
        logger.warning(f"Image not found: {image_path}")
        return None

    start = time.perf_counter()
    prompt_text = prompt or IMAGE_DESCRIBE_PROMPT

    try:
        llm_cfg = (config or {}).get("llm", {})
        base_url = llm_cfg.get("base_url", "http://localhost:11434")

        llm = ChatOllama(
            model=DEFAULT_VISION_MODEL,
            base_url=base_url,
            temperature=0.1,
            num_ctx=2048,
        )

        image_data = _encode_image_base64(image_path)
        mime_type = _get_image_mime_type(image_path)

        message = HumanMessage(content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}",
                },
            },
        ])

        parser = StrOutputParser()
        response = parser.invoke(llm.invoke([message]))

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"Image described: {os.path.basename(image_path)} "
            f"({len(response)} chars, {elapsed_ms}ms)"
        )
        return response.strip()

    except Exception as e:
        logger.warning(
            f"Image description failed for {image_path}: {e}"
        )
        return None


def describe_image_bytes(
    image_bytes: bytes,
    mime_type: str = "image/png",
    config: Optional[dict] = None,
    prompt: Optional[str] = None,
) -> Optional[str]:
    """Generate a text description from raw image bytes.

    Useful for images extracted from PDF/PPTX that are already in memory.

    Args:
        image_bytes: Raw image bytes.
        mime_type: MIME type of the image.
        config: Optional config dict for LLM settings.
        prompt: Optional custom prompt for image description.

    Returns:
        Text description string, or None if extraction fails.
    """
    start = time.perf_counter()
    prompt_text = prompt or IMAGE_DESCRIBE_PROMPT

    try:
        llm_cfg = (config or {}).get("llm", {})
        base_url = llm_cfg.get("base_url", "http://localhost:11434")

        llm = ChatOllama(
            model=DEFAULT_VISION_MODEL,
            base_url=base_url,
            temperature=0.1,
            num_ctx=2048,
        )

        image_data = base64.b64encode(image_bytes).decode("utf-8")

        message = HumanMessage(content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}",
                },
            },
        ])

        parser = StrOutputParser()
        response = parser.invoke(llm.invoke([message]))

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(f"Image bytes described ({len(response)} chars, {elapsed_ms}ms)")
        return response.strip()

    except Exception as e:
        logger.warning(f"Image bytes description failed: {e}")
        return None


# ── OCR functions ─────────────────────────────────────────────


def ocr_image(
    image_path: str,
    lang: str = "eng+chi_sim",
) -> Optional[str]:
    """Extract text from an image using Tesseract OCR.

    Args:
        image_path: Absolute path to the image file.
        lang: Tesseract language string (e.g., 'eng', 'chi_sim', 'eng+chi_sim').

    Returns:
        Extracted text string, or None if OCR is unavailable or fails.
    """
    if not _TESSERACT_AVAILABLE:
        logger.debug("OCR skipped: Tesseract not available")
        return None

    if not os.path.exists(image_path):
        logger.warning(f"Image not found for OCR: {image_path}")
        return None

    start = time.perf_counter()
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=lang)
        text = text.strip()

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        if text:
            logger.info(
                f"OCR extracted: {os.path.basename(image_path)} "
                f"({len(text)} chars, {elapsed_ms}ms)"
            )
        else:
            logger.debug(f"OCR returned empty for {os.path.basename(image_path)} ({elapsed_ms}ms)")

        return text if text else None

    except Exception as e:
        logger.warning(f"OCR failed for {image_path}: {e}")
        return None


def ocr_image_bytes(
    image_bytes: bytes,
    lang: str = "eng+chi_sim",
) -> Optional[str]:
    """Extract text from raw image bytes using Tesseract OCR.

    Args:
        image_bytes: Raw image bytes.
        lang: Tesseract language string.

    Returns:
        Extracted text string, or None if OCR is unavailable or fails.
    """
    if not _TESSERACT_AVAILABLE:
        logger.debug("OCR skipped: Tesseract not available")
        return None

    start = time.perf_counter()
    try:
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img, lang=lang)
        text = text.strip()

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        if text:
            logger.info(f"OCR bytes extracted ({len(text)} chars, {elapsed_ms}ms)")

        return text if text else None

    except Exception as e:
        logger.warning(f"OCR bytes extraction failed: {e}")
        return None


def extract_image_text(
    image_path: str,
    config: Optional[dict] = None,
    enable_ocr: bool = True,
    enable_llm: bool = True,
    ocr_lang: str = "eng+chi_sim",
) -> Optional[str]:
    """Extract text from an image using OCR and/or multimodal LLM description.

    Combines both approaches for comprehensive text extraction:
    - OCR extracts literal text content (best for text-heavy images)
    - LLM describes visual elements, charts, diagrams

    Args:
        image_path: Absolute path to the image file.
        config: Optional config dict for LLM settings.
        enable_ocr: Whether to run Tesseract OCR.
        enable_llm: Whether to run multimodal LLM description.
        ocr_lang: Tesseract language string.

    Returns:
        Combined text string, or None if all extraction fails.
    """
    parts = []

    if enable_ocr:
        ocr_text = ocr_image(image_path, lang=ocr_lang)
        if ocr_text:
            parts.append(f"[OCR Text]\n{ocr_text}")

    if enable_llm:
        llm_desc = describe_image(image_path, config=config)
        if llm_desc:
            parts.append(f"[Image Description]\n{llm_desc}")

    if not parts:
        return None

    return "\n\n".join(parts)


def extract_image_bytes_text(
    image_bytes: bytes,
    mime_type: str = "image/png",
    config: Optional[dict] = None,
    enable_ocr: bool = True,
    enable_llm: bool = True,
    ocr_lang: str = "eng+chi_sim",
) -> Optional[str]:
    """Extract text from raw image bytes using OCR and/or multimodal LLM.

    Args:
        image_bytes: Raw image bytes.
        mime_type: MIME type of the image.
        config: Optional config dict for LLM settings.
        enable_ocr: Whether to run Tesseract OCR.
        enable_llm: Whether to run multimodal LLM description.
        ocr_lang: Tesseract language string.

    Returns:
        Combined text string, or None if all extraction fails.
    """
    parts = []

    if enable_ocr:
        ocr_text = ocr_image_bytes(image_bytes, lang=ocr_lang)
        if ocr_text:
            parts.append(f"[OCR Text]\n{ocr_text}")

    if enable_llm:
        llm_desc = describe_image_bytes(image_bytes, mime_type=mime_type, config=config)
        if llm_desc:
            parts.append(f"[Image Description]\n{llm_desc}")

    if not parts:
        return None

    return "\n\n".join(parts)


def is_ocr_available() -> bool:
    """Check whether Tesseract OCR is available on this system.

    Returns:
        True if pytesseract and Tesseract binary are installed and accessible.
    """
    return _TESSERACT_AVAILABLE
