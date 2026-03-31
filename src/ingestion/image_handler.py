"""Image description generator for embedded images.

Uses multimodal LLM (llava via Ollama) to generate text descriptions
of images for indexing into the knowledge base.
"""

import base64
import os
import time
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.logging.logger import get_logger

logger = get_logger("ingestion.image_handler")

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
