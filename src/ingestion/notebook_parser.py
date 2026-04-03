"""Jupyter Notebook (.ipynb) parser for RAG ingestion.

Extracts code cells, markdown cells, and cell outputs from .ipynb files,
preserving cell ordering and metadata for meaningful chunking.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.logging.logger import get_logger

logger = get_logger("ingestion.notebook_parser")

# Cell types we process
_CODE_CELL = "code"
_MARKDOWN_CELL = "markdown"
_RAW_CELL = "raw"


def _extract_cell_source(cell: dict) -> str:
    """Extract source text from a notebook cell.

    Args:
        cell: Notebook cell dict with 'source' key.

    Returns:
        Cell source as a single string.
    """
    source = cell.get("source", [])
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _extract_cell_outputs(cell: dict, max_output_chars: int = 2000) -> str:
    """Extract text outputs from a code cell.

    Handles stream outputs, execute_result, and error tracebacks.

    Args:
        cell: Notebook code cell dict with 'outputs' key.
        max_output_chars: Maximum characters to extract from outputs.

    Returns:
        Concatenated output text, truncated if necessary.
    """
    outputs = cell.get("outputs", [])
    if not outputs:
        return ""

    parts = []
    for out in outputs:
        output_type = out.get("output_type", "")

        if output_type == "stream":
            text = out.get("text", [])
            if isinstance(text, list):
                text = "".join(text)
            parts.append(text)

        elif output_type in ("execute_result", "display_data"):
            data = out.get("data", {})
            # Prefer plain text representation
            if "text/plain" in data:
                text = data["text/plain"]
                if isinstance(text, list):
                    text = "".join(text)
                parts.append(text)

        elif output_type == "error":
            ename = out.get("ename", "Error")
            evalue = out.get("evalue", "")
            parts.append(f"{ename}: {evalue}")

    combined = "\n".join(parts)
    if len(combined) > max_output_chars:
        combined = combined[:max_output_chars] + "\n... (output truncated)"
    return combined


def _detect_notebook_language(nb: dict) -> str:
    """Detect the programming language of a notebook from its kernel spec.

    Args:
        nb: Parsed notebook dict.

    Returns:
        Language name string (e.g., 'python'), or 'unknown'.
    """
    metadata = nb.get("metadata", {})
    kernelspec = metadata.get("kernelspec", {})
    language = kernelspec.get("language", "")
    if language:
        return language.lower()

    language_info = metadata.get("language_info", {})
    name = language_info.get("name", "")
    if name:
        return name.lower()

    return "unknown"


def parse_notebook(
    file_path: str,
    include_outputs: bool = True,
    max_output_chars: int = 2000,
    extract_images: bool = False,
    config: Optional[dict] = None,
) -> List[Dict]:
    """Parse a Jupyter Notebook (.ipynb) file into document chunks.

    Each cell becomes a separate document with appropriate metadata.
    Code cells include their outputs (if enabled). Markdown cells
    are preserved as-is. Empty cells are skipped.

    Args:
        file_path: Absolute path to the .ipynb file.
        include_outputs: Whether to include code cell outputs.
        max_output_chars: Maximum characters per cell output.
        extract_images: Reserved for future image extraction from outputs.
        config: Optional config dict.

    Returns:
        List of dicts, each with 'content' and 'metadata' keys.
    """
    start = time.perf_counter()
    file_path = str(Path(file_path).resolve())

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid notebook JSON: {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to read notebook: {file_path}: {e}")
        return []

    cells = nb.get("cells", [])
    if not cells:
        logger.warning(f"Empty notebook (no cells): {file_path}")
        return []

    nb_language = _detect_notebook_language(nb)
    file_name = os.path.basename(file_path)
    documents = []

    for cell_idx, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "")
        source = _extract_cell_source(cell)

        if not source.strip():
            continue

        # Build cell content
        if cell_type == _CODE_CELL:
            # Wrap code in fenced block for clear context
            content_parts = [f"```{nb_language}\n{source}\n```"]
            if include_outputs:
                output_text = _extract_cell_outputs(cell, max_output_chars)
                if output_text:
                    content_parts.append(f"Output:\n```\n{output_text}\n```")
            content = "\n\n".join(content_parts)
            content_type = "code"

        elif cell_type == _MARKDOWN_CELL:
            content = source
            content_type = "markdown"

        elif cell_type == _RAW_CELL:
            content = source
            content_type = "raw"

        else:
            continue

        metadata = {
            "source_file": file_path,
            "file_name": file_name,
            "content_type": content_type,
            "format": "ipynb",
            "cell_index": cell_idx,
            "cell_type": cell_type,
            "notebook_language": nb_language,
        }

        # Add execution count for code cells
        exec_count = cell.get("execution_count")
        if exec_count is not None:
            metadata["execution_count"] = exec_count

        documents.append({"content": content, "metadata": metadata})

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    code_cells = sum(1 for d in documents if d["metadata"]["cell_type"] == _CODE_CELL)
    md_cells = sum(1 for d in documents if d["metadata"]["cell_type"] == _MARKDOWN_CELL)

    logger.info(
        f"Parsed notebook: {file_name} — "
        f"{len(documents)} cells ({code_cells} code, {md_cells} markdown), "
        f"lang={nb_language} ({elapsed_ms}ms)"
    )

    return documents
