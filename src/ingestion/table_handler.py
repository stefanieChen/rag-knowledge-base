"""Table extraction and processing handler.

Converts tables to Markdown format and generates natural language summaries
for improved retrieval.
"""

import time
from typing import Dict, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.logging.logger import get_logger

logger = get_logger("ingestion.table_handler")

TABLE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a data analyst. Given a Markdown table, write a concise "
     "natural language summary (2-4 sentences) describing what the table "
     "contains, its key columns, and notable values. "
     "Answer in the same language as the table content."),
    ("human", "Summarize this table:\n\n{table}"),
])


def table_to_markdown(table_data: List[List[str]]) -> str:
    """Convert a 2D table to Markdown format.

    Args:
        table_data: List of rows, each row is a list of cell strings.
                    First row is treated as header.

    Returns:
        Markdown-formatted table string.
    """
    if not table_data or not table_data[0]:
        return ""

    # Normalize: ensure all rows have the same number of columns
    max_cols = max(len(row) for row in table_data)
    rows = [row + [""] * (max_cols - len(row)) for row in table_data]

    # Header
    header = "| " + " | ".join(str(cell).strip() for cell in rows[0]) + " |"
    separator = "| " + " | ".join("---" for _ in rows[0]) + " |"

    # Body
    body_lines = []
    for row in rows[1:]:
        line = "| " + " | ".join(str(cell).strip() for cell in row) + " |"
        body_lines.append(line)

    parts = [header, separator] + body_lines
    return "\n".join(parts)


def summarize_table(
    markdown_table: str,
    config: Optional[dict] = None,
) -> Optional[str]:
    """Generate a natural language summary of a table using LLM.

    Args:
        markdown_table: Markdown-formatted table string.
        config: Optional config dict for LLM settings.

    Returns:
        Natural language summary string, or None on failure.
    """
    if not markdown_table.strip():
        return None

    start = time.perf_counter()

    try:
        llm_cfg = (config or {}).get("llm", {})
        llm = ChatOllama(
            model=llm_cfg.get("model", "qwen2.5:7b"),
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            temperature=0.1,
            num_ctx=2048,
        )
        chain = TABLE_SUMMARY_PROMPT | llm | StrOutputParser()
        summary = chain.invoke({"table": markdown_table})

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(f"Table summarized ({len(summary)} chars, {elapsed_ms}ms)")
        return summary.strip()

    except Exception as e:
        logger.warning(f"Table summarization failed: {e}")
        return None
