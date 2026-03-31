"""Query rewriting strategies for improving retrieval quality.

Supports multiple techniques:
- HyDE (Hypothetical Document Embeddings): generate hypothetical answer, use it for retrieval
- Multi-Query: rewrite query from multiple angles, merge retrieval results
"""

import time
from typing import List, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import load_config
from src.logging.logger import get_logger

logger = get_logger("generation.query_rewriter")

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Given a question, write a short paragraph "
     "that would be a plausible answer found in a knowledge base document. "
     "Do NOT answer the question directly — instead, write what a relevant "
     "document passage would look like. "
     "Answer in the same language as the question."),
    ("human", "{question}"),
])

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a search query optimization assistant. Given a user question, "
     "generate 3 alternative versions of the question that approach the topic "
     "from different angles. This helps retrieve more comprehensive results. "
     "Output exactly 3 queries, one per line, with no numbering or bullets. "
     "Answer in the same language as the question."),
    ("human", "{question}"),
])


class HyDERewriter:
    """Hypothetical Document Embeddings query rewriter.

    Generates a hypothetical answer to the query, then uses that
    answer text (instead of the original query) for embedding search.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()

        llm_cfg = config.get("llm", {})
        self._llm = ChatOllama(
            model=llm_cfg.get("model", "qwen2.5:3b"),
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            temperature=0.3,
            num_ctx=2048,
        )
        self._chain = HYDE_PROMPT | self._llm | StrOutputParser()
        logger.info("HyDERewriter initialized")

    def rewrite(self, question: str) -> str:
        """Generate a hypothetical document passage for the question.

        Args:
            question: Original user question.

        Returns:
            Hypothetical document passage to use for embedding search.
        """
        start = time.perf_counter()

        try:
            result = self._chain.invoke({"question": question})
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                f"HyDE rewrite: {len(question)} chars → "
                f"{len(result)} chars ({elapsed_ms}ms)"
            )
            return result.strip()
        except Exception as e:
            logger.warning(f"HyDE rewrite failed, using original query: {e}")
            return question


class MultiQueryRewriter:
    """Multi-query rewriter.

    Rewrites the original query into multiple perspectives,
    enabling broader retrieval coverage.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()

        llm_cfg = config.get("llm", {})
        self._llm = ChatOllama(
            model=llm_cfg.get("model", "qwen2.5:3b"),
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            temperature=0.5,
            num_ctx=2048,
        )
        self._chain = MULTI_QUERY_PROMPT | self._llm | StrOutputParser()
        logger.info("MultiQueryRewriter initialized")

    def rewrite(self, question: str) -> List[str]:
        """Generate multiple query variations for the given question.

        Args:
            question: Original user question.

        Returns:
            List of alternative query strings (including the original).
        """
        start = time.perf_counter()

        try:
            result = self._chain.invoke({"question": question})
            # Parse one query per line, filter empty
            queries = [
                q.strip() for q in result.strip().split("\n")
                if q.strip()
            ]
            # Always include original query
            queries = [question] + queries

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                f"Multi-query rewrite: 1 → {len(queries)} queries ({elapsed_ms}ms)"
            )
            return queries

        except Exception as e:
            logger.warning(f"Multi-query rewrite failed: {e}")
            return [question]
