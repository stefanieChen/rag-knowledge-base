"""Automatic test set generation using Ragas TestsetGenerator.

Generates synthetic QA pairs from the ingested knowledge base documents,
producing questions of varying complexity (simple, reasoning, multi-context).

Usage:
    python -m evaluation.generate_test_set --num-questions 20
    python -m evaluation.generate_test_set --num-questions 50 --output evaluation/test_set/generated_test_set.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import load_config
from src.logging.logger import get_logger, setup_logging

logger = get_logger("evaluation.generate_test_set")

DEFAULT_OUTPUT = "evaluation/test_set/generated_test_set.json"


def _extract_documents_from_store(
    config: Optional[dict] = None,
    max_docs: int = 200,
) -> List[Dict[str, Any]]:
    """Extract stored document chunks from ChromaDB for test generation.

    Args:
        config: Application config dict.
        max_docs: Maximum number of chunks to retrieve.

    Returns:
        List of dicts with 'content' and 'metadata' keys.
    """
    if config is None:
        config = load_config()

    from src.retrieval.vector_store import VectorStore

    vs = VectorStore(config=config)
    total = vs.count
    if total == 0:
        logger.warning("Vector store is empty — ingest documents before generating test set")
        return []

    limit = min(total, max_docs)
    result = vs._collection.get(
        include=["documents", "metadatas"],
        limit=limit,
    )

    docs = []
    if result and result["documents"]:
        for i, doc_text in enumerate(result["documents"]):
            meta = result["metadatas"][i] if result["metadatas"] else {}
            docs.append({
                "content": doc_text,
                "metadata": meta,
            })

    logger.info(f"Extracted {len(docs)} document chunks from vector store (total={total})")
    return docs


def _build_langchain_documents(chunks: List[Dict[str, Any]]):
    """Convert chunk dicts to LangChain Document objects for Ragas.

    Args:
        chunks: List of dicts with 'content' and 'metadata'.

    Returns:
        List of LangChain Document objects.
    """
    from langchain_core.documents import Document

    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["content"],
            metadata=chunk.get("metadata", {}),
        )
        documents.append(doc)

    return documents


def generate_test_set(
    config: Optional[dict] = None,
    num_questions: int = 20,
    max_docs: int = 200,
) -> List[Dict[str, Any]]:
    """Generate a synthetic test set from the knowledge base using Ragas.

    Uses the Ragas TestsetGenerator to create diverse QA pairs from
    ingested documents, with questions of varying complexity.

    Args:
        config: Application config dict.
        num_questions: Number of test questions to generate.
        max_docs: Maximum number of document chunks to use as source.

    Returns:
        List of test case dicts compatible with the evaluation runner format:
        [{"question": str, "ground_truth": str, "contexts": [str]}].
    """
    if config is None:
        config = load_config()

    logger.info(f"Generating {num_questions} test questions from knowledge base...")
    start = time.perf_counter()

    # 1. Extract documents from vector store
    chunks = _extract_documents_from_store(config=config, max_docs=max_docs)
    if not chunks:
        logger.error("No documents available for test generation")
        return []

    lc_documents = _build_langchain_documents(chunks)

    # 2. Build Ragas generator LLM and embeddings via Ollama OpenAI endpoint
    from openai import OpenAI
    from ragas.testset import TestsetGenerator

    llm_cfg = config.get("llm", {})
    base_url = llm_cfg.get("base_url", "http://localhost:11434")
    model = llm_cfg.get("model", "qwen2.5:7b")

    emb_cfg = config.get("embedding", {})
    emb_model = emb_cfg.get("model", "bge-m3")

    generator_client = OpenAI(
        base_url=f"{base_url}/v1",
        api_key="ollama",
    )

    # 3. Create TestsetGenerator with Ollama-backed LLM
    try:
        from ragas.llms import llm_factory
        from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings

        generator_llm = llm_factory(model, client=generator_client)
        generator_embeddings = RagasOpenAIEmbeddings(
            client=generator_client,
            model=emb_model,
        )

        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings,
        )

        logger.info(
            f"TestsetGenerator initialized: model={model}, "
            f"embeddings={emb_model}, docs={len(lc_documents)}"
        )

        # 4. Generate test set
        testset = generator.generate(
            documents=lc_documents,
            testset_size=num_questions,
        )

        # 5. Convert to our standard format
        test_cases = []
        df = testset.to_pandas()
        for _, row in df.iterrows():
            tc = {
                "question": row.get("user_input", row.get("question", "")),
                "ground_truth": row.get("reference", row.get("ground_truth", "")),
                "contexts": row.get("reference_contexts", row.get("contexts", [])),
            }
            # Ensure contexts is a list
            if isinstance(tc["contexts"], str):
                tc["contexts"] = [tc["contexts"]]
            if tc["question"]:
                test_cases.append(tc)

        elapsed_s = round(time.perf_counter() - start, 1)
        logger.info(
            f"Generated {len(test_cases)} test cases ({elapsed_s}s)"
        )
        return test_cases

    except ImportError as e:
        logger.error(f"Ragas TestsetGenerator not available: {e}")
        return []
    except Exception as e:
        logger.error(f"Test set generation failed: {e}")
        return []


def save_test_set(
    test_cases: List[Dict[str, Any]],
    output_path: str,
    merge_existing: bool = True,
) -> str:
    """Save generated test cases to a JSON file.

    Args:
        test_cases: List of test case dicts.
        output_path: Path to save the JSON file.
        merge_existing: If True and file exists, merge new cases with existing ones.

    Returns:
        The output file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if merge_existing and path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            logger.info(f"Merging with {len(existing)} existing test cases from {path}")
        except Exception:
            pass

    # Deduplicate by question text
    existing_questions = {tc["question"] for tc in existing}
    new_cases = [tc for tc in test_cases if tc["question"] not in existing_questions]
    merged = existing + new_cases

    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Saved {len(merged)} test cases to {path} "
        f"({len(new_cases)} new, {len(existing)} existing)"
    )
    return str(path)


def main() -> None:
    """CLI entry point for test set generation."""
    parser = argparse.ArgumentParser(description="RAG Test Set Generator")
    parser.add_argument(
        "--num-questions",
        type=int,
        default=20,
        help="Number of test questions to generate (default: 20)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=200,
        help="Maximum number of document chunks to use as source (default: 200)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Overwrite existing test set instead of merging",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config()

    test_cases = generate_test_set(
        config=config,
        num_questions=args.num_questions,
        max_docs=args.max_docs,
    )

    if not test_cases:
        logger.error("No test cases generated. Ensure the knowledge base has documents.")
        sys.exit(1)

    output_path = save_test_set(
        test_cases,
        output_path=args.output,
        merge_existing=not args.no_merge,
    )

    print(f"\n✅ Generated {len(test_cases)} test cases → {output_path}")


if __name__ == "__main__":
    main()
