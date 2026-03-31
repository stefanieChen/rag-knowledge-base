"""RAG evaluation using Ragas framework.

Evaluates RAG pipeline quality using four key metrics:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are relevant contexts ranked higher?
- Context Recall: Does the context cover the ground truth?

Uses Ollama via OpenAI-compatible endpoint as the LLM judge.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall,
    Faithfulness,
)

from src.config import load_config
from src.logging.logger import get_logger

logger = get_logger("evaluation.ragas")


def _build_ragas_llm(config: dict):
    """Create a Ragas-compatible LLM using Ollama's OpenAI endpoint.

    Args:
        config: Application config dict.

    Returns:
        InstructorLLM instance for Ragas metrics.
    """
    llm_cfg = config.get("llm", {})
    base_url = llm_cfg.get("base_url", "http://localhost:11434")
    model = llm_cfg.get("model", "qwen2.5:3b")

    client = OpenAI(
        base_url=f"{base_url}/v1",
        api_key="ollama",
    )
    return llm_factory(model, client=client)


def _build_ragas_embeddings(config: dict):
    """Create Ragas-compatible embeddings using Ollama's OpenAI endpoint.

    Args:
        config: Application config dict.

    Returns:
        Modern Ragas OpenAIEmbeddings for AnswerRelevancy metric.
    """
    emb_cfg = config.get("embedding", {})
    base_url = config.get("llm", {}).get("base_url", "http://localhost:11434")
    model = emb_cfg.get("model", "bge-m3")

    client = OpenAI(
        base_url=f"{base_url}/v1",
        api_key="ollama",
    )
    return RagasOpenAIEmbeddings(client=client, model=model)


def load_test_set(path: str) -> List[Dict]:
    """Load a test set from JSON file.

    Args:
        path: Path to test set JSON file.

    Returns:
        List of test case dicts with keys: question, ground_truth, contexts.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_evaluation_dataset_from_pipeline(
    test_cases: List[Dict],
    pipeline,
) -> EvaluationDataset:
    """Run the RAG pipeline on test questions and build a Ragas EvaluationDataset.

    Args:
        test_cases: List of test case dicts with 'question', 'ground_truth', 'contexts'.
        pipeline: RAGPipeline instance.

    Returns:
        EvaluationDataset ready for Ragas evaluate().
    """
    samples = []
    for tc in test_cases:
        question = tc["question"]
        ground_truth = tc["ground_truth"]

        # Run pipeline to get actual answer + retrieved contexts
        result = pipeline.query(question)
        answer = result["answer"]
        retrieved_contexts = [
            s.get("content_preview", "") for s in result.get("sources", [])
        ]

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=retrieved_contexts,
            reference=ground_truth,
        )
        samples.append(sample)
        logger.info(f"Generated sample for: '{question[:50]}'")

    return EvaluationDataset(samples=samples)


def build_evaluation_dataset_static(
    test_cases: List[Dict],
) -> EvaluationDataset:
    """Build a Ragas EvaluationDataset from pre-defined test data (no pipeline needed).

    Uses the ground_truth contexts and a placeholder response for
    context-only metrics evaluation.

    Args:
        test_cases: List of test case dicts with 'question', 'ground_truth', 'contexts'.

    Returns:
        EvaluationDataset ready for Ragas evaluate().
    """
    samples = []
    for tc in test_cases:
        sample = SingleTurnSample(
            user_input=tc["question"],
            response=tc.get("response", tc["ground_truth"]),
            retrieved_contexts=tc.get("contexts", []),
            reference=tc["ground_truth"],
        )
        samples.append(sample)

    return EvaluationDataset(samples=samples)


def run_ragas_evaluation(
    dataset: EvaluationDataset,
    config: Optional[dict] = None,
    metrics_to_run: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run Ragas evaluation on the given dataset.

    Args:
        dataset: EvaluationDataset with samples.
        config: Application config dict.
        metrics_to_run: List of metric names to evaluate. Options:
            'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall'.
            Defaults to all four.

    Returns:
        Dict with metric scores and evaluation metadata.
    """
    if config is None:
        config = load_config()

    if metrics_to_run is None:
        metrics_to_run = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

    llm = _build_ragas_llm(config)
    embeddings = _build_ragas_embeddings(config)

    # Build metric instances
    metric_map = {}
    if "faithfulness" in metrics_to_run:
        metric_map["faithfulness"] = Faithfulness(llm=llm)
    if "answer_relevancy" in metrics_to_run:
        metric_map["answer_relevancy"] = AnswerRelevancy(
            llm=llm, embeddings=embeddings
        )
    if "context_precision" in metrics_to_run:
        metric_map["context_precision"] = ContextPrecisionWithoutReference(llm=llm)
    if "context_recall" in metrics_to_run:
        metric_map["context_recall"] = ContextRecall(llm=llm)

    metrics = list(metric_map.values())

    logger.info(
        f"Running Ragas evaluation: {len(dataset)} samples, "
        f"metrics={list(metric_map.keys())}"
    )

    start = time.perf_counter()

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=False,
        show_progress=True,
    )

    elapsed_s = round(time.perf_counter() - start, 1)

    # Extract scores
    scores = {}
    for name in metric_map:
        score = result.get(name, None)
        if score is not None:
            scores[name] = round(float(score), 4)

    logger.info(f"Ragas evaluation complete ({elapsed_s}s): {scores}")

    return {
        "framework": "ragas",
        "scores": scores,
        "num_samples": len(dataset),
        "evaluation_time_s": elapsed_s,
        "per_sample": result.to_pandas().to_dict(orient="records")
        if hasattr(result, "to_pandas")
        else [],
    }


def save_results(results: Dict, output_path: str) -> None:
    """Save evaluation results to JSON file.

    Args:
        results: Evaluation results dict.
        output_path: Output file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Results saved to {output_path}")
