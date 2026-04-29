"""Quality benchmark harness for the RAG pipeline.

Measures retrieval accuracy (Recall@K, MRR) and generation quality
(Ragas + DeepEval metrics) against a benchmark test set with
ground-truth answers and expected source files.
"""

from typing import Any, Dict, List, Optional

from src.config import load_config
from src.logging.logger import get_logger

logger = get_logger("benchmarks.quality")


def _compute_recall_at_k(
    retrieved_files: List[str],
    expected_files: List[str],
    k: int,
) -> float:
    """Compute Recall@K for a single query.

    Args:
        retrieved_files: List of retrieved source file names (ordered by rank).
        expected_files: List of expected source file names (ground truth).
        k: Number of top results to consider.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not expected_files:
        return 1.0  # No expected files = vacuously correct
    top_k = set(retrieved_files[:k])
    expected = set(expected_files)
    return len(top_k & expected) / len(expected)


def _compute_mrr(
    retrieved_files: List[str],
    expected_files: List[str],
) -> float:
    """Compute Mean Reciprocal Rank for a single query.

    Args:
        retrieved_files: List of retrieved source file names (ordered by rank).
        expected_files: List of expected source file names (ground truth).

    Returns:
        Reciprocal rank (1/rank of first relevant result), or 0 if none found.
    """
    if not expected_files:
        return 1.0
    expected = set(expected_files)
    for i, f in enumerate(retrieved_files):
        if f in expected:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retrieval_quality(
    pipeline,
    test_cases: List[Dict],
    recall_k_values: List[int] = None,
    compute_mrr: bool = True,
) -> Dict[str, Any]:
    """Evaluate retrieval quality using Recall@K and MRR.

    Runs pipeline queries and compares retrieved source files against
    expected_source_files in the test set.

    Args:
        pipeline: Initialized RAGPipeline instance.
        test_cases: List of test case dicts with 'question' and 'expected_source_files'.
        recall_k_values: List of K values for Recall@K computation.
        compute_mrr: Whether to compute Mean Reciprocal Rank.

    Returns:
        Dict with retrieval quality metrics.
    """
    if recall_k_values is None:
        recall_k_values = [5, 10]

    # Filter test cases that have expected_source_files
    cases_with_files = [tc for tc in test_cases if tc.get("expected_source_files")]

    if not cases_with_files:
        logger.info("No test cases with expected_source_files, skipping retrieval recall evaluation")
        return {
            "recall": {f"recall@{k}": None for k in recall_k_values},
            "mrr": None,
            "note": "No test cases have expected_source_files defined",
        }

    recall_scores = {k: [] for k in recall_k_values}
    mrr_scores = []

    for tc in cases_with_files:
        question = tc["question"]
        expected = tc["expected_source_files"]

        result = pipeline.query(question)
        retrieved_files = [s.get("file", "") for s in result.get("sources", [])]

        for k in recall_k_values:
            recall_scores[k].append(_compute_recall_at_k(retrieved_files, expected, k))

        if compute_mrr:
            mrr_scores.append(_compute_mrr(retrieved_files, expected))

    # Aggregate
    recall_results = {}
    for k in recall_k_values:
        scores = recall_scores[k]
        avg = round(sum(scores) / len(scores), 4) if scores else 0.0
        recall_results[f"recall@{k}"] = avg

    mrr_avg = round(sum(mrr_scores) / len(mrr_scores), 4) if mrr_scores else None

    result = {
        "recall": recall_results,
        "mrr": mrr_avg,
        "num_cases_with_expected_files": len(cases_with_files),
    }
    logger.info(f"Retrieval quality: {recall_results}, MRR={mrr_avg}")
    return result


def evaluate_generation_quality_ragas(
    pipeline,
    test_cases: List[Dict],
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """Evaluate generation quality using Ragas framework.

    Delegates to evaluation.evaluate_ragas to avoid code duplication.

    Args:
        pipeline: Initialized RAGPipeline instance.
        test_cases: List of test case dicts with 'question', 'ground_truth', 'contexts'.
        config: Optional RAG system config dict.

    Returns:
        Dict with Ragas evaluation scores.
    """
    if config is None:
        config = load_config()

    try:
        from evaluation.evaluate_ragas import (
            build_evaluation_dataset_from_pipeline,
            run_ragas_evaluation,
        )
    except ImportError as e:
        logger.warning(f"Ragas not available, skipping: {e}")
        return {"error": str(e), "skipped": True}

    logger.info(f"Running Ragas evaluation on {len(test_cases)} test cases via pipeline...")
    dataset = build_evaluation_dataset_from_pipeline(test_cases, pipeline)
    return run_ragas_evaluation(dataset, config=config)


def evaluate_generation_quality_deepeval(
    pipeline,
    test_cases: List[Dict],
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """Evaluate generation quality using DeepEval framework.

    Delegates to evaluation.evaluate_deepeval to avoid code duplication.

    Args:
        pipeline: Initialized RAGPipeline instance.
        test_cases: List of test case dicts with 'question', 'ground_truth', 'contexts'.
        config: Optional RAG system config dict.

    Returns:
        Dict with DeepEval evaluation scores.
    """
    if config is None:
        config = load_config()

    try:
        from evaluation.evaluate_deepeval import (
            build_test_cases_from_pipeline,
            run_deepeval_evaluation,
        )
    except ImportError as e:
        logger.warning(f"DeepEval not available, skipping: {e}")
        return {"error": str(e), "skipped": True}

    logger.info(f"Running DeepEval evaluation on {len(test_cases)} test cases via pipeline...")
    deepeval_cases = build_test_cases_from_pipeline(test_cases, pipeline)
    return run_deepeval_evaluation(deepeval_cases, config=config)


def run_quality_benchmarks(
    pipeline,
    test_cases: List[Dict],
    bench_config: Dict[str, Any],
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """Run all quality benchmarks and return aggregated results.

    Args:
        pipeline: Initialized RAGPipeline instance.
        test_cases: List of test case dicts.
        bench_config: Benchmark configuration dict.
        config: Optional RAG system config dict.

    Returns:
        Dict with all quality measurement results.
    """
    quality_cfg = bench_config.get("quality", {})
    frameworks = quality_cfg.get("frameworks", ["ragas", "deepeval"])
    recall_k_values = quality_cfg.get("retrieval_recall_k", [5, 10])
    compute_mrr = quality_cfg.get("compute_mrr", True)

    results = {}

    # 1. Retrieval quality (Recall@K, MRR)
    logger.info("=== Evaluating retrieval quality ===")
    results["retrieval"] = evaluate_retrieval_quality(
        pipeline, test_cases, recall_k_values=recall_k_values, compute_mrr=compute_mrr
    )

    # 2. Ragas
    if "ragas" in frameworks:
        logger.info("=== Evaluating with Ragas ===")
        results["ragas"] = evaluate_generation_quality_ragas(pipeline, test_cases, config)

    # 3. DeepEval
    if "deepeval" in frameworks:
        logger.info("=== Evaluating with DeepEval ===")
        results["deepeval"] = evaluate_generation_quality_deepeval(pipeline, test_cases, config)

    return results
