"""Unified evaluation runner for the RAG system.

Runs both Ragas and DeepEval evaluations, either against
pre-defined test data (static mode) or against the live pipeline.

Usage:
    # Static evaluation (uses test set contexts as-is)
    python -m evaluation.run_evaluation --mode static

    # Pipeline evaluation (runs queries through the RAG pipeline)
    python -m evaluation.run_evaluation --mode pipeline

    # Only run specific framework
    python -m evaluation.run_evaluation --framework ragas
    python -m evaluation.run_evaluation --framework deepeval
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.config import load_config
from src.logging.logger import get_logger, setup_logging
from src.monitoring.mlflow_tracker import init_mlflow, log_evaluation_run

logger = get_logger("evaluation.runner")

DEFAULT_TEST_SET = "evaluation/test_set/sample_test_set.json"
DEFAULT_OUTPUT_DIR = "evaluation/results"


def run_ragas(test_data, config, mode, pipeline=None) -> Dict[str, Any]:
    """Run Ragas evaluation.

    Args:
        test_data: List of test case dicts.
        config: Application config dict.
        mode: 'static' or 'pipeline'.
        pipeline: RAGPipeline instance (required if mode='pipeline').

    Returns:
        Ragas evaluation results dict.
    """
    from evaluation.evaluate_ragas import (
        build_evaluation_dataset_from_pipeline,
        build_evaluation_dataset_static,
        run_ragas_evaluation,
    )

    if mode == "pipeline" and pipeline is not None:
        logger.info("Building Ragas dataset from pipeline...")
        dataset = build_evaluation_dataset_from_pipeline(test_data, pipeline)
    else:
        logger.info("Building Ragas dataset from static test data...")
        dataset = build_evaluation_dataset_static(test_data)

    return run_ragas_evaluation(dataset, config=config)


def run_deepeval(test_data, config, mode, pipeline=None) -> Dict[str, Any]:
    """Run DeepEval evaluation.

    Args:
        test_data: List of test case dicts.
        config: Application config dict.
        mode: 'static' or 'pipeline'.
        pipeline: RAGPipeline instance (required if mode='pipeline').

    Returns:
        DeepEval evaluation results dict.
    """
    from evaluation.evaluate_deepeval import (
        build_test_cases,
        build_test_cases_from_pipeline,
        run_deepeval_evaluation,
    )

    if mode == "pipeline" and pipeline is not None:
        logger.info("Building DeepEval test cases from pipeline...")
        test_cases = build_test_cases_from_pipeline(test_data, pipeline)
    else:
        logger.info("Building DeepEval test cases from static test data...")
        test_cases = build_test_cases(test_data)

    return run_deepeval_evaluation(test_cases, config=config)


def print_summary(results: Dict[str, Any]) -> None:
    """Print a formatted evaluation summary.

    Args:
        results: Combined results dict with 'ragas' and/or 'deepeval' keys.
    """
    print("\n" + "=" * 60)
    print("  RAG Evaluation Summary")
    print("=" * 60)

    for framework, data in results.items():
        if framework in ("ragas", "deepeval"):
            print(f"\n  [{framework.upper()}]")
            scores = data.get("scores", {})
            for metric, score in scores.items():
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"    {metric:<25s} {bar} {score:.4f}")
            print(f"    Samples: {data.get('num_samples', data.get('num_cases', 0))}")
            print(f"    Time: {data.get('evaluation_time_s', 0)}s")

    print("\n" + "=" * 60)


def main() -> None:
    """Run the evaluation pipeline."""
    parser = argparse.ArgumentParser(description="RAG Evaluation Runner")
    parser.add_argument(
        "--mode",
        choices=["static", "pipeline"],
        default="static",
        help="Evaluation mode: 'static' uses test set as-is, "
             "'pipeline' runs queries through the RAG pipeline",
    )
    parser.add_argument(
        "--framework",
        choices=["ragas", "deepeval", "all"],
        default="all",
        help="Which evaluation framework to run",
    )
    parser.add_argument(
        "--test-set",
        default=DEFAULT_TEST_SET,
        help="Path to test set JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save evaluation results",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config()

    # Initialize MLflow tracking if enabled
    init_mlflow(config)

    # Load test data
    test_set_path = Path(args.test_set)
    if not test_set_path.exists():
        logger.error(f"Test set not found: {test_set_path}")
        sys.exit(1)

    with open(test_set_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info(f"Loaded {len(test_data)} test cases from {test_set_path}")

    # Initialize pipeline if needed
    pipeline = None
    if args.mode == "pipeline":
        from src.pipeline import RAGPipeline
        pipeline = RAGPipeline(config)
        logger.info("Pipeline initialized for evaluation")

    # Run evaluations
    results = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "test_set": str(test_set_path),
        "num_test_cases": len(test_data),
    }

    start = time.perf_counter()

    # Build pipeline params snapshot for MLflow
    mlflow_params = {
        "mode": args.mode,
        "llm_model": config.get("llm", {}).get("model", ""),
        "embedding_model": config.get("embedding", {}).get("model", ""),
        "hybrid_mode": config.get("retrieval", {}).get("hybrid_mode", False),
        "top_k": config.get("retrieval", {}).get("top_k", 20),
        "top_n": config.get("retrieval", {}).get("top_n", 5),
        "chunk_size": config.get("chunking", {}).get("chunk_size", 512),
    }

    if args.framework in ("ragas", "all"):
        try:
            results["ragas"] = run_ragas(test_data, config, args.mode, pipeline)
            # Log to MLflow
            ragas_scores = results["ragas"].get("scores", {})
            if ragas_scores:
                log_evaluation_run(
                    framework="ragas",
                    scores=ragas_scores,
                    params=mlflow_params,
                    num_samples=results["ragas"].get("num_samples", len(test_data)),
                    evaluation_time_s=results["ragas"].get("evaluation_time_s", 0),
                )
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            results["ragas"] = {"error": str(e)}

    if args.framework in ("deepeval", "all"):
        try:
            results["deepeval"] = run_deepeval(test_data, config, args.mode, pipeline)
            # Log to MLflow
            deepeval_scores = results["deepeval"].get("scores", {})
            if deepeval_scores:
                log_evaluation_run(
                    framework="deepeval",
                    scores=deepeval_scores,
                    params=mlflow_params,
                    num_samples=results["deepeval"].get("num_cases", len(test_data)),
                    evaluation_time_s=results["deepeval"].get("evaluation_time_s", 0),
                )
        except Exception as e:
            logger.error(f"DeepEval evaluation failed: {e}")
            results["deepeval"] = {"error": str(e)}

    results["total_time_s"] = round(time.perf_counter() - start, 1)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_{args.mode}_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
