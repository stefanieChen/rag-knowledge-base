"""Unified evaluation & benchmark CLI for the RAG system.

Consolidates quality evaluation (Ragas/DeepEval), performance benchmarks,
and baseline comparison into a single entry point.

Usage:
    # Quality evaluation only (Ragas + DeepEval)
    python -m evaluation.run --suite quality

    # Performance benchmarks only (latency, memory, cache)
    python -m evaluation.run --suite perf

    # Full suite (quality + performance)
    python -m evaluation.run --suite full

    # Save as baseline for future comparison
    python -m evaluation.run --suite full --save-baseline

    # Specific framework
    python -m evaluation.run --suite quality --framework ragas

    # Static evaluation (no pipeline, uses test set contexts)
    python -m evaluation.run --suite quality --mode static
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import get_project_root, load_config
from src.logging.logger import get_logger, setup_logging

logger = get_logger("evaluation.unified")


def _load_test_set(path: str) -> list:
    """Load test set from a JSON file.

    Args:
        path: Path to the test set JSON file.

    Returns:
        List of test case dicts.
    """
    test_path = Path(path)
    if not test_path.is_absolute():
        test_path = get_project_root() / test_path
    if not test_path.exists():
        logger.error(f"Test set not found: {test_path}")
        sys.exit(1)
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} test cases from {test_path}")
    return data


def _run_quality(
    test_data: list,
    config: dict,
    mode: str,
    framework: str,
    pipeline=None,
) -> Dict[str, Any]:
    """Run quality evaluation (Ragas and/or DeepEval).

    Args:
        test_data: List of test case dicts.
        config: Application config dict.
        mode: 'static' or 'pipeline'.
        framework: 'ragas', 'deepeval', or 'all'.
        pipeline: RAGPipeline instance (required if mode='pipeline').

    Returns:
        Quality results dict.
    """
    from evaluation.run_evaluation import run_deepeval, run_ragas
    from src.monitoring.mlflow_tracker import init_mlflow, log_evaluation_run

    init_mlflow(config)

    # Retrieval metrics (Recall@K, MRR) — only in pipeline mode
    results: Dict[str, Any] = {}
    if mode == "pipeline" and pipeline is not None:
        try:
            from benchmarks.quality_harness import evaluate_retrieval_quality
            logger.info("=== Evaluating retrieval quality (Recall@K, MRR) ===")
            results["retrieval"] = evaluate_retrieval_quality(pipeline, test_data)
        except Exception as e:
            logger.warning(f"Retrieval quality eval failed: {e}")

    mlflow_params = {
        "mode": mode,
        "llm_model": config.get("llm", {}).get("model", ""),
        "embedding_model": config.get("embedding", {}).get("model", ""),
        "hybrid_mode": config.get("retrieval", {}).get("hybrid_mode", False),
        "top_k": config.get("retrieval", {}).get("top_k", 20),
        "top_n": config.get("retrieval", {}).get("top_n", 5),
        "chunk_size": config.get("chunking", {}).get("chunk_size", 512),
    }

    if framework in ("ragas", "all"):
        try:
            logger.info("=== Running Ragas evaluation ===")
            results["ragas"] = run_ragas(test_data, config, mode, pipeline)
            ragas_scores = results["ragas"].get("scores", {})
            if ragas_scores:
                log_evaluation_run(
                    framework="ragas", scores=ragas_scores,
                    params=mlflow_params,
                    num_samples=results["ragas"].get("num_samples", len(test_data)),
                    evaluation_time_s=results["ragas"].get("evaluation_time_s", 0),
                )
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            results["ragas"] = {"error": str(e)}

    if framework in ("deepeval", "all"):
        try:
            logger.info("=== Running DeepEval evaluation ===")
            results["deepeval"] = run_deepeval(test_data, config, mode, pipeline)
            deepeval_scores = results["deepeval"].get("scores", {})
            if deepeval_scores:
                log_evaluation_run(
                    framework="deepeval", scores=deepeval_scores,
                    params=mlflow_params,
                    num_samples=results["deepeval"].get("num_cases", len(test_data)),
                    evaluation_time_s=results["deepeval"].get("evaluation_time_s", 0),
                )
        except Exception as e:
            logger.error(f"DeepEval evaluation failed: {e}")
            results["deepeval"] = {"error": str(e)}

    return results


def _run_perf(
    queries: list,
    config: dict,
    bench_config: dict,
) -> tuple:
    """Run performance benchmarks.

    Args:
        queries: List of query strings.
        config: Application config dict.
        bench_config: Benchmark configuration dict.

    Returns:
        Tuple of (performance results dict, pipeline instance).
    """
    from benchmarks.perf_harness import run_performance_benchmarks

    logger.info("=== Running performance benchmarks ===")
    return run_performance_benchmarks(
        queries=queries, bench_config=bench_config, config=config,
    )


def _print_summary(result: Dict[str, Any]) -> None:
    """Print a formatted summary of results.

    Args:
        result: Combined result dict.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  RAG Evaluation Results — {result.get('timestamp', '')}")
    print(sep)

    # Quality results
    for fw in ("ragas", "deepeval"):
        fw_data = result.get("quality", {}).get(fw, {})
        scores = fw_data.get("scores", {})
        if scores:
            print(f"\n  [{fw.upper()}]")
            for metric, score in scores.items():
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"    {metric:<25s} {bar} {score:.4f}")

    retrieval = result.get("quality", {}).get("retrieval", {})
    recall = retrieval.get("recall", {})
    if recall:
        print(f"\n  [RETRIEVAL]")
        for k, v in recall.items():
            if v is not None:
                print(f"    {k:<25s} {v:.4f}")

    # Performance results
    perf = result.get("performance", {})
    if perf:
        print(f"\n  [PERFORMANCE]")
        for section, metric_key in [
            ("embedding", "embedding_latency_ms"),
            ("retrieval", "retrieval_latency_ms"),
            ("end_to_end", "total_latency_ms"),
        ]:
            stats = perf.get(section, {}).get(metric_key, {})
            if stats:
                print(f"    {f'{metric_key} (p50)':<35s} {stats.get('p50', 0):.1f}")
                print(f"    {f'{metric_key} (p95)':<35s} {stats.get('p95', 0):.1f}")

    print(f"\n  Total time: {result.get('total_time_s', 0):.1f}s")
    print(sep)


def main() -> None:
    """Unified evaluation & benchmark CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified RAG Evaluation & Benchmark Runner"
    )
    parser.add_argument(
        "--suite",
        choices=["quality", "perf", "full"],
        default="quality",
        help="Which suite to run: 'quality' (Ragas/DeepEval), "
             "'perf' (latency/memory), 'full' (both)",
    )
    parser.add_argument(
        "--mode",
        choices=["static", "pipeline"],
        default="pipeline",
        help="'static' uses test set contexts, 'pipeline' runs live queries",
    )
    parser.add_argument(
        "--framework",
        choices=["ragas", "deepeval", "all"],
        default="all",
        help="Which quality framework to run",
    )
    parser.add_argument(
        "--test-set",
        default=None,
        help="Path to test set JSON (default: evaluation or benchmark test set)",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save results as a new baseline",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/results",
        help="Directory to save results",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config()

    # Determine test set path
    if args.test_set:
        test_set_path = args.test_set
    elif args.suite == "perf":
        test_set_path = "benchmarks/test_sets/benchmark_qa.json"
    else:
        test_set_path = "evaluation/test_set/sample_test_set.json"

    test_data = _load_test_set(test_set_path)

    overall_start = time.perf_counter()
    result: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "suite": args.suite,
        "mode": args.mode,
        "test_set": test_set_path,
        "num_test_cases": len(test_data),
    }

    pipeline = None

    # Performance benchmarks
    if args.suite in ("perf", "full"):
        import yaml
        bench_config_path = get_project_root() / "benchmarks" / "config.yaml"
        bench_config = {}
        if bench_config_path.exists():
            with open(bench_config_path, "r", encoding="utf-8") as f:
                bench_config = yaml.safe_load(f) or {}

        queries = [tc["question"] for tc in test_data]
        perf_results, pipeline = _run_perf(queries, config, bench_config)
        result["performance"] = perf_results

    # Quality evaluation
    if args.suite in ("quality", "full"):
        if args.mode == "pipeline" and pipeline is None:
            from src.pipeline import RAGPipeline
            pipeline = RAGPipeline(config)

        result["quality"] = _run_quality(
            test_data, config, args.mode, args.framework, pipeline,
        )

    result["total_time_s"] = round(time.perf_counter() - overall_start, 1)

    # Save results
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = get_project_root() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "baseline" if args.save_baseline else "run"
    output_path = output_dir / f"{prefix}_{args.suite}_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Results saved to {output_path}")

    # Baseline handling
    if args.save_baseline:
        import shutil
        baselines_dir = get_project_root() / "benchmarks" / "baselines"
        baselines_dir.mkdir(parents=True, exist_ok=True)
        latest = baselines_dir / "latest.json"
        shutil.copy2(output_path, latest)
        logger.info(f"Baseline updated: {latest}")

    _print_summary(result)


if __name__ == "__main__":
    main()
