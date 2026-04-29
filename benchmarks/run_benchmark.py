"""Main benchmark entry point for the RAG system.

Orchestrates performance and quality benchmarks, saves results,
and auto-compares against the latest baseline if available.

Usage:
    # Full benchmark (perf + quality), save as baseline
    python -m benchmarks.run_benchmark --save-baseline

    # Full benchmark, compare against baseline
    python -m benchmarks.run_benchmark

    # Performance only (skip expensive LLM-judge quality eval)
    python -m benchmarks.run_benchmark --perf-only

    # Quality only (skip latency benchmarks)
    python -m benchmarks.run_benchmark --quality-only
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from src.config import get_project_root, load_config
from src.logging.logger import get_logger, setup_logging

logger = get_logger("benchmarks.runner")

BENCHMARK_CONFIG_PATH = "benchmarks/config.yaml"


def _load_bench_config() -> Dict[str, Any]:
    """Load benchmark configuration from benchmarks/config.yaml.

    Returns:
        Benchmark configuration dict.
    """
    root = get_project_root()
    config_path = root / BENCHMARK_CONFIG_PATH
    if not config_path.exists():
        logger.warning(f"Benchmark config not found at {config_path}, using defaults")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_test_set(bench_config: Dict[str, Any]) -> list:
    """Load the benchmark test set.

    Args:
        bench_config: Benchmark configuration dict.

    Returns:
        List of test case dicts.
    """
    root = get_project_root()
    test_set_path = root / bench_config.get("test_set", "benchmarks/test_sets/benchmark_qa.json")
    if not test_set_path.exists():
        logger.error(f"Test set not found: {test_set_path}")
        sys.exit(1)
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    logger.info(f"Loaded {len(test_cases)} test cases from {test_set_path}")
    return test_cases


def _save_result(
    result: Dict[str, Any],
    output_dir: Path,
    prefix: str = "run",
) -> Path:
    """Save benchmark result to a timestamped JSON file.

    Args:
        result: Benchmark result dict.
        output_dir: Directory to save the file.
        prefix: Filename prefix ("run" or "baseline").

    Returns:
        Path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Result saved to {output_path}")
    return output_path


def _get_latest_baseline(baselines_dir: Path) -> Path:
    """Find the latest baseline file.

    Args:
        baselines_dir: Directory containing baseline files.

    Returns:
        Path to the latest baseline, or None if not found.
    """
    latest = baselines_dir / "latest.json"
    if latest.exists():
        return latest
    return None


def main() -> None:
    """Run the benchmark suite."""
    parser = argparse.ArgumentParser(description="RAG System Benchmark Runner")
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save results as a new baseline (also copies to latest.json)",
    )
    parser.add_argument(
        "--perf-only",
        action="store_true",
        help="Run only performance benchmarks (skip quality evaluation)",
    )
    parser.add_argument(
        "--quality-only",
        action="store_true",
        help="Run only quality benchmarks (skip latency measurement)",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip automatic comparison against latest baseline",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    bench_config = _load_bench_config()
    test_cases = _load_test_set(bench_config)

    root = get_project_root()
    baselines_dir = root / bench_config.get("baselines_dir", "benchmarks/baselines")
    results_dir = root / bench_config.get("results_dir", "benchmarks/results")

    # Extract query strings for perf benchmarks
    queries = [tc["question"] for tc in test_cases]

    overall_start = time.perf_counter()

    result = {
        "timestamp": datetime.now().isoformat(),
        "config_snapshot": {
            "llm_model": config.get("llm", {}).get("model", ""),
            "embedding_model": config.get("embedding", {}).get("model", ""),
            "hybrid_mode": config.get("retrieval", {}).get("hybrid_mode", False),
            "enable_reranker": config.get("retrieval", {}).get("enable_reranker", True),
            "top_k": config.get("retrieval", {}).get("top_k", 20),
            "top_n": config.get("retrieval", {}).get("top_n", 5),
            "chunk_size": config.get("chunking", {}).get("chunk_size", 512),
            "chunk_overlap": config.get("chunking", {}).get("chunk_overlap", 64),
            "query_rewriting": config.get("query_rewriting", {}).get("strategy", "none"),
            "cache_enabled": config.get("cache", {}).get("enabled", True),
        },
        "num_test_cases": len(test_cases),
    }

    pipeline = None

    # --- Performance benchmarks ---
    if not args.quality_only:
        logger.info("=" * 60)
        logger.info("  RUNNING PERFORMANCE BENCHMARKS")
        logger.info("=" * 60)

        from benchmarks.perf_harness import run_performance_benchmarks
        perf_results, pipeline = run_performance_benchmarks(
            queries=queries,
            bench_config=bench_config,
            config=config,
        )
        result["performance"] = perf_results
    else:
        logger.info("Skipping performance benchmarks (--quality-only)")

    # --- Quality benchmarks ---
    if not args.perf_only:
        logger.info("=" * 60)
        logger.info("  RUNNING QUALITY BENCHMARKS")
        logger.info("=" * 60)

        # Initialize pipeline if not already done by perf benchmarks
        if pipeline is None:
            from src.pipeline import RAGPipeline
            pipeline = RAGPipeline(config=config)

        from benchmarks.quality_harness import run_quality_benchmarks
        quality_results = run_quality_benchmarks(
            pipeline=pipeline,
            test_cases=test_cases,
            bench_config=bench_config,
            config=config,
        )
        result["quality"] = quality_results
    else:
        logger.info("Skipping quality benchmarks (--perf-only)")

    result["total_benchmark_time_s"] = round(time.perf_counter() - overall_start, 1)

    # --- Save results ---
    if args.save_baseline:
        saved_path = _save_result(result, baselines_dir, prefix="baseline")
        # Copy to latest.json
        latest_path = baselines_dir / "latest.json"
        shutil.copy2(saved_path, latest_path)
        logger.info(f"Baseline saved: {saved_path}")
        logger.info(f"Latest baseline updated: {latest_path}")
        _print_summary(result)
    else:
        saved_path = _save_result(result, results_dir, prefix="run")

        # --- Auto-compare against baseline ---
        if not args.no_compare:
            baseline_path = _get_latest_baseline(baselines_dir)
            if baseline_path:
                logger.info(f"Comparing against baseline: {baseline_path}")
                from benchmarks.compare import (
                    compare_results,
                    format_report,
                    has_critical_regressions,
                )

                with open(baseline_path, "r", encoding="utf-8") as f:
                    baseline = json.load(f)

                thresholds = bench_config.get("thresholds")
                comparisons = compare_results(baseline, result, thresholds)
                report = format_report(
                    comparisons,
                    baseline_path=str(baseline_path),
                    current_path=str(saved_path),
                )
                print(report)

                if has_critical_regressions(comparisons):
                    logger.warning("CRITICAL REGRESSIONS DETECTED")
                    sys.exit(1)
            else:
                logger.info(
                    "No baseline found. Run with --save-baseline first to establish a baseline."
                )
                _print_summary(result)
        else:
            _print_summary(result)

    logger.info(f"Benchmark complete in {result['total_benchmark_time_s']}s")


def _print_summary(result: Dict[str, Any]) -> None:
    """Print a standalone summary when no comparison is available.

    Args:
        result: Benchmark result dict.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  RAG Benchmark Results — {result['timestamp']}")
    print(sep)

    if "performance" in result:
        perf = result["performance"]
        print("\n  PERFORMANCE")
        print(f"  {'-' * 56}")

        init = perf.get("init", {})
        if not init.get("skipped"):
            print(f"  {'pipeline_init_time_s':<35s} {init.get('pipeline_init_time_s', 0):.3f}")

        for section_key, metric_name in [
            ("embedding", "embedding_latency_ms"),
            ("retrieval", "retrieval_latency_ms"),
            ("end_to_end", "total_latency_ms"),
        ]:
            section = perf.get(section_key, {})
            stats = section.get(f"{metric_name}", {})
            if stats:
                print(f"  {f'{metric_name} (p50)':<35s} {stats.get('p50', 0):.1f}")
                print(f"  {f'{metric_name} (p95)':<35s} {stats.get('p95', 0):.1f}")

        bm25 = perf.get("bm25_rebuild", {})
        if not bm25.get("skipped"):
            print(f"  {'bm25_rebuild_time_ms':<35s} {bm25.get('bm25_rebuild_time_ms', 0):.1f}")

        cache = perf.get("cache", {})
        cache_stats = cache.get("cache_stats", {})
        if cache_stats:
            print(f"  {'cache_hit_rate':<35s} {cache_stats.get('hit_rate', 0):.4f}")
            print(f"  {'cache_speedup_ratio':<35s} {cache.get('speedup_ratio', 0):.2f}x")

        mem = perf.get("memory", {})
        if mem:
            print(f"  {'memory_peak_mb':<35s} {mem.get('memory_peak_mb', 0):.1f}")

    if "quality" in result:
        quality = result["quality"]
        print("\n  QUALITY")
        print(f"  {'-' * 56}")

        retrieval = quality.get("retrieval", {})
        recall = retrieval.get("recall", {})
        for k, v in recall.items():
            if v is not None:
                print(f"  {f'retrieval/{k}':<35s} {v:.4f}")
        mrr = retrieval.get("mrr")
        if mrr is not None:
            print(f"  {'retrieval/mrr':<35s} {mrr:.4f}")

        for framework in ["ragas", "deepeval"]:
            fw_result = quality.get(framework, {})
            scores = fw_result.get("scores", {})
            for metric_name, score in scores.items():
                print(f"  {f'{framework}/{metric_name}':<35s} {score:.4f}")

    print(f"\n{sep}")
    print(f"  Total benchmark time: {result.get('total_benchmark_time_s', 0):.1f}s")
    print(sep)


if __name__ == "__main__":
    main()
