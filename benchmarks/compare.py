"""Benchmark comparison engine.

Compares two benchmark result JSON files and produces a formatted diff
report showing performance regressions, improvements, and stable metrics.

Usage:
    python -m benchmarks.compare baselines/baseline_X.json results/run_Y.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Metric definitions: (json_path, display_name, metric_type)
# metric_type: "latency" (lower=better), "quality" (higher=better), "ratio" (higher=better)
METRIC_DEFINITIONS: List[Tuple[str, str, str]] = [
    # Performance — latency (lower is better)
    ("performance.init.pipeline_init_time_s", "pipeline_init_time_s", "latency"),
    ("performance.embedding.embedding_latency_ms.p50", "embedding_latency_ms (p50)", "latency"),
    ("performance.embedding.embedding_latency_ms.p95", "embedding_latency_ms (p95)", "latency"),
    ("performance.retrieval.retrieval_latency_ms.p50", "retrieval_latency_ms (p50)", "latency"),
    ("performance.retrieval.retrieval_latency_ms.p95", "retrieval_latency_ms (p95)", "latency"),
    ("performance.end_to_end.total_latency_ms.p50", "total_latency_ms (p50)", "latency"),
    ("performance.end_to_end.total_latency_ms.p95", "total_latency_ms (p95)", "latency"),
    ("performance.bm25_rebuild.bm25_rebuild_time_ms", "bm25_rebuild_time_ms", "latency"),
    # Performance — memory (lower is better)
    ("performance.memory.memory_peak_mb", "memory_peak_mb", "latency"),
    # Performance — cache (higher is better)
    ("performance.cache.cache_stats.hit_rate", "cache_hit_rate", "quality"),
    ("performance.cache.speedup_ratio", "cache_speedup_ratio", "ratio"),
    # Quality — retrieval (higher is better)
    ("quality.retrieval.recall.recall@5", "retrieval/recall@5", "quality"),
    ("quality.retrieval.recall.recall@10", "retrieval/recall@10", "quality"),
    ("quality.retrieval.mrr", "retrieval/mrr", "quality"),
    # Quality — Ragas (higher is better)
    ("quality.ragas.scores.faithfulness", "ragas/faithfulness", "quality"),
    ("quality.ragas.scores.answer_relevancy", "ragas/answer_relevancy", "quality"),
    ("quality.ragas.scores.context_precision", "ragas/context_precision", "quality"),
    ("quality.ragas.scores.context_recall", "ragas/context_recall", "quality"),
    # Quality — DeepEval (higher is better)
    ("quality.deepeval.scores.hallucination", "deepeval/hallucination", "quality"),
    ("quality.deepeval.scores.answer_relevancy", "deepeval/answer_relevancy", "quality"),
    ("quality.deepeval.scores.faithfulness", "deepeval/faithfulness", "quality"),
]


def _get_nested(data: Dict, path: str) -> Optional[float]:
    """Get a value from a nested dict using dot-separated path.

    Args:
        data: Nested dictionary.
        path: Dot-separated key path (e.g., "performance.init.pipeline_init_time_s").

    Returns:
        The value at the path, or None if not found.
    """
    keys = path.split(".")
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _classify_delta(
    delta_pct: float,
    metric_type: str,
    thresholds: Dict[str, Any],
) -> str:
    """Classify a delta percentage into a status symbol.

    Args:
        delta_pct: Percentage change (positive = increased, negative = decreased).
        metric_type: "latency" (lower=better) or "quality" (higher=better).
        thresholds: Dict with 'latency' and 'quality' threshold sections.

    Returns:
        Status string: "CRITICAL", "WARNING", "IMPROVED", "STABLE".
    """
    if metric_type == "latency":
        # For latency: positive delta = regression (slower)
        threshold_cfg = thresholds.get("latency", {})
        critical = threshold_cfg.get("critical", 20.0)
        warning = threshold_cfg.get("warning", 5.0)
        if delta_pct > critical:
            return "CRITICAL"
        elif delta_pct > warning:
            return "WARNING"
        elif delta_pct < -warning:
            return "IMPROVED"
        return "STABLE"
    else:
        # For quality/ratio: negative delta = regression (worse)
        threshold_cfg = thresholds.get("quality", {})
        critical = threshold_cfg.get("critical", 10.0)
        warning = threshold_cfg.get("warning", 3.0)
        if delta_pct < -critical:
            return "CRITICAL"
        elif delta_pct < -warning:
            return "WARNING"
        elif delta_pct > warning:
            return "IMPROVED"
        return "STABLE"


_STATUS_SYMBOLS = {
    "CRITICAL": "\u274c",   # Red X
    "WARNING": "\u26a0\ufe0f",    # Warning sign
    "IMPROVED": "\u2705",   # Green check
    "STABLE": "\u2796",     # Dash
}


def compare_results(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
    thresholds: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Compare baseline and current benchmark results.

    Args:
        baseline: Baseline benchmark result dict.
        current: Current benchmark result dict.
        thresholds: Regression threshold config.

    Returns:
        List of comparison dicts with metric name, values, delta, and status.
    """
    if thresholds is None:
        thresholds = {
            "latency": {"critical": 20.0, "warning": 5.0},
            "quality": {"critical": 10.0, "warning": 3.0},
        }

    comparisons = []

    for json_path, display_name, metric_type in METRIC_DEFINITIONS:
        base_val = _get_nested(baseline, json_path)
        curr_val = _get_nested(current, json_path)

        if base_val is None and curr_val is None:
            continue

        if base_val is None:
            comparisons.append({
                "metric": display_name,
                "baseline": None,
                "current": curr_val,
                "delta_pct": None,
                "status": "NEW",
                "type": metric_type,
            })
            continue

        if curr_val is None:
            comparisons.append({
                "metric": display_name,
                "baseline": base_val,
                "current": None,
                "delta_pct": None,
                "status": "MISSING",
                "type": metric_type,
            })
            continue

        # Compute delta percentage
        if base_val == 0:
            delta_pct = 0.0 if curr_val == 0 else 100.0
        else:
            delta_pct = round(((curr_val - base_val) / abs(base_val)) * 100, 1)

        status = _classify_delta(delta_pct, metric_type, thresholds)

        comparisons.append({
            "metric": display_name,
            "baseline": round(base_val, 4),
            "current": round(curr_val, 4),
            "delta_pct": delta_pct,
            "status": status,
            "type": metric_type,
        })

    return comparisons


def format_report(
    comparisons: List[Dict[str, Any]],
    baseline_path: str = "",
    current_path: str = "",
) -> str:
    """Format a comparison report as a human-readable string.

    Args:
        comparisons: List of comparison dicts from compare_results().
        baseline_path: Path to baseline file (for display).
        current_path: Path to current file (for display).

    Returns:
        Formatted report string.
    """
    lines = []
    sep = "=" * 72
    thin_sep = "-" * 72

    lines.append(sep)
    lines.append(f"  RAG Benchmark Comparison Report")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if baseline_path:
        lines.append(f"  Baseline:  {Path(baseline_path).name}")
    if current_path:
        lines.append(f"  Current:   {Path(current_path).name}")
    lines.append(sep)

    # Split into performance and quality sections
    perf_metrics = [c for c in comparisons if c["type"] == "latency" or c["metric"].startswith("cache_")]
    quality_metrics = [c for c in comparisons if c not in perf_metrics]

    header = f"  {'Metric':<32s} {'Baseline':>10s} {'Current':>10s} {'Delta':>8s}  Status"

    if perf_metrics:
        lines.append("")
        lines.append("  PERFORMANCE")
        lines.append(f"  {thin_sep[2:]}")
        lines.append(header)
        lines.append(f"  {thin_sep[2:]}")
        for c in perf_metrics:
            lines.append(_format_row(c))

    if quality_metrics:
        lines.append("")
        lines.append("  QUALITY")
        lines.append(f"  {thin_sep[2:]}")
        lines.append(header)
        lines.append(f"  {thin_sep[2:]}")
        for c in quality_metrics:
            lines.append(_format_row(c))

    lines.append("")
    lines.append(sep)

    # Summary counts
    statuses = [c["status"] for c in comparisons if c["status"] not in ("NEW", "MISSING")]
    critical = sum(1 for s in statuses if s == "CRITICAL")
    warnings = sum(1 for s in statuses if s == "WARNING")
    improved = sum(1 for s in statuses if s == "IMPROVED")
    stable = sum(1 for s in statuses if s == "STABLE")

    lines.append(
        f"  Summary: {critical} critical, {warnings} warnings, "
        f"{improved} improved, {stable} stable"
    )

    has_regression = critical > 0
    if has_regression:
        lines.append("  RESULT: REGRESSION DETECTED")
    else:
        lines.append("  RESULT: OK")
    lines.append(sep)

    return "\n".join(lines)


def _format_row(c: Dict[str, Any]) -> str:
    """Format a single comparison row.

    Args:
        c: Comparison dict.

    Returns:
        Formatted string for one metric row.
    """
    metric = c["metric"]
    base_str = f"{c['baseline']:.4f}" if c["baseline"] is not None else "N/A"
    curr_str = f"{c['current']:.4f}" if c["current"] is not None else "N/A"

    if c["delta_pct"] is not None:
        sign = "+" if c["delta_pct"] > 0 else ""
        delta_str = f"{sign}{c['delta_pct']:.1f}%"
    elif c["status"] == "NEW":
        delta_str = "new"
    elif c["status"] == "MISSING":
        delta_str = "gone"
    else:
        delta_str = "-"

    symbol = _STATUS_SYMBOLS.get(c["status"], "")

    # Adjust formatting for large numbers
    if c["baseline"] is not None and abs(c["baseline"]) >= 100:
        base_str = f"{c['baseline']:.1f}"
    if c["current"] is not None and abs(c["current"]) >= 100:
        curr_str = f"{c['current']:.1f}"

    return f"  {metric:<32s} {base_str:>10s} {curr_str:>10s} {delta_str:>8s}  {symbol}"


def has_critical_regressions(comparisons: List[Dict[str, Any]]) -> bool:
    """Check if any metrics have critical regressions.

    Args:
        comparisons: List of comparison dicts.

    Returns:
        True if any metric has CRITICAL status.
    """
    return any(c["status"] == "CRITICAL" for c in comparisons)


def main() -> None:
    """CLI entry point for comparing two benchmark result files."""
    parser = argparse.ArgumentParser(description="Compare RAG benchmark results")
    parser.add_argument("baseline", help="Path to baseline JSON file")
    parser.add_argument("current", help="Path to current run JSON file")
    parser.add_argument(
        "--threshold-file",
        default=None,
        help="Path to benchmark config.yaml for custom thresholds",
    )
    args = parser.parse_args()

    with open(args.baseline, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(args.current, "r", encoding="utf-8") as f:
        current = json.load(f)

    thresholds = None
    if args.threshold_file:
        import yaml
        with open(args.threshold_file, "r", encoding="utf-8") as f:
            bench_cfg = yaml.safe_load(f)
        thresholds = bench_cfg.get("thresholds")

    comparisons = compare_results(baseline, current, thresholds)
    report = format_report(comparisons, args.baseline, args.current)
    print(report)

    if has_critical_regressions(comparisons):
        sys.exit(1)


if __name__ == "__main__":
    main()
