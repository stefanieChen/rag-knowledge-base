"""Performance benchmark harness for the RAG pipeline.

Measures latency (per-component and end-to-end), pipeline init time,
BM25 rebuild time, cache effectiveness, and peak memory usage.
All measurements run against live Ollama models.
"""

import statistics
import time
import tracemalloc
from typing import Any, Dict, List, Optional

from src.config import load_config, reset_config
from src.logging.logger import get_logger

logger = get_logger("benchmarks.perf")


def _percentile(data: List[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list.

    Args:
        data: List of numeric values.
        p: Percentile value between 0 and 100.

    Returns:
        The p-th percentile value.
    """
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of measurements.

    Args:
        values: List of numeric measurements.

    Returns:
        Dict with mean, p50, p95, min, max, stdev.
    """
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    return {
        "mean": round(statistics.mean(values), 2),
        "p50": round(_percentile(values, 50), 2),
        "p95": round(_percentile(values, 95), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
    }


def benchmark_pipeline_init(config: Optional[dict] = None) -> Dict[str, Any]:
    """Measure pipeline cold-start initialization time.

    Creates a fresh RAGPipeline from scratch and measures total init time.

    Args:
        config: Optional config dict.

    Returns:
        Dict with init_time_s and component details.
    """
    if config is None:
        config = load_config()

    # Force fresh config to simulate cold start
    reset_config()
    config = load_config()

    start = time.perf_counter()
    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline(config=config)
    elapsed_s = round(time.perf_counter() - start, 3)

    result = {
        "pipeline_init_time_s": elapsed_s,
        "doc_store_count": pipeline.doc_store.count,
        "code_store_count": pipeline.code_store.count,
        "hybrid_mode": pipeline._hybrid_mode,
    }

    logger.info(f"Pipeline init benchmark: {elapsed_s}s")
    return result, pipeline


def benchmark_embedding(
    pipeline,
    queries: List[str],
    warmup: int = 2,
) -> Dict[str, Any]:
    """Measure embedding latency for individual queries.

    Args:
        pipeline: Initialized RAGPipeline instance.
        queries: List of query strings to embed.
        warmup: Number of warmup iterations to discard.

    Returns:
        Dict with embedding latency stats.
    """
    embedder = pipeline.doc_store._embedder
    latencies_ms = []

    # Warmup
    for i in range(min(warmup, len(queries))):
        embedder.embed_query(queries[i])

    # Measure
    for query in queries:
        start = time.perf_counter()
        embedder.embed_query(query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

    stats = _compute_stats(latencies_ms)
    logger.info(f"Embedding benchmark: p50={stats['p50']}ms, p95={stats['p95']}ms over {len(queries)} queries")
    return {"embedding_latency_ms": stats, "num_queries": len(queries)}


def benchmark_retrieval(
    pipeline,
    queries: List[str],
    warmup: int = 2,
) -> Dict[str, Any]:
    """Measure retrieval-only latency (dense + hybrid, no generation).

    Uses the pipeline's internal retrieval components directly.

    Args:
        pipeline: Initialized RAGPipeline instance.
        queries: List of query strings.
        warmup: Number of warmup iterations to discard.

    Returns:
        Dict with retrieval latency stats.
    """
    ret_cfg = pipeline._config.get("retrieval", {})
    top_n = ret_cfg.get("top_n", 5)
    latencies_ms = []

    # Warmup
    for i in range(min(warmup, len(queries))):
        if pipeline._hybrid_retriever:
            pipeline._hybrid_retriever.search(query=queries[i], top_n=top_n)
        else:
            pipeline._dense_search(queries[i], top_n)

    # Measure
    for query in queries:
        start = time.perf_counter()
        if pipeline._hybrid_retriever:
            results = pipeline._hybrid_retriever.search(query=query, top_n=top_n)
        else:
            results = pipeline._dense_search(query, top_n)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

    stats = _compute_stats(latencies_ms)
    logger.info(
        f"Retrieval benchmark: p50={stats['p50']}ms, p95={stats['p95']}ms "
        f"over {len(queries)} queries"
    )
    return {"retrieval_latency_ms": stats, "num_queries": len(queries)}


def benchmark_end_to_end(
    pipeline,
    queries: List[str],
    warmup: int = 2,
    template_name: str = "default_v1",
) -> Dict[str, Any]:
    """Measure full end-to-end pipeline latency.

    Runs pipeline.query() for each query and collects per-query timing.

    Args:
        pipeline: Initialized RAGPipeline instance.
        queries: List of query strings.
        warmup: Number of warmup iterations to discard.
        template_name: Prompt template to use.

    Returns:
        Dict with total, retrieval, and generation latency stats.
    """
    # Disable cache for accurate measurement
    original_cache_enabled = pipeline._query_cache._enabled
    pipeline._query_cache._enabled = False

    total_latencies = []
    retrieval_latencies = []
    generation_latencies = []
    answers = []

    # Warmup
    for i in range(min(warmup, len(queries))):
        pipeline.query(queries[i], template_name=template_name)

    # Measure
    for query in queries:
        start = time.perf_counter()
        result = pipeline.query(query, template_name=template_name)
        total_ms = (time.perf_counter() - start) * 1000
        total_latencies.append(total_ms)
        answers.append(result.get("answer", ""))

        # Extract sub-latencies from the result trace if available
        latency_ms = result.get("latency_ms", 0)
        if latency_ms > 0:
            total_latencies[-1] = latency_ms

    # Restore cache
    pipeline._query_cache._enabled = original_cache_enabled

    stats = {
        "total_latency_ms": _compute_stats(total_latencies),
        "num_queries": len(queries),
        "avg_answer_length": round(
            statistics.mean([len(a) for a in answers]), 1
        ) if answers else 0,
    }
    logger.info(
        f"End-to-end benchmark: p50={stats['total_latency_ms']['p50']}ms, "
        f"p95={stats['total_latency_ms']['p95']}ms over {len(queries)} queries"
    )
    return stats


def benchmark_bm25_rebuild(pipeline) -> Dict[str, Any]:
    """Measure BM25 index rebuild time.

    Args:
        pipeline: Initialized RAGPipeline with hybrid retriever.

    Returns:
        Dict with rebuild_time_ms and document counts.
    """
    if not pipeline._hybrid_retriever:
        return {"bm25_rebuild_time_ms": 0, "skipped": True, "reason": "hybrid mode disabled"}

    doc_count = pipeline.doc_store.count
    code_count = pipeline.code_store.count

    start = time.perf_counter()
    pipeline._hybrid_retriever._bm25_doc.build_from_vector_store(pipeline._doc_store)
    if code_count > 0:
        pipeline._hybrid_retriever._bm25_code.build_from_vector_store(pipeline._code_store)
    elapsed_ms = (time.perf_counter() - start) * 1000

    result = {
        "bm25_rebuild_time_ms": round(elapsed_ms, 2),
        "doc_count": doc_count,
        "code_count": code_count,
    }
    logger.info(f"BM25 rebuild benchmark: {elapsed_ms:.1f}ms for {doc_count + code_count} docs")
    return result


def benchmark_cache_effectiveness(
    pipeline,
    queries: List[str],
) -> Dict[str, Any]:
    """Measure cache hit/miss behavior with repeated queries.

    Runs each query twice: first pass populates cache, second pass should hit cache.

    Args:
        pipeline: Initialized RAGPipeline.
        queries: List of query strings.

    Returns:
        Dict with cache stats and timing comparison.
    """
    # Enable cache
    original_cache_enabled = pipeline._query_cache._enabled
    pipeline._query_cache._enabled = True
    pipeline._query_cache.invalidate()
    pipeline._query_cache._hits = 0
    pipeline._query_cache._misses = 0

    # First pass: cold (cache miss)
    cold_latencies = []
    for query in queries[:5]:  # Use subset to keep benchmark fast
        start = time.perf_counter()
        pipeline.query(query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        cold_latencies.append(elapsed_ms)

    # Second pass: warm (cache hit)
    warm_latencies = []
    for query in queries[:5]:
        start = time.perf_counter()
        pipeline.query(query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        warm_latencies.append(elapsed_ms)

    cache_stats = pipeline._query_cache.stats

    # Restore
    pipeline._query_cache._enabled = original_cache_enabled

    result = {
        "cache_stats": cache_stats,
        "cold_latency_ms": _compute_stats(cold_latencies),
        "warm_latency_ms": _compute_stats(warm_latencies),
        "speedup_ratio": round(
            statistics.mean(cold_latencies) / max(statistics.mean(warm_latencies), 0.01), 2
        ) if cold_latencies and warm_latencies else 0,
    }
    logger.info(
        f"Cache benchmark: hit_rate={cache_stats['hit_rate']}, "
        f"speedup={result['speedup_ratio']}x"
    )
    return result


def measure_memory(pipeline, queries: List[str]) -> Dict[str, Any]:
    """Measure peak memory usage during query execution.

    Args:
        pipeline: Initialized RAGPipeline.
        queries: List of query strings.

    Returns:
        Dict with peak memory in MB.
    """
    tracemalloc.start()

    for query in queries[:5]:
        pipeline.query(query)

    current_mb, peak_mb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "memory_current_mb": round(current_mb / (1024 * 1024), 2),
        "memory_peak_mb": round(peak_mb / (1024 * 1024), 2),
    }
    logger.info(f"Memory benchmark: peak={result['memory_peak_mb']}MB")
    return result


def run_performance_benchmarks(
    queries: List[str],
    bench_config: Dict[str, Any],
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """Run all performance benchmarks and return aggregated results.

    Args:
        queries: List of benchmark query strings.
        bench_config: Benchmark configuration dict.
        config: Optional RAG system config dict.

    Returns:
        Dict with all performance measurement results.
    """
    perf_cfg = bench_config.get("performance", {})
    warmup = perf_cfg.get("warmup", 2)
    iterations = perf_cfg.get("iterations", 10)

    results = {}

    # 1. Pipeline init
    if perf_cfg.get("measure_init", True):
        logger.info("=== Benchmarking pipeline init ===")
        init_result, pipeline = benchmark_pipeline_init(config)
        results["init"] = init_result
    else:
        if config is None:
            config = load_config()
        from src.pipeline import RAGPipeline
        pipeline = RAGPipeline(config=config)
        results["init"] = {"pipeline_init_time_s": 0, "skipped": True}

    # Use a subset of queries up to the configured iteration count
    benchmark_queries = queries[:iterations] if len(queries) > iterations else queries

    # 2. Embedding latency
    logger.info("=== Benchmarking embedding latency ===")
    results["embedding"] = benchmark_embedding(pipeline, benchmark_queries, warmup=warmup)

    # 3. Retrieval latency
    logger.info("=== Benchmarking retrieval latency ===")
    results["retrieval"] = benchmark_retrieval(pipeline, benchmark_queries, warmup=warmup)

    # 4. End-to-end latency
    logger.info("=== Benchmarking end-to-end latency ===")
    results["end_to_end"] = benchmark_end_to_end(pipeline, benchmark_queries, warmup=warmup)

    # 5. BM25 rebuild
    if perf_cfg.get("measure_bm25_rebuild", True):
        logger.info("=== Benchmarking BM25 rebuild ===")
        results["bm25_rebuild"] = benchmark_bm25_rebuild(pipeline)

    # 6. Cache effectiveness
    logger.info("=== Benchmarking cache effectiveness ===")
    results["cache"] = benchmark_cache_effectiveness(pipeline, benchmark_queries)

    # 7. Memory
    if perf_cfg.get("measure_memory", True):
        logger.info("=== Benchmarking memory usage ===")
        results["memory"] = measure_memory(pipeline, benchmark_queries)

    return results, pipeline
