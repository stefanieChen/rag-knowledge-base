# RAG Benchmark & Regression Framework

Automated benchmark suite for measuring and tracking performance and quality of the RAG pipeline across code changes.

## Quick Start

```bash
# Activate venv
source venv/Scripts/activate

# Ensure Ollama is running with required models
ollama pull qwen2.5:7b
ollama pull bge-m3

# 1. Establish a baseline (first run)
python -m benchmarks.run_benchmark --save-baseline

# 2. After making code changes, run benchmarks and compare
python -m benchmarks.run_benchmark

# 3. Quick perf-only check (skips expensive LLM-judge evaluation)
python -m benchmarks.run_benchmark --perf-only

# 4. Quality-only check (skips latency measurement)
python -m benchmarks.run_benchmark --quality-only

# 5. Compare two arbitrary result files
python -m benchmarks.compare benchmarks/baselines/baseline_X.json benchmarks/results/run_Y.json
```

## What Gets Measured

### Performance Metrics
| Metric | Description |
|--------|-------------|
| `pipeline_init_time_s` | Cold-start pipeline initialization |
| `embedding_latency_ms` | Single query embedding time (p50/p95) |
| `retrieval_latency_ms` | Dense + BM25 + RRF retrieval time (p50/p95) |
| `total_latency_ms` | Full end-to-end query time (p50/p95) |
| `bm25_rebuild_time_ms` | BM25 index rebuild from ChromaDB |
| `cache_hit_rate` | Query cache effectiveness |
| `cache_speedup_ratio` | Cache hit vs miss latency ratio |
| `memory_peak_mb` | Peak memory during query execution |

### Quality Metrics
| Metric | Source | Description |
|--------|--------|-------------|
| `retrieval/recall@K` | Custom | Fraction of expected sources found in top-K |
| `retrieval/mrr` | Custom | Mean Reciprocal Rank of first relevant result |
| `ragas/faithfulness` | Ragas | Is the answer grounded in context? |
| `ragas/answer_relevancy` | Ragas | Is the answer relevant to the question? |
| `ragas/context_precision` | Ragas | Are relevant contexts ranked higher? |
| `ragas/context_recall` | Ragas | Does context cover the ground truth? |
| `deepeval/hallucination` | DeepEval | Does the answer contain fabricated info? |
| `deepeval/answer_relevancy` | DeepEval | Is the answer relevant? |
| `deepeval/faithfulness` | DeepEval | Is the answer supported by context? |

## Regression Detection

The comparison engine classifies each metric delta:

| Symbol | Status | Threshold |
|--------|--------|-----------|
| :x: | CRITICAL | Latency >20% slower or Quality >10% drop |
| :warning: | WARNING | Latency >5% slower or Quality >3% drop |
| :white_check_mark: | IMPROVED | Measurable improvement beyond threshold |
| :heavy_minus_sign: | STABLE | Within acceptable range |

Thresholds are configurable in `benchmarks/config.yaml`.

## File Structure

```
benchmarks/
├── config.yaml              # Benchmark settings (iterations, thresholds)
├── test_sets/
│   └── benchmark_qa.json    # 22 diverse Q/A test cases
├── baselines/
│   ├── baseline_*.json      # Saved baselines
│   └── latest.json          # Most recent baseline (auto-updated)
├── results/
│   └── run_*.json           # Per-run results
├── run_benchmark.py         # Main entry point
├── perf_harness.py          # Performance measurement
├── quality_harness.py       # Quality evaluation
├── compare.py               # Baseline comparison engine
└── README.md
```

## Customizing the Test Set

Edit `benchmarks/test_sets/benchmark_qa.json` to add domain-specific questions:

```json
{
    "question": "Your question here",
    "ground_truth": "Expected answer",
    "contexts": ["Relevant context chunk 1"],
    "expected_source_files": ["data/raw/doc.pdf"]
}
```

The `expected_source_files` field enables retrieval Recall@K and MRR computation. Leave it empty if you only want generation quality metrics.
