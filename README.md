# Local RAG Knowledge Base System

基于 Ollama 本地大模型 + 多格式文档解析 + 混合检索的端到端 RAG 系统。

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │            Document Ingestion               │
                         │                                             │
                         │  PDF ──→ pymupdf4llm (Markdown per page)    │
                         │  PPTX ──→ python-pptx (text + tables)       │
                         │  MD ──→ section-based parser                │
                         │  TXT ──→ plain text reader                  │
                         │  OneNote ──→ HTML export + BeautifulSoup    │
                         │  Images ──→ llava multimodal LLM + OCR      │
                         │  Tables ──→ Markdown + LLM summary          │
                         │  Code ──→ ASTChunk/LangChain + RepoMap      │
                         │  Notebook ──→ ipynb cell extractor          │
                         │                                             │
                         │       ↓ RecursiveCharacterTextSplitter      │
                         │       ↓ (512 chars, 64 overlap)             │
                         └───────────────┬─────────────────────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────────┐
                         │     Embedding (bge-m3 via Ollama) │
                         │     → ChromaDB (cosine HNSW)      │
                         └───────────────┬───────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                    Query Pipeline                               │
        │                                                                 │
        │  User Question                                                  │
        │       │                                                         │
        │       ├──→ Language Detection (langdetect)                     │
        │       ├──→ Cache Lookup (embedding similarity)                   │
        │       ├──→ (optional) HyDE / Multi-Query rewrite                │
        │       │                                                         │
        │       ├──→ Dense Retrieval (bge-m3 cosine) ──→ top-k            │
        │       │                                                         │
        │       ├──→ Sparse Retrieval (BM25 keyword) ──→ top-k            │
        │       │                                                         │
        │       └──→ Reciprocal Rank Fusion (RRF) ──→ merged ranking      │
        │                        │                                        │
        │                        ▼                                        │
        │            Cross-Encoder Rerank (bge-reranker-v2-m3)            │
        │                        │                                        │
        │                        ▼                                        │
        │               Top-n context chunks                              │
        │                        │                                        │
        │                        ▼                                        │
        │     ChatPromptTemplate (system/human roles) + context           │
        │                        │                                        │
        │                        ▼                                        │
        │           ChatOllama (qwen2.5) → LCEL chain → Answer            │
        │                        │                                        │
        │                        ▼                                        │
        │              Cache Store (LRU + TTL)                             │
        │                                                                 │
        └──────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   RAGTracer (JSON)  │
                    │   Full query trace  │
                    └─────────────────────┘
```

## Features

- **多格式文档支持**: PDF, PowerPoint, OneNote (HTML export), TXT, Markdown, Jupyter Notebook (.ipynb) (含表格/图片/OCR处理)
- **代码知识支持**: 源代码解析与AST分块 (Python/Java/C#/TypeScript等), RepoMap符号关系图+依赖图动态上下文拉取, 代码专用Prompt模板
- **Web UI**: Streamlit 交互界面，支持聊天式问答、文件上传、来源展示
- **混合检索**: Dense (bge-m3) + Sparse (BM25) + RRF Fusion + Cross-Encoder Rerank
- **查询增强**: HyDE/Multi-Query 查询改写 (pipeline集成) + 语言自动检测 + 查询缓存 (LRU+TTL+embedding相似度)
- **图片处理**: 多模态 LLM (llava) 语义描述 + Tesseract OCR 文字提取，可组合使用
- **结构化 Prompt**: ChatPromptTemplate + LCEL chain，支持模板注册与版本管理
- **评测体系**: Ragas + DeepEval 双框架 + 自动测试集生成 (Ragas TestsetGenerator)
- **全链路日志追踪**: 每次查询生成结构化 JSON trace，便于效果回溯
- **本地部署**: Ollama LLM + ChromaDB，无需外部 API

## Quick Start

```bash
# 1. Create venv and install dependencies
python -m venv venv
source venv/Scripts/activate # Windows (git bash)
# source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt

# 2. Ensure Ollama is running with required models
ollama pull qwen2.5:7b
ollama pull bge-m3
ollama pull llava          # for image description

# 3. Place documents in data/raw/

# 4. Ingest documents
python -m src.ingest

# 5a. Run interactive Q&A (CLI)
python -m src.main

# 5b. Or launch Web UI
streamlit run app.py

# Optional: Start monitoring services (Phoenix + MLflow)
python scripts/start_monitoring.py
```

## Usage

### CLI Mode

```bash
python -m src.main
```

Type questions at the prompt. The system retrieves relevant chunks, generates an answer, and displays source files with similarity scores. Type `quit` to exit.

### Web UI (Streamlit)

```bash
streamlit run app.py
```

The sidebar provides:
- **Top-N slider**: Control how many context chunks feed the LLM (1–20)
- **Prompt template selector**: Pick from versioned prompt templates (loaded from `config/prompts/*.yaml`)
- **Source toggle**: Show/hide retrieved sources per answer
- **Document upload**: Drag-and-drop files (PDF, PPTX, TXT, MD, HTM/HTML) to ingest directly into the knowledge base
- **Code repository upload**: Ingest entire code repositories with AST-based chunking and RepoMap generation

### Ingesting Source Code

For code repositories, use the Web UI or direct function call:

**Option 1: Web UI (Recommended)**
```bash
streamlit run app.py
# In the sidebar, use "Code Repository" upload section
```

**Option 2: Direct Function Call**
```bash
python -m src.code_ingest
python src/ingest_code.py /path/to/code/repo
python src/ingest_code.py /path/to/code/repo "my-repo-name"
python src/ingest_code.py https://github.com/user/repo
python src/ingest_code.py https://github.com/user/repo "custom-name"
```

**Code-specific features:**
- **AST-based chunking**: Uses ASTChunk for Python, Java, C#, TypeScript to preserve code structure
- **LangChain fallback**: Language-aware splitting for other supported languages
- **RepoMap generation**: Extracts symbols (classes, functions) and builds relationship graphs with PageRank ranking
- **Incremental indexing**: Tracks file hashes to skip unchanged files
- **Code metadata**: Includes file paths, language types, and AST node information

### Code Knowledge Retrieval

When code files are ingested, use the `code_v1` prompt template for best results:

```yaml
# In config/settings.yaml
generation:
  prompt_template: "code_v1"  # Code-aware prompt with file structure
```

The code template provides:
- File path and hierarchy awareness
- Precise code citation format [Source N: filepath]
- Structure-aware context preservation

### Ingesting Documents

Place files in `data/raw/` and run:

```bash
python -m src.ingest
```

Supported formats: `.txt`, `.md`, `.pdf`, `.pptx`, `.htm`, `.html` (OneNote HTML exports), `.ipynb` (Jupyter Notebooks).

**Note**: Source code files are handled separately via the Web UI or `src\ingest_code.py` script.

### Prompt Version Management

Prompt templates are YAML files in `config/prompts/`. To create a new version:

1. Copy an existing template (e.g., `default_v1.yaml` → `default_v2.yaml`)
2. Edit the `version`, `system_prompt`, and `description` fields
3. The new version appears automatically in both the CLI and Web UI

Programmatically:

```python
from src.generation.prompt_version_manager import PromptVersionManager
mgr = PromptVersionManager()
mgr.create_version("default_v1", system_prompt="...", description="concise variant")
```

### Switching Retrieval Mode

Edit `config/settings.yaml`:

```yaml
retrieval:
  hybrid_mode: true    # true = Dense + BM25 + RRF + Rerank; false = Dense only
  enable_reranker: true
```

## Evaluation & Benchmarking (Unified CLI)

Quality evaluation, performance benchmarks, and baseline comparison are consolidated into a single entry point:

```bash
# Quality evaluation only (Ragas + DeepEval)
python -m evaluation.run --suite quality

# Performance benchmarks only (latency, memory, cache)
python -m evaluation.run --suite perf

# Full suite (quality + performance)
python -m evaluation.run --suite full

# Save results as a new baseline
python -m evaluation.run --suite full --save-baseline

# Specific framework only
python -m evaluation.run --suite quality --framework ragas

# Static evaluation (uses test set contexts, no live pipeline)
python -m evaluation.run --suite quality --mode static
```

The original standalone runners are still available:

```bash
# Standalone evaluation runner
python -m evaluation.run_evaluation --mode static
python -m evaluation.run_evaluation --framework ragas

# Standalone benchmark runner
python -m benchmarks.run_benchmark --save-baseline
python -m benchmarks.run_benchmark --perf-only
```

Results are saved as JSON in `evaluation/results/`.

### Metrics

**Ragas** (via Ollama as LLM judge):
- **Faithfulness**: Is the answer grounded in the retrieved context?
- **Answer Relevancy**: Is the answer relevant to the question?
- **Context Precision**: Are relevant contexts ranked higher?
- **Context Recall**: Does the context cover the ground truth?

**DeepEval** (via custom Ollama wrapper):
- **Hallucination**: Does the answer contain fabricated information?
- **Answer Relevancy**: Is the answer relevant to the question?
- **Faithfulness**: Is the answer supported by the context?

### Test Dataset

Edit `evaluation/test_set/sample_test_set.json` to add your own Q/A/context triples:

```json
[
  {
    "question": "Your question here",
    "ground_truth": "Expected answer",
    "contexts": ["Relevant context chunk 1", "Relevant context chunk 2"]
  }
]
```

## Testing

### Running All Tests

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run a specific test module
python -m pytest tests/test_ingestion.py -v

# Run a specific test function
python -m pytest tests/test_retrieval.py::test_bm25_store -v
```

### Test Structure

```
tests/
├── test_ingestion.py      # Parsers (TXT, MD, PDF, PPTX, OneNote), chunker, ingest registry
├── test_retrieval.py      # BM25, RRF fusion, reranker import, hybrid config
├── test_generation.py     # Prompt templates, prompt versioning, query rewriters
├── test_evaluation.py     # Ragas + DeepEval metric creation, test set loading
├── test_pipeline.py       # RAG tracer, pipeline config
└── test_new_features.py   # Notebook parser, query cache, language detector, OCR, pipeline integration
```

### After Code Changes

Always run the test suite before committing:

```bash
python -m pytest tests/ -v
```

If you modified a specific component, run only the relevant module for faster feedback (see above). All tests are designed to run without Ollama or a live LLM — they verify imports, data structures, and algorithmic logic only.

## Regression Detection

The benchmark framework measures pipeline performance and quality against live Ollama, establishes baselines, and detects regressions after code changes.

> **Tip**: Use the unified CLI (`python -m evaluation.run --suite full`) or the standalone runner below.

```bash
# Compare two arbitrary result files
python -m benchmarks.compare benchmarks/baselines/baseline_X.json benchmarks/results/run_Y.json
```

### What Gets Measured

**Performance** (per-query, against live Ollama):
- `pipeline_init_time_s` — cold-start initialization
- `embedding_latency_ms` — single query embedding (p50/p95)
- `retrieval_latency_ms` — dense + BM25 + RRF retrieval (p50/p95)
- `total_latency_ms` — full end-to-end pipeline (p50/p95)
- `bm25_rebuild_time_ms` — BM25 index rebuild
- `cache_hit_rate`, `cache_speedup_ratio` — query cache effectiveness
- `memory_peak_mb` — peak process memory

**Quality** (aggregated over 22 test cases):
- Ragas: faithfulness, answer_relevancy, context_precision, context_recall
- DeepEval: hallucination, answer_relevancy, faithfulness
- Retrieval: Recall@5, Recall@10, MRR

### Regression Thresholds

| Symbol | Status | Threshold |
|--------|--------|----------|
| :x: | Critical regression | Latency >20% slower or Quality >10% drop |
| :warning: | Warning | Latency >5% slower or Quality >3% drop |
| :white_check_mark: | Improved | Measurable improvement beyond threshold |
| :heavy_minus_sign: | Stable | Within acceptable range |

Thresholds are configurable in `benchmarks/config.yaml`. Exit code 1 is returned when critical regressions are detected (CI-friendly).

Baseline results are saved in `benchmarks/baselines/` and run results in `benchmarks/results/`. See `benchmarks/README.md` for full details.

## Project Structure

```
rag_2/
├── config/
│   ├── settings.yaml            # All configuration (LLM, embedding, retrieval, etc.)
│   └── prompts/                 # Versioned prompt templates (YAML)
├── src/
│   ├── logging/
│   │   ├── logger.py            # Unified logging
│   │   └── rag_tracer.py        # RAG full-chain tracer
│   ├── ingestion/
│   │   ├── pdf_parser.py        # PDF → Markdown (pymupdf4llm)
│   │   ├── pptx_parser.py       # PPTX → text + tables (python-pptx)
│   │   ├── markdown_parser.py   # Markdown section parser
│   │   ├── txt_parser.py        # Plain text reader
│   │   ├── onenote_parser.py    # OneNote HTML export parser
│   │   ├── code_parser.py       # Source code parser with language detection
│   │   ├── image_handler.py     # Image description (llava) + OCR (Tesseract)
│   │   ├── table_handler.py     # Table → Markdown + LLM summary
│   │   ├── notebook_parser.py   # Jupyter Notebook (.ipynb) parser
│   │   ├── chunker.py           # RecursiveCharacterTextSplitter
│   │   └── code_chunker.py      # AST-based code chunking with LangChain fallback
│   ├── embedding/
│   │   └── embedder.py          # bge-m3 embedding via Ollama
│   ├── retrieval/
│   │   ├── vector_store.py      # ChromaDB operations
│   │   ├── bm25_store.py        # BM25 sparse index (rank-bm25)
│   │   ├── hybrid.py            # Hybrid retrieval + RRF fusion
│   │   ├── reranker.py          # Cross-Encoder reranker (bge-reranker-v2-m3)
│   │   ├── repo_map.py          # Repository symbol mapping with PageRank + dependency context
│   │   └── query_cache.py       # LRU query cache with embedding similarity
│   ├── generation/
│   │   ├── prompt_templates.py  # ChatPromptTemplate registry
│   │   ├── query_rewriter.py    # HyDE + Multi-Query rewriting
│   │   ├── language_detector.py # Auto language detection (langdetect)
│   │   └── generator.py         # ChatOllama + LCEL chain
│   ├── ingest.py                # Document ingestion script
│   ├── pipeline.py              # End-to-end RAG pipeline
│   └── main.py                  # Interactive Q&A entry point
├── evaluation/                  # RAG evaluation scripts (Phase 4)
│   ├── run.py                   # **Unified CLI** (quality / perf / full)
│   ├── evaluate_ragas.py        # Ragas metrics evaluation
│   ├── evaluate_deepeval.py     # DeepEval metrics evaluation
│   ├── run_evaluation.py        # Standalone evaluation runner
│   ├── generate_test_set.py     # Auto test set generation (Ragas TestsetGenerator)
│   ├── test_set/                # Test datasets (JSON)
│   └── results/                 # Evaluation & benchmark result files
├── benchmarks/                  # Performance & quality benchmarking (Phase 9)
│   ├── config.yaml              # Benchmark settings (iterations, warmup, thresholds)
│   ├── perf_harness.py          # Latency, memory, cache benchmarks
│   ├── quality_harness.py       # Retrieval recall (delegates to evaluation/ for Ragas/DeepEval)
│   ├── compare.py               # Baseline vs current comparison + diff report
│   ├── run_benchmark.py         # Standalone benchmark runner
│   ├── test_sets/               # Benchmark Q/A test cases (JSON)
│   ├── baselines/               # Saved baseline snapshots
│   └── results/                 # Per-run result files
├── .streamlit/config.toml       # Streamlit server config (timeouts, WebSocket)
├── app.py                       # Streamlit Web UI (cached pipeline init)
├── logs/rag_traces/             # RAG trace JSON files
├── data/raw/                    # Place your documents here
├── requirements.txt
└── README.md
```

## Configuration

Edit `config/settings.yaml` to adjust:
- **LLM**: model, temperature, context window
- **Embedding**: model, batch size
- **Chunking**: chunk size, overlap, separators
- **Retrieval**: top_k, top_n, similarity threshold, hybrid mode, RRF constant
- **Query Rewriting**: enable/disable, strategy (hyde/multi_query/none)
- **Cache**: enable/disable, max_size, TTL, similarity threshold
- **Reranker**: model, top_n (lazy-loaded on first query for faster startup)
- **Logging**: level, trace directory, retention
- **Evaluation**: test set path, output directory, frameworks

Streamlit server settings are in `.streamlit/config.toml` (WebSocket message size, compression).

## Monitoring & Observability

The system supports two local, free observability tools:

### Arize Phoenix (RAG Trace Visualization)

Visualize full RAG pipeline traces (retrieval → rerank → generation) in a local web UI.

```bash
# Install
pip install arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-langchain

# Enable in config/settings.yaml
monitoring.phoenix.enabled: true

# Start Phoenix UI
python scripts/start_monitoring.py --phoenix
# Open http://localhost:6006
```

### MLflow (Evaluation Experiment Tracking)

Track evaluation metrics (Ragas/DeepEval scores) across runs with parameter snapshots.

```bash
# Install
pip install mlflow

# Enable in config/settings.yaml
# monitoring.mlflow.enabled: true

# Start MLflow UI
python scripts/start_monitoring.py --mlflow
# Open http://localhost:5000
```

### Start Both Services

```bash
python scripts/start_monitoring.py
```

## Implementation Roadmap

| Phase | Content | Status |
|-------|---------|--------|
| **Phase 1** | Basic pipeline: TXT + MD → Embedding → ChromaDB → Ollama Q&A | ✅ Done |
| **Phase 1.5** | Prompt upgrade: ChatPromptTemplate + LCEL chains | ✅ Done |
| **Phase 2** | PDF + PPTX parsing, image description, table handling | ✅ Done |
| **Phase 3** | Hybrid retrieval (BM25 + RRF + Rerank) + query rewriting | ✅ Done |
| **Phase 4** | Evaluation (Ragas + DeepEval) + DSPy prompt optimization | ✅ Done |
| **Phase 5** | OneNote support + Streamlit Web UI + Local Prompt Versioning | ✅ Done |
| **Phase 6** | Code knowledge ingestion + RepoMap | ✅ Done |
| **Phase 7** | Observability (Phoenix tracing + MLflow evaluation tracking) | ✅ Done |
| **Phase 8** | Query rewriter pipeline integration, OCR, query cache, language detection, notebook parsing, dependency graph RAG, auto test set generation | ✅ Done |
| **Phase 9** | Benchmark & regression framework: perf/quality harness, comparison engine, critical bug fixes (cache key, cosine similarity, evaluation context) | ✅ Done |
| **Phase 10** | Bug fixes & consolidation: fix answer not displaying on Web UI (latency_ms=0, session_state ordering), `@st.cache_resource` for pipeline init, lazy reranker loading, per-step retrieval timing, unified evaluation/benchmark CLI (`evaluation/run.py`), deduplicated quality_harness | ✅ Done |
