"""End-to-end RAG pipeline orchestrating retrieval and generation.

Supports two retrieval modes:
- **Dense-only** (Phase 1): VectorStore cosine similarity search
- **Hybrid** (Phase 3): Dense + BM25 sparse → RRF fusion → cross-encoder rerank
"""

import time
from typing import Dict, List, Optional

from src.config import load_config
from src.generation.generator import Generator
from src.generation.language_detector import detect_language
from src.generation.query_rewriter import HyDERewriter, MultiQueryRewriter
from src.logging.logger import get_logger
from src.logging.rag_tracer import RAGTracer
from src.monitoring.phoenix_tracer import init_phoenix_tracing
from src.retrieval.context_compressor import ContextCompressor
from src.retrieval.query_cache import QueryCache
from src.retrieval.repo_map import RepoMap
from src.retrieval.vector_store import VectorStore

logger = get_logger("pipeline")


class RAGPipeline:
    """Orchestrates the full RAG pipeline: query → retrieve → generate.

    Each query is traced via RAGTracer for retrospection.
    When hybrid_mode is enabled, uses HybridRetriever (dense + BM25 + RRF + rerank).
    Supports separate document and code collections with search scope filtering.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()

        self._config = config

        # Separate vector stores for documents and code
        vs_cfg = config.get("vector_store", {})
        code_collection = vs_cfg.get("code_collection_name", "rag_code_base")

        self._doc_store = VectorStore(config=config)
        self._code_store = VectorStore(
            config=config,
            collection_name=code_collection,
            embedder=self._doc_store._embedder,  # Share embedder instance
        )
        self._generator = Generator(config=config)
        self._context_compressor = ContextCompressor(config=config)

        ret_cfg = config.get("retrieval", {})
        self._top_n = ret_cfg.get("top_n", 5)
        self._hybrid_mode = ret_cfg.get("hybrid_mode", False)

        # Initialize hybrid retriever if enabled
        self._hybrid_retriever = None
        if self._hybrid_mode:
            try:
                from src.retrieval.hybrid import HybridRetriever
                enable_reranker = ret_cfg.get("enable_reranker", True)
                self._hybrid_retriever = HybridRetriever(
                    vector_store=self._doc_store,
                    code_store=self._code_store,
                    config=config,
                    enable_reranker=enable_reranker,
                )
                logger.info("Pipeline using HYBRID retrieval (dense + BM25 + RRF + rerank)")
            except Exception as e:
                logger.warning(
                    f"Hybrid retriever init failed, falling back to dense-only: {e}"
                )
                self._hybrid_mode = False

        if not self._hybrid_mode:
            logger.info("Pipeline using DENSE-ONLY retrieval")

        # Initialize query rewriter if enabled
        self._query_rewriter = None
        self._query_rewrite_strategy = "none"
        qr_cfg = config.get("query_rewriting", {})
        if qr_cfg.get("enabled", False):
            strategy = qr_cfg.get("strategy", "none")
            try:
                if strategy == "hyde":
                    self._query_rewriter = HyDERewriter(config=config)
                    self._query_rewrite_strategy = "hyde"
                    logger.info("Query rewriting enabled: HyDE")
                elif strategy == "multi_query":
                    self._query_rewriter = MultiQueryRewriter(config=config)
                    self._query_rewrite_strategy = "multi_query"
                    logger.info("Query rewriting enabled: MultiQuery")
            except Exception as e:
                logger.warning(f"Query rewriter init failed, continuing without: {e}")

        # Initialize query cache
        self._query_cache = QueryCache(config=config)

        # Initialize Phoenix tracing if enabled
        init_phoenix_tracing(config)

        # RepoMap for code navigation context
        self._repo_map: Optional[RepoMap] = None
        self._repo_map_text: str = ""

        logger.info(
            f"RAG pipeline initialized: "
            f"docs={self._doc_store.count}, code={self._code_store.count}"
        )

    def query(
        self,
        question: str,
        top_n: Optional[int] = None,
        template_name: str = "default_v1",
        search_scope: str = "all",
    ) -> Dict:
        """Run a full RAG query: retrieve context, generate answer.

        Args:
            question: User's natural language question.
            top_n: Number of top context chunks to feed to LLM.
                   Defaults to config retrieval.top_n.
            template_name: Prompt template name to use.
            search_scope: Search scope filter — "all", "docs", or "code".

        Returns:
            Dict with keys: answer, sources, trace_id, latency_ms, retrieval_mode.
        """
        if top_n is None:
            top_n = self._top_n

        # Check cache before running full pipeline
        cache_key_params = f"{question}|{top_n}|{template_name}|{search_scope}"
        cached, cache_embedding = self._query_cache.get(cache_key_params)
        if cached is not None:
            logger.info(f"Returning cached result for: '{question[:50]}...'")
            return cached

        pipeline_start = time.perf_counter()

        with RAGTracer(self._config) as tracer:
            # 1. Query rewriting
            rewrite_start = time.perf_counter()
            rewritten_query = None
            search_queries = [question]

            if self._query_rewriter:
                try:
                    if self._query_rewrite_strategy == "hyde":
                        rewritten_query = self._query_rewriter.rewrite(question)
                        search_queries = [rewritten_query]
                    elif self._query_rewrite_strategy == "multi_query":
                        search_queries = self._query_rewriter.rewrite(question)
                        rewritten_query = " | ".join(search_queries)
                except Exception as e:
                    logger.warning(f"Query rewrite failed, using original: {e}")
                    search_queries = [question]

            rewrite_ms = int((time.perf_counter() - rewrite_start) * 1000)
            logger.info(
                f"Query rewrite ({self._query_rewrite_strategy}): "
                f"{len(search_queries)} queries ({rewrite_ms}ms)"
            )

            # Detect query language
            query_language = detect_language(question)

            tracer.log_query(
                raw_query=question,
                rewritten_query=rewritten_query,
                language=query_language,
            )

            # 2. Retrieve (multi-query: run each query, deduplicate by chunk_id)
            retrieval_start = time.perf_counter()

            if len(search_queries) > 1:
                # Multi-query: merge results from all queries
                retrieved_chunks = self._multi_query_search(
                    search_queries, top_n, search_scope
                )
            elif self._hybrid_retriever:
                # Hybrid: dense + BM25 → RRF → rerank
                retrieved_chunks = self._hybrid_retriever.search(
                    query=search_queries[0],
                    top_n=top_n,
                    search_scope=search_scope,
                )
            else:
                # Dense-only fallback with scope filtering
                retrieved_chunks = self._dense_search(
                    search_queries[0], top_n, search_scope
                )

            context_chunks = self._context_compressor.compress(question, retrieved_chunks)
            if len(context_chunks) > top_n:
                context_chunks = context_chunks[:top_n]

            retrieval_ms = int(
                (time.perf_counter() - retrieval_start) * 1000
            )

            tracer.log_retrieval_simple(
                results=[
                    {
                        "chunk_id": r["chunk_id"],
                        "score": r.get("rerank_score", r.get("rrf_score", r.get("score", 0))),
                        "source": r.get("metadata", {}).get("file_name", ""),
                    }
                    for r in retrieved_chunks
                ],
                retrieval_latency_ms=retrieval_ms,
            )

            # 3. Generate
            if not context_chunks:
                answer_text = (
                    "No relevant documents found for your question. "
                    "Please try rephrasing or check if the knowledge base "
                    "contains related content."
                )
                tracer.log_generation(
                    model=self._generator.model_name,
                    prompt_template=template_name,
                    context_token_count=0,
                    answer=answer_text,
                    generation_latency_ms=0,
                )
            else:
                # Inject repo map + dynamic dependency context for code searches
                rmap = None
                if search_scope in ("all", "code") and self._repo_map_text:
                    rmap = self._repo_map_text

                    # Dynamic context pulling from dependency graph
                    if self._repo_map:
                        dep_context = self._get_dependency_context(context_chunks)
                        if dep_context:
                            rmap = f"{rmap}\n\n{dep_context}"

                gen_result = self._generator.generate(
                    query=question,
                    context_chunks=context_chunks,
                    template_name=template_name,
                    repo_map_text=rmap,
                )
                answer_text = gen_result["answer"]
                tracer.log_generation(
                    model=gen_result["model"],
                    prompt_template=gen_result["prompt_template"],
                    context_token_count=gen_result["context_token_count"],
                    answer=answer_text,
                    generation_latency_ms=gen_result["generation_latency_ms"],
                )

            # 4. Build response
            sources = []
            for chunk in context_chunks:
                meta = chunk.get("metadata", {})
                sources.append({
                    "file": meta.get("file_name", "unknown"),
                    "content_type": meta.get("content_type", "unknown"),
                    "score": chunk.get("rerank_score",
                                       chunk.get("rrf_score",
                                                  chunk.get("score", 0))),
                    "content_preview": chunk.get("content", "")[:200],
                    "content_full": chunk.get("content", ""),
                })

            total_latency_ms = int(
                (time.perf_counter() - pipeline_start) * 1000
            )

            result = {
                "answer": answer_text,
                "sources": sources,
                "trace_id": tracer.trace_id,
                "latency_ms": total_latency_ms,
                "retrieval_mode": "hybrid" if self._hybrid_retriever else "dense",
            }

            # Store in cache (reuse embedding from get() to avoid redundant API call)
            self._query_cache.put(
                cache_key_params, result, query_embedding=cache_embedding
            )

            return result

    def _multi_query_search(
        self,
        queries: List[str],
        top_n: int,
        search_scope: str = "all",
    ) -> List[Dict]:
        """Run multiple queries and merge results using RRF.

        Each query is run through either the hybrid or dense retriever,
        and results are fused via Reciprocal Rank Fusion for deduplication.

        Args:
            queries: List of query strings (original + rewrites).
            top_n: Number of final results to return.
            search_scope: "all", "docs", or "code".

        Returns:
            Merged and deduplicated list of result dicts.
        """
        from src.retrieval.hybrid import reciprocal_rank_fusion

        all_result_lists = []
        for q in queries:
            if self._hybrid_retriever:
                results = self._hybrid_retriever.search(
                    query=q, top_n=top_n, search_scope=search_scope,
                )
            else:
                results = self._dense_search(q, top_n, search_scope)
            if results:
                all_result_lists.append(results)

        if not all_result_lists:
            return []

        if len(all_result_lists) == 1:
            return all_result_lists[0]

        rrf_k = self._config.get("retrieval", {}).get("rrf_k", 60)
        fused = reciprocal_rank_fusion(all_result_lists, k=rrf_k)

        logger.info(
            f"Multi-query fusion: {len(queries)} queries → "
            f"{len(fused)} unique chunks (returning top {top_n})"
        )
        return fused[:top_n]

    def _get_dependency_context(self, context_chunks: List[Dict]) -> str:
        """Extract identifiers from code chunks and pull related definitions.

        Scans retrieved code chunks for identifier-like tokens, then uses
        the RepoMap dependency graph to find and return source code of
        referenced definitions.

        Args:
            context_chunks: Retrieved code chunks with 'content' and 'metadata'.

        Returns:
            Formatted string of related source code, or "" if none found.
        """
        if not self._repo_map:
            return ""

        import re

        # Only process code-type chunks
        identifiers = set()
        for chunk in context_chunks:
            meta = chunk.get("metadata", {})
            if meta.get("content_type") not in ("code", "function", "class", "method"):
                continue

            content = chunk.get("content", "")
            # Extract potential identifiers: words with 2+ chars that look like
            # function/class names (starts with letter or underscore)
            tokens = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]{1,}\b', content)
            identifiers.update(tokens)

        if not identifiers:
            return ""

        # Filter out common language keywords to reduce noise
        _COMMON_KEYWORDS = {
            "def", "class", "return", "import", "from", "self", "None",
            "True", "False", "if", "else", "elif", "for", "while", "try",
            "except", "finally", "with", "as", "in", "not", "and", "or",
            "is", "lambda", "yield", "raise", "pass", "break", "continue",
            "global", "nonlocal", "assert", "del", "print", "range", "len",
            "str", "int", "float", "bool", "list", "dict", "set", "tuple",
            "type", "super", "isinstance", "hasattr", "getattr", "setattr",
            "public", "private", "protected", "static", "void", "string",
            "var", "let", "const", "function", "new", "this", "null",
            "undefined", "async", "await", "export", "default",
        }
        identifiers -= _COMMON_KEYWORDS

        return self._repo_map.get_related_context(
            symbol_names=list(identifiers),
            max_symbols=8,
            max_chars=2000,
        )

    def _dense_search(
        self,
        query: str,
        top_n: int,
        search_scope: str = "all",
    ) -> List[Dict]:
        """Dense-only search across doc/code stores based on scope.

        Args:
            query: User query string.
            top_n: Number of results to return.
            search_scope: "all", "docs", or "code".

        Returns:
            Sorted list of result dicts.
        """
        top_k = self._config.get("retrieval", {}).get("top_k", 20)
        results = []

        if search_scope in ("all", "docs"):
            results.extend(self._doc_store.search(query=query, top_k=top_k))
        if search_scope in ("all", "code"):
            results.extend(self._code_store.search(query=query, top_k=top_k))

        # Sort by score descending and take top_n
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:top_n]

    @property
    def vector_store(self) -> VectorStore:
        """Access the document vector store (backward compatible)."""
        return self._doc_store

    @property
    def code_store(self) -> VectorStore:
        """Access the code vector store."""
        return self._code_store

    @property
    def doc_store(self) -> VectorStore:
        """Access the document vector store."""
        return self._doc_store

    def rebuild_bm25_index(self) -> None:
        """Rebuild BM25 index after new documents are ingested."""
        if self._hybrid_retriever:
            self._hybrid_retriever.rebuild_bm25_index()
            logger.info("BM25 index rebuilt after ingestion")
        # Invalidate cache since knowledge base changed
        self._query_cache.invalidate()

    def invalidate_cache(self) -> int:
        """Invalidate the query cache. Call after ingesting new documents.

        Returns:
            Number of cache entries cleared.
        """
        return self._query_cache.invalidate()

    @property
    def cache_stats(self) -> Dict:
        """Return query cache statistics."""
        return self._query_cache.stats

    def build_repo_map(
        self,
        files: Optional[List[Dict[str, str]]] = None,
        max_chars: int = 3000,
    ) -> str:
        """Build or rebuild the RepoMap from code files.

        Args:
            files: List of dicts with absolute_path, relative_path, language.
                   If None, attempts to reconstruct from code store metadata.
            max_chars: Max character budget for the generated map text.

        Returns:
            The generated repo map text.
        """
        start = time.perf_counter()

        self._repo_map = RepoMap()

        if files is None:
            files = self._reconstruct_file_list_from_store()

        if not files:
            logger.info("No code files available for repo map")
            self._repo_map_text = ""
            return ""

        self._repo_map.add_files(files)
        self._repo_map.build_graph()
        self._repo_map.compute_pagerank()
        self._repo_map_text = self._repo_map.generate_map(max_chars=max_chars)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"RepoMap built: {self._repo_map.file_count} files, "
            f"{self._repo_map.definition_count} defs, "
            f"map_chars={len(self._repo_map_text)} ({elapsed_ms}ms)"
        )

        return self._repo_map_text

    def _reconstruct_file_list_from_store(self) -> List[Dict[str, str]]:
        """Reconstruct file list from code store metadata for RepoMap.

        Returns:
            List of file info dicts with absolute_path, relative_path, language.
        """
        from src.ingestion.code_parser import detect_language

        file_hashes = self._code_store.get_file_hashes()
        files = []
        seen = set()

        for file_path in file_hashes:
            if file_path in seen:
                continue
            seen.add(file_path)

            lang_info = detect_language(file_path)
            if lang_info["language"] != "unknown":
                files.append({
                    "absolute_path": file_path,
                    "relative_path": file_path,
                    "language": lang_info["language"],
                })

        return files

    @property
    def repo_map(self) -> Optional[RepoMap]:
        """Access the current RepoMap instance."""
        return self._repo_map

    @property
    def repo_map_text(self) -> str:
        """Access the current repo map text."""
        return self._repo_map_text
