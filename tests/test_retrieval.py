"""Tests for retrieval: BM25, RRF fusion, reranker, hybrid config."""

import pytest


# ---------------------------------------------------------------------------
# BM25 tokenizer
# ---------------------------------------------------------------------------

class TestBM25Tokenizer:
    """Tests for the bilingual BM25 tokenizer."""

    def test_english(self):
        from src.retrieval.bm25_store import _tokenize

        tokens = _tokenize("Hello World 123")
        assert tokens == ["hello", "world", "123"]

    def test_chinese(self):
        from src.retrieval.bm25_store import _tokenize

        tokens = _tokenize("检索增强生成")
        assert len(tokens) == 6

    def test_mixed(self):
        from src.retrieval.bm25_store import _tokenize

        tokens = _tokenize("RAG 检索系统 v2")
        assert "rag" in tokens
        assert "v2" in tokens


# ---------------------------------------------------------------------------
# BM25 store
# ---------------------------------------------------------------------------

class TestBM25Store:
    """Tests for BM25 index build and search."""

    def _sample_docs(self):
        return [
            {"chunk_id": "c1", "content": "RAG combines retrieval with generation for better answers", "metadata": {"file": "a.txt"}},
            {"chunk_id": "c2", "content": "BM25 is a keyword-based ranking algorithm used in search engines", "metadata": {"file": "b.txt"}},
            {"chunk_id": "c3", "content": "Vector embeddings capture semantic meaning of text", "metadata": {"file": "c.txt"}},
            {"chunk_id": "c4", "content": "Hybrid search combines BM25 sparse and dense vector retrieval", "metadata": {"file": "d.txt"}},
            {"chunk_id": "c5", "content": "Cross-encoder rerankers improve retrieval precision", "metadata": {"file": "e.txt"}},
        ]

    def test_build_and_count(self):
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store()
        assert store.document_count == 0
        store.build(self._sample_docs())
        assert store.document_count == 5

    def test_search_relevance(self):
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store()
        store.build(self._sample_docs())
        results = store.search("BM25 keyword search", top_k=3)
        assert len(results) >= 1
        # BM25 should favor docs containing "BM25" keyword
        assert results[0]["chunk_id"] in ("c2", "c4")


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class TestRRF:
    """Tests for Reciprocal Rank Fusion."""

    def test_fusion(self):
        from src.retrieval.hybrid import reciprocal_rank_fusion

        dense = [
            {"chunk_id": "c1", "content": "doc1", "score": 0.9},
            {"chunk_id": "c2", "content": "doc2", "score": 0.8},
            {"chunk_id": "c3", "content": "doc3", "score": 0.7},
        ]
        sparse = [
            {"chunk_id": "c2", "content": "doc2", "score": 5.0},
            {"chunk_id": "c4", "content": "doc4", "score": 3.0},
            {"chunk_id": "c1", "content": "doc1", "score": 2.0},
        ]

        fused = reciprocal_rank_fusion([dense, sparse], k=60)
        chunk_ids = [r["chunk_id"] for r in fused]

        # c2 appears at rank 2 in dense and rank 1 in sparse -> should rank highest
        assert "c2" in chunk_ids[:2]
        assert "c1" in chunk_ids[:2]
        assert len(fused) == 4


# ---------------------------------------------------------------------------
# Reranker import
# ---------------------------------------------------------------------------

class TestReranker:
    """Tests for reranker module import."""

    def test_import(self):
        from src.retrieval.reranker import Reranker, DEFAULT_RERANKER_MODEL

        assert DEFAULT_RERANKER_MODEL is not None


# ---------------------------------------------------------------------------
# Hybrid config
# ---------------------------------------------------------------------------

class TestHybridConfig:
    """Tests for hybrid retrieval configuration loading."""

    def test_config_values(self):
        from src.config import load_config

        config = load_config()
        ret_cfg = config.get("retrieval", {})
        assert ret_cfg.get("hybrid_mode") is True
        assert ret_cfg.get("bm25_top_k") == 20
        assert ret_cfg.get("rrf_k") == 60

        rerank_cfg = config.get("reranker", {})
        assert rerank_cfg.get("model") is not None
        assert rerank_cfg.get("top_n") is not None
