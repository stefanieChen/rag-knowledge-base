"""Tests for pipeline: RAG tracer and pipeline configuration."""

import pytest


# ---------------------------------------------------------------------------
# RAG tracer
# ---------------------------------------------------------------------------

class TestRAGTracer:
    """Tests for RAG trace logging."""

    def test_tracer_context_manager(self):
        from src.logging.rag_tracer import RAGTracer

        with RAGTracer() as tracer:
            tracer.log_query("What is RAG?", language="en")
            tracer.log_retrieval_simple(
                results=[
                    {"chunk_id": "abc123", "score": 0.85, "source": "sample.txt"},
                    {"chunk_id": "def456", "score": 0.72, "source": "sample_zh.md"},
                ],
                retrieval_latency_ms=50,
            )
            tracer.log_generation(
                model="test",
                prompt_template="default_v1",
                context_token_count=200,
                answer="RAG is Retrieval-Augmented Generation.",
                generation_latency_ms=100,
            )
            assert tracer.trace_id is not None
            assert len(tracer.trace_id) > 0


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    """Tests for pipeline-level configuration loading."""

    def test_retrieval_config(self):
        from src.config import load_config

        config = load_config()
        ret_cfg = config.get("retrieval", {})
        assert ret_cfg.get("top_k") is not None
        assert ret_cfg.get("top_n") is not None
        assert ret_cfg.get("similarity_threshold") is not None

    def test_llm_config(self):
        from src.config import load_config

        config = load_config()
        llm_cfg = config.get("llm", {})
        assert llm_cfg.get("model") is not None
        assert llm_cfg.get("base_url") is not None
        assert llm_cfg.get("temperature") is not None

    def test_embedding_config(self):
        from src.config import load_config

        config = load_config()
        emb_cfg = config.get("embedding", {})
        assert emb_cfg.get("model") is not None

    def test_streamlit_import(self):
        import streamlit
        assert streamlit.__version__ is not None
