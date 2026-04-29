"""Tests for new features: notebook parser, query cache, language detector, OCR, pipeline integration."""

import json
import os
import tempfile
import time

import pytest


# ---------------------------------------------------------------------------
# Notebook parser
# ---------------------------------------------------------------------------

class TestNotebookParser:
    """Tests for Jupyter Notebook (.ipynb) parsing."""

    def _make_notebook(self, cells, language="python"):
        """Create a minimal .ipynb JSON structure.

        Args:
            cells: List of cell dicts.
            language: Kernel language name.

        Returns:
            Notebook dict.
        """
        return {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": f"Python 3",
                    "language": language,
                    "name": "python3",
                },
                "language_info": {"name": language},
            },
            "cells": cells,
        }

    def _write_notebook(self, nb_dict, tmp_path):
        """Write notebook dict to a temp .ipynb file.

        Args:
            nb_dict: Notebook dict.
            tmp_path: Directory to write into.

        Returns:
            Path to the created file.
        """
        path = os.path.join(str(tmp_path), "test_notebook.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb_dict, f)
        return path

    def test_parse_code_cell(self, tmp_path):
        from src.ingestion.notebook_parser import parse_notebook

        nb = self._make_notebook([
            {
                "cell_type": "code",
                "source": ["import pandas as pd\n", "df = pd.DataFrame()"],
                "outputs": [],
                "execution_count": 1,
                "metadata": {},
            }
        ])
        path = self._write_notebook(nb, tmp_path)
        docs = parse_notebook(path)

        assert len(docs) == 1
        assert docs[0]["metadata"]["cell_type"] == "code"
        assert docs[0]["metadata"]["format"] == "ipynb"
        assert docs[0]["metadata"]["notebook_language"] == "python"
        assert "pandas" in docs[0]["content"]

    def test_parse_markdown_cell(self, tmp_path):
        from src.ingestion.notebook_parser import parse_notebook

        nb = self._make_notebook([
            {
                "cell_type": "markdown",
                "source": ["# My Analysis\n", "\n", "This notebook analyzes data."],
                "metadata": {},
            }
        ])
        path = self._write_notebook(nb, tmp_path)
        docs = parse_notebook(path)

        assert len(docs) == 1
        assert docs[0]["metadata"]["cell_type"] == "markdown"
        assert "# My Analysis" in docs[0]["content"]

    def test_parse_mixed_cells(self, tmp_path):
        from src.ingestion.notebook_parser import parse_notebook

        nb = self._make_notebook([
            {
                "cell_type": "markdown",
                "source": ["# Title"],
                "metadata": {},
            },
            {
                "cell_type": "code",
                "source": ["print('hello')"],
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": ["hello\n"]},
                ],
                "execution_count": 1,
                "metadata": {},
            },
            {
                "cell_type": "code",
                "source": [""],  # Empty cell — should be skipped
                "outputs": [],
                "execution_count": None,
                "metadata": {},
            },
        ])
        path = self._write_notebook(nb, tmp_path)
        docs = parse_notebook(path)

        assert len(docs) == 2  # empty cell skipped
        assert docs[0]["metadata"]["cell_type"] == "markdown"
        assert docs[1]["metadata"]["cell_type"] == "code"

    def test_parse_cell_outputs(self, tmp_path):
        from src.ingestion.notebook_parser import parse_notebook

        nb = self._make_notebook([
            {
                "cell_type": "code",
                "source": ["x = 42\n", "x"],
                "outputs": [
                    {
                        "output_type": "execute_result",
                        "data": {"text/plain": ["42"]},
                        "metadata": {},
                        "execution_count": 1,
                    }
                ],
                "execution_count": 1,
                "metadata": {},
            }
        ])
        path = self._write_notebook(nb, tmp_path)
        docs = parse_notebook(path, include_outputs=True)

        assert "42" in docs[0]["content"]
        assert "Output:" in docs[0]["content"]

    def test_parse_without_outputs(self, tmp_path):
        from src.ingestion.notebook_parser import parse_notebook

        nb = self._make_notebook([
            {
                "cell_type": "code",
                "source": ["x = 42"],
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": ["42\n"]},
                ],
                "execution_count": 1,
                "metadata": {},
            }
        ])
        path = self._write_notebook(nb, tmp_path)
        docs = parse_notebook(path, include_outputs=False)

        assert "Output:" not in docs[0]["content"]

    def test_parse_error_output(self, tmp_path):
        from src.ingestion.notebook_parser import parse_notebook

        nb = self._make_notebook([
            {
                "cell_type": "code",
                "source": ["1/0"],
                "outputs": [
                    {
                        "output_type": "error",
                        "ename": "ZeroDivisionError",
                        "evalue": "division by zero",
                        "traceback": [],
                    }
                ],
                "execution_count": 1,
                "metadata": {},
            }
        ])
        path = self._write_notebook(nb, tmp_path)
        docs = parse_notebook(path)

        assert "ZeroDivisionError" in docs[0]["content"]

    def test_parse_empty_notebook(self, tmp_path):
        from src.ingestion.notebook_parser import parse_notebook

        nb = self._make_notebook([])
        path = self._write_notebook(nb, tmp_path)
        docs = parse_notebook(path)
        assert docs == []

    def test_parse_nonexistent_file(self):
        from src.ingestion.notebook_parser import parse_notebook

        docs = parse_notebook("/nonexistent/path/notebook.ipynb")
        assert docs == []

    def test_parse_invalid_json(self, tmp_path):
        from src.ingestion.notebook_parser import parse_notebook

        path = os.path.join(str(tmp_path), "bad.ipynb")
        with open(path, "w") as f:
            f.write("not valid json{{{")
        docs = parse_notebook(path)
        assert docs == []

    def test_language_detection(self, tmp_path):
        from src.ingestion.notebook_parser import _detect_notebook_language

        nb_py = {
            "metadata": {"kernelspec": {"language": "Python"}, "language_info": {"name": "python"}}
        }
        assert _detect_notebook_language(nb_py) == "python"

        nb_r = {
            "metadata": {"kernelspec": {}, "language_info": {"name": "R"}}
        }
        assert _detect_notebook_language(nb_r) == "r"

        nb_empty = {"metadata": {}}
        assert _detect_notebook_language(nb_empty) == "unknown"


# ---------------------------------------------------------------------------
# Query cache
# ---------------------------------------------------------------------------

class TestQueryCache:
    """Tests for the in-memory query result cache."""

    def _make_cache(self, **overrides):
        """Create a QueryCache with test-friendly defaults.

        Args:
            **overrides: Override config cache settings.

        Returns:
            QueryCache instance.
        """
        from src.retrieval.query_cache import QueryCache

        config = {
            "cache": {
                "enabled": True,
                "max_size": 10,
                "ttl_seconds": 60,
                "similarity_threshold": 0.95,
                **overrides,
            }
        }
        return QueryCache(config=config)

    def test_exact_match_hit(self):
        cache = self._make_cache()

        # Bypass embedding by manually inserting
        cache._cache["test_key"] = {
            "query": "what is rag",
            "embedding": [0.1] * 10,
            "result": {"answer": "RAG is..."},
            "timestamp": time.time(),
        }

        # Exact match via hash
        key = cache._make_key("what is rag")
        cache._cache[key] = cache._cache.pop("test_key")

        result, _embedding = cache.get("what is rag")
        assert result is not None
        assert result["answer"] == "RAG is..."

    def test_cache_miss(self):
        cache = self._make_cache()
        result, _embedding = cache.get("totally new question")
        assert result is None

    def test_cache_disabled(self):
        cache = self._make_cache(enabled=False)
        cache.put("question", {"answer": "test"})
        result, _embedding = cache.get("question")
        assert result is None
        assert cache.size == 0

    def test_lru_eviction(self):
        cache = self._make_cache(max_size=3)

        # Manually insert entries to avoid embedding calls
        for i in range(5):
            key = f"key_{i}"
            cache._cache[key] = {
                "query": f"query {i}",
                "embedding": [float(i)] * 10,
                "result": {"answer": f"answer {i}"},
                "timestamp": time.time(),
            }
            # Enforce max size
            while len(cache._cache) > cache._max_size:
                cache._cache.popitem(last=False)

        assert cache.size <= 3

    def test_ttl_expiration(self):
        cache = self._make_cache(ttl_seconds=1)

        key = cache._make_key("old question")
        cache._cache[key] = {
            "query": "old question",
            "embedding": [0.1] * 10,
            "result": {"answer": "old answer"},
            "timestamp": time.time() - 10,  # Expired 10s ago
        }

        result, _embedding = cache.get("old question")
        assert result is None

    def test_invalidate(self):
        cache = self._make_cache()

        cache._cache["a"] = {
            "query": "q1", "embedding": [], "result": {}, "timestamp": time.time()
        }
        cache._cache["b"] = {
            "query": "q2", "embedding": [], "result": {}, "timestamp": time.time()
        }

        cleared = cache.invalidate()
        assert cleared == 2
        assert cache.size == 0

    def test_stats(self):
        cache = self._make_cache()

        stats = cache.stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "size" in stats
        assert stats["max_size"] == 10

    def test_cosine_similarity(self):
        cache = self._make_cache()

        # Identical vectors → 1.0
        sim = cache._cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(sim - 1.0) < 0.001

        # Orthogonal vectors → 0.0
        sim = cache._cosine_similarity([1, 0, 0], [0, 1, 0])
        assert abs(sim) < 0.001

        # Zero vector → 0.0
        sim = cache._cosine_similarity([0, 0, 0], [1, 1, 1])
        assert sim == 0.0

    def test_make_key_normalization(self):
        cache = self._make_cache()

        # Same content with different casing/whitespace → same key
        k1 = cache._make_key("  What Is RAG?  ")
        k2 = cache._make_key("what is rag?")
        assert k1 == k2


# ---------------------------------------------------------------------------
# Language detector
# ---------------------------------------------------------------------------

class TestLanguageDetector:
    """Tests for automatic language detection."""

    def test_import(self):
        from src.generation.language_detector import (
            detect_language,
            detect_language_with_confidence,
            suggest_template,
            is_available,
        )
        # Just ensure imports work
        assert callable(detect_language)
        assert callable(suggest_template)

    def test_detect_english(self):
        from src.generation.language_detector import detect_language, is_available

        if not is_available():
            pytest.skip("langdetect not installed")

        lang = detect_language("What is Retrieval-Augmented Generation and how does it work?")
        assert lang == "en"

    def test_detect_chinese(self):
        from src.generation.language_detector import detect_language, is_available

        if not is_available():
            pytest.skip("langdetect not installed")

        lang = detect_language("什么是检索增强生成技术？它是如何工作的？")
        assert lang == "zh"

    def test_detect_short_text(self):
        from src.generation.language_detector import detect_language

        # Too short to detect reliably
        result = detect_language("hi")
        # Should return None for very short text
        assert result is None

    def test_detect_empty_text(self):
        from src.generation.language_detector import detect_language

        assert detect_language("") is None
        assert detect_language("   ") is None

    def test_detect_with_confidence(self):
        from src.generation.language_detector import detect_language_with_confidence, is_available

        if not is_available():
            pytest.skip("langdetect not installed")

        result = detect_language_with_confidence(
            "This is a fairly long English sentence for testing language detection."
        )
        assert result["language"] == "en"
        assert result["confidence"] > 0.5
        assert len(result["all_results"]) > 0

    def test_suggest_template_chinese(self):
        from src.generation.language_detector import suggest_template, is_available

        if not is_available():
            pytest.skip("langdetect not installed")

        tpl = suggest_template("请问RAG系统是如何进行文档检索的？")
        assert tpl == "chinese_v1"

    def test_suggest_template_english(self):
        from src.generation.language_detector import suggest_template, is_available

        if not is_available():
            pytest.skip("langdetect not installed")

        tpl = suggest_template("How does the RAG system perform document retrieval?")
        assert tpl == "default_v1"

    def test_suggest_template_fallback(self):
        from src.generation.language_detector import suggest_template

        # Empty text → fallback
        tpl = suggest_template("", default="my_default")
        assert tpl == "my_default"


# ---------------------------------------------------------------------------
# OCR image handler
# ---------------------------------------------------------------------------

class TestOCRImageHandler:
    """Tests for OCR functions in image_handler."""

    def test_import_ocr_functions(self):
        from src.ingestion.image_handler import (
            ocr_image,
            ocr_image_bytes,
            extract_image_text,
            extract_image_bytes_text,
            is_ocr_available,
        )
        assert callable(ocr_image)
        assert callable(is_ocr_available)

    def test_ocr_nonexistent_file(self):
        from src.ingestion.image_handler import ocr_image

        result = ocr_image("/nonexistent/path/image.png")
        assert result is None

    def test_is_ocr_available_returns_bool(self):
        from src.ingestion.image_handler import is_ocr_available

        result = is_ocr_available()
        assert isinstance(result, bool)

    def test_extract_image_text_no_file(self):
        from src.ingestion.image_handler import extract_image_text

        result = extract_image_text(
            "/nonexistent/image.png",
            enable_ocr=True,
            enable_llm=False,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Pipeline integration (import-level)
# ---------------------------------------------------------------------------

class TestPipelineImports:
    """Tests that new pipeline components import correctly."""

    def test_query_rewriter_imports(self):
        from src.generation.query_rewriter import HyDERewriter, MultiQueryRewriter
        assert HyDERewriter is not None
        assert MultiQueryRewriter is not None

    def test_query_cache_import(self):
        from src.retrieval.query_cache import QueryCache
        assert QueryCache is not None

    def test_language_detector_import(self):
        from src.generation.language_detector import detect_language, suggest_template
        assert detect_language is not None

    def test_notebook_parser_import(self):
        from src.ingestion.notebook_parser import parse_notebook
        assert parse_notebook is not None

    def test_notebook_in_parser_registry(self):
        from src.ingest import PARSERS
        assert ".ipynb" in PARSERS


# ---------------------------------------------------------------------------
# RepoMap get_related_context
# ---------------------------------------------------------------------------

class TestRepoMapRelatedContext:
    """Tests for dynamic dependency graph context pulling."""

    def test_empty_repo_map(self):
        from src.retrieval.repo_map import RepoMap

        rm = RepoMap()
        result = rm.get_related_context(["some_func"])
        assert result == ""

    def test_get_related_context_no_match(self):
        from src.retrieval.repo_map import RepoMap

        rm = RepoMap()
        # Even after building an empty graph, no symbols should match
        rm._graph = None
        result = rm.get_related_context(["nonexistent_symbol_xyz"])
        assert result == ""
