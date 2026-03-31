"""Tests for generation: prompt templates, prompt versioning, query rewriters."""

import os
import shutil
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

class TestPromptTemplates:
    """Tests for ChatPromptTemplate registry and formatting."""

    def test_get_default_template(self):
        from src.generation.prompt_templates import get_template

        template = get_template("default_v1")
        assert "ChatPromptTemplate" in type(template).__name__

    def test_get_chinese_template(self):
        from src.generation.prompt_templates import get_template

        template = get_template("chinese_v1")
        assert "ChatPromptTemplate" in type(template).__name__

    def test_template_input_variables(self):
        from src.generation.prompt_templates import get_template

        template = get_template("default_v1")
        assert "context" in template.input_variables
        assert "question" in template.input_variables

    def test_format_messages(self):
        from src.generation.prompt_templates import get_template

        template = get_template("default_v1")
        messages = template.format_messages(context="test context", question="What is RAG?")
        assert len(messages) == 3  # system + context + question

    def test_format_context(self):
        from src.generation.prompt_templates import format_context

        chunks = [
            {"content": "RAG is great.", "metadata": {"file_name": "a.txt"}},
            {"content": "LLMs are powerful.", "metadata": {"file_name": "b.txt"}},
        ]
        result = format_context(chunks)
        assert "[Source 1: a.txt]" in result
        assert "[Source 2: b.txt]" in result

    def test_register_template(self):
        from langchain_core.prompts import ChatPromptTemplate
        from src.generation.prompt_templates import get_template, register_template

        custom = ChatPromptTemplate.from_messages([
            ("system", "You are a test bot."),
            ("human", "{context}\n{question}"),
        ])
        register_template("_test_custom_v1", custom)
        assert get_template("_test_custom_v1") is custom

    def test_nonexistent_template(self):
        from src.generation.prompt_templates import get_template

        with pytest.raises(KeyError):
            get_template("nonexistent_v99")

    def test_list_available_templates(self):
        from src.generation.prompt_templates import list_available_templates

        available = list_available_templates()
        keys = [t["key"] for t in available]
        assert "default_v1" in keys
        assert "chinese_v1" in keys


# ---------------------------------------------------------------------------
# Prompt version manager
# ---------------------------------------------------------------------------

class TestPromptVersionManager:
    """Tests for YAML-based prompt version management."""

    def _make_temp_manager(self):
        """Create a PromptVersionManager with a temp directory copy of prompts."""
        from src.generation.prompt_version_manager import (
            DEFAULT_PROMPTS_DIR,
            PromptVersionManager,
        )

        tmpdir = tempfile.mkdtemp()
        for f in os.listdir(DEFAULT_PROMPTS_DIR):
            if f.endswith((".yaml", ".yml")):
                shutil.copy2(
                    os.path.join(DEFAULT_PROMPTS_DIR, f),
                    os.path.join(tmpdir, f),
                )
        return PromptVersionManager(prompts_dir=tmpdir), tmpdir

    def test_load_templates(self):
        from src.generation.prompt_version_manager import PromptVersionManager

        mgr = PromptVersionManager()
        keys = mgr.get_keys(active_only=False)
        assert "default_v1" in keys
        assert "chinese_v1" in keys

    def test_list_templates_metadata(self):
        from src.generation.prompt_version_manager import PromptVersionManager

        mgr = PromptVersionManager()
        templates = mgr.list_templates()
        assert len(templates) >= 2
        for t in templates:
            assert "key" in t
            assert "name" in t
            assert "version" in t

    def test_build_chat_prompt(self):
        from src.generation.prompt_version_manager import PromptVersionManager

        mgr = PromptVersionManager()
        prompt = mgr.build_chat_prompt("default_v1")
        assert "ChatPromptTemplate" in type(prompt).__name__
        assert "context" in prompt.input_variables
        assert "question" in prompt.input_variables

    def test_create_version(self):
        mgr, tmpdir = self._make_temp_manager()
        try:
            initial_count = len(mgr.get_keys(active_only=False))
            new_key = mgr.create_version(
                base_key="default_v1",
                system_prompt="You are a concise assistant.",
                description="Concise variant",
                author="test",
            )
            assert new_key == "default_v2"
            assert len(mgr.get_keys(active_only=False)) == initial_count + 1
            assert os.path.exists(os.path.join(tmpdir, "default_v2.yaml"))

            data = mgr.get_template_data("default_v2")
            assert data["author"] == "test"
            assert "concise" in data["system_prompt"].lower()
        finally:
            shutil.rmtree(tmpdir)

    def test_auto_increment_version(self):
        mgr, tmpdir = self._make_temp_manager()
        try:
            mgr.create_version("default_v1", description="v2")
            key3 = mgr.create_version("default_v2", description="v3")
            assert key3 == "default_v3"
        finally:
            shutil.rmtree(tmpdir)

    def test_version_history(self):
        mgr, tmpdir = self._make_temp_manager()
        try:
            mgr.create_version("default_v1", description="v2 test")
            history = mgr.get_version_history("default")
            assert len(history) == 2
            assert history[0]["key"] == "default_v1"
            assert history[1]["key"] == "default_v2"
        finally:
            shutil.rmtree(tmpdir)

    def test_set_active(self):
        mgr, tmpdir = self._make_temp_manager()
        try:
            mgr.set_active("chinese_v1", False)
            assert "chinese_v1" not in mgr.get_keys(active_only=True)

            mgr.set_active("chinese_v1", True)
            assert "chinese_v1" in mgr.get_keys(active_only=True)
        finally:
            shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Query rewriters
# ---------------------------------------------------------------------------

class TestQueryRewriters:
    """Tests for query rewriter module imports."""

    def test_imports(self):
        from src.generation.query_rewriter import HyDERewriter, MultiQueryRewriter
        from src.generation.query_rewriter import HYDE_PROMPT, MULTI_QUERY_PROMPT

        assert HYDE_PROMPT is not None
        assert MULTI_QUERY_PROMPT is not None
