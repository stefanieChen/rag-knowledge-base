"""Prompt templates for the RAG generation pipeline.

Uses LangChain ChatPromptTemplate for structured system/human role separation.
Templates are versioned and registered for A/B testing via RAG trace logging.

Supports two sources:
- YAML files in config/prompts/ (managed by PromptVersionManager)
- Hardcoded fallbacks in this module (legacy)
"""

from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# System prompts (role definition + behavioral constraints)
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that answers questions strictly based "
    "on the provided context documents.\n\n"
    "## Rules:\n"
    "- Only use information from the provided context to answer.\n"
    "- Answer in the same language as the question.\n"
    "- Be concise and accurate.\n"
    "- Cite sources using the format [Source N] where N is the source number.\n"
    "- If the context does not contain enough information, say: "
    "\"Based on the provided documents, I cannot find the answer to this question.\""
)

CHINESE_SYSTEM_PROMPT = (
    "你是一个知识库助手，严格基于提供的上下文文档回答问题。\n\n"
    "## 规则：\n"
    "- 只使用提供的上下文中的信息来回答。\n"
    "- 用中文回答。\n"
    "- 简洁准确。\n"
    "- 使用 [来源 N] 格式引用来源。\n"
    "- 如果上下文中没有足够的信息，请说："
    "\"根据提供的文档，我无法找到该问题的答案。\""
)

# ---------------------------------------------------------------------------
# Context formatting template (human message with retrieved chunks)
# ---------------------------------------------------------------------------

CONTEXT_TEMPLATE = (
    "Below are relevant documents retrieved from the knowledge base. "
    "Use them to answer the question that follows.\n\n"
    "## Context:\n{context}\n"
)

# ---------------------------------------------------------------------------
# ChatPromptTemplate builders
# ---------------------------------------------------------------------------

def _build_chat_prompt(system_prompt: str) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate with system/human role separation.

    Args:
        system_prompt: The system-level instruction text.

    Returns:
        A ChatPromptTemplate with (system, human-context, human-question) messages.
    """
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", CONTEXT_TEMPLATE),
        ("human", "Question: {question}"),
    ])


# ---------------------------------------------------------------------------
# Template registry (versioned for A/B testing)
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, ChatPromptTemplate] = {
    "default_v1": _build_chat_prompt(DEFAULT_SYSTEM_PROMPT),
    "chinese_v1": _build_chat_prompt(CHINESE_SYSTEM_PROMPT),
}


# ---------------------------------------------------------------------------
# Lazy-loaded PromptVersionManager (YAML-based)
# ---------------------------------------------------------------------------

_manager = None


def _get_manager():
    """Get or initialize the PromptVersionManager singleton.

    Returns:
        PromptVersionManager instance.
    """
    global _manager
    if _manager is None:
        from src.generation.prompt_version_manager import PromptVersionManager
        _manager = PromptVersionManager()
    return _manager


def get_template(name: str = "default_v1") -> ChatPromptTemplate:
    """Get a ChatPromptTemplate by version name.

    Checks YAML-based prompt store first, then falls back to
    hardcoded TEMPLATES dict.

    Args:
        name: Template version key (e.g., 'default_v1', 'chinese_v1').

    Returns:
        The corresponding ChatPromptTemplate.

    Raises:
        KeyError: If template name not found in either source.
    """
    # Try YAML-based version manager first
    manager = _get_manager()
    if manager.get_template_data(name) is not None:
        return manager.build_chat_prompt(name)

    # Fall back to hardcoded templates
    if name in TEMPLATES:
        return TEMPLATES[name]

    all_keys = list(set(manager.get_keys(active_only=False)) | set(TEMPLATES.keys()))
    raise KeyError(
        f"Template '{name}' not found. "
        f"Available: {sorted(all_keys)}"
    )


def register_template(name: str, template: ChatPromptTemplate) -> None:
    """Register a new prompt template version in the in-memory registry.

    Args:
        name: Version name for the template.
        template: A ChatPromptTemplate instance.
    """
    TEMPLATES[name] = template


def list_available_templates(active_only: bool = True) -> List[Dict[str, Any]]:
    """List all available prompt templates from both YAML and hardcoded sources.

    Args:
        active_only: If True, only return active YAML templates.

    Returns:
        List of template info dicts with 'key', 'source', and metadata.
    """
    result = []

    # YAML-based templates
    manager = _get_manager()
    for info in manager.list_templates(active_only=active_only):
        info["source"] = "yaml"
        result.append(info)

    # Hardcoded templates (only add if not already in YAML)
    yaml_keys = {r["key"] for r in result}
    for key in TEMPLATES:
        if key not in yaml_keys:
            result.append({
                "key": key,
                "source": "hardcoded",
                "name": key,
                "version": "",
                "description": "Built-in template",
            })

    return result


def format_context(chunks: list) -> str:
    """Format retrieved chunks into a context string for the prompt.

    Args:
        chunks: List of result dicts with 'content' and 'metadata' keys.

    Returns:
        Formatted context string with source annotations.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("file_name", "unknown")
        content = chunk.get("content", "")
        context_parts.append(f"[Source {i}: {source}]\n{content}")

    return "\n\n".join(context_parts)
