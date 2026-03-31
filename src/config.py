"""Configuration loader for the RAG system."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


_CONFIG_CACHE: Dict[str, Any] = {}


def get_project_root() -> Path:
    """Return the project root directory (where config/ lives)."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Optional absolute path to config file.
                     Defaults to config/settings.yaml in project root.

    Returns:
        Dictionary of configuration values.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = str(get_project_root() / "config" / "settings.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG_CACHE = yaml.safe_load(f)

    _resolve_paths(_CONFIG_CACHE)
    return _CONFIG_CACHE


def _resolve_paths(config: Dict[str, Any]) -> None:
    """Resolve relative paths in config to absolute paths based on project root."""
    root = get_project_root()
    path_keys = [
        ("vector_store", "persist_directory"),
        ("logging", "log_dir"),
        ("logging", "trace_dir"),
        ("data", "raw_dir"),
        ("data", "processed_dir"),
        ("reranker", "local_model_path"),
        ("code_ingestion", "repos_dir"),
    ]
    for section, key in path_keys:
        if section in config and key in config[section]:
            val = config[section][key]
            if val.startswith("./"):
                config[section][key] = str(root / val)


def reset_config() -> None:
    """Reset cached config (useful for testing)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = {}
