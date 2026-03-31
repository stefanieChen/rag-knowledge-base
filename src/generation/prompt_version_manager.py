"""Local prompt version manager.

Manages prompt templates stored as YAML files in config/prompts/.
Provides version tracking, listing, creation, and activation without
requiring any external services (LangSmith alternative).

Each prompt YAML file has the structure:
    name: <template_name>
    version: <version_string>
    description: <human-readable description>
    created_at: <ISO timestamp>
    author: <author name>
    tags: [<tag1>, <tag2>]
    active: true/false
    system_prompt: |
        <system prompt text>
    context_template: |
        <context template with {context} placeholder>
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langchain_core.prompts import ChatPromptTemplate

from src.logging.logger import get_logger

logger = get_logger("generation.prompt_version_manager")

DEFAULT_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config",
    "prompts",
)


class PromptVersionManager:
    """Manages versioned prompt templates stored as YAML files.

    Provides:
    - Load all templates from config/prompts/
    - List available templates with metadata
    - Get a specific template by name_version key
    - Create new versions (copies existing + increments version)
    - Activate/deactivate templates
    - Build LangChain ChatPromptTemplate from YAML data
    """

    def __init__(self, prompts_dir: Optional[str] = None) -> None:
        self._prompts_dir = prompts_dir or DEFAULT_PROMPTS_DIR
        Path(self._prompts_dir).mkdir(parents=True, exist_ok=True)
        self._templates: Dict[str, Dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        """Reload all prompt templates from the prompts directory."""
        self._templates.clear()

        for fname in sorted(os.listdir(self._prompts_dir)):
            if not fname.endswith((".yaml", ".yml")):
                continue

            fpath = os.path.join(self._prompts_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if not isinstance(data, dict):
                    logger.warning(f"Skipping invalid prompt file: {fname}")
                    continue

                name = data.get("name", "")
                version = data.get("version", "v1")
                key = f"{name}_{version}"
                data["_file"] = fname
                data["_path"] = fpath
                self._templates[key] = data

            except Exception as e:
                logger.error(f"Failed to load prompt {fname}: {e}")

        logger.info(
            f"Loaded {len(self._templates)} prompt templates "
            f"from {self._prompts_dir}"
        )

    def list_templates(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """List all available prompt templates with metadata.

        Args:
            active_only: If True, only return active templates.

        Returns:
            List of template metadata dicts (without full prompt text).
        """
        result = []
        for key, data in self._templates.items():
            if active_only and not data.get("active", True):
                continue

            result.append({
                "key": key,
                "name": data.get("name", ""),
                "version": data.get("version", ""),
                "description": data.get("description", ""),
                "author": data.get("author", ""),
                "tags": data.get("tags", []),
                "active": data.get("active", True),
                "created_at": data.get("created_at", ""),
                "file": data.get("_file", ""),
            })

        return result

    def get_keys(self, active_only: bool = True) -> List[str]:
        """Get list of template keys (name_version format).

        Args:
            active_only: If True, only return keys of active templates.

        Returns:
            Sorted list of template key strings.
        """
        if active_only:
            return sorted(
                k for k, v in self._templates.items()
                if v.get("active", True)
            )
        return sorted(self._templates.keys())

    def get_template_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Get the raw YAML data for a template.

        Args:
            key: Template key in name_version format.

        Returns:
            Template data dict, or None if not found.
        """
        return self._templates.get(key)

    def build_chat_prompt(self, key: str) -> ChatPromptTemplate:
        """Build a LangChain ChatPromptTemplate from a stored template.

        Args:
            key: Template key in name_version format.

        Returns:
            ChatPromptTemplate with system/human messages.

        Raises:
            KeyError: If template key not found.
        """
        data = self._templates.get(key)
        if data is None:
            raise KeyError(
                f"Prompt template '{key}' not found. "
                f"Available: {self.get_keys(active_only=False)}"
            )

        system_prompt = data.get("system_prompt", "").strip()
        context_template = data.get("context_template", "").strip()

        if not context_template:
            context_template = (
                "Below are relevant documents retrieved from the knowledge base. "
                "Use them to answer the question that follows.\n\n"
                "## Context:\n{context}"
            )

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", context_template),
            ("human", "Question: {question}"),
        ])

    def create_version(
        self,
        base_key: str,
        new_version: Optional[str] = None,
        system_prompt: Optional[str] = None,
        context_template: Optional[str] = None,
        description: Optional[str] = None,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Create a new version of an existing template.

        Args:
            base_key: Key of the template to base the new version on.
            new_version: Version string for the new template (auto-increments if None).
            system_prompt: New system prompt (keeps original if None).
            context_template: New context template (keeps original if None).
            description: New description (keeps original if None).
            author: Author of the new version.
            tags: Tags for the new version (keeps original if None).

        Returns:
            The key of the newly created template.

        Raises:
            KeyError: If base_key not found.
        """
        base = self._templates.get(base_key)
        if base is None:
            raise KeyError(f"Base template '{base_key}' not found")

        name = base["name"]

        # Auto-increment version
        if new_version is None:
            existing_versions = [
                v["version"] for v in self._templates.values()
                if v.get("name") == name
            ]
            max_num = 0
            for v in existing_versions:
                try:
                    num = int(v.lstrip("v"))
                    max_num = max(max_num, num)
                except ValueError:
                    pass
            new_version = f"v{max_num + 1}"

        new_key = f"{name}_{new_version}"
        new_data = {
            "name": name,
            "version": new_version,
            "description": description or base.get("description", ""),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "author": author or "user",
            "tags": tags or base.get("tags", []),
            "active": True,
            "system_prompt": system_prompt or base.get("system_prompt", ""),
            "context_template": context_template or base.get("context_template", ""),
        }

        # Write to file
        fname = f"{name}_{new_version}.yaml"
        fpath = os.path.join(self._prompts_dir, fname)

        with open(fpath, "w", encoding="utf-8") as f:
            yaml.dump(
                new_data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        # Reload to pick up new file
        self.reload()

        logger.info(f"Created new prompt version: {new_key} ({fname})")
        return new_key

    def set_active(self, key: str, active: bool) -> None:
        """Activate or deactivate a prompt template.

        Args:
            key: Template key.
            active: True to activate, False to deactivate.

        Raises:
            KeyError: If key not found.
        """
        data = self._templates.get(key)
        if data is None:
            raise KeyError(f"Template '{key}' not found")

        data["active"] = active
        fpath = data["_path"]

        # Update the file
        save_data = {
            k: v for k, v in data.items() if not k.startswith("_")
        }
        with open(fpath, "w", encoding="utf-8") as f:
            yaml.dump(
                save_data, f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logger.info(f"Template '{key}' active={active}")

    def get_version_history(self, name: str) -> List[Dict[str, Any]]:
        """Get all versions of a template by name, sorted by version.

        Args:
            name: Template name (without version).

        Returns:
            List of template metadata dicts sorted by version.
        """
        versions = [
            {
                "key": key,
                "version": data.get("version", ""),
                "description": data.get("description", ""),
                "created_at": data.get("created_at", ""),
                "author": data.get("author", ""),
                "active": data.get("active", True),
            }
            for key, data in self._templates.items()
            if data.get("name") == name
        ]
        return sorted(versions, key=lambda x: x["version"])
