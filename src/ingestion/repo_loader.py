"""Repository loader for code ingestion.

Handles local folder traversal (with .gitignore support) and
GitHub repository cloning. Filters files by supported extensions,
size limits, and exclude patterns.
"""

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import pathspec

from src.config import load_config
from src.logging.logger import get_logger

logger = get_logger("ingestion.repo_loader")

# Default exclude patterns (directories/files to always skip)
DEFAULT_EXCLUDE_PATTERNS = [
    "node_modules", "venv", ".venv", "__pycache__", ".git",
    "dist", "build", ".tox", ".egg-info", ".eggs",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "*.min.js", "*.min.css", "*.map",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
]

# Default supported code extensions
DEFAULT_CODE_EXTENSIONS = [
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".cs", ".go", ".rs", ".cpp", ".c", ".h", ".hpp",
    ".rb", ".php", ".sh", ".sql",
    ".yaml", ".yml", ".json", ".toml", ".xml",
    ".css", ".scss",
]

# Regex to detect GitHub/GitLab URLs
GIT_URL_PATTERN = re.compile(
    r"^(https?://|git@)(github\.com|gitlab\.com|bitbucket\.org)[/:]"
)


def is_git_url(path: str) -> bool:
    """Check if the given path is a Git repository URL.

    Args:
        path: A local path or URL string.

    Returns:
        True if the path looks like a Git URL.
    """
    return bool(GIT_URL_PATTERN.match(path))


def extract_repo_name(url: str) -> str:
    """Extract a repository name from a Git URL.

    Args:
        url: Git repository URL.

    Returns:
        Repository name (e.g. 'owner_repo').
    """
    # Remove trailing .git and slashes
    clean = url.rstrip("/").removesuffix(".git")
    parts = re.split(r"[/:]", clean)
    # Take last two parts: owner/repo
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    return parts[-1] if parts else "unknown_repo"


def clone_repo(url: str, repos_dir: str) -> str:
    """Clone a Git repository (shallow) into the repos directory.

    Args:
        url: Git repository URL.
        repos_dir: Directory to clone into.

    Returns:
        Path to the cloned repository folder.

    Raises:
        RuntimeError: If git clone fails.
    """
    repo_name = extract_repo_name(url)
    target_dir = os.path.join(repos_dir, repo_name)

    if os.path.exists(target_dir):
        logger.info(f"Repository already exists: {target_dir}")
        return target_dir

    Path(repos_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Cloning {url} → {target_dir}")
    start = time.perf_counter()

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, target_dir],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git clone failed (exit {result.returncode}): {result.stderr.strip()}"
            )
    except FileNotFoundError:
        raise RuntimeError(
            "git is not installed or not on PATH. "
            "Please install git to clone repositories."
        )

    elapsed = time.perf_counter() - start
    logger.info(f"Cloned {repo_name} in {elapsed:.1f}s")
    return target_dir


def _load_gitignore_spec(folder: str) -> Optional[pathspec.PathSpec]:
    """Load .gitignore patterns from a folder if present.

    Args:
        folder: Root folder to look for .gitignore.

    Returns:
        PathSpec instance or None if no .gitignore found.
    """
    gitignore_path = os.path.join(folder, ".gitignore")
    if not os.path.isfile(gitignore_path):
        return None

    try:
        with open(gitignore_path, "r", encoding="utf-8") as f:
            patterns = f.read()
        spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns.splitlines())
        logger.info(f"Loaded .gitignore from {gitignore_path}")
        return spec
    except Exception as e:
        logger.warning(f"Failed to parse .gitignore: {e}")
        return None


def _build_exclude_spec(exclude_patterns: List[str]) -> pathspec.PathSpec:
    """Build a PathSpec from exclude pattern list.

    Args:
        exclude_patterns: List of glob patterns to exclude.

    Returns:
        PathSpec instance for matching excluded paths.
    """
    return pathspec.PathSpec.from_lines("gitwildmatch", exclude_patterns)


def discover_code_files(
    folder: str,
    config: Optional[dict] = None,
) -> List[Dict[str, str]]:
    """Walk a folder and discover supported code files.

    Respects .gitignore and configured exclude patterns.
    Skips files exceeding the max size limit.

    Args:
        folder: Root folder to scan.
        config: Optional config dict with code_ingestion settings.

    Returns:
        List of dicts with keys: 'absolute_path', 'relative_path'.
    """
    if config is None:
        config = load_config()

    code_cfg = config.get("code_ingestion", {})
    supported_ext = set(
        code_cfg.get("supported_extensions", DEFAULT_CODE_EXTENSIONS)
    )
    max_size_kb = code_cfg.get("max_file_size_kb", 500)
    max_size_bytes = max_size_kb * 1024
    exclude_patterns = code_cfg.get("exclude_patterns", DEFAULT_EXCLUDE_PATTERNS)

    start = time.perf_counter()
    folder = str(Path(folder).resolve())

    # Load .gitignore and exclude patterns
    gitignore_spec = _load_gitignore_spec(folder)
    exclude_spec = _build_exclude_spec(exclude_patterns)

    files = []
    skipped_ext = 0
    skipped_size = 0
    skipped_exclude = 0

    for root, dirs, filenames in os.walk(folder):
        # Compute relative path from root folder
        rel_root = os.path.relpath(root, folder)

        # Skip excluded directories (modify dirs in-place to prune os.walk)
        dirs[:] = [
            d for d in dirs
            if not exclude_spec.match_file(os.path.join(rel_root, d) + "/")
            and not (gitignore_spec and gitignore_spec.match_file(
                os.path.join(rel_root, d) + "/"
            ))
        ]

        for filename in filenames:
            ext = Path(filename).suffix.lower()

            # Check extension
            if ext not in supported_ext:
                skipped_ext += 1
                continue

            abs_path = os.path.join(root, filename)
            rel_path = os.path.relpath(abs_path, folder)

            # Check exclude patterns
            if exclude_spec.match_file(rel_path):
                skipped_exclude += 1
                continue

            # Check .gitignore
            if gitignore_spec and gitignore_spec.match_file(rel_path):
                skipped_exclude += 1
                continue

            # Check file size
            try:
                file_size = os.path.getsize(abs_path)
                if file_size > max_size_bytes:
                    skipped_size += 1
                    logger.debug(
                        f"Skipped (too large: {file_size // 1024}KB): {rel_path}"
                    )
                    continue
            except OSError:
                continue

            files.append({
                "absolute_path": abs_path,
                "relative_path": rel_path.replace("\\", "/"),
            })

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        f"Discovered {len(files)} code files in {folder} "
        f"(skipped: {skipped_ext} ext, {skipped_size} size, "
        f"{skipped_exclude} excluded) ({elapsed_ms}ms)"
    )
    return files


def load_repo(
    path: str,
    repo_name: Optional[str] = None,
    config: Optional[dict] = None,
) -> tuple:
    """Load a code repository from a local folder or Git URL.

    If path is a Git URL, clones the repo first. Then discovers
    all supported code files.

    Args:
        path: Local folder path or Git repository URL.
        repo_name: Optional name to identify the repo. Auto-detected if None.
        config: Optional config dict.

    Returns:
        Tuple of (repo_name, folder_path, list_of_file_dicts).
    """
    if config is None:
        config = load_config()

    code_cfg = config.get("code_ingestion", {})
    repos_dir = code_cfg.get("repos_dir", "./data/repos")

    if is_git_url(path):
        if repo_name is None:
            repo_name = extract_repo_name(path)
        folder = clone_repo(path, repos_dir)
    else:
        folder = str(Path(path).resolve())
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Directory not found: {folder}")
        if repo_name is None:
            repo_name = Path(folder).name

    files = discover_code_files(folder, config)
    return repo_name, folder, files
