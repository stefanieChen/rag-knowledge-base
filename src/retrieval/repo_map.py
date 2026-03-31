"""RepoMap: Aider-style repository map with tree-sitter def/ref extraction + PageRank.

Extracts definitions (classes, functions, methods) and references from source files,
builds a directed graph of symbol relationships, and ranks symbols by importance
using PageRank. Generates a concise repo map text for LLM context injection.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import tree_sitter as ts

from src.logging.logger import get_logger

logger = get_logger("retrieval.repo_map")

# ── Tree-sitter language registry ────────────────────────────────

# Map language name → (module_import_name, language_func)
_LANG_MODULES = {
    "python": "tree_sitter_python",
    "java": "tree_sitter_java",
    "csharp": "tree_sitter_c_sharp",
    "typescript": "tree_sitter_typescript",
}

# Cache loaded parsers
_PARSER_CACHE: Dict[str, ts.Parser] = {}


def _get_parser(language: str) -> Optional[ts.Parser]:
    """Get or create a tree-sitter parser for the given language.

    Args:
        language: Language name (python, java, csharp, typescript).

    Returns:
        A tree-sitter Parser, or None if the language is not supported.
    """
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]

    module_name = _LANG_MODULES.get(language)
    if not module_name:
        return None

    try:
        mod = __import__(module_name)
        if language == "typescript":
            # tree_sitter_typescript has .language_typescript()
            lang_func = getattr(mod, "language_typescript", None)
            if lang_func is None:
                lang_func = mod.language
        else:
            lang_func = mod.language
        lang = ts.Language(lang_func())
        parser = ts.Parser(lang)
        _PARSER_CACHE[language] = parser
        return parser
    except Exception as e:
        logger.warning(f"Failed to load tree-sitter parser for {language}: {e}")
        return None


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class SymbolDef:
    """A symbol definition extracted from source code."""
    name: str
    kind: str  # "class", "function", "method"
    file_path: str
    line: int
    end_line: int
    parent: Optional[str] = None  # parent class name for methods

    @property
    def qualified_name(self) -> str:
        """Return fully qualified name (e.g., 'ClassName.method_name')."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name


@dataclass
class SymbolRef:
    """A symbol reference (usage) extracted from source code."""
    name: str
    file_path: str
    line: int


@dataclass
class FileSymbols:
    """All symbols extracted from a single file."""
    file_path: str
    relative_path: str
    language: str
    definitions: List[SymbolDef] = field(default_factory=list)
    references: List[SymbolRef] = field(default_factory=list)


# ── Tree-sitter queries per language ─────────────────────────────

# Node types that represent definitions per language
_DEF_NODE_TYPES = {
    "python": {
        "class_definition": "class",
        "function_definition": "function",
    },
    "java": {
        "class_declaration": "class",
        "interface_declaration": "class",
        "method_declaration": "method",
        "constructor_declaration": "method",
    },
    "csharp": {
        "class_declaration": "class",
        "interface_declaration": "class",
        "method_declaration": "method",
        "constructor_declaration": "method",
    },
    "typescript": {
        "class_declaration": "class",
        "function_declaration": "function",
        "method_definition": "method",
    },
}

# Node types that contain identifiers we want to track as references
_REF_IDENTIFIER_TYPES = {"identifier", "type_identifier"}


def _extract_name_from_node(node: ts.Node, language: str) -> Optional[str]:
    """Extract the name identifier from a definition node.

    Args:
        node: Tree-sitter node of a definition.
        language: Programming language.

    Returns:
        The symbol name, or None if not found.
    """
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier"):
            return child.text.decode("utf-8") if child.text else None
    return None


def _find_parent_class(node: ts.Node, language: str) -> Optional[str]:
    """Walk up the tree to find if this node is inside a class definition.

    Args:
        node: Current tree-sitter node.
        language: Programming language.

    Returns:
        Parent class name, or None.
    """
    class_types = {
        "python": {"class_definition"},
        "java": {"class_declaration", "interface_declaration"},
        "csharp": {"class_declaration", "interface_declaration"},
        "typescript": {"class_declaration"},
    }
    valid_types = class_types.get(language, set())

    current = node.parent
    while current is not None:
        if current.type in valid_types:
            return _extract_name_from_node(current, language)
        current = current.parent
    return None


def _walk_tree(node: ts.Node):
    """Yield all nodes in the tree via depth-first traversal."""
    yield node
    for child in node.children:
        yield from _walk_tree(child)


def extract_symbols(
    file_path: str,
    relative_path: str,
    language: str,
    content: Optional[str] = None,
) -> Optional[FileSymbols]:
    """Extract definitions and references from a source file using tree-sitter.

    Args:
        file_path: Absolute path to the source file.
        relative_path: Relative path (for display).
        language: Programming language name.
        content: Optional file content. If None, reads from file_path.

    Returns:
        FileSymbols with defs and refs, or None on failure.
    """
    parser = _get_parser(language)
    if parser is None:
        return None

    if content is None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="gbk") as f:
                    content = f.read()
            except Exception:
                return None
        except Exception:
            return None

    if not content.strip():
        return None

    try:
        tree = parser.parse(content.encode("utf-8"))
    except Exception as e:
        logger.warning(f"Tree-sitter parse failed for {file_path}: {e}")
        return None

    def_types = _DEF_NODE_TYPES.get(language, {})
    fs = FileSymbols(
        file_path=file_path,
        relative_path=relative_path,
        language=language,
    )

    # Collect all definition names for reference filtering
    defined_names: Set[str] = set()

    # Pass 1: extract definitions
    for node in _walk_tree(tree.root_node):
        if node.type in def_types:
            name = _extract_name_from_node(node, language)
            if not name:
                continue

            kind = def_types[node.type]

            # In Python, a function inside a class is a method
            parent = None
            if language == "python" and kind == "function":
                parent = _find_parent_class(node, language)
                if parent:
                    kind = "method"

            # For Java/C#/TS, methods already have class parent
            if kind == "method" and parent is None:
                parent = _find_parent_class(node, language)

            sym = SymbolDef(
                name=name,
                kind=kind,
                file_path=file_path,
                line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                parent=parent,
            )
            fs.definitions.append(sym)
            defined_names.add(name)

    # Pass 2: extract references (identifiers not part of definitions)
    for node in _walk_tree(tree.root_node):
        if node.type in _REF_IDENTIFIER_TYPES:
            name = node.text.decode("utf-8") if node.text else None
            if not name:
                continue

            # Skip very short names and Python builtins/keywords
            if len(name) <= 1:
                continue

            # Check if this identifier is the name child of a definition node
            parent = node.parent
            if parent and parent.type in def_types:
                continue

            fs.references.append(SymbolRef(
                name=name,
                file_path=file_path,
                line=node.start_point.row + 1,
            ))

    return fs


# ── RepoMap builder ──────────────────────────────────────────────

class RepoMap:
    """Builds and maintains a repo-level symbol map with PageRank ranking.

    Usage:
        repo_map = RepoMap()
        repo_map.add_files(file_list)  # file_list: [{absolute_path, relative_path, language}]
        text = repo_map.generate_map(max_tokens=2000)
    """

    def __init__(self) -> None:
        self._file_symbols: Dict[str, FileSymbols] = {}
        self._graph: Optional[nx.DiGraph] = None
        self._pagerank: Dict[str, float] = {}
        self._all_defs: Dict[str, List[SymbolDef]] = defaultdict(list)

    def add_files(
        self,
        files: List[Dict[str, str]],
    ) -> int:
        """Extract symbols from a list of source files.

        Args:
            files: List of dicts with keys: absolute_path, relative_path, language.

        Returns:
            Number of files successfully parsed.
        """
        start = time.perf_counter()
        parsed = 0

        for file_info in files:
            abs_path = file_info["absolute_path"]
            rel_path = file_info.get("relative_path", abs_path)
            language = file_info.get("language", "")

            if not language:
                continue

            fs = extract_symbols(abs_path, rel_path, language)
            if fs:
                self._file_symbols[abs_path] = fs
                for d in fs.definitions:
                    self._all_defs[d.name].append(d)
                parsed += 1

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        total_defs = sum(len(fs.definitions) for fs in self._file_symbols.values())
        total_refs = sum(len(fs.references) for fs in self._file_symbols.values())
        logger.info(
            f"Symbol extraction: {parsed}/{len(files)} files, "
            f"{total_defs} defs, {total_refs} refs ({elapsed_ms}ms)"
        )

        return parsed

    def build_graph(self) -> nx.DiGraph:
        """Build a directed graph of symbol relationships.

        Nodes are (file_path, symbol_name) tuples.
        Edges go from reference sites to definition sites.

        Returns:
            The constructed NetworkX DiGraph.
        """
        start = time.perf_counter()
        G = nx.DiGraph()

        # Add definition nodes
        for file_path, fs in self._file_symbols.items():
            for d in fs.definitions:
                node_id = f"{fs.relative_path}::{d.qualified_name}"
                G.add_node(node_id, **{
                    "name": d.name,
                    "qualified_name": d.qualified_name,
                    "kind": d.kind,
                    "file": fs.relative_path,
                    "line": d.line,
                    "end_line": d.end_line,
                })

        # Add edges from reference files to definition nodes
        for file_path, fs in self._file_symbols.items():
            for ref in fs.references:
                # Find matching definitions
                if ref.name in self._all_defs:
                    for target_def in self._all_defs[ref.name]:
                        target_fs = self._file_symbols.get(target_def.file_path)
                        if target_fs is None:
                            continue
                        target_id = f"{target_fs.relative_path}::{target_def.qualified_name}"
                        source_id = f"{fs.relative_path}::__ref_{ref.name}_L{ref.line}"

                        # Skip self-references within the same file
                        if target_fs.relative_path == fs.relative_path:
                            continue

                        if not G.has_node(source_id):
                            G.add_node(source_id, **{
                                "name": ref.name,
                                "kind": "reference",
                                "file": fs.relative_path,
                                "line": ref.line,
                            })

                        G.add_edge(source_id, target_id, weight=1.0)

        self._graph = G

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"Symbol graph built: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges ({elapsed_ms}ms)"
        )

        return G

    def compute_pagerank(self, alpha: float = 0.85) -> Dict[str, float]:
        """Compute PageRank scores for all symbol nodes.

        Args:
            alpha: PageRank damping factor.

        Returns:
            Dict mapping node ID to PageRank score.
        """
        if self._graph is None:
            self.build_graph()

        if self._graph.number_of_nodes() == 0:
            return {}

        start = time.perf_counter()

        try:
            self._pagerank = nx.pagerank(
                self._graph, alpha=alpha, max_iter=100
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank did not converge, using degree centrality")
            self._pagerank = nx.degree_centrality(self._graph)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(f"PageRank computed: {len(self._pagerank)} nodes ({elapsed_ms}ms)")

        return self._pagerank

    def get_ranked_definitions(
        self,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get definitions ranked by PageRank importance.

        Args:
            top_n: Maximum number of results. None returns all.

        Returns:
            List of dicts with symbol info and rank score, sorted by importance.
        """
        if not self._pagerank:
            self.compute_pagerank()

        if self._graph is None:
            return []

        # Filter to only definition nodes (not reference nodes)
        ranked = []
        for node_id, score in self._pagerank.items():
            node_data = self._graph.nodes.get(node_id, {})
            if node_data.get("kind") in ("class", "function", "method"):
                ranked.append({
                    "node_id": node_id,
                    "name": node_data.get("name", ""),
                    "qualified_name": node_data.get("qualified_name", ""),
                    "kind": node_data.get("kind", ""),
                    "file": node_data.get("file", ""),
                    "line": node_data.get("line", 0),
                    "end_line": node_data.get("end_line", 0),
                    "pagerank": round(score, 6),
                    "in_degree": self._graph.in_degree(node_id),
                })

        ranked.sort(key=lambda x: x["pagerank"], reverse=True)

        if top_n:
            ranked = ranked[:top_n]

        return ranked

    def generate_map(
        self,
        max_chars: int = 4000,
        top_n: Optional[int] = None,
    ) -> str:
        """Generate a concise repo map text for LLM context injection.

        Groups symbols by file and shows the most important ones first.

        Args:
            max_chars: Approximate character budget for the output.
            top_n: Max number of symbols to include. Defaults to auto-fit.

        Returns:
            Formatted repo map string.
        """
        ranked = self.get_ranked_definitions(top_n=top_n)
        if not ranked:
            return ""

        # Group by file, preserving rank order
        file_symbols: Dict[str, List[Dict]] = defaultdict(list)
        file_max_rank: Dict[str, float] = {}

        for sym in ranked:
            f = sym["file"]
            file_symbols[f].append(sym)
            file_max_rank[f] = max(file_max_rank.get(f, 0), sym["pagerank"])

        # Sort files by their highest-ranked symbol
        sorted_files = sorted(
            file_symbols.keys(),
            key=lambda f: file_max_rank[f],
            reverse=True,
        )

        lines = ["# Repository Map", ""]
        char_count = 20  # header

        for filepath in sorted_files:
            symbols = file_symbols[filepath]
            # Sort within file by line number
            symbols.sort(key=lambda s: s["line"])

            file_header = f"## {filepath}"
            if char_count + len(file_header) + 2 > max_chars:
                break

            lines.append(file_header)
            char_count += len(file_header) + 1

            for sym in symbols:
                kind_icon = {"class": "●", "function": "ƒ", "method": "→"}.get(
                    sym["kind"], "·"
                )
                line_text = (
                    f"  {kind_icon} {sym['qualified_name']} "
                    f"(L{sym['line']}-{sym['end_line']}) "
                    f"[rank: {sym['pagerank']:.4f}, refs: {sym['in_degree']}]"
                )

                if char_count + len(line_text) + 1 > max_chars:
                    lines.append("  ... (truncated)")
                    return "\n".join(lines)

                lines.append(line_text)
                char_count += len(line_text) + 1

            lines.append("")
            char_count += 1

        return "\n".join(lines)

    @property
    def file_count(self) -> int:
        """Number of files with extracted symbols."""
        return len(self._file_symbols)

    @property
    def definition_count(self) -> int:
        """Total number of definitions across all files."""
        return sum(len(fs.definitions) for fs in self._file_symbols.values())

    @property
    def reference_count(self) -> int:
        """Total number of references across all files."""
        return sum(len(fs.references) for fs in self._file_symbols.values())

    def get_file_tree(self) -> str:
        """Generate a simple file tree with definition counts.

        Returns:
            Formatted file tree string.
        """
        lines = []
        for fp in sorted(self._file_symbols.keys()):
            fs = self._file_symbols[fp]
            defs = len(fs.definitions)
            refs = len(fs.references)
            lines.append(f"  {fs.relative_path}  ({defs} defs, {refs} refs)")
        return "\n".join(lines)
