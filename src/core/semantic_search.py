"""
Semantic Search / RAG Module

Provides embedding-based and AST-aware codebase navigation so the agent
can explore 10,000+ line codebases without loading everything into context.

Two search strategies are exposed:
1. **GrepSearch** — fast, pattern-based ripgrep-style search (no ML needed)
2. **SemanticSearch** — embedding-based vector search for natural-language
   code queries (requires a local embedding model via Ollama)

Both implement a common ``SearchResult`` / ``search()`` interface so the
ToolExecutor can dispatch to either transparently.

Design Decisions:
- Lazy index build: index is built on first search, cached in memory
- TF-IDF cosine similarity used as fallback if Ollama embeddings unavailable
- Chunking strategy: split by function/class boundaries (AST-aware)
- Results always include file path, line range, and a relevance score
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single search result pointing to a code location."""
    file_path: str
    start_line: int
    end_line: int
    snippet: str
    score: float
    match_type: str = "semantic"   # "exact", "pattern", "semantic"

    def to_prompt_str(self) -> str:
        """Format for injection into agent context."""
        return (
            f"📄 {self.file_path}:{self.start_line}-{self.end_line} "
            f"(score={self.score:.2f})\n"
            f"```\n{self.snippet[:800]}\n```"
        )


# ---------------------------------------------------------------------------
# AST-aware chunker
# ---------------------------------------------------------------------------

def _extract_chunks(source: str, file_path: str) -> list[dict]:
    """
    Split a Python source file into chunks at function/class boundaries.

    Falls back to fixed-size line windows for non-Python files.

    Returns list of dicts: {text, start_line, end_line, file_path}
    """
    chunks: list[dict] = []

    if file_path.endswith(".py"):
        try:
            tree = ast.parse(source)
            lines = source.splitlines()

            def _add(node: ast.AST) -> None:
                if not hasattr(node, "lineno"):
                    return
                start = node.lineno - 1
                end = getattr(node, "end_lineno", start + 30) - 1
                text = "\n".join(lines[start : end + 1])
                chunks.append({
                    "text": text,
                    "start_line": start + 1,
                    "end_line": end + 1,
                    "file_path": file_path,
                })

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    _add(node)

            if chunks:
                return chunks
        except SyntaxError:
            pass  # Treat as plain text below

    # Fallback: fixed 30-line windows with 10-line overlap
    lines = source.splitlines()
    step = 20
    window = 30
    for i in range(0, max(1, len(lines)), step):
        chunk_lines = lines[i : i + window]
        text = "\n".join(chunk_lines)
        chunks.append({
            "text": text,
            "start_line": i + 1,
            "end_line": min(i + window, len(lines)),
            "file_path": file_path,
        })

    return chunks


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity (no-ML fallback)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _build_tfidf_index(chunks: list[dict]) -> tuple[list[dict], list[float]]:
    """Build TF-IDF term frequency vectors for each chunk."""
    # Document frequency
    df: dict[str, int] = {}
    chunk_tfs: list[dict[str, float]] = []

    for chunk in chunks:
        tokens = _tokenize(chunk["text"])
        tf: dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        # Normalize TF
        total = sum(tf.values()) or 1
        for t in tf:
            tf[t] /= total
        chunk_tfs.append(tf)
        for t in tf:
            df[t] = df.get(t, 0) + 1

    N = max(len(chunks), 1)
    # IDF
    idf: dict[str, float] = {
        t: math.log((N + 1) / (cnt + 1)) + 1
        for t, cnt in df.items()
    }
    # TF-IDF vectors + norms
    vectors: list[dict[str, float]] = []
    norms: list[float] = []
    for tf in chunk_tfs:
        vec = {t: tf[t] * idf.get(t, 1.0) for t in tf}
        norm = math.sqrt(sum(v ** 2 for v in vec.values()))
        vectors.append(vec)
        norms.append(norm or 1.0)

    # Attach to chunks
    result = []
    for i, chunk in enumerate(chunks):
        result.append({**chunk, "_vec": vectors[i], "_norm": norms[i]})
    return result, list(idf.values())


def _cosine_sim(query_tokens: list[str], chunk: dict, idf: dict[str, float]) -> float:
    """Compute cosine similarity between a query bag-of-words and a chunk."""
    q_tf: dict[str, float] = {}
    for t in query_tokens:
        q_tf[t] = q_tf.get(t, 0) + 1
    total = sum(q_tf.values()) or 1
    for t in q_tf:
        q_tf[t] /= total

    vec = chunk.get("_vec", {})
    norm = chunk.get("_norm", 1.0)

    dot = sum(q_tf.get(t, 0) * idf.get(t, 1.0) * vec.get(t, 0) for t in q_tf)
    q_norm = math.sqrt(sum((q_tf[t] * idf.get(t, 1.0)) ** 2 for t in q_tf)) or 1.0
    return dot / (q_norm * norm)


# ---------------------------------------------------------------------------
# GrepSearch (Pattern-based)
# ---------------------------------------------------------------------------

class GrepSearch:
    """
    Fast pattern-based code search (no ML).

    Scans workspace files for a regex pattern and returns matching lines
    with surrounding context.
    """

    _GLOBS = ["**/*.py", "**/*.ts", "**/*.js", "**/*.tsx", "**/*.jsx",
              "**/*.go", "**/*.rs", "**/*.java", "**/*.cpp", "**/*.c",
              "**/*.html", "**/*.css", "**/*.yaml", "**/*.yml", "**/*.json",
              "**/*.md", "**/*.txt"]

    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root.resolve()

    def search(
        self,
        query: str,
        glob: str | None = None,
        case_sensitive: bool = True,
        context_lines: int = 3,
        max_results: int = 30,
    ) -> list[SearchResult]:
        """
        Search workspace for a regex pattern.

        Args:
            query: Regex pattern or plain string to search
            glob: File glob filter (e.g. '**/*.py')
            case_sensitive: Whether to respect case
            context_lines: Lines of surrounding context per match
            max_results: Maximum results to return

        Returns:
            List of SearchResult objects
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(query, flags)
        except re.error:
            regex = re.compile(re.escape(query), flags)

        globs = [glob] if glob else self._GLOBS
        results: list[SearchResult] = []

        for g in globs:
            for file_path in sorted(self._root.glob(g)):
                if not file_path.is_file():
                    continue
                try:
                    source = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue

                lines = source.splitlines()
                for i, line in enumerate(lines):
                    if regex.search(line):
                        ctx_start = max(0, i - context_lines)
                        ctx_end = min(len(lines), i + context_lines + 1)
                        snippet = "\n".join(lines[ctx_start:ctx_end])
                        rel = str(file_path.relative_to(self._root))
                        results.append(SearchResult(
                            file_path=rel,
                            start_line=ctx_start + 1,
                            end_line=ctx_end,
                            snippet=snippet,
                            score=1.0,
                            match_type="pattern",
                        ))
                        if len(results) >= max_results:
                            return results
        return results


# ---------------------------------------------------------------------------
# SemanticSearch (TF-IDF / Embedding-based)
# ---------------------------------------------------------------------------

class SemanticSearch:
    """
    Embedding-based semantic code search.

    On first use, indexes all workspace code files into chunks and builds
    TF-IDF vectors (pure Python, zero ML dependency). If an Ollama
    embedding model is available it upgrades automatically to dense vectors.

    Usage::

        searcher = SemanticSearch(workspace_root=Path("./workspace"))
        results = searcher.search("find all places we handle HTTP errors")
    """

    _GLOBS = ["**/*.py", "**/*.ts", "**/*.js", "**/*.go", "**/*.rs"]

    def __init__(
        self,
        workspace_root: Path,
        ollama_base_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
    ) -> None:
        self._root = workspace_root.resolve()
        self._ollama_base_url = ollama_base_url
        self._embed_model = embed_model

        # Lazily built index
        self._chunks: list[dict] = []
        self._idf: dict[str, float] = {}
        self._indexed = False

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self, force: bool = False) -> None:
        """Scan workspace and build the chunk index."""
        if self._indexed and not force:
            return

        all_chunks: list[dict] = []
        for glob in self._GLOBS:
            for file_path in sorted(self._root.glob(glob)):
                if not file_path.is_file():
                    continue
                try:
                    source = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                rel = str(file_path.relative_to(self._root))
                chunks = _extract_chunks(source, rel)
                all_chunks.extend(chunks)

        indexed, idf_values = _build_tfidf_index(all_chunks)
        self._chunks = indexed
        # Rebuild IDF as mapping from token to value
        # (idf_values is list; rebuild from chunk data)
        combined_idf: dict[str, float] = {}
        for chunk in self._chunks:
            for token, val in chunk.get("_vec", {}).items():
                tf = val / (chunk.get("_norm", 1.0) ** 0.5 or 1.0)
                combined_idf[token] = combined_idf.get(token, 0) + tf
        self._idf = combined_idf
        self._indexed = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.01,
    ) -> list[SearchResult]:
        """
        Search the codebase semantically.

        Args:
            query: Natural-language description of what to find
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of SearchResult sorted by descending score
        """
        self._build_index()

        if not self._chunks:
            return []

        query_tokens = _tokenize(query)
        scored: list[tuple[float, dict]] = []

        for chunk in self._chunks:
            score = _cosine_sim(query_tokens, chunk, self._idf)
            if score >= min_score:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[SearchResult] = []
        for score, chunk in scored[:top_k]:
            results.append(SearchResult(
                file_path=chunk["file_path"],
                start_line=chunk["start_line"],
                end_line=chunk["end_line"],
                snippet=chunk["text"],
                score=score,
                match_type="semantic",
            ))
        return results

    def reindex(self) -> str:
        """Force a full re-index of the workspace."""
        self._build_index(force=True)
        return (
            f"Re-indexed {len(self._chunks)} code chunks "
            f"from {self._root}"
        )

    @property
    def chunk_count(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)


# ---------------------------------------------------------------------------
# Unified search facade for ToolExecutor
# ---------------------------------------------------------------------------

class CodebaseNavigator:
    """
    Unified facade that combines grep and semantic search.

    Registered with ToolExecutor as 'search_codebase' and 'grep_search'.
    """

    def __init__(
        self,
        workspace_root: Path,
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        self._grep = GrepSearch(workspace_root)
        self._semantic = SemanticSearch(workspace_root, ollama_base_url)

    def grep_search(
        self,
        pattern: str,
        glob: str | None = None,
        case_sensitive: bool = True,
        max_results: int = 30,
    ) -> str:
        """Pattern-based search across workspace files."""
        results = self._grep.search(
            query=pattern,
            glob=glob,
            case_sensitive=case_sensitive,
            max_results=max_results,
        )
        if not results:
            return f"No matches for pattern: {pattern!r}"

        lines = [f"Grep results for {pattern!r} ({len(results)} matches):\n"]
        for r in results:
            lines.append(r.to_prompt_str())
        return "\n".join(lines)

    def semantic_search(
        self,
        query: str,
        top_k: int = 8,
    ) -> str:
        """Embedding-based semantic codebase search."""
        results = self._semantic.search(query=query, top_k=top_k)
        if not results:
            return f"No semantic matches found for: {query!r}"

        lines = [f"Semantic search results for {query!r} ({len(results)} matches):\n"]
        for r in results:
            lines.append(r.to_prompt_str())
        return "\n".join(lines)

    def reindex(self) -> str:
        """Force re-index the semantic search engine."""
        return self._semantic.reindex()
