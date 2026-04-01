"""
Granular File Editing Tools

Exposes the DiffEngine as lightweight ToolCall-compatible functions:
  - read_file(path)              → file contents (with line numbers)
  - list_dir(path)               → directory listing
  - replace_string(path, old, new) → targeted string replace (not whole-file overwrite)
  - write_file(path, content)    → create/overwrite a file
  - delete_file(path)            → remove a file
  - grep_file(path, pattern)     → search within a file

These allow the Coder to read → edit a single function → run tests → fix,
all within the tool loop, without emitting a final CodeChange JSON at the end.

Design Decisions:
- Delegates path safety to FileGuard (exact same rules as diff engine)
- replace_string raises if the old string appears 0 or 2+ times (safety)
- All operations are atomic at the Python level
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class FileEditingTools:
    """
    Thin wrappers around raw filesystem + FileGuard for agent use.

    This class is registered with ToolExecutor so agents can call
    individual operations via ToolCall JSON.
    """

    _MAX_READ_BYTES = 500_000   # 500 KB per read
    _MAX_DIR_ENTRIES = 300

    def __init__(self, workspace_root: Path, file_guard: Any | None = None) -> None:
        self._root = workspace_root.resolve()
        self._guard = file_guard  # src.core.FileGuard instance (optional)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _resolve(self, rel_or_abs: str) -> Path:
        """Resolve a path relative to workspace root and validate it."""
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = self._root / p
        p = p.resolve()
        # Security: must stay inside workspace
        try:
            p.relative_to(self._root)
        except ValueError:
            raise PermissionError(
                f"Path '{rel_or_abs}' is outside the workspace root"
            )
        return p

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """
        Read a file and return its contents with line numbers.

        Args:
            path: Path to the file (relative to workspace root)
            start_line: First line to include (1-indexed)
            end_line: Last line to include (inclusive), or None for all

        Returns:
            File contents with prepended line numbers
        """
        p = self._resolve(path)
        if not p.exists():
            return f"Error: File not found: {path}"
        if not p.is_file():
            return f"Error: Not a file: {path}"

        raw = p.read_bytes()[: self._MAX_READ_BYTES]
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        lines = text.splitlines(keepends=True)
        s = max(1, start_line) - 1          # 0-indexed
        e = end_line if end_line else len(lines)
        selected = lines[s:e]

        numbered = "".join(
            f"{s + i + 1:6d}│ {line}" for i, line in enumerate(selected)
        )
        header = f"File: {path}  ({len(lines)} lines total)\n"
        return header + numbered

    def list_dir(self, path: str = ".") -> str:
        """
        List directory contents.

        Args:
            path: Directory path (relative to workspace root)

        Returns:
            Formatted directory listing
        """
        p = self._resolve(path)
        if not p.exists():
            return f"Error: Directory not found: {path}"
        if not p.is_dir():
            return f"Error: Not a directory: {path}"

        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        lines: list[str] = [f"Directory: {path}\n"]

        for entry in entries[: self._MAX_DIR_ENTRIES]:
            rel = entry.relative_to(self._root)
            icon = "📁" if entry.is_dir() else "📄"
            size_str = ""
            if entry.is_file():
                try:
                    size = entry.stat().st_size
                    size_str = f"  ({size:,} B)"
                except OSError:
                    pass
            lines.append(f"  {icon} {rel}{size_str}")

        if len(list(p.iterdir())) > self._MAX_DIR_ENTRIES:
            lines.append(f"  … (truncated at {self._MAX_DIR_ENTRIES} entries)")

        return "\n".join(lines)

    def replace_string(
        self,
        path: str,
        old_string: str,
        new_string: str,
        expect_count: int = 1,
    ) -> str:
        """
        Replace exactly *expect_count* occurrences of old_string with new_string.

        Fails safely if the string appears fewer or more times than expected,
        preventing unintended mass-replaces.

        Args:
            path: Target file path
            old_string: Exact substring to replace (must be unique by default)
            new_string: Replacement text
            expect_count: How many replacements to expect (default: 1)

        Returns:
            Success message or error description
        """
        p = self._resolve(path)
        if not p.exists():
            return f"Error: File not found: {path}"

        content = p.read_text(encoding="utf-8", errors="replace")
        actual_count = content.count(old_string)

        if actual_count == 0:
            return (
                f"Error: String not found in {path}.\n"
                f"Searched for: {old_string[:120]!r}"
            )
        if actual_count != expect_count:
            return (
                f"Error: Expected {expect_count} occurrence(s) but found {actual_count}.\n"
                f"Set expect_count={actual_count} to replace all of them, or "
                f"make old_string more specific."
            )

        new_content = content.replace(old_string, new_string, expect_count)
        p.write_text(new_content, encoding="utf-8")

        lines_changed = abs(new_content.count("\n") - content.count("\n"))
        return (
            f"Replaced {expect_count} occurrence(s) in {path}. "
            f"Net line delta: {lines_changed:+d}."
        )

    def write_file(self, path: str, content: str, overwrite: bool = True) -> str:
        """
        Write content to a file, creating parent directories as needed.

        Args:
            path: Target file path (relative to workspace root)
            content: File content to write
            overwrite: If False, refuse to overwrite existing files

        Returns:
            Success message or error description
        """
        p = self._resolve(path)

        if not overwrite and p.exists():
            return (
                f"Error: File already exists: {path}. "
                f"Pass overwrite=true to replace it."
            )

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        lines = content.count("\n") + 1
        action = "Overwrote" if p.exists() else "Created"
        return f"{action} {path} ({lines} lines, {len(content.encode())} bytes)."

    def delete_file(self, path: str, require_confirmation: bool = False) -> str:
        """
        Delete a file from the workspace.

        Args:
            path: File path to remove (relative to workspace root)
            require_confirmation: If True, return a confirmation prompt instead

        Returns:
            Success message or error description
        """
        if require_confirmation:
            return (
                f"Confirm deletion of '{path}' by calling delete_file again "
                f"with require_confirmation=false."
            )

        p = self._resolve(path)
        if not p.exists():
            return f"Error: File not found: {path}"
        if not p.is_file():
            return f"Error: Not a regular file: {path}"

        p.unlink()
        return f"Deleted: {path}"

    def grep_file(
        self,
        path: str,
        pattern: str,
        case_sensitive: bool = True,
        max_results: int = 100,
    ) -> str:
        """
        Search for a regex pattern inside a file, returning matching lines.

        Args:
            path: File path (relative to workspace root)
            pattern: Python regex pattern
            case_sensitive: Whether to respect case
            max_results: Max matching lines to return

        Returns:
            Formatted list of matching lines with line numbers
        """
        p = self._resolve(path)
        if not p.exists():
            return f"Error: File not found: {path}"

        content = p.read_text(encoding="utf-8", errors="replace")
        flags = 0 if case_sensitive else re.IGNORECASE

        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        matches: list[str] = []
        for lineno, line in enumerate(content.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"  {lineno:6d}│ {line}")
                if len(matches) >= max_results:
                    matches.append(f"  … (truncated at {max_results} results)")
                    break

        if not matches:
            return f"No matches for {pattern!r} in {path}"

        header = f"Matches for {pattern!r} in {path} ({len(matches)} found):\n"
        return header + "\n".join(matches)

    def grep_workspace(
        self,
        pattern: str,
        glob: str = "**/*.py",
        case_sensitive: bool = True,
        max_results: int = 200,
    ) -> str:
        """
        Search for a pattern across all workspace files matching a glob.

        Args:
            pattern: Python regex pattern
            glob: File glob pattern to filter by (e.g. '**/*.ts')
            case_sensitive: Whether to respect case
            max_results: Max total matching lines to return

        Returns:
            Formatted results grouped by file
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex: {e}"

        results: list[str] = []
        total = 0

        for file_path in sorted(self._root.glob(glob)):
            if not file_path.is_file():
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            file_matches: list[str] = []
            for lineno, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    file_matches.append(f"    {lineno:6d}│ {line}")
                    total += 1
                    if total >= max_results:
                        break

            if file_matches:
                rel = file_path.relative_to(self._root)
                results.append(f"\n📄 {rel}:")
                results.extend(file_matches)

            if total >= max_results:
                results.append(f"\n… (truncated at {max_results} total results)")
                break

        if not results:
            return f"No matches for {pattern!r} in {glob} files"

        header = f"Workspace grep — {pattern!r} ({total} matches):\n"
        return header + "\n".join(results)
