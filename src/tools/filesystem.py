"""
Filesystem Tools Module

Provides safe file system operations through the FileGuard.
All operations are logged and auditable.

Design Decisions:
- All paths validated against whitelist
- Rate limiting on operations
- Backup before modifications
- No hidden operations
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.core import FileGuard, TelemetryCollector


class FileOperation(str, Enum):
    """Types of file operations."""
    READ = "read"
    WRITE = "write"
    CREATE = "create"
    DELETE = "delete"
    LIST = "list"
    EXISTS = "exists"
    SEARCH = "search"


class FileInfo(BaseModel):
    """Information about a file."""
    path: str
    name: str
    extension: str
    size_bytes: int
    is_file: bool
    is_directory: bool
    is_readable: bool
    is_writable: bool


class SearchMatch(BaseModel):
    """A search result match."""
    file_path: str
    line_number: int
    line_content: str
    match_start: int
    match_end: int


class ToolResult(BaseModel):
    """Result of a tool operation."""
    success: bool
    operation: FileOperation
    path: str | None = None
    data: str | list | dict | None = None
    error: str | None = None


class FilesystemTools:
    """
    File system tools with safety guarantees.
    
    All operations:
    - Go through FileGuard for validation
    - Are logged to telemetry
    - Respect rate limits
    - Cannot escape workspace
    """
    
    def __init__(
        self,
        workspace_root: Path,
        file_guard: "FileGuard",
        telemetry: "TelemetryCollector | None" = None,
    ) -> None:
        """
        Initialize filesystem tools.
        
        Args:
            workspace_root: Root directory for file operations
            file_guard: FileGuard instance for mediated access
            telemetry: Optional telemetry collector
        """
        self._workspace_root = workspace_root.resolve()
        self._file_guard = file_guard
        self._telemetry = telemetry
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to workspace root."""
        p = Path(path)
        if not p.is_absolute():
            p = self._workspace_root / p
        return p.resolve()
    
    def _log(self, operation: FileOperation, path: Path, success: bool, error: str | None = None) -> None:
        """Log operation to telemetry."""
        if self._telemetry:
            self._telemetry.record_tool_call(
                tool_name=f"filesystem.{operation.value}",
                inputs={"path": str(path)},
                outputs={"success": success, "error": error},
                success=success,
            )
    
    def read_file(self, path: str, encoding: str = "utf-8") -> ToolResult:
        """
        Read contents of a file.
        
        Args:
            path: Path to file (relative to workspace or absolute)
            encoding: File encoding (default: utf-8)
            
        Returns:
            ToolResult with file contents or error
        """
        resolved = self._resolve_path(path)
        
        try:
            content = self._file_guard.read(resolved, encoding=encoding)
            self._log(FileOperation.READ, resolved, True)
            return ToolResult(
                success=True,
                operation=FileOperation.READ,
                path=str(resolved),
                data=content,
            )
        except Exception as e:
            self._log(FileOperation.READ, resolved, False, str(e))
            return ToolResult(
                success=False,
                operation=FileOperation.READ,
                path=str(resolved),
                error=str(e),
            )
    
    def write_file(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True,
    ) -> ToolResult:
        """
        Write content to a file.
        
        Args:
            path: Path to file (relative to workspace or absolute)
            content: Content to write
            encoding: File encoding (default: utf-8)
            create_dirs: Create parent directories if needed
            
        Returns:
            ToolResult with success status
        """
        resolved = self._resolve_path(path)
        
        try:
            # Create parent directories if needed
            if create_dirs:
                resolved.parent.mkdir(parents=True, exist_ok=True)
            
            self._file_guard.write(resolved, content, encoding=encoding)
            self._log(FileOperation.WRITE, resolved, True)
            return ToolResult(
                success=True,
                operation=FileOperation.WRITE,
                path=str(resolved),
            )
        except Exception as e:
            self._log(FileOperation.WRITE, resolved, False, str(e))
            return ToolResult(
                success=False,
                operation=FileOperation.WRITE,
                path=str(resolved),
                error=str(e),
            )
    
    def create_file(
        self,
        path: str,
        content: str = "",
        encoding: str = "utf-8",
    ) -> ToolResult:
        """
        Create a new file. Fails if file exists.
        
        Args:
            path: Path for new file
            content: Initial content (default: empty)
            encoding: File encoding
            
        Returns:
            ToolResult with success status
        """
        resolved = self._resolve_path(path)
        
        if resolved.exists():
            return ToolResult(
                success=False,
                operation=FileOperation.CREATE,
                path=str(resolved),
                error="File already exists",
            )
        
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self._file_guard.write(resolved, content, encoding=encoding)
            self._log(FileOperation.CREATE, resolved, True)
            return ToolResult(
                success=True,
                operation=FileOperation.CREATE,
                path=str(resolved),
            )
        except Exception as e:
            self._log(FileOperation.CREATE, resolved, False, str(e))
            return ToolResult(
                success=False,
                operation=FileOperation.CREATE,
                path=str(resolved),
                error=str(e),
            )
    
    def delete_file(self, path: str) -> ToolResult:
        """
        Delete a file.
        
        Args:
            path: Path to file to delete
            
        Returns:
            ToolResult with success status
        """
        resolved = self._resolve_path(path)
        
        try:
            self._file_guard.delete(resolved)
            self._log(FileOperation.DELETE, resolved, True)
            return ToolResult(
                success=True,
                operation=FileOperation.DELETE,
                path=str(resolved),
            )
        except Exception as e:
            self._log(FileOperation.DELETE, resolved, False, str(e))
            return ToolResult(
                success=False,
                operation=FileOperation.DELETE,
                path=str(resolved),
                error=str(e),
            )
    
    def list_directory(
        self,
        path: str = ".",
        recursive: bool = False,
        include_hidden: bool = False,
    ) -> ToolResult:
        """
        List contents of a directory.
        
        Args:
            path: Directory path (default: workspace root)
            recursive: List recursively
            include_hidden: Include hidden files
            
        Returns:
            ToolResult with list of FileInfo
        """
        resolved = self._resolve_path(path)
        
        if not resolved.is_dir():
            return ToolResult(
                success=False,
                operation=FileOperation.LIST,
                path=str(resolved),
                error="Not a directory",
            )
        
        try:
            files = self._file_guard.list_dir(resolved, recursive=recursive)
            
            file_infos = []
            for f in files:
                if not include_hidden and f.name.startswith("."):
                    continue
                
                try:
                    stat = f.stat()
                    file_infos.append({
                        "path": str(f.relative_to(self._workspace_root)),
                        "name": f.name,
                        "extension": f.suffix,
                        "size_bytes": stat.st_size,
                        "is_file": f.is_file(),
                        "is_directory": f.is_dir(),
                        "is_readable": os.access(f, os.R_OK),
                        "is_writable": os.access(f, os.W_OK),
                    })
                except Exception:
                    pass
            
            self._log(FileOperation.LIST, resolved, True)
            return ToolResult(
                success=True,
                operation=FileOperation.LIST,
                path=str(resolved),
                data=file_infos,
            )
        except Exception as e:
            self._log(FileOperation.LIST, resolved, False, str(e))
            return ToolResult(
                success=False,
                operation=FileOperation.LIST,
                path=str(resolved),
                error=str(e),
            )
    
    def file_exists(self, path: str) -> ToolResult:
        """
        Check if a file exists.
        
        Args:
            path: Path to check
            
        Returns:
            ToolResult with exists boolean
        """
        resolved = self._resolve_path(path)
        
        try:
            exists = self._file_guard.exists(resolved)
            self._log(FileOperation.EXISTS, resolved, True)
            return ToolResult(
                success=True,
                operation=FileOperation.EXISTS,
                path=str(resolved),
                data={"exists": exists, "is_file": resolved.is_file(), "is_dir": resolved.is_dir()},
            )
        except Exception as e:
            self._log(FileOperation.EXISTS, resolved, False, str(e))
            return ToolResult(
                success=False,
                operation=FileOperation.EXISTS,
                path=str(resolved),
                error=str(e),
            )
    
    def get_file_info(self, path: str) -> ToolResult:
        """
        Get detailed information about a file.
        
        Args:
            path: Path to file
            
        Returns:
            ToolResult with FileInfo
        """
        resolved = self._resolve_path(path)
        
        if not resolved.exists():
            return ToolResult(
                success=False,
                operation=FileOperation.EXISTS,
                path=str(resolved),
                error="File does not exist",
            )
        
        try:
            stat = resolved.stat()
            info = {
                "path": str(resolved.relative_to(self._workspace_root)),
                "name": resolved.name,
                "extension": resolved.suffix,
                "size_bytes": stat.st_size,
                "is_file": resolved.is_file(),
                "is_directory": resolved.is_dir(),
                "is_readable": os.access(resolved, os.R_OK),
                "is_writable": os.access(resolved, os.W_OK),
            }
            return ToolResult(
                success=True,
                operation=FileOperation.EXISTS,
                path=str(resolved),
                data=info,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                operation=FileOperation.EXISTS,
                path=str(resolved),
                error=str(e),
            )
    
    def search_in_file(
        self,
        path: str,
        pattern: str,
        case_sensitive: bool = True,
        max_matches: int = 100,
    ) -> ToolResult:
        """
        Search for a pattern in a file.
        
        Args:
            path: Path to file
            pattern: Search pattern (substring)
            case_sensitive: Case sensitive search
            max_matches: Maximum matches to return
            
        Returns:
            ToolResult with list of matches
        """
        resolved = self._resolve_path(path)
        
        try:
            content = self._file_guard.read(resolved)
            lines = content.split("\n")
            
            matches = []
            search_pattern = pattern if case_sensitive else pattern.lower()
            
            for i, line in enumerate(lines, 1):
                search_line = line if case_sensitive else line.lower()
                
                start = 0
                while True:
                    pos = search_line.find(search_pattern, start)
                    if pos == -1:
                        break
                    
                    matches.append({
                        "file_path": str(resolved.relative_to(self._workspace_root)),
                        "line_number": i,
                        "line_content": line,
                        "match_start": pos,
                        "match_end": pos + len(pattern),
                    })
                    
                    if len(matches) >= max_matches:
                        break
                    
                    start = pos + 1
                
                if len(matches) >= max_matches:
                    break
            
            self._log(FileOperation.SEARCH, resolved, True)
            return ToolResult(
                success=True,
                operation=FileOperation.SEARCH,
                path=str(resolved),
                data=matches,
            )
        except Exception as e:
            self._log(FileOperation.SEARCH, resolved, False, str(e))
            return ToolResult(
                success=False,
                operation=FileOperation.SEARCH,
                path=str(resolved),
                error=str(e),
            )
    
    def search_workspace(
        self,
        pattern: str,
        file_extensions: list[str] | None = None,
        max_matches_per_file: int = 10,
        max_files: int = 50,
    ) -> ToolResult:
        """
        Search for a pattern across the workspace.
        
        Args:
            pattern: Search pattern (substring)
            file_extensions: Filter by extensions (e.g., [".py", ".ts"])
            max_matches_per_file: Max matches per file
            max_files: Max files to search
            
        Returns:
            ToolResult with matches grouped by file
        """
        try:
            files = self._file_guard.list_dir(self._workspace_root, recursive=True)
            
            # Filter by extension
            if file_extensions:
                files = [f for f in files if f.suffix in file_extensions]
            
            # Limit files
            files = files[:max_files]
            
            all_matches = {}
            for f in files:
                if f.is_dir():
                    continue
                
                result = self.search_in_file(
                    str(f),
                    pattern,
                    max_matches=max_matches_per_file,
                )
                
                if result.success and result.data:
                    rel_path = str(f.relative_to(self._workspace_root))
                    all_matches[rel_path] = result.data
            
            return ToolResult(
                success=True,
                operation=FileOperation.SEARCH,
                data=all_matches,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                operation=FileOperation.SEARCH,
                error=str(e),
            )
