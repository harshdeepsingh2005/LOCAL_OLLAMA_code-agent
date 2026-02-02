"""
File Guard Module

Mediates all file system access to enforce safety policies.
No agent can directly read/write files - all access goes through this guard.

Design Decisions:
- Whitelist-based path validation
- Explicit sandbox boundaries
- Operation audit trail
- No path traversal attacks
- Rate limiting for file operations
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class FileOperation(str, Enum):
    """Types of file operations."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    CREATE = "create"
    MODIFY = "modify"
    LIST = "list"
    STAT = "stat"
    COPY = "copy"
    MOVE = "move"


class AccessDeniedError(Exception):
    """Raised when file access is denied."""
    pass


class PathViolationError(Exception):
    """Raised when path escapes sandbox."""
    pass


class RateLimitError(Exception):
    """Raised when operation rate limit is exceeded."""
    pass


class FileLimitError(Exception):
    """Raised when file count limit is exceeded."""
    pass


class FileGuardPolicy(BaseModel):
    """Policy configuration for file guard."""
    # Sandbox configuration
    allowed_roots: list[Path] = Field(default_factory=list)
    blocked_patterns: list[str] = Field(
        default_factory=lambda: [
            "*.pem", "*.key", "*.env", "*.secret*",
            "*password*", "*credential*", "*.ssh/*",
            "__pycache__", "*.pyc", ".git/objects/*",
        ]
    )
    
    # Limits
    max_file_size_bytes: int = Field(default=10 * 1024 * 1024)  # 10MB
    max_files_per_operation: int = Field(default=10)
    max_files_per_run: int = Field(default=50)
    max_operations_per_minute: int = Field(default=100)
    
    # Allowed extensions for modification
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [
            ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml",
            ".md", ".txt", ".html", ".css", ".sql", ".sh", ".bash",
            ".toml", ".ini", ".cfg", ".conf", ".xml", ".rs", ".go",
        ]
    )
    
    # Read-only paths (can read but not modify)
    read_only_paths: list[str] = Field(default_factory=list)


@dataclass
class FileAccessRecord:
    """Record of a file access operation."""
    timestamp: datetime
    operation: FileOperation
    path: Path
    success: bool
    size_bytes: int | None = None
    hash: str | None = None
    error: str | None = None


@dataclass
class FileGuardState:
    """State tracking for file guard."""
    files_modified: set[Path] = field(default_factory=set)
    files_created: set[Path] = field(default_factory=set)
    files_deleted: set[Path] = field(default_factory=set)
    access_records: list[FileAccessRecord] = field(default_factory=list)
    operations_this_minute: int = 0
    last_minute_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FileGuard:
    """
    Mediates all file system access with policy enforcement.
    
    This is the ONLY authorized way for agents to interact with
    the file system. Direct file access is prohibited.
    
    Thread Safety: NOT thread-safe. Designed for sequential execution.
    """
    
    def __init__(
        self,
        workspace_root: Path,
        policy: FileGuardPolicy | None = None,
        telemetry: Any | None = None,  # TelemetryCollector
    ) -> None:
        """
        Initialize the file guard.
        
        Args:
            workspace_root: Root directory for file operations
            policy: File guard policy configuration
            telemetry: Optional telemetry collector
        """
        self._workspace_root = workspace_root.resolve()
        self._policy = policy or FileGuardPolicy(allowed_roots=[self._workspace_root])
        self._telemetry = telemetry
        self._state = FileGuardState()
        
        # Ensure workspace exists
        self._workspace_root.mkdir(parents=True, exist_ok=True)
        
        # Add workspace to allowed roots if not present
        if self._workspace_root not in self._policy.allowed_roots:
            self._policy.allowed_roots.append(self._workspace_root)
    
    def _check_rate_limit(self) -> None:
        """Check and enforce operation rate limiting."""
        now = datetime.now(timezone.utc)
        
        # Reset counter if minute has passed
        elapsed = (now - self._state.last_minute_reset).total_seconds()
        if elapsed >= 60:
            self._state.operations_this_minute = 0
            self._state.last_minute_reset = now
        
        # Check limit
        if self._state.operations_this_minute >= self._policy.max_operations_per_minute:
            raise RateLimitError(
                f"Rate limit exceeded: {self._policy.max_operations_per_minute} ops/minute"
            )
        
        self._state.operations_this_minute += 1
    
    def _validate_path(self, path: Path, operation: FileOperation) -> Path:
        """
        Validate and resolve a path against security policies.
        
        Args:
            path: Path to validate
            operation: Operation being performed
            
        Returns:
            Resolved, validated path
            
        Raises:
            PathViolationError: If path escapes sandbox
            AccessDeniedError: If path matches blocked pattern
        """
        # Resolve to absolute path
        if not path.is_absolute():
            resolved = (self._workspace_root / path).resolve()
        else:
            resolved = path.resolve()
        
        # Check if within allowed roots
        in_allowed_root = any(
            self._is_subpath(resolved, root.resolve())
            for root in self._policy.allowed_roots
        )
        
        if not in_allowed_root:
            raise PathViolationError(
                f"Path '{resolved}' is outside allowed roots"
            )
        
        # Check blocked patterns
        path_str = str(resolved)
        for pattern in self._policy.blocked_patterns:
            if self._matches_pattern(path_str, pattern):
                raise AccessDeniedError(
                    f"Path '{resolved}' matches blocked pattern '{pattern}'"
                )
        
        # Check read-only for write operations
        if operation in (FileOperation.WRITE, FileOperation.MODIFY, 
                        FileOperation.DELETE, FileOperation.CREATE):
            for read_only in self._policy.read_only_paths:
                if self._matches_pattern(path_str, read_only):
                    raise AccessDeniedError(
                        f"Path '{resolved}' is read-only"
                    )
            
            # Check extension for write operations
            if resolved.suffix and resolved.suffix not in self._policy.allowed_extensions:
                raise AccessDeniedError(
                    f"Extension '{resolved.suffix}' not allowed for modification"
                )
        
        return resolved
    
    def _is_subpath(self, path: Path, parent: Path) -> bool:
        """Check if path is a subpath of parent."""
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False
    
    def _matches_pattern(self, path_str: str, pattern: str) -> bool:
        """Simple glob-like pattern matching."""
        import fnmatch
        return fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(
            os.path.basename(path_str), pattern
        )
    
    def _record_access(
        self,
        operation: FileOperation,
        path: Path,
        success: bool,
        size_bytes: int | None = None,
        content_hash: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record a file access operation."""
        record = FileAccessRecord(
            timestamp=datetime.now(timezone.utc),
            operation=operation,
            path=path,
            success=success,
            size_bytes=size_bytes,
            hash=content_hash,
            error=error,
        )
        self._state.access_records.append(record)
        
        # Update state tracking
        if success:
            if operation == FileOperation.CREATE:
                self._state.files_created.add(path)
            elif operation in (FileOperation.WRITE, FileOperation.MODIFY):
                self._state.files_modified.add(path)
            elif operation == FileOperation.DELETE:
                self._state.files_deleted.add(path)
        
        # Telemetry
        if self._telemetry:
            self._telemetry.record_file_access(
                operation=operation.value,
                path=str(path),
                success=success,
            )
    
    def _compute_hash(self, content: bytes) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content).hexdigest()[:16]
    
    def _check_file_limits(self) -> None:
        """Check if file limits have been reached."""
        total_modified = len(
            self._state.files_modified | 
            self._state.files_created | 
            self._state.files_deleted
        )
        if total_modified >= self._policy.max_files_per_run:
            raise FileLimitError(
                f"File limit exceeded: {self._policy.max_files_per_run} files per run"
            )
    
    def read(self, path: Path) -> str:
        """
        Read file contents as text.
        
        Args:
            path: Path to read
            
        Returns:
            File contents as string
            
        Raises:
            PathViolationError: If path escapes sandbox
            AccessDeniedError: If access is denied
            FileNotFoundError: If file doesn't exist
        """
        self._check_rate_limit()
        resolved = self._validate_path(path, FileOperation.READ)
        
        try:
            content = resolved.read_text(encoding="utf-8")
            size = len(content.encode("utf-8"))
            
            if size > self._policy.max_file_size_bytes:
                raise AccessDeniedError(
                    f"File too large: {size} bytes (max: {self._policy.max_file_size_bytes})"
                )
            
            self._record_access(
                FileOperation.READ, resolved, True,
                size_bytes=size,
                content_hash=self._compute_hash(content.encode("utf-8")),
            )
            return content
        except UnicodeDecodeError as e:
            self._record_access(FileOperation.READ, resolved, False, error=str(e))
            raise AccessDeniedError(f"Cannot read binary file: {resolved}") from e
        except FileNotFoundError:
            self._record_access(FileOperation.READ, resolved, False, error="File not found")
            raise
    
    def read_bytes(self, path: Path) -> bytes:
        """
        Read file contents as bytes.
        
        Args:
            path: Path to read
            
        Returns:
            File contents as bytes
        """
        self._check_rate_limit()
        resolved = self._validate_path(path, FileOperation.READ)
        
        try:
            content = resolved.read_bytes()
            size = len(content)
            
            if size > self._policy.max_file_size_bytes:
                raise AccessDeniedError(
                    f"File too large: {size} bytes (max: {self._policy.max_file_size_bytes})"
                )
            
            self._record_access(
                FileOperation.READ, resolved, True,
                size_bytes=size,
                content_hash=self._compute_hash(content),
            )
            return content
        except FileNotFoundError:
            self._record_access(FileOperation.READ, resolved, False, error="File not found")
            raise
    
    def write(self, path: Path, content: str) -> None:
        """
        Write text content to file (creates or overwrites).
        
        Args:
            path: Path to write
            content: Content to write
        """
        self._check_rate_limit()
        self._check_file_limits()
        
        content_bytes = content.encode("utf-8")
        if len(content_bytes) > self._policy.max_file_size_bytes:
            raise AccessDeniedError(
                f"Content too large: {len(content_bytes)} bytes"
            )
        
        is_create = not path.exists() if path.is_absolute() else not (self._workspace_root / path).exists()
        operation = FileOperation.CREATE if is_create else FileOperation.WRITE
        
        resolved = self._validate_path(path, operation)
        
        try:
            # Ensure parent directory exists
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            
            self._record_access(
                operation, resolved, True,
                size_bytes=len(content_bytes),
                content_hash=self._compute_hash(content_bytes),
            )
        except Exception as e:
            self._record_access(operation, resolved, False, error=str(e))
            raise
    
    def delete(self, path: Path) -> None:
        """
        Delete a file.
        
        Args:
            path: Path to delete
        """
        self._check_rate_limit()
        resolved = self._validate_path(path, FileOperation.DELETE)
        
        try:
            if resolved.is_dir():
                raise AccessDeniedError("Cannot delete directories")
            
            resolved.unlink()
            self._record_access(FileOperation.DELETE, resolved, True)
        except FileNotFoundError:
            self._record_access(FileOperation.DELETE, resolved, False, error="File not found")
            raise
        except Exception as e:
            self._record_access(FileOperation.DELETE, resolved, False, error=str(e))
            raise
    
    def exists(self, path: Path) -> bool:
        """
        Check if path exists.
        
        Args:
            path: Path to check
            
        Returns:
            True if path exists
        """
        self._check_rate_limit()
        resolved = self._validate_path(path, FileOperation.STAT)
        result = resolved.exists()
        self._record_access(FileOperation.STAT, resolved, True)
        return result
    
    def list_dir(self, path: Path) -> list[Path]:
        """
        List directory contents.
        
        Args:
            path: Directory path
            
        Returns:
            List of paths in directory
        """
        self._check_rate_limit()
        resolved = self._validate_path(path, FileOperation.LIST)
        
        try:
            entries = list(resolved.iterdir())
            
            # Filter out blocked patterns
            filtered = []
            for entry in entries:
                try:
                    self._validate_path(entry, FileOperation.LIST)
                    filtered.append(entry)
                except (AccessDeniedError, PathViolationError):
                    continue
            
            self._record_access(FileOperation.LIST, resolved, True)
            return filtered
        except Exception as e:
            self._record_access(FileOperation.LIST, resolved, False, error=str(e))
            raise
    
    def copy(self, src: Path, dst: Path) -> None:
        """
        Copy a file.
        
        Args:
            src: Source path
            dst: Destination path
        """
        self._check_rate_limit()
        self._check_file_limits()
        
        resolved_src = self._validate_path(src, FileOperation.READ)
        resolved_dst = self._validate_path(dst, FileOperation.CREATE)
        
        try:
            shutil.copy2(resolved_src, resolved_dst)
            self._record_access(FileOperation.COPY, resolved_dst, True)
        except Exception as e:
            self._record_access(FileOperation.COPY, resolved_dst, False, error=str(e))
            raise
    
    def move(self, src: Path, dst: Path) -> None:
        """
        Move a file.
        
        Args:
            src: Source path
            dst: Destination path
        """
        self._check_rate_limit()
        
        resolved_src = self._validate_path(src, FileOperation.DELETE)
        resolved_dst = self._validate_path(dst, FileOperation.CREATE)
        
        try:
            shutil.move(str(resolved_src), str(resolved_dst))
            self._record_access(FileOperation.MOVE, resolved_dst, True)
            
            # Update state: moved file is deleted from old, created at new
            self._state.files_deleted.add(resolved_src)
            self._state.files_created.add(resolved_dst)
        except Exception as e:
            self._record_access(FileOperation.MOVE, resolved_dst, False, error=str(e))
            raise
    
    def get_file_info(self, path: Path) -> dict[str, Any]:
        """
        Get file metadata.
        
        Args:
            path: Path to get info for
            
        Returns:
            Dictionary with file information
        """
        self._check_rate_limit()
        resolved = self._validate_path(path, FileOperation.STAT)
        
        try:
            stat = resolved.stat()
            info = {
                "path": str(resolved),
                "name": resolved.name,
                "extension": resolved.suffix,
                "size_bytes": stat.st_size,
                "is_file": resolved.is_file(),
                "is_dir": resolved.is_dir(),
                "modified_time": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "created_time": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
            }
            self._record_access(FileOperation.STAT, resolved, True)
            return info
        except Exception as e:
            self._record_access(FileOperation.STAT, resolved, False, error=str(e))
            raise
    
    def create_backup(self, path: Path) -> Path:
        """
        Create a backup of a file.
        
        Args:
            path: Path to backup
            
        Returns:
            Path to backup file
        """
        resolved = self._validate_path(path, FileOperation.READ)
        
        if not resolved.exists():
            raise FileNotFoundError(f"Cannot backup non-existent file: {resolved}")
        
        # Create backup in temp directory
        backup_dir = self._workspace_root / ".backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"{resolved.stem}_{timestamp}{resolved.suffix}.bak"
        backup_path = backup_dir / backup_name
        
        content = resolved.read_bytes()
        backup_path.write_bytes(content)
        
        return backup_path
    
    @property
    def state(self) -> FileGuardState:
        """Get current file guard state."""
        return self._state
    
    @property
    def access_records(self) -> list[FileAccessRecord]:
        """Get all access records."""
        return self._state.access_records.copy()
    
    def get_modified_files(self) -> set[Path]:
        """Get set of files modified during this session."""
        return self._state.files_modified | self._state.files_created
    
    def reset_state(self) -> None:
        """Reset state tracking (for new run)."""
        self._state = FileGuardState()
