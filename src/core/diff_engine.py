"""
Diff Engine Module

Handles diff-based file modifications with validation and preview capabilities.
All file edits MUST go through this engine to ensure reversibility.

Design Decisions:
- Unified diff format for readability
- Pre-application validation
- Dry-run support
- Atomic operations
- Full reversibility
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from diff_match_patch import diff_match_patch
from pydantic import BaseModel, ConfigDict, Field


class DiffType(str, Enum):
    """Types of diff operations."""
    ADD = "add"
    DELETE = "delete"
    MODIFY = "modify"
    REPLACE = "replace"


class DiffValidationError(Exception):
    """Raised when diff validation fails."""
    pass


class DiffApplicationError(Exception):
    """Raised when diff application fails."""
    pass


class DiffHunk(BaseModel):
    """A single hunk (change region) in a diff."""
    start_line: int = Field(..., ge=0)
    end_line: int = Field(..., ge=0)
    original_content: str
    new_content: str
    context_before: list[str] = Field(default_factory=list)
    context_after: list[str] = Field(default_factory=list)


class FileDiff(BaseModel):
    """
    A complete diff for a single file.
    
    Contains all changes to be applied to a file along with
    metadata for validation and auditing.
    """
    file_path: Path
    diff_type: DiffType
    original_hash: str | None = None
    hunks: list[DiffHunk] = Field(default_factory=list)
    
    # Full content for create/replace operations
    full_new_content: str | None = None
    
    # Metadata
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @property
    def lines_added(self) -> int:
        """Count of lines added."""
        if self.full_new_content:
            return len(self.full_new_content.splitlines())
        return sum(
            len(h.new_content.splitlines()) - len(h.original_content.splitlines())
            for h in self.hunks
            if len(h.new_content.splitlines()) > len(h.original_content.splitlines())
        )
    
    @property
    def lines_removed(self) -> int:
        """Count of lines removed."""
        if self.diff_type == DiffType.DELETE:
            return 0  # Whole file deleted
        return sum(
            len(h.original_content.splitlines()) - len(h.new_content.splitlines())
            for h in self.hunks
            if len(h.original_content.splitlines()) > len(h.new_content.splitlines())
        )


@dataclass
class DiffResult:
    """Result of a diff application."""
    success: bool
    file_path: Path
    diff_type: DiffType
    lines_added: int
    lines_removed: int
    backup_path: Path | None = None
    error: str | None = None
    new_hash: str | None = None


@dataclass
class DiffPreview:
    """Preview of what a diff would change."""
    file_path: Path
    current_content: str | None
    new_content: str
    unified_diff: str
    lines_added: int
    lines_removed: int
    is_valid: bool
    validation_errors: list[str] = field(default_factory=list)


class DiffEngine:
    """
    Engine for creating, validating, and applying file diffs.
    
    This is the ONLY way file modifications should happen.
    All edits are diff-based to ensure reversibility.
    
    Thread Safety: NOT thread-safe. Designed for sequential execution.
    """
    
    # Maximum diff size limits
    MAX_HUNKS_PER_FILE = 50
    MAX_LINES_PER_HUNK = 500
    MAX_TOTAL_LINES_CHANGED = 1000
    
    def __init__(
        self,
        file_guard: Any,  # FileGuard
        telemetry: Any | None = None,  # TelemetryCollector
    ) -> None:
        """
        Initialize the diff engine.
        
        Args:
            file_guard: FileGuard instance for file access
            telemetry: Optional telemetry collector
        """
        self._file_guard = file_guard
        self._telemetry = telemetry
        self._dmp = diff_match_patch()
        
        # Applied diffs for rollback
        self._applied_diffs: list[tuple[FileDiff, str]] = []  # (diff, original_content)
    
    def _compute_hash(self, content: str) -> str:
        """Compute hash of content for verification."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    
    def create_diff(
        self,
        file_path: Path,
        new_content: str,
        description: str = "",
    ) -> FileDiff:
        """
        Create a diff from current file content to new content.
        
        Args:
            file_path: Path to the file
            new_content: New content for the file
            description: Human-readable description of the change
            
        Returns:
            FileDiff representing the change
        """
        # Check if file exists
        try:
            current_content = self._file_guard.read(file_path)
            diff_type = DiffType.MODIFY
            original_hash = self._compute_hash(current_content)
        except FileNotFoundError:
            current_content = ""
            diff_type = DiffType.ADD
            original_hash = None
        
        # Create hunks from differences
        hunks = self._create_hunks(current_content, new_content)
        
        return FileDiff(
            file_path=file_path,
            diff_type=diff_type,
            original_hash=original_hash,
            hunks=hunks,
            full_new_content=new_content,
            description=description,
        )
    
    def create_deletion_diff(
        self,
        file_path: Path,
        description: str = "",
    ) -> FileDiff:
        """
        Create a diff for file deletion.
        
        Args:
            file_path: Path to the file
            description: Human-readable description
            
        Returns:
            FileDiff for deletion
        """
        current_content = self._file_guard.read(file_path)
        original_hash = self._compute_hash(current_content)
        
        return FileDiff(
            file_path=file_path,
            diff_type=DiffType.DELETE,
            original_hash=original_hash,
            hunks=[],
            description=description,
        )
    
    def _create_hunks(
        self,
        original: str,
        new: str,
    ) -> list[DiffHunk]:
        """Create hunks from original and new content."""
        if not original and not new:
            return []
        
        if not original:
            # New file
            return [DiffHunk(
                start_line=0,
                end_line=0,
                original_content="",
                new_content=new,
            )]
        
        if not new:
            # Deletion
            return [DiffHunk(
                start_line=0,
                end_line=len(original.splitlines()),
                original_content=original,
                new_content="",
            )]
        
        # Use diff-match-patch for semantic diff
        diffs = self._dmp.diff_main(original, new)
        self._dmp.diff_cleanupSemantic(diffs)
        
        # Convert to line-based hunks
        hunks: list[DiffHunk] = []
        original_lines = original.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        
        # Simple approach: if there are changes, create a single hunk
        # More sophisticated: group consecutive changes into hunks
        if original != new:
            hunks.append(DiffHunk(
                start_line=0,
                end_line=len(original_lines),
                original_content=original,
                new_content=new,
                context_before=[],
                context_after=[],
            ))
        
        return hunks
    
    def validate_diff(
        self,
        diff: FileDiff,
        strict: bool = True,
    ) -> tuple[bool, list[str]]:
        """
        Validate a diff before application.
        
        Args:
            diff: FileDiff to validate
            strict: Enable strict validation
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors: list[str] = []
        
        # Check hunk count limit
        if len(diff.hunks) > self.MAX_HUNKS_PER_FILE:
            errors.append(
                f"Too many hunks: {len(diff.hunks)} > {self.MAX_HUNKS_PER_FILE}"
            )
        
        # Check lines changed limit
        total_lines = diff.lines_added + diff.lines_removed
        if total_lines > self.MAX_TOTAL_LINES_CHANGED:
            errors.append(
                f"Too many lines changed: {total_lines} > {self.MAX_TOTAL_LINES_CHANGED}"
            )
        
        # Check per-hunk limits
        for i, hunk in enumerate(diff.hunks):
            hunk_lines = max(
                len(hunk.original_content.splitlines()),
                len(hunk.new_content.splitlines())
            )
            if hunk_lines > self.MAX_LINES_PER_HUNK:
                errors.append(
                    f"Hunk {i} too large: {hunk_lines} > {self.MAX_LINES_PER_HUNK}"
                )
        
        # Verify file hash if modifying
        if strict and diff.diff_type == DiffType.MODIFY:
            try:
                current_content = self._file_guard.read(diff.file_path)
                current_hash = self._compute_hash(current_content)
                if diff.original_hash and current_hash != diff.original_hash:
                    errors.append(
                        "File has been modified since diff was created "
                        f"(expected hash {diff.original_hash}, got {current_hash})"
                    )
            except FileNotFoundError:
                errors.append("File no longer exists")
        
        # Check file exists for deletion
        if diff.diff_type == DiffType.DELETE:
            if not self._file_guard.exists(diff.file_path):
                errors.append("Cannot delete non-existent file")
        
        # Check new content is valid
        if diff.full_new_content is not None:
            # Basic syntax checks could go here
            pass
        
        return len(errors) == 0, errors
    
    def preview(self, diff: FileDiff) -> DiffPreview:
        """
        Generate a preview of what the diff would change.
        
        Args:
            diff: FileDiff to preview
            
        Returns:
            DiffPreview with current and new content
        """
        # Get current content
        try:
            current_content = self._file_guard.read(diff.file_path)
        except FileNotFoundError:
            current_content = None
        
        # Determine new content
        if diff.diff_type == DiffType.DELETE:
            new_content = ""
        elif diff.full_new_content is not None:
            new_content = diff.full_new_content
        else:
            # Apply hunks to reconstruct
            new_content = self._apply_hunks(current_content or "", diff.hunks)
        
        # Generate unified diff
        unified = self._generate_unified_diff(
            str(diff.file_path),
            current_content or "",
            new_content,
        )
        
        # Validate
        is_valid, errors = self.validate_diff(diff)
        
        return DiffPreview(
            file_path=diff.file_path,
            current_content=current_content,
            new_content=new_content,
            unified_diff=unified,
            lines_added=diff.lines_added,
            lines_removed=diff.lines_removed,
            is_valid=is_valid,
            validation_errors=errors,
        )
    
    def _generate_unified_diff(
        self,
        file_name: str,
        original: str,
        new: str,
    ) -> str:
        """Generate unified diff format."""
        import difflib
        
        original_lines = original.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        
        diff_lines = difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=f"a/{file_name}",
            tofile=f"b/{file_name}",
            lineterm="",
        )
        
        return "".join(diff_lines)
    
    def _apply_hunks(self, content: str, hunks: list[DiffHunk]) -> str:
        """Apply hunks to content."""
        if not hunks:
            return content
        
        # For simplicity, if we have full new content in first hunk, use it
        if len(hunks) == 1 and hunks[0].start_line == 0:
            return hunks[0].new_content
        
        # Otherwise, apply hunks in order
        lines = content.splitlines(keepends=True)
        
        # Apply hunks in reverse order to preserve line numbers
        for hunk in reversed(hunks):
            # Replace lines
            new_lines = hunk.new_content.splitlines(keepends=True)
            lines[hunk.start_line:hunk.end_line] = new_lines
        
        return "".join(lines)
    
    def apply(
        self,
        diff: FileDiff,
        create_backup: bool = True,
        validate: bool = True,
    ) -> DiffResult:
        """
        Apply a diff to the file system.
        
        Args:
            diff: FileDiff to apply
            create_backup: Whether to create a backup before applying
            validate: Whether to validate before applying
            
        Returns:
            DiffResult with application status
            
        Raises:
            DiffValidationError: If validation fails
            DiffApplicationError: If application fails
        """
        # Validate first
        if validate:
            is_valid, errors = self.validate_diff(diff)
            if not is_valid:
                raise DiffValidationError(
                    f"Diff validation failed: {'; '.join(errors)}"
                )
        
        backup_path = None
        original_content = None
        
        try:
            # Create backup if requested and file exists
            if create_backup and diff.diff_type in (DiffType.MODIFY, DiffType.DELETE):
                if self._file_guard.exists(diff.file_path):
                    backup_path = self._file_guard.create_backup(diff.file_path)
                    original_content = self._file_guard.read(diff.file_path)
            
            # Apply the diff
            if diff.diff_type == DiffType.DELETE:
                self._file_guard.delete(diff.file_path)
                new_content = ""
            else:
                # Determine new content
                if diff.full_new_content is not None:
                    new_content = diff.full_new_content
                else:
                    current = ""
                    if self._file_guard.exists(diff.file_path):
                        current = self._file_guard.read(diff.file_path)
                    new_content = self._apply_hunks(current, diff.hunks)
                
                # Write new content
                self._file_guard.write(diff.file_path, new_content)
            
            # Store for rollback
            self._applied_diffs.append((diff, original_content or ""))
            
            # Record telemetry
            if self._telemetry:
                self._telemetry.record_diff_apply(
                    file_path=str(diff.file_path),
                    lines_added=diff.lines_added,
                    lines_removed=diff.lines_removed,
                    success=True,
                )
            
            return DiffResult(
                success=True,
                file_path=diff.file_path,
                diff_type=diff.diff_type,
                lines_added=diff.lines_added,
                lines_removed=diff.lines_removed,
                backup_path=backup_path,
                new_hash=self._compute_hash(new_content) if new_content else None,
            )
            
        except Exception as e:
            # Record telemetry
            if self._telemetry:
                self._telemetry.record_diff_apply(
                    file_path=str(diff.file_path),
                    lines_added=diff.lines_added,
                    lines_removed=diff.lines_removed,
                    success=False,
                )
            
            return DiffResult(
                success=False,
                file_path=diff.file_path,
                diff_type=diff.diff_type,
                lines_added=0,
                lines_removed=0,
                backup_path=backup_path,
                error=str(e),
            )
    
    def apply_multiple(
        self,
        diffs: list[FileDiff],
        atomic: bool = True,
    ) -> list[DiffResult]:
        """
        Apply multiple diffs.
        
        Args:
            diffs: List of FileDiffs to apply
            atomic: If True, rollback all on any failure
            
        Returns:
            List of DiffResults
        """
        results: list[DiffResult] = []
        
        for diff in diffs:
            result = self.apply(diff)
            results.append(result)
            
            if not result.success and atomic:
                # Rollback previously applied diffs
                for prev_result in reversed(results[:-1]):
                    if prev_result.success:
                        self.rollback_last()
                break
        
        return results
    
    def rollback_last(self) -> bool:
        """
        Rollback the last applied diff.
        
        Returns:
            True if rollback succeeded
        """
        if not self._applied_diffs:
            return False
        
        diff, original_content = self._applied_diffs.pop()
        
        try:
            if diff.diff_type == DiffType.DELETE:
                # Restore deleted file
                self._file_guard.write(diff.file_path, original_content)
            elif diff.diff_type == DiffType.ADD:
                # Delete created file
                self._file_guard.delete(diff.file_path)
            else:
                # Restore original content
                self._file_guard.write(diff.file_path, original_content)
            
            if self._telemetry:
                self._telemetry.record_rollback(
                    checkpoint_id="diff",
                    reason=f"Rollback diff on {diff.file_path}",
                )
            
            return True
        except Exception:
            return False
    
    def rollback_all(self) -> int:
        """
        Rollback all applied diffs.
        
        Returns:
            Number of diffs rolled back
        """
        count = 0
        while self._applied_diffs:
            if self.rollback_last():
                count += 1
            else:
                break
        return count
    
    @property
    def applied_diffs_count(self) -> int:
        """Number of diffs that have been applied."""
        return len(self._applied_diffs)
    
    def clear_history(self) -> None:
        """Clear the applied diffs history."""
        self._applied_diffs.clear()


# Helper functions for creating common diffs

def create_line_replacement_diff(
    file_path: Path,
    line_number: int,
    old_line: str,
    new_line: str,
    description: str = "",
) -> DiffHunk:
    """
    Create a hunk for replacing a single line.
    
    Args:
        file_path: Path to the file
        line_number: 1-indexed line number
        old_line: Expected current line content
        new_line: New line content
        description: Description of the change
        
    Returns:
        DiffHunk for the replacement
    """
    return DiffHunk(
        start_line=line_number - 1,  # Convert to 0-indexed
        end_line=line_number,
        original_content=old_line,
        new_content=new_line,
    )


def create_insertion_diff(
    file_path: Path,
    after_line: int,
    new_content: str,
    description: str = "",
) -> DiffHunk:
    """
    Create a hunk for inserting content after a line.
    
    Args:
        file_path: Path to the file
        after_line: 1-indexed line number to insert after
        new_content: Content to insert
        description: Description of the change
        
    Returns:
        DiffHunk for the insertion
    """
    return DiffHunk(
        start_line=after_line,
        end_line=after_line,
        original_content="",
        new_content=new_content,
    )
