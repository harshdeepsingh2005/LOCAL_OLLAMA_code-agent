"""
Rollback Module

Provides state restoration capabilities for recovering from failures.
All operations can be rolled back to any checkpoint.

Design Decisions:
- Full state capture at checkpoints
- File system snapshot support
- Agent output history
- Efficient incremental storage
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.file_guard import FileGuard


@dataclass
class FileSnapshot:
    """Snapshot of a single file's state."""
    path: Path
    existed: bool
    content_hash: str | None = None
    content: str | None = None  # Only stored if small enough
    backup_path: Path | None = None  # For large files
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "path": str(self.path),
            "existed": self.existed,
            "content_hash": self.content_hash,
            "content": self.content,
            "backup_path": str(self.backup_path) if self.backup_path else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileSnapshot":
        """Deserialize from dictionary."""
        return cls(
            path=Path(data["path"]),
            existed=data["existed"],
            content_hash=data.get("content_hash"),
            content=data.get("content"),
            backup_path=Path(data["backup_path"]) if data.get("backup_path") else None,
        )


@dataclass
class Checkpoint:
    """
    A snapshot of system state at a point in time.
    
    Captures:
    - Task graph state
    - File system changes
    - Agent execution history
    - Context state
    """
    id: str
    run_id: str
    created_at: datetime
    description: str
    
    # Task state
    task_graph_state: dict[str, Any] = field(default_factory=dict)
    current_task_id: str | None = None
    
    # File state
    file_snapshots: dict[str, FileSnapshot] = field(default_factory=dict)
    
    # Agent state
    agent_history: list[dict[str, Any]] = field(default_factory=list)
    
    # Metrics at checkpoint time
    tokens_used: int = 0
    files_modified: int = 0
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "id": self.id,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "task_graph_state": self.task_graph_state,
            "current_task_id": self.current_task_id,
            "file_snapshots": {
                k: v.to_dict() for k, v in self.file_snapshots.items()
            },
            "agent_history": self.agent_history,
            "tokens_used": self.tokens_used,
            "files_modified": self.files_modified,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """Deserialize checkpoint from dictionary."""
        return cls(
            id=data["id"],
            run_id=data["run_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            description=data["description"],
            task_graph_state=data.get("task_graph_state", {}),
            current_task_id=data.get("current_task_id"),
            file_snapshots={
                k: FileSnapshot.from_dict(v) 
                for k, v in data.get("file_snapshots", {}).items()
            },
            agent_history=data.get("agent_history", []),
            tokens_used=data.get("tokens_used", 0),
            files_modified=data.get("files_modified", 0),
            metadata=data.get("metadata", {}),
        )


class RollbackError(Exception):
    """Raised when rollback fails."""
    pass


class RollbackManager:
    """
    Manages checkpoints and rollback operations.
    
    Provides:
    - Checkpoint creation
    - State restoration
    - Cleanup of old checkpoints
    
    Thread Safety: NOT thread-safe. Designed for sequential execution.
    """
    
    # Maximum file size to store inline (10KB)
    MAX_INLINE_FILE_SIZE = 10 * 1024
    
    def __init__(
        self,
        workspace_root: Path,
        checkpoint_dir: Path,
        file_guard: FileGuard,
        max_checkpoints: int = 20,
    ) -> None:
        """
        Initialize the rollback manager.
        
        Args:
            workspace_root: Root of the workspace
            checkpoint_dir: Directory to store checkpoints
            file_guard: FileGuard instance for file operations
            max_checkpoints: Maximum checkpoints to retain
        """
        self._workspace_root = workspace_root
        self._checkpoint_dir = checkpoint_dir
        self._file_guard = file_guard
        self._max_checkpoints = max_checkpoints
        
        # Checkpoints indexed by ID
        self._checkpoints: dict[str, Checkpoint] = {}
        self._checkpoint_order: list[str] = []
        
        # Ensure checkpoint directory exists
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def create_checkpoint(
        self,
        run_id: str,
        description: str,
        task_graph_state: dict[str, Any],
        current_task_id: str | None = None,
        modified_files: list[Path] | None = None,
        agent_history: list[dict[str, Any]] | None = None,
        tokens_used: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """
        Create a new checkpoint.
        
        Args:
            run_id: ID of the current run
            description: Human-readable description
            task_graph_state: Current task graph state
            current_task_id: ID of current task being executed
            modified_files: List of files that have been modified
            agent_history: History of agent executions
            tokens_used: Total tokens used so far
            metadata: Additional metadata
            
        Returns:
            Created Checkpoint
        """
        # Generate checkpoint ID
        checkpoint_id = f"cp_{run_id}_{len(self._checkpoint_order):03d}"
        
        # Snapshot modified files
        file_snapshots: dict[str, FileSnapshot] = {}
        if modified_files:
            for file_path in modified_files:
                snapshot = self._snapshot_file(file_path)
                file_snapshots[str(file_path)] = snapshot
        
        # Create checkpoint
        checkpoint = Checkpoint(
            id=checkpoint_id,
            run_id=run_id,
            created_at=datetime.now(timezone.utc),
            description=description,
            task_graph_state=task_graph_state,
            current_task_id=current_task_id,
            file_snapshots=file_snapshots,
            agent_history=agent_history or [],
            tokens_used=tokens_used,
            files_modified=len(file_snapshots),
            metadata=metadata or {},
        )
        
        # Store checkpoint
        self._checkpoints[checkpoint_id] = checkpoint
        self._checkpoint_order.append(checkpoint_id)
        
        # Save to disk
        self._save_checkpoint(checkpoint)
        
        # Clean up old checkpoints if needed
        self._cleanup_old_checkpoints()
        
        return checkpoint
    
    def _snapshot_file(self, file_path: Path) -> FileSnapshot:
        """Create a snapshot of a file."""
        try:
            if not self._file_guard.exists(file_path):
                return FileSnapshot(
                    path=file_path,
                    existed=False,
                )
            
            content = self._file_guard.read(file_path)
            content_bytes = content.encode("utf-8")
            
            import hashlib
            content_hash = hashlib.sha256(content_bytes).hexdigest()[:16]
            
            # If file is small, store inline
            if len(content_bytes) <= self.MAX_INLINE_FILE_SIZE:
                return FileSnapshot(
                    path=file_path,
                    existed=True,
                    content_hash=content_hash,
                    content=content,
                )
            
            # For large files, create backup
            backup_path = self._file_guard.create_backup(file_path)
            return FileSnapshot(
                path=file_path,
                existed=True,
                content_hash=content_hash,
                backup_path=backup_path,
            )
            
        except Exception as e:
            # If we can't snapshot, mark as not existed
            return FileSnapshot(
                path=file_path,
                existed=False,
            )
    
    def _save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to disk."""
        checkpoint_file = self._checkpoint_dir / f"{checkpoint.id}.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
    
    def _load_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Load checkpoint from disk."""
        checkpoint_file = self._checkpoint_dir / f"{checkpoint_id}.json"
        if not checkpoint_file.exists():
            return None
        
        with open(checkpoint_file) as f:
            data = json.load(f)
        return Checkpoint.from_dict(data)
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints exceeding the limit."""
        while len(self._checkpoint_order) > self._max_checkpoints:
            old_id = self._checkpoint_order.pop(0)
            
            # Remove from memory
            if old_id in self._checkpoints:
                del self._checkpoints[old_id]
            
            # Remove from disk
            checkpoint_file = self._checkpoint_dir / f"{old_id}.json"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
    
    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """
        Get a checkpoint by ID.
        
        Args:
            checkpoint_id: ID of the checkpoint
            
        Returns:
            Checkpoint if found, None otherwise
        """
        # Check memory first
        if checkpoint_id in self._checkpoints:
            return self._checkpoints[checkpoint_id]
        
        # Try loading from disk
        return self._load_checkpoint(checkpoint_id)
    
    def list_checkpoints(self, run_id: str | None = None) -> list[Checkpoint]:
        """
        List all checkpoints, optionally filtered by run.
        
        Args:
            run_id: Optional run ID to filter by
            
        Returns:
            List of checkpoints in chronological order
        """
        checkpoints = []
        for cp_id in self._checkpoint_order:
            cp = self.get_checkpoint(cp_id)
            if cp and (run_id is None or cp.run_id == run_id):
                checkpoints.append(cp)
        return checkpoints
    
    def rollback_to(self, checkpoint_id: str) -> bool:
        """
        Rollback to a specific checkpoint.
        
        Restores:
        - File system state
        - Returns checkpoint for task graph restoration
        
        Args:
            checkpoint_id: ID of checkpoint to rollback to
            
        Returns:
            True if rollback succeeded
            
        Raises:
            RollbackError: If rollback fails
        """
        checkpoint = self.get_checkpoint(checkpoint_id)
        if not checkpoint:
            raise RollbackError(f"Checkpoint {checkpoint_id} not found")
        
        # Restore file state
        for path_str, snapshot in checkpoint.file_snapshots.items():
            self._restore_file(snapshot)
        
        # Remove checkpoints after this one
        if checkpoint_id in self._checkpoint_order:
            idx = self._checkpoint_order.index(checkpoint_id)
            for old_id in self._checkpoint_order[idx + 1:]:
                if old_id in self._checkpoints:
                    del self._checkpoints[old_id]
                checkpoint_file = self._checkpoint_dir / f"{old_id}.json"
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
            self._checkpoint_order = self._checkpoint_order[:idx + 1]
        
        return True
    
    def _restore_file(self, snapshot: FileSnapshot) -> None:
        """Restore a file from snapshot."""
        try:
            if not snapshot.existed:
                # File shouldn't exist - delete if it does
                if self._file_guard.exists(snapshot.path):
                    self._file_guard.delete(snapshot.path)
                return
            
            # File should exist - restore content
            if snapshot.content is not None:
                # Content stored inline
                self._file_guard.write(snapshot.path, snapshot.content)
            elif snapshot.backup_path and snapshot.backup_path.exists():
                # Content in backup file
                content = snapshot.backup_path.read_text(encoding="utf-8")
                self._file_guard.write(snapshot.path, content)
            
        except Exception as e:
            raise RollbackError(f"Failed to restore {snapshot.path}: {e}")
    
    def rollback_last(self) -> Checkpoint | None:
        """
        Rollback to the previous checkpoint.
        
        Returns:
            The checkpoint rolled back to, or None if no checkpoints
        """
        if len(self._checkpoint_order) < 2:
            return None
        
        # Rollback to second-to-last checkpoint
        prev_id = self._checkpoint_order[-2]
        self.rollback_to(prev_id)
        return self.get_checkpoint(prev_id)
    
    @property
    def checkpoint_count(self) -> int:
        """Number of checkpoints."""
        return len(self._checkpoint_order)
    
    @property
    def latest_checkpoint(self) -> Checkpoint | None:
        """Get the most recent checkpoint."""
        if not self._checkpoint_order:
            return None
        return self.get_checkpoint(self._checkpoint_order[-1])
    
    def clear_all(self) -> None:
        """Remove all checkpoints."""
        for cp_id in self._checkpoint_order:
            checkpoint_file = self._checkpoint_dir / f"{cp_id}.json"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
        
        self._checkpoints.clear()
        self._checkpoint_order.clear()
