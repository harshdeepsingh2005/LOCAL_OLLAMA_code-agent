"""
Checkpoints Module

Provides checkpoint management for run recovery.
Integrates with RunState and RollbackManager.

Design Decisions:
- Checkpoints are immutable once created
- Include both file system and run state
- Support for incremental checkpoints
- Pruning of old checkpoints
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CheckpointMetadata(BaseModel):
    """Metadata for a checkpoint."""
    checkpoint_id: str
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Context
    description: str
    phase: str
    task_id: str | None = None
    iteration: int = 0
    
    # Content
    files_count: int = 0
    state_hash: str = ""
    size_bytes: int = 0
    
    # Relationships
    parent_checkpoint_id: str | None = None
    is_incremental: bool = False


class CheckpointStore:
    """
    Persistent checkpoint storage.
    
    Stores:
    - Checkpoint metadata
    - File snapshots
    - Run state snapshots
    """
    
    def __init__(self, store_dir: Path, max_checkpoints_per_run: int = 10) -> None:
        """
        Initialize checkpoint store.
        
        Args:
            store_dir: Directory for checkpoint storage
            max_checkpoints_per_run: Maximum checkpoints to keep per run
        """
        self._store_dir = store_dir
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._max_per_run = max_checkpoints_per_run
        
        # Metadata index
        self._index_path = self._store_dir / "index.json"
        self._index: dict[str, list[str]] = {}  # run_id -> [checkpoint_ids]
        self._load_index()
    
    def _load_index(self) -> None:
        """Load checkpoint index."""
        if self._index_path.exists():
            try:
                with open(self._index_path, "r") as f:
                    self._index = json.load(f)
            except Exception:
                self._index = {}
    
    def _save_index(self) -> None:
        """Save checkpoint index."""
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)
    
    def _get_checkpoint_dir(self, checkpoint_id: str) -> Path:
        """Get directory for a checkpoint."""
        return self._store_dir / checkpoint_id
    
    def _compute_hash(self, data: bytes) -> str:
        """Compute hash of data."""
        return hashlib.sha256(data).hexdigest()[:16]
    
    def create_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
        description: str,
        phase: str,
        task_id: str | None,
        iteration: int,
        run_state: dict[str, Any],
        files: dict[Path, bytes],
        parent_checkpoint_id: str | None = None,
    ) -> CheckpointMetadata:
        """
        Create a new checkpoint.
        
        Args:
            run_id: Run ID
            checkpoint_id: Unique checkpoint ID
            description: Human-readable description
            phase: Current phase
            task_id: Current task ID
            iteration: Current iteration
            run_state: Run state dictionary
            files: Dictionary of file path to content
            parent_checkpoint_id: Parent checkpoint for incremental
            
        Returns:
            CheckpointMetadata for the created checkpoint
        """
        checkpoint_dir = self._get_checkpoint_dir(checkpoint_id)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save run state
        state_path = checkpoint_dir / "state.json"
        state_bytes = json.dumps(run_state, indent=2, default=str).encode()
        with open(state_path, "wb") as f:
            f.write(state_bytes)
        
        # Save files
        files_dir = checkpoint_dir / "files"
        files_dir.mkdir(exist_ok=True)
        
        total_size = len(state_bytes)
        for file_path, content in files.items():
            # Use relative path for storage
            rel_path = file_path.name if file_path.is_absolute() else str(file_path)
            safe_name = hashlib.sha256(str(file_path).encode()).hexdigest()[:16]
            
            file_entry = files_dir / f"{safe_name}.bin"
            with open(file_entry, "wb") as f:
                f.write(content)
            
            # Save path mapping
            mapping_file = files_dir / f"{safe_name}.path"
            with open(mapping_file, "w") as f:
                f.write(str(file_path))
            
            total_size += len(content)
        
        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            description=description,
            phase=phase,
            task_id=task_id,
            iteration=iteration,
            files_count=len(files),
            state_hash=self._compute_hash(state_bytes),
            size_bytes=total_size,
            parent_checkpoint_id=parent_checkpoint_id,
            is_incremental=parent_checkpoint_id is not None,
        )
        
        # Save metadata
        meta_path = checkpoint_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata.model_dump(mode="json"), f, indent=2, default=str)
        
        # Update index
        if run_id not in self._index:
            self._index[run_id] = []
        self._index[run_id].append(checkpoint_id)
        self._save_index()
        
        # Prune old checkpoints
        self._prune_run_checkpoints(run_id)
        
        return metadata
    
    def get_checkpoint_metadata(self, checkpoint_id: str) -> CheckpointMetadata | None:
        """Get metadata for a checkpoint."""
        checkpoint_dir = self._get_checkpoint_dir(checkpoint_id)
        meta_path = checkpoint_dir / "metadata.json"
        
        if not meta_path.exists():
            return None
        
        try:
            with open(meta_path, "r") as f:
                data = json.load(f)
            return CheckpointMetadata.model_validate(data)
        except Exception:
            return None
    
    def get_checkpoint_state(self, checkpoint_id: str) -> dict[str, Any] | None:
        """Get run state from a checkpoint."""
        checkpoint_dir = self._get_checkpoint_dir(checkpoint_id)
        state_path = checkpoint_dir / "state.json"
        
        if not state_path.exists():
            return None
        
        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    
    def get_checkpoint_files(self, checkpoint_id: str) -> dict[Path, bytes]:
        """Get files from a checkpoint."""
        checkpoint_dir = self._get_checkpoint_dir(checkpoint_id)
        files_dir = checkpoint_dir / "files"
        
        if not files_dir.exists():
            return {}
        
        files = {}
        for path_file in files_dir.glob("*.path"):
            try:
                with open(path_file, "r") as f:
                    original_path = Path(f.read().strip())
                
                content_file = path_file.with_suffix(".bin")
                if content_file.exists():
                    with open(content_file, "rb") as f:
                        files[original_path] = f.read()
            except Exception:
                pass
        
        return files
    
    def list_checkpoints_for_run(self, run_id: str) -> list[CheckpointMetadata]:
        """List all checkpoints for a run."""
        checkpoint_ids = self._index.get(run_id, [])
        
        checkpoints = []
        for cid in checkpoint_ids:
            meta = self.get_checkpoint_metadata(cid)
            if meta:
                checkpoints.append(meta)
        
        return sorted(checkpoints, key=lambda c: c.created_at)
    
    def get_latest_checkpoint(self, run_id: str) -> CheckpointMetadata | None:
        """Get the most recent checkpoint for a run."""
        checkpoints = self.list_checkpoints_for_run(run_id)
        return checkpoints[-1] if checkpoints else None
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        checkpoint_dir = self._get_checkpoint_dir(checkpoint_id)
        
        if not checkpoint_dir.exists():
            return False
        
        try:
            # Get run_id from metadata
            meta = self.get_checkpoint_metadata(checkpoint_id)
            if meta:
                # Update index
                if meta.run_id in self._index:
                    self._index[meta.run_id] = [
                        cid for cid in self._index[meta.run_id]
                        if cid != checkpoint_id
                    ]
                    self._save_index()
            
            # Delete directory
            shutil.rmtree(checkpoint_dir)
            return True
        except Exception:
            return False
    
    def _prune_run_checkpoints(self, run_id: str) -> None:
        """Remove old checkpoints for a run, keeping the most recent."""
        checkpoint_ids = self._index.get(run_id, [])
        
        if len(checkpoint_ids) <= self._max_per_run:
            return
        
        # Get checkpoints with metadata
        checkpoints = []
        for cid in checkpoint_ids:
            meta = self.get_checkpoint_metadata(cid)
            if meta:
                checkpoints.append((cid, meta.created_at))
        
        # Sort by creation time
        checkpoints.sort(key=lambda x: x[1])
        
        # Delete oldest
        to_delete = checkpoints[:-self._max_per_run]
        for cid, _ in to_delete:
            self.delete_checkpoint(cid)
    
    def cleanup_run(self, run_id: str) -> int:
        """Delete all checkpoints for a run."""
        checkpoint_ids = self._index.get(run_id, []).copy()
        
        deleted = 0
        for cid in checkpoint_ids:
            if self.delete_checkpoint(cid):
                deleted += 1
        
        if run_id in self._index:
            del self._index[run_id]
            self._save_index()
        
        return deleted
    
    def get_total_size(self) -> int:
        """Get total size of all checkpoints."""
        total = 0
        for run_id, checkpoint_ids in self._index.items():
            for cid in checkpoint_ids:
                meta = self.get_checkpoint_metadata(cid)
                if meta:
                    total += meta.size_bytes
        return total
    
    def list_all_runs(self) -> list[str]:
        """List all runs with checkpoints."""
        return list(self._index.keys())
