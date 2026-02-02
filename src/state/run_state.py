"""
Run State Module

Manages the state of a single execution run.
Provides serialization and restoration capabilities.

Design Decisions:
- Immutable snapshots
- JSON-serializable
- Versioned for migration
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RunPhase(str, Enum):
    """Phases of a run."""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    FIXING = "fixing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class TokenUsage(BaseModel):
    """Token usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, prompt: int, completion: int) -> None:
        """Add token usage."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion


class AgentExecution(BaseModel):
    """Record of a single agent execution."""
    agent_type: str
    task_id: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    success: bool = False
    tokens_used: int = 0
    error: str | None = None
    output_summary: str | None = None


class TaskState(BaseModel):
    """State of a single task."""
    task_id: str
    description: str
    status: str = "pending"
    dependencies: list[str] = Field(default_factory=list)
    assigned_files: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


class RunState(BaseModel):
    """
    Complete state of an execution run.
    
    This is the authoritative record of a run's progress
    and can be used to resume or analyze runs.
    """
    # Identity
    run_id: str
    version: str = "1.0.0"
    
    # Task
    task_description: str
    workspace_root: str
    
    # Timing
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    # Progress
    phase: RunPhase = RunPhase.INITIALIZING
    current_task_id: str | None = None
    current_agent: str | None = None
    iteration_count: int = 0
    fix_count: int = 0
    
    # Tasks
    tasks: list[TaskState] = Field(default_factory=list)
    
    # Execution history
    agent_executions: list[AgentExecution] = Field(default_factory=list)
    
    # Metrics
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    
    # Files
    files_created: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    files_deleted: list[str] = Field(default_factory=list)
    
    # Checkpoints
    checkpoint_ids: list[str] = Field(default_factory=list)
    last_checkpoint_id: str | None = None
    
    # Outcome
    success: bool | None = None
    error: str | None = None
    summary: str | None = None
    
    def mark_started(self) -> None:
        """Mark run as started."""
        self.started_at = datetime.now(timezone.utc)
        self.phase = RunPhase.PLANNING
    
    def mark_planning(self) -> None:
        """Mark run as in planning phase."""
        self.phase = RunPhase.PLANNING
        self.current_agent = "planner"
    
    def mark_executing(self, task_id: str) -> None:
        """Mark run as executing a task."""
        self.phase = RunPhase.EXECUTING
        self.current_task_id = task_id
        self.current_agent = "coder"
    
    def mark_reviewing(self, task_id: str) -> None:
        """Mark run as reviewing."""
        self.phase = RunPhase.REVIEWING
        self.current_task_id = task_id
        self.current_agent = "reviewer"
    
    def mark_fixing(self, task_id: str) -> None:
        """Mark run as fixing."""
        self.phase = RunPhase.FIXING
        self.current_task_id = task_id
        self.current_agent = "fixer"
        self.fix_count += 1
    
    def mark_completed(self, success: bool, summary: str | None = None) -> None:
        """Mark run as completed."""
        self.completed_at = datetime.now(timezone.utc)
        self.phase = RunPhase.COMPLETED if success else RunPhase.FAILED
        self.success = success
        self.summary = summary
        self.current_task_id = None
        self.current_agent = None
    
    def mark_aborted(self, reason: str) -> None:
        """Mark run as aborted."""
        self.completed_at = datetime.now(timezone.utc)
        self.phase = RunPhase.ABORTED
        self.success = False
        self.error = reason
        self.current_task_id = None
        self.current_agent = None
    
    def increment_iteration(self) -> None:
        """Increment iteration count."""
        self.iteration_count += 1
    
    def add_task(self, task: TaskState) -> None:
        """Add a task to the state."""
        self.tasks.append(task)
    
    def update_task_status(
        self,
        task_id: str,
        status: str,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Update a task's status."""
        for task in self.tasks:
            if task.task_id == task_id:
                task.status = status
                if status == "running" and not task.started_at:
                    task.started_at = datetime.now(timezone.utc)
                elif status in ("completed", "failed", "blocked"):
                    task.completed_at = datetime.now(timezone.utc)
                if error:
                    task.error = error
                if result:
                    task.result = result
                break
    
    def add_agent_execution(self, execution: AgentExecution) -> None:
        """Record an agent execution."""
        self.agent_executions.append(execution)
        if execution.tokens_used:
            self.token_usage.add(execution.tokens_used, 0)
    
    def add_checkpoint(self, checkpoint_id: str) -> None:
        """Record a checkpoint."""
        self.checkpoint_ids.append(checkpoint_id)
        self.last_checkpoint_id = checkpoint_id
    
    def add_file_created(self, path: str) -> None:
        """Record a file creation."""
        if path not in self.files_created:
            self.files_created.append(path)
    
    def add_file_modified(self, path: str) -> None:
        """Record a file modification."""
        if path not in self.files_modified and path not in self.files_created:
            self.files_modified.append(path)
    
    def add_file_deleted(self, path: str) -> None:
        """Record a file deletion."""
        if path not in self.files_deleted:
            self.files_deleted.append(path)
    
    def get_duration_ms(self) -> float | None:
        """Get run duration in milliseconds."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds() * 1000
    
    def get_task_stats(self) -> dict[str, int]:
        """Get task statistics."""
        stats = {
            "total": len(self.tasks),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "blocked": 0,
        }
        
        for task in self.tasks:
            status = task.status.lower()
            if status in stats:
                stats[status] += 1
        
        return stats
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return self.model_dump(mode="json")
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        """Deserialize from dictionary."""
        return cls.model_validate(data)
    
    def save(self, path: Path) -> None:
        """Save state to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> RunState:
        """Load state from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class RunStateManager:
    """
    Manages run state persistence and retrieval.
    
    Handles:
    - Creating new run states
    - Saving and loading states
    - Listing historical runs
    - Cleanup of old runs
    """
    
    def __init__(self, state_dir: Path) -> None:
        """
        Initialize state manager.
        
        Args:
            state_dir: Directory for state files
        """
        self._state_dir = state_dir
        self._state_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_state_path(self, run_id: str) -> Path:
        """Get path to state file for a run."""
        return self._state_dir / f"{run_id}.json"
    
    def create(
        self,
        run_id: str,
        task_description: str,
        workspace_root: str | Path,
    ) -> RunState:
        """Create a new run state."""
        state = RunState(
            run_id=run_id,
            task_description=task_description,
            workspace_root=str(workspace_root),
        )
        self.save(state)
        return state
    
    def save(self, state: RunState) -> None:
        """Save run state to disk."""
        path = self._get_state_path(state.run_id)
        state.save(path)
    
    def load(self, run_id: str) -> RunState | None:
        """Load run state from disk."""
        path = self._get_state_path(run_id)
        if not path.exists():
            return None
        return RunState.load(path)
    
    def exists(self, run_id: str) -> bool:
        """Check if a run state exists."""
        return self._get_state_path(run_id).exists()
    
    def delete(self, run_id: str) -> bool:
        """Delete a run state."""
        path = self._get_state_path(run_id)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def list_runs(
        self,
        limit: int = 100,
        include_completed: bool = True,
        include_failed: bool = True,
    ) -> list[RunState]:
        """List run states."""
        runs = []
        
        for path in sorted(self._state_dir.glob("*.json"), reverse=True):
            if len(runs) >= limit:
                break
            
            try:
                state = RunState.load(path)
                
                # Filter by status
                if state.phase == RunPhase.COMPLETED and not include_completed:
                    continue
                if state.phase == RunPhase.FAILED and not include_failed:
                    continue
                
                runs.append(state)
            except Exception:
                pass
        
        return runs
    
    def get_latest(self) -> RunState | None:
        """Get the most recent run state."""
        runs = self.list_runs(limit=1)
        return runs[0] if runs else None
    
    def cleanup_old_runs(self, keep_count: int = 50) -> int:
        """
        Remove old run states, keeping the most recent.
        
        Returns:
            Number of runs deleted
        """
        runs = sorted(
            self._state_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        
        deleted = 0
        for path in runs[keep_count:]:
            try:
                path.unlink()
                deleted += 1
            except Exception:
                pass
        
        return deleted
