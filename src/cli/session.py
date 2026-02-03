"""
CLI Session Module

Manages interactive session state, persistence, and recovery.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionState(str, Enum):
    """Session lifecycle states."""
    INIT = "init"
    ACTIVE = "active"
    PENDING_APPROVAL = "pending_approval"
    PAUSED = "paused"
    ENDED = "ended"
    ERROR = "error"


class PendingChange(BaseModel):
    """A pending file change awaiting approval."""
    file_path: str
    change_type: str  # create, modify, delete
    description: str
    diff_content: str
    new_content: str
    original_content: Optional[str] = None
    lines_added: int = 0
    lines_removed: int = 0


class ConversationMessage(BaseModel):
    """A message in the conversation history."""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionConfig(BaseModel):
    """Session configuration."""
    workspace: Path
    model: str = "qwen2.5-coder:7b-instruct-q4_K_M"
    auto_checkpoint: bool = True
    require_approval: bool = True
    max_tokens_per_run: int = 50000
    
    class Config:
        arbitrary_types_allowed = True


class Session:
    """
    Manages an interactive agent session.
    
    Handles:
    - Session lifecycle (init, active, paused, ended)
    - Conversation history
    - Pending changes
    - Checkpoints
    - Persistence and recovery
    """
    
    SESSIONS_DIR = Path.home() / ".local" / "share" / "agent" / "sessions"
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        config: Optional[SessionConfig] = None,
    ) -> None:
        """
        Initialize a session.
        
        Args:
            session_id: Existing session ID to resume, or None for new
            config: Session configuration
        """
        self.id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.config = config or SessionConfig(workspace=Path.cwd())
        
        self.state = SessionState.INIT
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        
        # Task tracking
        self.current_task: Optional[str] = None
        self.task_plan: list[dict[str, Any]] = []
        self.tasks_completed: int = 0
        self.tasks_total: int = 0
        
        # Resource tracking
        self.tokens_used: int = 0
        self.iterations: int = 0
        
        # Conversation history
        self.messages: list[ConversationMessage] = []
        
        # Pending changes
        self.pending_changes: list[PendingChange] = []
        
        # Checkpoints
        self.checkpoints: list[str] = []
        self.current_checkpoint: Optional[str] = None
        
        # Error state
        self.last_error: Optional[str] = None
        
        # Ensure sessions directory exists
        self.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    @property
    def session_dir(self) -> Path:
        """Get the directory for this session's data."""
        return self.SESSIONS_DIR / self.id
    
    @property
    def is_active(self) -> bool:
        """Check if session is in an active state."""
        return self.state in (SessionState.ACTIVE, SessionState.PENDING_APPROVAL)
    
    @property
    def has_pending_changes(self) -> bool:
        """Check if there are pending changes."""
        return len(self.pending_changes) > 0
    
    @property
    def token_percentage(self) -> float:
        """Get token usage as percentage."""
        if self.config.max_tokens_per_run <= 0:
            return 0.0
        return (self.tokens_used / self.config.max_tokens_per_run) * 100
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    def start(self) -> None:
        """Start the session."""
        self.state = SessionState.ACTIVE
        self.updated_at = datetime.now(timezone.utc)
    
    def pause(self) -> None:
        """Pause the session for later resumption."""
        self.state = SessionState.PAUSED
        self.updated_at = datetime.now(timezone.utc)
        self.save()
    
    def end(self, success: bool = True) -> None:
        """End the session."""
        self.state = SessionState.ENDED
        self.updated_at = datetime.now(timezone.utc)
        self.save()
    
    def set_error(self, error: str) -> None:
        """Set session to error state."""
        self.state = SessionState.ERROR
        self.last_error = error
        self.updated_at = datetime.now(timezone.utc)
    
    def set_pending_approval(self) -> None:
        """Set session to pending approval state."""
        self.state = SessionState.PENDING_APPROVAL
        self.updated_at = datetime.now(timezone.utc)
    
    def resume_active(self) -> None:
        """Resume from pending approval to active."""
        if self.state in (SessionState.PENDING_APPROVAL, SessionState.PAUSED):
            self.state = SessionState.ACTIVE
            self.updated_at = datetime.now(timezone.utc)
    
    # =========================================================================
    # Conversation
    # =========================================================================
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.messages.append(ConversationMessage(
            role="user",
            content=content,
        ))
        self.updated_at = datetime.now(timezone.utc)
    
    def add_assistant_message(self, content: str, metadata: Optional[dict] = None) -> None:
        """Add an assistant message to history."""
        self.messages.append(ConversationMessage(
            role="assistant",
            content=content,
            metadata=metadata or {},
        ))
        self.updated_at = datetime.now(timezone.utc)
    
    def add_system_message(self, content: str) -> None:
        """Add a system message to history."""
        self.messages.append(ConversationMessage(
            role="system",
            content=content,
        ))
    
    def get_context_messages(self, max_messages: int = 20) -> list[dict[str, str]]:
        """Get recent messages for context."""
        recent = self.messages[-max_messages:]
        return [{"role": m.role, "content": m.content} for m in recent]
    
    def clear_context(self) -> None:
        """Clear conversation context (keep only system messages)."""
        self.messages = [m for m in self.messages if m.role == "system"]
    
    # =========================================================================
    # Plan Management
    # =========================================================================
    
    def set_plan(self, plan: list[dict[str, Any]]) -> None:
        """Set the execution plan."""
        self.task_plan = plan
        self.tasks_total = len(plan)
        self.tasks_completed = 0
        for task in self.task_plan:
            task["status"] = "not_started"
    
    def update_task_status(self, task_index: int, status: str) -> None:
        """Update a task's status."""
        if 0 <= task_index < len(self.task_plan):
            self.task_plan[task_index]["status"] = status
            if status == "completed":
                self.tasks_completed += 1
    
    def get_current_task(self) -> Optional[dict[str, Any]]:
        """Get the current task being executed."""
        for task in self.task_plan:
            if task.get("status") == "in_progress":
                return task
        return None
    
    def get_next_task(self) -> Optional[dict[str, Any]]:
        """Get the next task to execute."""
        for i, task in enumerate(self.task_plan):
            if task.get("status") == "not_started":
                return task
        return None
    
    # =========================================================================
    # Pending Changes
    # =========================================================================
    
    def add_pending_change(self, change: PendingChange) -> None:
        """Add a pending change."""
        self.pending_changes.append(change)
        self.set_pending_approval()
    
    def clear_pending_changes(self) -> None:
        """Clear all pending changes."""
        self.pending_changes = []
        if self.state == SessionState.PENDING_APPROVAL:
            self.state = SessionState.ACTIVE
    
    def get_pending_summary(self) -> list[dict[str, Any]]:
        """Get summary of pending changes."""
        return [
            {
                "path": c.file_path,
                "change_type": c.change_type,
                "lines_added": c.lines_added,
                "lines_removed": c.lines_removed,
                "lines": c.lines_added if c.change_type == "create" else None,
            }
            for c in self.pending_changes
        ]
    
    # =========================================================================
    # Checkpoints
    # =========================================================================
    
    def add_checkpoint(self, checkpoint_id: str) -> None:
        """Record a checkpoint."""
        self.checkpoints.append(checkpoint_id)
        self.current_checkpoint = checkpoint_id
    
    # =========================================================================
    # Token Tracking
    # =========================================================================
    
    def add_tokens(self, count: int) -> None:
        """Add to token count."""
        self.tokens_used += count
    
    def is_near_token_limit(self, threshold: float = 0.9) -> bool:
        """Check if near token limit."""
        return self.token_percentage >= (threshold * 100)
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def save(self) -> None:
        """Save session to disk."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        state_file = self.session_dir / "state.json"
        state_data = {
            "id": self.id,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "config": {
                "workspace": str(self.config.workspace),
                "model": self.config.model,
                "auto_checkpoint": self.config.auto_checkpoint,
                "require_approval": self.config.require_approval,
                "max_tokens_per_run": self.config.max_tokens_per_run,
            },
            "current_task": self.current_task,
            "task_plan": self.task_plan,
            "tasks_completed": self.tasks_completed,
            "tasks_total": self.tasks_total,
            "tokens_used": self.tokens_used,
            "iterations": self.iterations,
            "checkpoints": self.checkpoints,
            "current_checkpoint": self.current_checkpoint,
            "last_error": self.last_error,
        }
        
        with open(state_file, "w") as f:
            json.dump(state_data, f, indent=2)
        
        # Save conversation
        context_file = self.session_dir / "context.json"
        messages_data = [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
                "metadata": m.metadata,
            }
            for m in self.messages
        ]
        
        with open(context_file, "w") as f:
            json.dump(messages_data, f, indent=2)
        
        # Save pending changes
        if self.pending_changes:
            diffs_dir = self.session_dir / "diffs"
            diffs_dir.mkdir(exist_ok=True)
            
            for i, change in enumerate(self.pending_changes):
                diff_file = diffs_dir / f"change_{i:03d}.json"
                with open(diff_file, "w") as f:
                    json.dump(change.model_dump(), f, indent=2)
    
    @classmethod
    def load(cls, session_id: str) -> Optional["Session"]:
        """Load a session from disk."""
        session_dir = cls.SESSIONS_DIR / session_id
        state_file = session_dir / "state.json"
        
        if not state_file.exists():
            return None
        
        try:
            with open(state_file) as f:
                state_data = json.load(f)
            
            config = SessionConfig(
                workspace=Path(state_data["config"]["workspace"]),
                model=state_data["config"]["model"],
                auto_checkpoint=state_data["config"]["auto_checkpoint"],
                require_approval=state_data["config"]["require_approval"],
                max_tokens_per_run=state_data["config"]["max_tokens_per_run"],
            )
            
            session = cls(session_id=session_id, config=config)
            session.state = SessionState(state_data["state"])
            session.created_at = datetime.fromisoformat(state_data["created_at"])
            session.updated_at = datetime.fromisoformat(state_data["updated_at"])
            session.current_task = state_data.get("current_task")
            session.task_plan = state_data.get("task_plan", [])
            session.tasks_completed = state_data.get("tasks_completed", 0)
            session.tasks_total = state_data.get("tasks_total", 0)
            session.tokens_used = state_data.get("tokens_used", 0)
            session.iterations = state_data.get("iterations", 0)
            session.checkpoints = state_data.get("checkpoints", [])
            session.current_checkpoint = state_data.get("current_checkpoint")
            session.last_error = state_data.get("last_error")
            
            # Load conversation
            context_file = session_dir / "context.json"
            if context_file.exists():
                with open(context_file) as f:
                    messages_data = json.load(f)
                session.messages = [
                    ConversationMessage(
                        role=m["role"],
                        content=m["content"],
                        timestamp=datetime.fromisoformat(m["timestamp"]),
                        metadata=m.get("metadata", {}),
                    )
                    for m in messages_data
                ]
            
            # Load pending changes
            diffs_dir = session_dir / "diffs"
            if diffs_dir.exists():
                for diff_file in sorted(diffs_dir.glob("change_*.json")):
                    with open(diff_file) as f:
                        change_data = json.load(f)
                    session.pending_changes.append(PendingChange(**change_data))
            
            return session
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return None
    
    @classmethod
    def list_sessions(cls, limit: int = 20) -> list[dict[str, Any]]:
        """List available sessions."""
        sessions = []
        
        if not cls.SESSIONS_DIR.exists():
            return sessions
        
        for session_dir in sorted(cls.SESSIONS_DIR.iterdir(), reverse=True):
            if not session_dir.is_dir():
                continue
            
            state_file = session_dir / "state.json"
            if not state_file.exists():
                continue
            
            try:
                with open(state_file) as f:
                    state_data = json.load(f)
                
                sessions.append({
                    "id": state_data["id"],
                    "state": state_data["state"],
                    "created_at": state_data["created_at"],
                    "updated_at": state_data["updated_at"],
                    "tasks_completed": state_data.get("tasks_completed", 0),
                    "tasks_total": state_data.get("tasks_total", 0),
                    "workspace": state_data["config"]["workspace"],
                })
                
                if len(sessions) >= limit:
                    break
                    
            except (json.JSONDecodeError, KeyError):
                continue
        
        return sessions
    
    @classmethod
    def cleanup_old_sessions(cls, max_age_days: int = 7) -> int:
        """Clean up old sessions."""
        import shutil
        from datetime import timedelta
        
        cleaned = 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        
        if not cls.SESSIONS_DIR.exists():
            return 0
        
        for session_dir in cls.SESSIONS_DIR.iterdir():
            if not session_dir.is_dir():
                continue
            
            state_file = session_dir / "state.json"
            if not state_file.exists():
                shutil.rmtree(session_dir)
                cleaned += 1
                continue
            
            try:
                with open(state_file) as f:
                    state_data = json.load(f)
                
                updated = datetime.fromisoformat(state_data["updated_at"])
                if updated < cutoff and state_data["state"] != "paused":
                    shutil.rmtree(session_dir)
                    cleaned += 1
                    
            except (json.JSONDecodeError, KeyError):
                shutil.rmtree(session_dir)
                cleaned += 1
        
        return cleaned
    
    def to_summary(self) -> dict[str, Any]:
        """Get session summary."""
        return {
            "id": self.id,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "tasks_completed": self.tasks_completed,
            "tasks_total": self.tasks_total,
            "tokens_used": self.tokens_used,
            "pending_changes": len(self.pending_changes),
            "checkpoints": len(self.checkpoints),
        }
