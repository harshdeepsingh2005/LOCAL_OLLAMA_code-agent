"""
Summaries Module

Provides run summary generation and reporting.
Creates human-readable and machine-parseable summaries.

Design Decisions:
- Multiple output formats (text, JSON, HTML)
- Configurable verbosity levels
- Include actionable insights
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.state.run_state import RunPhase, RunState


class SummaryVerbosity(str, Enum):
    """Verbosity levels for summaries."""
    BRIEF = "brief"
    NORMAL = "normal"
    DETAILED = "detailed"


class TaskSummary(BaseModel):
    """Summary of a single task."""
    task_id: str
    description: str
    status: str
    duration_ms: float | None = None
    files_affected: list[str] = Field(default_factory=list)
    error: str | None = None


class AgentSummary(BaseModel):
    """Summary of agent executions."""
    agent_type: str
    executions: int = 0
    successful: int = 0
    failed: int = 0
    total_tokens: int = 0
    avg_duration_ms: float = 0


class RunSummary(BaseModel):
    """Complete summary of a run."""
    # Identity
    run_id: str
    task_description: str
    
    # Status
    status: str
    success: bool
    
    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float | None = None
    
    # Tasks
    tasks_total: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    task_details: list[TaskSummary] = Field(default_factory=list)
    
    # Agents
    agent_summaries: list[AgentSummary] = Field(default_factory=list)
    total_iterations: int = 0
    
    # Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Files
    files_created: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    files_deleted: list[str] = Field(default_factory=list)
    
    # Checkpoints
    checkpoint_count: int = 0
    
    # Errors
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SummaryGenerator:
    """
    Generates summaries from run state.
    
    Supports multiple output formats and verbosity levels.
    """
    
    def __init__(self, verbosity: SummaryVerbosity = SummaryVerbosity.NORMAL) -> None:
        """
        Initialize summary generator.
        
        Args:
            verbosity: Default verbosity level
        """
        self._verbosity = verbosity
    
    def from_run_state(self, state: RunState) -> RunSummary:
        """
        Generate summary from run state.
        
        Args:
            state: RunState to summarize
            
        Returns:
            RunSummary
        """
        # Task statistics
        task_stats = state.get_task_stats()
        
        # Task details
        task_details = []
        for task in state.tasks:
            duration = None
            if task.started_at and task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds() * 1000
            
            task_details.append(TaskSummary(
                task_id=task.task_id,
                description=task.description,
                status=task.status,
                duration_ms=duration,
                files_affected=task.assigned_files,
                error=task.error,
            ))
        
        # Agent summaries
        agent_stats: dict[str, dict] = {}
        for exec in state.agent_executions:
            if exec.agent_type not in agent_stats:
                agent_stats[exec.agent_type] = {
                    "executions": 0,
                    "successful": 0,
                    "failed": 0,
                    "total_tokens": 0,
                    "total_duration": 0,
                }
            
            stats = agent_stats[exec.agent_type]
            stats["executions"] += 1
            if exec.success:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
            stats["total_tokens"] += exec.tokens_used
            
            if exec.started_at and exec.completed_at:
                duration = (exec.completed_at - exec.started_at).total_seconds() * 1000
                stats["total_duration"] += duration
        
        agent_summaries = []
        for agent_type, stats in agent_stats.items():
            avg_duration = stats["total_duration"] / stats["executions"] if stats["executions"] > 0 else 0
            agent_summaries.append(AgentSummary(
                agent_type=agent_type,
                executions=stats["executions"],
                successful=stats["successful"],
                failed=stats["failed"],
                total_tokens=stats["total_tokens"],
                avg_duration_ms=avg_duration,
            ))
        
        # Collect errors
        errors = []
        if state.error:
            errors.append(state.error)
        for task in state.tasks:
            if task.error:
                errors.append(f"Task {task.task_id}: {task.error}")
        
        return RunSummary(
            run_id=state.run_id,
            task_description=state.task_description,
            status=state.phase.value,
            success=state.success or False,
            started_at=state.started_at,
            completed_at=state.completed_at,
            duration_ms=state.get_duration_ms(),
            tasks_total=task_stats["total"],
            tasks_completed=task_stats["completed"],
            tasks_failed=task_stats["failed"],
            task_details=task_details,
            agent_summaries=agent_summaries,
            total_iterations=state.iteration_count,
            prompt_tokens=state.token_usage.prompt_tokens,
            completion_tokens=state.token_usage.completion_tokens,
            total_tokens=state.token_usage.total_tokens,
            files_created=state.files_created,
            files_modified=state.files_modified,
            files_deleted=state.files_deleted,
            checkpoint_count=len(state.checkpoint_ids),
            errors=errors,
        )
    
    def to_text(
        self,
        summary: RunSummary,
        verbosity: SummaryVerbosity | None = None,
    ) -> str:
        """
        Format summary as text.
        
        Args:
            summary: RunSummary to format
            verbosity: Verbosity level (uses default if not specified)
            
        Returns:
            Formatted text string
        """
        v = verbosity or self._verbosity
        lines = []
        
        # Header
        status_icon = "✓" if summary.success else "✗"
        lines.append(f"{'=' * 60}")
        lines.append(f"Run Summary: {summary.run_id}")
        lines.append(f"{'=' * 60}")
        lines.append("")
        
        # Status
        lines.append(f"Status: {status_icon} {summary.status.upper()}")
        lines.append(f"Task: {summary.task_description[:100]}...")
        lines.append("")
        
        # Timing
        if summary.duration_ms:
            duration_s = summary.duration_ms / 1000
            lines.append(f"Duration: {duration_s:.1f}s")
        
        # Tasks
        lines.append(f"Tasks: {summary.tasks_completed}/{summary.tasks_total} completed")
        if summary.tasks_failed > 0:
            lines.append(f"  Failed: {summary.tasks_failed}")
        lines.append("")
        
        # Task details (for normal and detailed)
        if v in (SummaryVerbosity.NORMAL, SummaryVerbosity.DETAILED):
            if summary.task_details:
                lines.append("Task Details:")
                for task in summary.task_details:
                    icon = "✓" if task.status == "completed" else "✗"
                    lines.append(f"  {icon} [{task.task_id}] {task.description[:50]}")
                    if task.error and v == SummaryVerbosity.DETAILED:
                        lines.append(f"      Error: {task.error}")
                lines.append("")
        
        # Tokens
        lines.append(f"Tokens: {summary.total_tokens:,} total")
        if v == SummaryVerbosity.DETAILED:
            lines.append(f"  Prompt: {summary.prompt_tokens:,}")
            lines.append(f"  Completion: {summary.completion_tokens:,}")
        lines.append("")
        
        # Files
        total_files = len(summary.files_created) + len(summary.files_modified)
        lines.append(f"Files Modified: {total_files}")
        if v in (SummaryVerbosity.NORMAL, SummaryVerbosity.DETAILED):
            if summary.files_created:
                lines.append(f"  Created: {len(summary.files_created)}")
                if v == SummaryVerbosity.DETAILED:
                    for f in summary.files_created[:5]:
                        lines.append(f"    + {f}")
                    if len(summary.files_created) > 5:
                        lines.append(f"    ... and {len(summary.files_created) - 5} more")
            if summary.files_modified:
                lines.append(f"  Modified: {len(summary.files_modified)}")
                if v == SummaryVerbosity.DETAILED:
                    for f in summary.files_modified[:5]:
                        lines.append(f"    ~ {f}")
                    if len(summary.files_modified) > 5:
                        lines.append(f"    ... and {len(summary.files_modified) - 5} more")
        lines.append("")
        
        # Agent stats (detailed only)
        if v == SummaryVerbosity.DETAILED and summary.agent_summaries:
            lines.append("Agent Statistics:")
            for agent in summary.agent_summaries:
                lines.append(f"  {agent.agent_type}:")
                lines.append(f"    Executions: {agent.executions}")
                lines.append(f"    Success Rate: {agent.successful}/{agent.executions}")
                lines.append(f"    Tokens: {agent.total_tokens:,}")
            lines.append("")
        
        # Errors
        if summary.errors:
            lines.append("Errors:")
            for error in summary.errors[:5]:
                lines.append(f"  • {error[:100]}")
            if len(summary.errors) > 5:
                lines.append(f"  ... and {len(summary.errors) - 5} more")
            lines.append("")
        
        lines.append(f"{'=' * 60}")
        
        return "\n".join(lines)
    
    def to_json(self, summary: RunSummary, indent: int = 2) -> str:
        """
        Format summary as JSON.
        
        Args:
            summary: RunSummary to format
            indent: JSON indentation
            
        Returns:
            JSON string
        """
        return json.dumps(summary.model_dump(mode="json"), indent=indent, default=str)
    
    def to_dict(self, summary: RunSummary) -> dict[str, Any]:
        """
        Convert summary to dictionary.
        
        Args:
            summary: RunSummary to convert
            
        Returns:
            Dictionary representation
        """
        return summary.model_dump(mode="json")
    
    def to_markdown(self, summary: RunSummary) -> str:
        """
        Format summary as Markdown.
        
        Args:
            summary: RunSummary to format
            
        Returns:
            Markdown string
        """
        lines = []
        
        status_icon = "✅" if summary.success else "❌"
        
        lines.append(f"# Run Summary: {summary.run_id}")
        lines.append("")
        lines.append(f"**Status:** {status_icon} {summary.status.upper()}")
        lines.append(f"**Task:** {summary.task_description}")
        lines.append("")
        
        lines.append("## Overview")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Duration | {(summary.duration_ms or 0) / 1000:.1f}s |")
        lines.append(f"| Tasks | {summary.tasks_completed}/{summary.tasks_total} |")
        lines.append(f"| Tokens | {summary.total_tokens:,} |")
        lines.append(f"| Files Changed | {len(summary.files_created) + len(summary.files_modified)} |")
        lines.append("")
        
        if summary.task_details:
            lines.append("## Tasks")
            lines.append("")
            for task in summary.task_details:
                icon = "✅" if task.status == "completed" else "❌"
                lines.append(f"- {icon} **{task.task_id}**: {task.description}")
                if task.error:
                    lines.append(f"  - Error: {task.error}")
            lines.append("")
        
        if summary.files_created or summary.files_modified:
            lines.append("## Files")
            lines.append("")
            if summary.files_created:
                lines.append("### Created")
                for f in summary.files_created:
                    lines.append(f"- `{f}`")
            if summary.files_modified:
                lines.append("### Modified")
                for f in summary.files_modified:
                    lines.append(f"- `{f}`")
            lines.append("")
        
        if summary.errors:
            lines.append("## Errors")
            lines.append("")
            for error in summary.errors:
                lines.append(f"- {error}")
            lines.append("")
        
        return "\n".join(lines)
    
    def save(
        self,
        summary: RunSummary,
        output_path: Path,
        format: str = "text",
    ) -> None:
        """
        Save summary to file.
        
        Args:
            summary: RunSummary to save
            output_path: Output file path
            format: Output format (text, json, markdown)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            content = self.to_json(summary)
        elif format == "markdown":
            content = self.to_markdown(summary)
        else:
            content = self.to_text(summary)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
