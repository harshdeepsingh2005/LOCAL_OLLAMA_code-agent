"""
Telemetry Module

Provides structured observability for the agent system without any external
data exfiltration. All telemetry is stored locally in structured logs.

Design Decisions:
- No external services, no network calls
- Structured logging with structlog
- Per-run isolation
- Human-readable and machine-parseable output
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of telemetry events."""
    RUN_START = "run_start"
    RUN_END = "run_end"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    MODEL_LOAD = "model_load"
    COMPLETION = "completion"
    TOOL_CALL = "tool_call"
    CHECKPOINT = "checkpoint"
    ROLLBACK = "rollback"
    ERROR = "error"
    WARNING = "warning"
    FILE_ACCESS = "file_access"
    DIFF_APPLY = "diff_apply"


class TelemetryEvent(BaseModel):
    """A single telemetry event."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: EventType
    run_id: str
    agent_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None
    tokens_used: int | None = None
    
    def to_log_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "run_id": self.run_id,
        }
        if self.agent_id:
            result["agent_id"] = self.agent_id
        if self.data:
            result["data"] = self.data
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.tokens_used is not None:
            result["tokens_used"] = self.tokens_used
        return result


@dataclass
class TokenUsage:
    """Tracks token usage across a run."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens
    
    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion


@dataclass
class AgentMetrics:
    """Metrics for a single agent execution."""
    agent_id: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    tokens: TokenUsage = field(default_factory=TokenUsage)
    completions: int = 0
    errors: int = 0
    
    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


@dataclass  
class RunMetrics:
    """Metrics for an entire run."""
    run_id: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    tokens: TokenUsage = field(default_factory=TokenUsage)
    agents_executed: int = 0
    checkpoints_created: int = 0
    files_modified: int = 0
    diffs_applied: int = 0
    errors: int = 0
    planned_tools: int = 0
    executed_tools: int = 0
    fallback_count: int = 0
    plan_adherence_score_sum: float = 0.0
    plan_adherence_samples: int = 0
    
    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


class TelemetryCollector:
    """
    Collects and persists telemetry data for observability.
    
    All data is stored locally in structured log files.
    No external network calls are made.
    
    Thread Safety: NOT thread-safe. Designed for sequential execution.
    """
    
    def __init__(
        self,
        run_id: str,
        log_dir: Path,
        enable_console: bool = True,
    ) -> None:
        """
        Initialize the telemetry collector.
        
        Args:
            run_id: Unique identifier for this run
            log_dir: Directory to store log files
            enable_console: Whether to output to console
        """
        self._run_id = run_id
        self._log_dir = log_dir
        self._enable_console = enable_console
        self._events: list[TelemetryEvent] = []
        self._run_metrics = RunMetrics(run_id=run_id)
        self._agent_metrics: dict[str, AgentMetrics] = {}
        self._current_agent: str | None = None
        
        # Ensure log directory exists
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure structlog
        self._logger = self._setup_logger()
    
    def _setup_logger(self) -> structlog.BoundLogger:
        """Configure structured logging."""
        processors = [
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        if self._enable_console:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(structlog.processors.JSONRenderer())
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(0),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
        )
        
        return structlog.get_logger().bind(run_id=self._run_id)
    
    def _record_event(self, event: TelemetryEvent) -> None:
        """Record an event to memory and log file."""
        self._events.append(event)
        
        # Log to file
        log_file = self._log_dir / f"{self._run_id}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(event.to_log_dict()) + "\n")
    
    def record_run_start(self, task: str, config: dict[str, Any]) -> None:
        """Record the start of a run."""
        event = TelemetryEvent(
            event_type=EventType.RUN_START,
            run_id=self._run_id,
            data={"task": task, "config": config},
        )
        self._record_event(event)
        self._logger.info("Run started", task=task[:100])
    
    def record_run_end(self, success: bool, summary: str) -> None:
        """Record the end of a run."""
        self._run_metrics.end_time = time.perf_counter()
        adherence_avg = (
            self._run_metrics.plan_adherence_score_sum / self._run_metrics.plan_adherence_samples
            if self._run_metrics.plan_adherence_samples > 0
            else 0.0
        )
        event = TelemetryEvent(
            event_type=EventType.RUN_END,
            run_id=self._run_id,
            data={
                "success": success,
                "summary": summary,
                "metrics": {
                    "duration_ms": self._run_metrics.duration_ms,
                    "total_tokens": self._run_metrics.tokens.total,
                    "agents_executed": self._run_metrics.agents_executed,
                    "files_modified": self._run_metrics.files_modified,
                    "errors": self._run_metrics.errors,
                    "planned_tools": self._run_metrics.planned_tools,
                    "executed_tools": self._run_metrics.executed_tools,
                    "fallback_count": self._run_metrics.fallback_count,
                    "plan_adherence_score": round(adherence_avg, 4),
                },
            },
            duration_ms=self._run_metrics.duration_ms,
            tokens_used=self._run_metrics.tokens.total,
        )
        self._record_event(event)
        self._logger.info(
            "Run completed",
            success=success,
            duration_ms=self._run_metrics.duration_ms,
            tokens=self._run_metrics.tokens.total,
        )
    
    def record_agent_start(self, agent_id: str, input_summary: str) -> None:
        """Record the start of an agent execution."""
        self._current_agent = agent_id
        self._agent_metrics[agent_id] = AgentMetrics(agent_id=agent_id)
        event = TelemetryEvent(
            event_type=EventType.AGENT_START,
            run_id=self._run_id,
            agent_id=agent_id,
            data={"input_summary": input_summary[:500]},
        )
        self._record_event(event)
        self._logger.info("Agent started", agent_id=agent_id)
    
    def record_agent_end(self, agent_id: str, success: bool, output_summary: str) -> None:
        """Record the end of an agent execution."""
        metrics = self._agent_metrics.get(agent_id)
        if metrics:
            metrics.end_time = time.perf_counter()
            self._run_metrics.agents_executed += 1
        
        event = TelemetryEvent(
            event_type=EventType.AGENT_END,
            run_id=self._run_id,
            agent_id=agent_id,
            data={"success": success, "output_summary": output_summary[:500]},
            duration_ms=metrics.duration_ms if metrics else None,
            tokens_used=metrics.tokens.total if metrics else None,
        )
        self._record_event(event)
        self._logger.info(
            "Agent completed",
            agent_id=agent_id,
            success=success,
            duration_ms=metrics.duration_ms if metrics else None,
        )
        self._current_agent = None
    
    def record_model_load(self, model_name: str) -> None:
        """Record a model load event."""
        event = TelemetryEvent(
            event_type=EventType.MODEL_LOAD,
            run_id=self._run_id,
            agent_id=self._current_agent,
            data={"model_name": model_name},
        )
        self._record_event(event)
        self._logger.debug("Model loaded", model=model_name)
    
    def record_completion(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
    ) -> None:
        """Record an LLM completion."""
        # Update metrics
        self._run_metrics.tokens.add(prompt_tokens, completion_tokens)
        if self._current_agent and self._current_agent in self._agent_metrics:
            agent_metrics = self._agent_metrics[self._current_agent]
            agent_metrics.tokens.add(prompt_tokens, completion_tokens)
            agent_metrics.completions += 1
        
        event = TelemetryEvent(
            event_type=EventType.COMPLETION,
            run_id=self._run_id,
            agent_id=self._current_agent,
            data={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            duration_ms=latency_ms,
            tokens_used=prompt_tokens + completion_tokens,
        )
        self._record_event(event)
    
    def record_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result_summary: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record a tool invocation."""
        event = TelemetryEvent(
            event_type=EventType.TOOL_CALL,
            run_id=self._run_id,
            agent_id=self._current_agent,
            data={
                "tool_name": tool_name,
                "args": args,
                "result_summary": result_summary[:200],
                "success": success,
            },
            duration_ms=duration_ms,
        )
        self._record_event(event)
        self._logger.debug("Tool called", tool=tool_name, success=success)
    
    def record_checkpoint(self, checkpoint_id: str, description: str) -> None:
        """Record a checkpoint creation."""
        self._run_metrics.checkpoints_created += 1
        event = TelemetryEvent(
            event_type=EventType.CHECKPOINT,
            run_id=self._run_id,
            data={"checkpoint_id": checkpoint_id, "description": description},
        )
        self._record_event(event)
        self._logger.info("Checkpoint created", checkpoint_id=checkpoint_id)
    
    def record_rollback(self, checkpoint_id: str, reason: str) -> None:
        """Record a rollback operation."""
        event = TelemetryEvent(
            event_type=EventType.ROLLBACK,
            run_id=self._run_id,
            data={"checkpoint_id": checkpoint_id, "reason": reason},
        )
        self._record_event(event)
        self._logger.warning("Rollback performed", checkpoint_id=checkpoint_id, reason=reason)
    
    def record_error(self, error: str, context: dict[str, Any] | None = None) -> None:
        """Record an error."""
        self._run_metrics.errors += 1
        if self._current_agent and self._current_agent in self._agent_metrics:
            self._agent_metrics[self._current_agent].errors += 1
        
        event = TelemetryEvent(
            event_type=EventType.ERROR,
            run_id=self._run_id,
            agent_id=self._current_agent,
            data={"error": error, "context": context or {}},
        )
        self._record_event(event)
        self._logger.error("Error occurred", error=error)
    
    def record_warning(self, warning: str, context: dict[str, Any] | None = None) -> None:
        """Record a warning."""
        event = TelemetryEvent(
            event_type=EventType.WARNING,
            run_id=self._run_id,
            agent_id=self._current_agent,
            data={"warning": warning, "context": context or {}},
        )
        self._record_event(event)
        self._logger.warning("Warning", message=warning)
    
    def record_file_access(
        self,
        operation: str,
        path: str,
        success: bool,
    ) -> None:
        """Record a file access operation."""
        if success and operation in ("write", "modify"):
            self._run_metrics.files_modified += 1
        
        event = TelemetryEvent(
            event_type=EventType.FILE_ACCESS,
            run_id=self._run_id,
            agent_id=self._current_agent,
            data={"operation": operation, "path": path, "success": success},
        )
        self._record_event(event)
    
    def record_diff_apply(
        self,
        file_path: str,
        lines_added: int,
        lines_removed: int,
        success: bool,
    ) -> None:
        """Record a diff application."""
        if success:
            self._run_metrics.diffs_applied += 1
        
        event = TelemetryEvent(
            event_type=EventType.DIFF_APPLY,
            run_id=self._run_id,
            agent_id=self._current_agent,
            data={
                "file_path": file_path,
                "lines_added": lines_added,
                "lines_removed": lines_removed,
                "success": success,
            },
        )
        self._record_event(event)

    def record_tool_plan_metrics(
        self,
        planned_tools: int,
        executed_tools: int,
        fallback_count: int,
        adherence_score: float,
    ) -> None:
        """Record aggregate metrics for planned-vs-executed tool usage."""
        self._run_metrics.planned_tools += max(0, planned_tools)
        self._run_metrics.executed_tools += max(0, executed_tools)
        self._run_metrics.fallback_count += max(0, fallback_count)
        self._run_metrics.plan_adherence_score_sum += max(0.0, min(1.0, adherence_score))
        self._run_metrics.plan_adherence_samples += 1

        event = TelemetryEvent(
            event_type=EventType.WARNING,
            run_id=self._run_id,
            agent_id=self._current_agent,
            data={
                "warning": "tool_plan_metrics",
                "context": {
                    "planned_tools": planned_tools,
                    "executed_tools": executed_tools,
                    "fallback_count": fallback_count,
                    "adherence_score": round(adherence_score, 3),
                },
            },
        )
        self._record_event(event)

    def record_tool_plan_violation(self, reason: str, context: dict[str, Any] | None = None) -> None:
        """Emit structured tool-plan violation event (non-blocking)."""
        self.record_warning("tool_plan_violation", {"reason": reason, **(context or {})})

    def record_fallback_invoked(
        self,
        primary_tool: str,
        fallback_tool: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Emit structured fallback-invoked event (non-blocking)."""
        payload = {
            "primary_tool": primary_tool,
            "fallback_tool": fallback_tool,
        }
        if context:
            payload.update(context)
        self.record_warning("fallback_invoked", payload)
    
    @property
    def run_metrics(self) -> RunMetrics:
        """Get current run metrics."""
        return self._run_metrics
    
    @property
    def events(self) -> list[TelemetryEvent]:
        """Get all recorded events."""
        return self._events.copy()
    
    def get_agent_metrics(self, agent_id: str) -> AgentMetrics | None:
        """Get metrics for a specific agent."""
        return self._agent_metrics.get(agent_id)
    
    def export_summary(self) -> dict[str, Any]:
        """Export a human-readable summary of the run."""
        adherence_avg = (
            self._run_metrics.plan_adherence_score_sum / self._run_metrics.plan_adherence_samples
            if self._run_metrics.plan_adherence_samples > 0
            else 0.0
        )
        return {
            "run_id": self._run_id,
            "duration_ms": self._run_metrics.duration_ms,
            "tokens": {
                "prompt": self._run_metrics.tokens.prompt_tokens,
                "completion": self._run_metrics.tokens.completion_tokens,
                "total": self._run_metrics.tokens.total,
            },
            "agents_executed": self._run_metrics.agents_executed,
            "checkpoints_created": self._run_metrics.checkpoints_created,
            "files_modified": self._run_metrics.files_modified,
            "diffs_applied": self._run_metrics.diffs_applied,
            "errors": self._run_metrics.errors,
            "planned_tools": self._run_metrics.planned_tools,
            "executed_tools": self._run_metrics.executed_tools,
            "fallback_count": self._run_metrics.fallback_count,
            "plan_adherence_score": round(adherence_avg, 4),
            "events_count": len(self._events),
        }
