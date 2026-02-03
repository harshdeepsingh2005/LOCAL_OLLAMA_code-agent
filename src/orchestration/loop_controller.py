"""
Loop Controller Module

Controls the execution flow of agents with iteration limits,
retry logic, and termination conditions.

Design Decisions:
- Deterministic execution order
- Hard iteration limits
- Explicit state machine
- Audit-friendly state transitions
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from src.config import Configuration


class LoopState(str, Enum):
    """States of the execution loop."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    FIXING = "fixing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ABORTED = "aborted"
    PAUSED = "paused"  # New: paused waiting for user input


class TerminationReason(str, Enum):
    """Reasons for loop termination."""
    SUCCESS = "success"
    MAX_ITERATIONS = "max_iterations"
    MAX_FIX_ITERATIONS = "max_fix_iterations"
    TIMEOUT = "timeout"
    FATAL_ERROR = "fatal_error"
    USER_ABORT = "user_abort"
    ALL_TASKS_COMPLETE = "all_tasks_complete"
    UNRECOVERABLE_FAILURE = "unrecoverable_failure"
    PAUSED_FOR_USER = "paused_for_user"  # New: paused waiting for user decision


@dataclass
class LoopIteration:
    """Record of a single loop iteration."""
    iteration_number: int
    state: LoopState
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: float | None = None
    agent_type: str | None = None
    task_id: str | None = None
    success: bool = False
    error: str | None = None
    tokens_used: int = 0
    
    def complete(self, success: bool, error: str | None = None) -> None:
        """Mark iteration as complete."""
        self.completed_at = datetime.now(timezone.utc)
        self.duration_ms = (
            self.completed_at - self.started_at
        ).total_seconds() * 1000
        self.success = success
        self.error = error


@dataclass
class LoopStatistics:
    """Statistics about loop execution."""
    total_iterations: int = 0
    successful_iterations: int = 0
    failed_iterations: int = 0
    total_duration_ms: float = 0
    total_tokens: int = 0
    planning_iterations: int = 0
    coding_iterations: int = 0
    review_iterations: int = 0
    fix_iterations: int = 0


class LoopLimitExceededError(Exception):
    """Raised when loop limit is exceeded."""
    pass


class LoopTimeoutError(Exception):
    """Raised when loop times out."""
    pass


class LoopController:
    """
    Controls the agent execution loop with limits and state tracking.
    
    Responsibilities:
    - Enforce iteration limits
    - Track state transitions
    - Handle retries
    - Detect termination conditions
    
    Thread Safety: NOT thread-safe. Designed for sequential execution.
    """
    
    def __init__(
        self,
        config: Configuration,
        run_id: str,
    ) -> None:
        """
        Initialize the loop controller.
        
        Args:
            config: Configuration with limits
            run_id: ID of the current run
        """
        self._config = config
        self._run_id = run_id
        
        # Limits from config
        self._max_iterations = config.limits.iterations.max_loop_iterations
        self._max_agent_retries = config.limits.iterations.max_agent_retries
        self._max_fix_iterations = config.limits.iterations.max_fix_iterations
        self._max_planning_cycles = config.limits.iterations.max_planning_cycles
        self._max_run_seconds = config.limits.time.max_run_seconds
        
        # State tracking
        self._state = LoopState.IDLE
        self._iteration_count = 0
        self._fix_iteration_count = 0
        self._planning_cycle_count = 0
        self._iterations: list[LoopIteration] = []
        
        # Timing
        self._start_time: float | None = None
        self._total_tokens = 0
        
        # Retry tracking per agent
        self._agent_retries: dict[str, int] = {}
        
        # Termination
        self._termination_reason: TerminationReason | None = None
        self._termination_message: str | None = None
        
        # User continuation flag (for max iterations)
        self._needs_user_continue: bool = False
        
        # Callbacks
        self._on_state_change: Callable[[LoopState, LoopState], None] | None = None
    
    def start(self) -> None:
        """Start the loop controller."""
        self._start_time = time.perf_counter()
        self._transition_to(LoopState.PLANNING)
    
    def _transition_to(self, new_state: LoopState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        
        if self._on_state_change:
            self._on_state_change(old_state, new_state)
    
    def begin_iteration(
        self,
        agent_type: str,
        task_id: str | None = None,
    ) -> LoopIteration | None:
        """
        Begin a new loop iteration.
        
        Args:
            agent_type: Type of agent being executed
            task_id: Optional task ID
            
        Returns:
            LoopIteration record, or None if limit reached (check needs_user_continue)
            
        Raises:
            LoopTimeoutError: If timeout exceeded
        """
        # Check timeout (this is still a hard error)
        if self._start_time:
            elapsed = time.perf_counter() - self._start_time
            if elapsed > self._max_run_seconds:
                self._terminate(
                    TerminationReason.TIMEOUT,
                    f"Run timeout after {elapsed:.1f}s"
                )
                raise LoopTimeoutError(f"Run timeout exceeded: {elapsed:.1f}s")
        
        # Check iteration limit - don't raise, signal for user input
        if self._iteration_count >= self._max_iterations:
            self._needs_user_continue = True
            self._transition_to(LoopState.PAUSED)
            return None
        
        # Determine state based on agent type
        state_map = {
            "planner": LoopState.PLANNING,
            "coder": LoopState.EXECUTING,
            "reviewer": LoopState.REVIEWING,
            "fixer": LoopState.FIXING,
        }
        self._transition_to(state_map.get(agent_type, LoopState.EXECUTING))
        
        # Create iteration record
        self._iteration_count += 1
        iteration = LoopIteration(
            iteration_number=self._iteration_count,
            state=self._state,
            started_at=datetime.now(timezone.utc),
            agent_type=agent_type,
            task_id=task_id,
        )
        self._iterations.append(iteration)
        
        # Track specific iteration types
        if agent_type == "planner":
            self._planning_cycle_count += 1
        elif agent_type == "fixer":
            self._fix_iteration_count += 1
        
        return iteration
    
    def end_iteration(
        self,
        iteration: LoopIteration,
        success: bool,
        error: str | None = None,
        tokens_used: int = 0,
    ) -> None:
        """
        End a loop iteration.
        
        Args:
            iteration: The iteration to end
            success: Whether the iteration succeeded
            error: Optional error message
            tokens_used: Tokens used in this iteration
        """
        iteration.complete(success, error)
        iteration.tokens_used = tokens_used
        self._total_tokens += tokens_used
        
        # Track agent retries
        if not success and iteration.agent_type:
            retry_key = f"{iteration.agent_type}_{iteration.task_id or 'default'}"
            self._agent_retries[retry_key] = self._agent_retries.get(retry_key, 0) + 1
    
    def can_retry_agent(self, agent_type: str, task_id: str | None = None) -> bool:
        """
        Check if an agent can be retried.
        
        Args:
            agent_type: Type of agent
            task_id: Optional task ID
            
        Returns:
            True if retry is allowed
        """
        retry_key = f"{agent_type}_{task_id or 'default'}"
        current_retries = self._agent_retries.get(retry_key, 0)
        return current_retries < self._max_agent_retries
    
    def can_fix_again(self) -> bool:
        """Check if another fix iteration is allowed."""
        return self._fix_iteration_count < self._max_fix_iterations
    
    def can_replan(self) -> bool:
        """Check if another planning cycle is allowed."""
        return self._planning_cycle_count < self._max_planning_cycles
    
    def reset_fix_counter(self) -> None:
        """Reset the fix iteration counter (e.g., when moving to new task)."""
        self._fix_iteration_count = 0
    
    @property
    def needs_user_continue(self) -> bool:
        """Check if max iterations reached and waiting for user decision."""
        return self._needs_user_continue
    
    def extend_iterations(self, additional: int = 10) -> None:
        """
        Extend the maximum iterations (called when user chooses to continue).
        
        Args:
            additional: Number of additional iterations to allow
        """
        self._max_iterations += additional
        self._needs_user_continue = False
        self._transition_to(LoopState.EXECUTING)  # Resume execution
    
    def decline_continue(self) -> None:
        """User declined to continue - terminate gracefully."""
        self._needs_user_continue = False
        self._terminate(
            TerminationReason.MAX_ITERATIONS,
            f"Max iterations ({self._iteration_count}) reached - user declined to continue"
        )
    
    def _terminate(self, reason: TerminationReason, message: str) -> None:
        """Set termination reason and state."""
        self._termination_reason = reason
        self._termination_message = message
        
        if reason == TerminationReason.SUCCESS:
            self._transition_to(LoopState.COMPLETED)
        elif reason == TerminationReason.TIMEOUT:
            self._transition_to(LoopState.TIMEOUT)
        elif reason == TerminationReason.USER_ABORT:
            self._transition_to(LoopState.ABORTED)
        elif reason == TerminationReason.PAUSED_FOR_USER:
            self._transition_to(LoopState.PAUSED)
        else:
            self._transition_to(LoopState.FAILED)
    
    def complete_success(self, message: str = "All tasks completed successfully") -> None:
        """Mark loop as successfully completed."""
        self._terminate(TerminationReason.SUCCESS, message)
    
    def complete_failure(self, reason: TerminationReason, message: str) -> None:
        """Mark loop as failed."""
        self._terminate(reason, message)
    
    def abort(self, message: str = "Aborted by user") -> None:
        """Abort the loop."""
        self._terminate(TerminationReason.USER_ABORT, message)
    
    @property
    def state(self) -> LoopState:
        """Current loop state."""
        return self._state
    
    @property
    def is_running(self) -> bool:
        """Check if loop is still running (not paused or terminated)."""
        return self._state not in (
            LoopState.IDLE,
            LoopState.COMPLETED,
            LoopState.FAILED,
            LoopState.TIMEOUT,
            LoopState.ABORTED,
            LoopState.PAUSED,
        )
    
    @property
    def is_paused(self) -> bool:
        """Check if loop is paused waiting for user input."""
        return self._state == LoopState.PAUSED
    
    @property
    def is_terminal(self) -> bool:
        """Check if loop is in terminal state (not paused)."""
        return self._state in (
            LoopState.COMPLETED,
            LoopState.FAILED,
            LoopState.TIMEOUT,
            LoopState.ABORTED,
        )
    
    @property
    def iteration_count(self) -> int:
        """Total number of iterations."""
        return self._iteration_count
    
    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since start."""
        if self._start_time is None:
            return 0
        return time.perf_counter() - self._start_time
    
    @property
    def remaining_seconds(self) -> float:
        """Seconds remaining before timeout."""
        return max(0, self._max_run_seconds - self.elapsed_seconds)
    
    @property
    def termination_reason(self) -> TerminationReason | None:
        """Reason for termination."""
        return self._termination_reason
    
    @property
    def termination_message(self) -> str | None:
        """Termination message."""
        return self._termination_message
    
    def get_statistics(self) -> LoopStatistics:
        """Get execution statistics."""
        stats = LoopStatistics(
            total_iterations=self._iteration_count,
            total_tokens=self._total_tokens,
        )
        
        for iteration in self._iterations:
            if iteration.success:
                stats.successful_iterations += 1
            else:
                stats.failed_iterations += 1
            
            if iteration.duration_ms:
                stats.total_duration_ms += iteration.duration_ms
            
            if iteration.agent_type == "planner":
                stats.planning_iterations += 1
            elif iteration.agent_type == "coder":
                stats.coding_iterations += 1
            elif iteration.agent_type == "reviewer":
                stats.review_iterations += 1
            elif iteration.agent_type == "fixer":
                stats.fix_iterations += 1
        
        return stats
    
    def set_state_change_callback(
        self,
        callback: Callable[[LoopState, LoopState], None],
    ) -> None:
        """Set callback for state changes."""
        self._on_state_change = callback
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize controller state to dictionary."""
        return {
            "run_id": self._run_id,
            "state": self._state.value,
            "iteration_count": self._iteration_count,
            "fix_iteration_count": self._fix_iteration_count,
            "planning_cycle_count": self._planning_cycle_count,
            "total_tokens": self._total_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "termination_reason": self._termination_reason.value if self._termination_reason else None,
            "termination_message": self._termination_message,
            "agent_retries": self._agent_retries,
        }
