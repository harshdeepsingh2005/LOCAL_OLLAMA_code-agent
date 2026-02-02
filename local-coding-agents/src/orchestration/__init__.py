"""
Orchestration Package

Provides execution coordination, task management, and state recovery.
"""

from src.orchestration.executor import (
    ExecutionError,
    ExecutionResult,
    Executor,
)
from src.orchestration.loop_controller import (
    Iteration,
    LoopController,
    LoopState,
    TerminationReason,
)
from src.orchestration.rollback import (
    Checkpoint,
    FileSnapshot,
    RollbackManager,
)
from src.orchestration.task_graph import (
    TaskGraph,
    TaskNode,
    TaskStats,
    TaskStatus,
)

__all__ = [
    # Executor
    "Executor",
    "ExecutionResult",
    "ExecutionError",
    # Task Graph
    "TaskGraph",
    "TaskNode",
    "TaskStatus",
    "TaskStats",
    # Loop Controller
    "LoopController",
    "LoopState",
    "TerminationReason",
    "Iteration",
    # Rollback
    "RollbackManager",
    "Checkpoint",
    "FileSnapshot",
]
