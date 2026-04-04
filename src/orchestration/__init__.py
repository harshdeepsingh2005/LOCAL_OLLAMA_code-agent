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
    LoopController,
    LoopIteration,
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
    TaskGraphStats,
    TaskNode,
    TaskStatus,
)
from src.orchestration.large_project import (
    LargeProjectHandler,
    ProjectMetrics,
    ShardConfig,
    ShardSummary,
    TaskShard,
)
from src.orchestration.context_pipeline import (
    ContextBuilder,
    ContextPacket,
    TaskDomain,
    TaskRoute,
    TaskRouter,
    ValidationLayer,
)
from src.orchestration.meta_agent import MetaAgentReflector, MetaReflection
from src.orchestration.workspace_manager import WorkspaceContext, WorkspaceManager

__all__ = [
    # Executor
    "Executor",
    "ExecutionResult",
    "ExecutionError",
    # Task Graph
    "TaskGraph",
    "TaskNode",
    "TaskStatus",
    "TaskGraphStats",
    # Loop Controller
    "LoopController",
    "LoopState",
    "TerminationReason",
    "LoopIteration",
    # Rollback
    "RollbackManager",
    "Checkpoint",
    "FileSnapshot",
    # Large Project
    "LargeProjectHandler",
    "ProjectMetrics",
    "ShardConfig",
    "ShardSummary",
    "TaskShard",
    # Context pipeline
    "TaskDomain",
    "TaskRoute",
    "TaskRouter",
    "ContextPacket",
    "ContextBuilder",
    "ValidationLayer",
    "MetaAgentReflector",
    "MetaReflection",
    # Multi-workspace orchestration
    "WorkspaceManager",
    "WorkspaceContext",
]
