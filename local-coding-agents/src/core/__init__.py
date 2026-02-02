"""
Core Module Package

Provides foundational infrastructure for the agent system.
"""

from src.core.context_manager import (
    ContextBudget,
    ContextItem,
    ContextManager,
    ContextPriority,
    ContextType,
)
from src.core.diff_engine import (
    DiffEngine,
    DiffHunk,
    DiffResult,
    DiffType,
    FileDiff,
)
from src.core.file_guard import (
    AccessDeniedError,
    FileGuard,
    FileGuardPolicy,
    FileOperation,
    FileLimitError,
    PathViolationError,
    RateLimitError,
)
from src.core.llm_client import (
    CompletionRequest,
    CompletionResponse,
    LLMClient,
    LLMClientError,
    Message,
    ModelConfig,
    ModelNotAvailableError,
    TokenLimitExceededError,
)
from src.core.telemetry import (
    EventType,
    TelemetryCollector,
    TelemetryEvent,
)

__all__ = [
    # Context Manager
    "ContextBudget",
    "ContextItem",
    "ContextManager",
    "ContextPriority",
    "ContextType",
    # Diff Engine
    "DiffEngine",
    "DiffHunk",
    "DiffResult",
    "DiffType",
    "FileDiff",
    # File Guard
    "AccessDeniedError",
    "FileGuard",
    "FileGuardPolicy",
    "FileOperation",
    "FileLimitError",
    "PathViolationError",
    "RateLimitError",
    # LLM Client
    "CompletionRequest",
    "CompletionResponse",
    "LLMClient",
    "LLMClientError",
    "Message",
    "ModelConfig",
    "ModelNotAvailableError",
    "TokenLimitExceededError",
    # Telemetry
    "EventType",
    "TelemetryCollector",
    "TelemetryEvent",
]
