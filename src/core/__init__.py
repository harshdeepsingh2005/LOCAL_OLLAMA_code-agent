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
from src.core.memory import MemoryManager
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
from src.core.contracts import (
    ContractEnforcer,
    ContractViolation,
    ValidationResult,
    ViolationType,
)
from src.core.file_lock import (
    AtomicFileWriter,
    FileLockError,
    FileLockManager,
    FileLockTimeout,
)
# ── New capability modules ──
from src.core.pty_shell import PTYSession, PTYShellManager, PTYShellError
from src.core.mcp_client import (
    MCPClient,
    MCPServerConfig,
    MCPToolSchema,
    MCPCallResult,
)
from src.core.hitl import (
    HITLGate,
    HITLConfig,
    HITLResult,
    HITLDecision,
    PermissionLevel,
    classify_command,
)
from src.core.file_editing_tools import FileEditingTools
from src.core.semantic_search import (
    GrepSearch,
    SemanticSearch,
    CodebaseNavigator,
    SearchResult,
)
from src.core.agent_tools import ToolExecutor, TOOL_SCHEMAS, get_tools_system_prompt

__all__ = [
    # Context Manager
    "ContextBudget",
    "ContextItem",
    "ContextManager",
    "ContextPriority",
    "ContextType",
    # Memory
    "MemoryManager",
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
    # Contracts
    "ContractEnforcer",
    "ContractViolation",
    "ValidationResult",
    "ViolationType",
    # File Locks
    "AtomicFileWriter",
    "FileLockError",
    "FileLockManager",
    "FileLockTimeout",
    # ── Feature 1: Persistent PTY shell ──
    "PTYSession",
    "PTYShellManager",
    "PTYShellError",
    # ── Feature 2: Native MCP client ──
    "MCPClient",
    "MCPServerConfig",
    "MCPToolSchema",
    "MCPCallResult",
    # ── Feature 3: HITL security ──
    "HITLGate",
    "HITLConfig",
    "HITLResult",
    "HITLDecision",
    "PermissionLevel",
    "classify_command",
    # ── Feature 4: Granular file editing tools ──
    "FileEditingTools",
    # ── Feature 6: Semantic codebase navigation ──
    "GrepSearch",
    "SemanticSearch",
    "CodebaseNavigator",
    "SearchResult",
    # ── Upgraded tool executor ──
    "ToolExecutor",
    "TOOL_SCHEMAS",
    "get_tools_system_prompt",
]
