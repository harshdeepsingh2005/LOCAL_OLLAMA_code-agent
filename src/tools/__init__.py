"""
Tools Package

Provides safe, mediated access to external resources.
All tools are auditable and respect workspace boundaries.
"""

from src.tools.filesystem import (
    FileInfo,
    FileOperation,
    FilesystemTools,
    SearchMatch,
    ToolResult,
)
from src.tools.shell import (
    CommandCategory,
    CommandResult,
    CommandStatus,
    ShellExecutor,
)
from src.tools.testing import (
    TestResult,
    TestRunner,
    TestRunResult,
    TestStatus,
    TestSuiteResult,
    TypeChecker,
)
from src.tools.base import ToolExecutionContext, ToolPlugin
from src.tools.registry import ToolRegistry

__all__ = [
    # Filesystem
    "FilesystemTools",
    "FileOperation",
    "FileInfo",
    "SearchMatch",
    "ToolResult",
    # Shell
    "ShellExecutor",
    "CommandCategory",
    "CommandStatus",
    "CommandResult",
    # Testing
    "TestRunner",
    "TypeChecker",
    "TestStatus",
    "TestResult",
    "TestSuiteResult",
    "TestRunResult",
    # Plugin architecture
    "ToolPlugin",
    "ToolExecutionContext",
    "ToolRegistry",
]
