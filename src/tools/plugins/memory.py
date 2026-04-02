"""Memory and MCP tool plugins."""

from __future__ import annotations

from typing import Any, Callable

from src.tools.base import ToolExecutionContext, ToolPlugin


class MemoryPlugin(ToolPlugin):
    """Plugin wrapper for memory-related operations."""

    def __init__(
        self,
        name: str,
        executor: Callable[[dict[str, Any]], str],
        required_args: list[str] | None = None,
    ) -> None:
        self._name = name
        self._executor = executor
        self._required_args = required_args or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> list[str]:
        return ["memory"]

    def validate(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        if not isinstance(args, dict):
            return False, "Tool arguments must be an object"
        for key in self._required_args:
            if key not in args:
                return False, f"Missing required argument: {key}"
        return True, None

    def policy_check(
        self,
        context: ToolExecutionContext,
        args: dict[str, Any],
    ) -> tuple[bool, str | None]:
        _ = context
        _ = args
        return True, None

    def execute(self, args: dict[str, Any]) -> str:
        return self._executor(args)


class MCPPlugin(ToolPlugin):
    """Plugin wrapper for explicitly registered MCP operations."""

    def __init__(
        self,
        name: str,
        executor: Callable[[dict[str, Any]], str],
        required_args: list[str] | None = None,
    ) -> None:
        self._name = name
        self._executor = executor
        self._required_args = required_args or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> list[str]:
        return ["mcp"]

    def validate(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        if not isinstance(args, dict):
            return False, "Tool arguments must be an object"
        for key in self._required_args:
            if key not in args:
                return False, f"Missing required argument: {key}"
        return True, None

    def policy_check(
        self,
        context: ToolExecutionContext,
        args: dict[str, Any],
    ) -> tuple[bool, str | None]:
        _ = context
        _ = args
        return True, None

    def execute(self, args: dict[str, Any]) -> str:
        return self._executor(args)
