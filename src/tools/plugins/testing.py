"""Testing plugin wrappers."""

from __future__ import annotations

from typing import Any, Callable

from src.tools.base import ToolExecutionContext, ToolPlugin


class TestingPlugin(ToolPlugin):
    """Plugin wrapper for test execution tools."""

    def __init__(self, name: str, executor: Callable[[dict[str, Any]], str]) -> None:
        self._name = name
        self._executor = executor

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> list[str]:
        return ["testing"]

    def validate(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        if not isinstance(args, dict):
            return False, "Tool arguments must be an object"
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
