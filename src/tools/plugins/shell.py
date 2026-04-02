"""Shell execution plugin."""

from __future__ import annotations

from typing import Any, Callable

from src.tools.base import ToolExecutionContext, ToolPlugin


class ShellPlugin(ToolPlugin):
    """Policy-aware shell plugin wrapper."""

    def __init__(self, executor: Callable[[dict[str, Any]], str]) -> None:
        self._executor = executor

    @property
    def name(self) -> str:
        return "run_command"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> list[str]:
        return ["shell", "command-execution"]

    def validate(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        if not isinstance(args, dict):
            return False, "Tool arguments must be an object"
        command = str(args.get("command", "")).strip()
        if not command:
            return False, "Missing required argument: command"
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
