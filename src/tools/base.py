"""Base abstractions for policy-aware tool plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolExecutionContext:
    """Execution context passed to plugins for policy and telemetry decisions."""

    run_id: str
    workspace_root: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolPlugin(ABC):
    """Contract for all tool plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name used for registration and dispatch."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin semantic version string."""

    @property
    @abstractmethod
    def capabilities(self) -> list[str]:
        """Declared plugin capabilities used for policy and audit."""

    @abstractmethod
    def validate(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate raw input arguments before policy and execution."""

    @abstractmethod
    def policy_check(
        self,
        context: ToolExecutionContext,
        args: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Perform policy checks before execution."""

    @abstractmethod
    def execute(self, args: dict[str, Any]) -> str:
        """Execute tool call and return a serialized result."""
