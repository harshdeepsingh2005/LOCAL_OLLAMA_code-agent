"""Multi-workspace orchestration primitives (v1, sequential only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.agents import Subtask
from src.core.memory import MemoryManager
from src.orchestration.context_pipeline import ContextBuilder, TaskRoute, TaskRouter


@dataclass
class WorkspaceContext:
    """Isolated per-workspace execution context."""

    name: str
    root: Path
    memory: MemoryManager
    context_builder: ContextBuilder


class WorkspaceManager:
    """Coordinates deterministic sequential execution across isolated workspaces."""

    def __init__(self, workspace_roots: list[Path]) -> None:
        if not workspace_roots:
            raise ValueError("At least one workspace is required")

        contexts: dict[str, WorkspaceContext] = {}
        for root in sorted((p.resolve() for p in workspace_roots), key=lambda p: p.as_posix()):
            name = root.name
            if name in contexts:
                raise ValueError(f"Duplicate workspace name: {name}")
            memory = MemoryManager(root)
            builder = ContextBuilder(root, memory)
            contexts[name] = WorkspaceContext(name=name, root=root, memory=memory, context_builder=builder)

        self._contexts = contexts
        self._router = TaskRouter()

    def list_workspaces(self) -> list[str]:
        """Return deterministic workspace ordering."""
        return sorted(self._contexts.keys())

    def get_workspace(self, name: str) -> WorkspaceContext:
        """Get isolated context for a workspace."""
        context = self._contexts.get(name)
        if context is None:
            raise KeyError(name)
        return context

    def build_workspace_context(self, workspace_name: str, task_description: str) -> str:
        """Build prompt context for one workspace with strict isolation."""
        context = self.get_workspace(workspace_name)
        route: TaskRoute = self._router.route(task_description)
        packet = context.context_builder.build(task_description, route)
        return packet.to_prompt_context(max_chars=1800)

    def assign_subtasks_to_workspaces(
        self,
        subtasks: list[Subtask],
        default_workspace: str | None = None,
    ) -> dict[str, list[Subtask]]:
        """Assign subtasks to workspaces using deterministic path-prefix rules."""
        names = self.list_workspaces()
        fallback = default_workspace or names[0]
        if fallback not in self._contexts:
            raise KeyError(fallback)

        assignments: dict[str, list[Subtask]] = {name: [] for name in names}

        for subtask in subtasks:
            assigned = fallback
            for file_path in subtask.target_files:
                prefix = file_path.split("/", 1)[0]
                if prefix in assignments:
                    assigned = prefix
                    break
            assignments[assigned].append(subtask)

        return assignments

    def execute_sequential(
        self,
        assignments: dict[str, list[Any]],
        runner: Callable[[str, Any], Any],
    ) -> list[tuple[str, Any]]:
        """Execute assignments deterministically workspace-by-workspace."""
        results: list[tuple[str, Any]] = []
        for workspace_name in self.list_workspaces():
            for item in assignments.get(workspace_name, []):
                results.append((workspace_name, runner(workspace_name, item)))
        return results
