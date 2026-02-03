"""
Task Graph Module

Manages the directed acyclic graph of tasks for execution.
Handles dependencies, ordering, and state tracking.

Design Decisions:
- Explicit DAG structure
- Topological ordering for execution
- Immutable once built
- Supports checkpointing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterator

from pydantic import BaseModel, Field

from src.agents.base import Subtask


class TaskStatus(str, Enum):
    """Status of a task in the graph."""
    PENDING = "pending"
    READY = "ready"        # Dependencies satisfied, can execute
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"    # Dependency failed


class TaskNode(BaseModel):
    """A single node in the task graph."""
    id: str
    subtask: Subtask
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = Field(default_factory=list)
    dependents: list[str] = Field(default_factory=list)
    
    # Execution metadata
    started_at: datetime | None = None
    completed_at: datetime | None = None
    execution_time_ms: float | None = None
    error: str | None = None
    
    # Results
    result: dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
    
    def is_ready(self, graph: "TaskGraph") -> bool:
        """Check if all dependencies are satisfied."""
        for dep_id in self.dependencies:
            dep_node = graph.get_node(dep_id)
            if dep_node is None or dep_node.status != TaskStatus.COMPLETED:
                return False
        return True
    
    def mark_running(self) -> None:
        """Mark task as running."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
    
    def mark_completed(self, result: dict[str, Any] | None = None) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            self.execution_time_ms = (
                self.completed_at - self.started_at
            ).total_seconds() * 1000
        if result:
            self.result = result
    
    def mark_failed(self, error: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error = error
        if self.started_at:
            self.execution_time_ms = (
                self.completed_at - self.started_at
            ).total_seconds() * 1000
    
    def mark_skipped(self, reason: str) -> None:
        """Mark task as skipped."""
        self.status = TaskStatus.SKIPPED
        self.error = reason
    
    def mark_blocked(self, reason: str) -> None:
        """Mark task as blocked due to dependency failure."""
        self.status = TaskStatus.BLOCKED
        self.error = reason


@dataclass
class TaskGraphStats:
    """Statistics about task graph execution."""
    total_tasks: int = 0
    pending: int = 0
    ready: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    blocked: int = 0
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_tasks == 0:
            return 0.0
        done = self.completed + self.skipped + self.failed + self.blocked
        return (done / self.total_tasks) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if all tasks are in terminal state."""
        return (
            self.completed + self.failed + self.skipped + self.blocked
            == self.total_tasks
        )


class CycleDetectedError(Exception):
    """Raised when a cycle is detected in the task graph."""
    pass


class TaskNotFoundError(Exception):
    """Raised when a task is not found in the graph."""
    pass


class TaskGraph:
    """
    Directed acyclic graph of tasks for execution.
    
    Manages:
    - Task dependencies
    - Execution ordering
    - Status tracking
    - Topological sort
    
    Thread Safety: NOT thread-safe. Designed for sequential execution.
    """
    
    def __init__(self) -> None:
        """Initialize an empty task graph."""
        self._nodes: dict[str, TaskNode] = {}
        self._execution_order: list[str] | None = None
    
    def add_task(self, subtask: Subtask) -> TaskNode:
        """
        Add a task to the graph.
        
        Args:
            subtask: The subtask to add
            
        Returns:
            The created TaskNode
            
        Raises:
            ValueError: If task ID already exists
        """
        if subtask.id in self._nodes:
            raise ValueError(f"Task {subtask.id} already exists")
        
        node = TaskNode(
            id=subtask.id,
            subtask=subtask,
            dependencies=subtask.dependencies.copy(),
        )
        self._nodes[subtask.id] = node
        
        # Update dependents of dependencies
        for dep_id in subtask.dependencies:
            if dep_id in self._nodes:
                self._nodes[dep_id].dependents.append(subtask.id)
        
        # Invalidate cached execution order
        self._execution_order = None
        
        return node
    
    def add_tasks(self, subtasks: list[Subtask]) -> list[TaskNode]:
        """
        Add multiple tasks to the graph.
        
        Args:
            subtasks: List of subtasks to add
            
        Returns:
            List of created TaskNodes
        """
        nodes = []
        for subtask in subtasks:
            node = self.add_task(subtask)
            nodes.append(node)
        return nodes
    
    def get_node(self, task_id: str) -> TaskNode | None:
        """Get a task node by ID."""
        return self._nodes.get(task_id)
    
    def get_all_nodes(self) -> list[TaskNode]:
        """Get all nodes in the graph."""
        return list(self._nodes.values())
    
    def get_ready_tasks(self) -> list[TaskNode]:
        """Get all tasks that are ready to execute."""
        ready = []
        for node in self._nodes.values():
            if node.status == TaskStatus.PENDING and node.is_ready(self):
                node.status = TaskStatus.READY
                ready.append(node)
            elif node.status == TaskStatus.READY:
                ready.append(node)
        return ready
    
    def _detect_cycle(self) -> bool:
        """
        Detect if there's a cycle in the graph using DFS.
        
        Returns:
            True if cycle detected, False otherwise
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node_id: WHITE for node_id in self._nodes}
        
        def dfs(node_id: str) -> bool:
            colors[node_id] = GRAY
            node = self._nodes[node_id]
            
            for dep_id in node.dependents:
                if dep_id not in colors:
                    continue
                if colors[dep_id] == GRAY:
                    return True  # Cycle detected
                if colors[dep_id] == WHITE:
                    if dfs(dep_id):
                        return True
            
            colors[node_id] = BLACK
            return False
        
        for node_id in self._nodes:
            if colors[node_id] == WHITE:
                if dfs(node_id):
                    return True
        
        return False
    
    def topological_sort(self) -> list[str]:
        """
        Get tasks in topological order (respecting dependencies).
        
        Returns:
            List of task IDs in execution order
            
        Raises:
            CycleDetectedError: If graph contains cycles
        """
        if self._execution_order is not None:
            return self._execution_order
        
        if self._detect_cycle():
            raise CycleDetectedError("Task graph contains cycles")
        
        # Kahn's algorithm for topological sort
        in_degree = {node_id: len(node.dependencies) 
                    for node_id, node in self._nodes.items()}
        
        # Start with nodes that have no dependencies
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Process in ID order for determinism
            queue.sort()
            node_id = queue.pop(0)
            result.append(node_id)
            
            node = self._nodes[node_id]
            for dependent_id in node.dependents:
                if dependent_id in in_degree:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        queue.append(dependent_id)
        
        # Check if all nodes were processed
        if len(result) != len(self._nodes):
            raise CycleDetectedError("Could not process all nodes - cycle likely exists")
        
        self._execution_order = result
        return result
    
    def iter_execution_order(self) -> Iterator[TaskNode]:
        """
        Iterate over tasks in execution order.
        
        Yields:
            TaskNodes in topologically sorted order
        """
        order = self.topological_sort()
        for task_id in order:
            yield self._nodes[task_id]
    
    def propagate_failure(self, failed_task_id: str) -> list[str]:
        """
        Mark all tasks dependent on a failed task as blocked.
        
        Args:
            failed_task_id: ID of the failed task
            
        Returns:
            List of task IDs that were blocked
        """
        blocked = []
        visited = set()
        
        def block_dependents(task_id: str) -> None:
            node = self._nodes.get(task_id)
            if not node:
                return
            
            for dependent_id in node.dependents:
                if dependent_id in visited:
                    continue
                visited.add(dependent_id)
                
                dependent = self._nodes.get(dependent_id)
                if dependent and dependent.status == TaskStatus.PENDING:
                    dependent.mark_blocked(f"Dependency {task_id} failed")
                    blocked.append(dependent_id)
                    block_dependents(dependent_id)
        
        block_dependents(failed_task_id)
        return blocked
    
    def get_stats(self) -> TaskGraphStats:
        """Get current statistics about the graph."""
        stats = TaskGraphStats(total_tasks=len(self._nodes))
        
        for node in self._nodes.values():
            if node.status == TaskStatus.PENDING:
                stats.pending += 1
            elif node.status == TaskStatus.READY:
                stats.ready += 1
            elif node.status == TaskStatus.RUNNING:
                stats.running += 1
            elif node.status == TaskStatus.COMPLETED:
                stats.completed += 1
            elif node.status == TaskStatus.FAILED:
                stats.failed += 1
            elif node.status == TaskStatus.SKIPPED:
                stats.skipped += 1
            elif node.status == TaskStatus.BLOCKED:
                stats.blocked += 1
        
        return stats
    
    def reset(self) -> None:
        """Reset all task statuses to PENDING."""
        for node in self._nodes.values():
            node.status = TaskStatus.PENDING
            node.started_at = None
            node.completed_at = None
            node.execution_time_ms = None
            node.error = None
            node.result = {}
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dictionary for checkpointing."""
        return {
            "nodes": {
                node_id: node.model_dump(mode="json")
                for node_id, node in self._nodes.items()
            },
            "execution_order": self._execution_order,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskGraph":
        """Deserialize graph from dictionary."""
        graph = cls()
        
        for node_id, node_data in data.get("nodes", {}).items():
            node = TaskNode(**node_data)
            graph._nodes[node_id] = node
        
        graph._execution_order = data.get("execution_order")
        return graph
    
    @classmethod
    def from_subtasks(cls, subtasks: list[Subtask]) -> "TaskGraph":
        """Create a task graph from a list of subtasks."""
        graph = cls()
        graph.add_tasks(subtasks)
        return graph
    
    def __len__(self) -> int:
        return len(self._nodes)
    
    def __contains__(self, task_id: str) -> bool:
        return task_id in self._nodes
