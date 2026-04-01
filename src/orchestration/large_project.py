"""
Large Project Mode Module

Handles large projects by sharding tasks and managing context efficiently.
Ensures the system can handle codebases larger than context window limits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class ProjectMetrics:
    """Metrics about a project's size and complexity."""
    total_files: int = 0
    total_lines: int = 0
    total_size_bytes: int = 0
    file_types: dict[str, int] = field(default_factory=dict)
    directories: int = 0
    
    @property
    def is_large(self) -> bool:
        """Check if project qualifies as 'large'."""
        return (
            self.total_files > 100 or
            self.total_lines > 50000 or
            self.total_size_bytes > 10 * 1024 * 1024  # 10MB
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "total_size_bytes": self.total_size_bytes,
            "file_types": self.file_types,
            "directories": self.directories,
            "is_large": self.is_large,
        }


class TaskShard(BaseModel):
    """A shard of tasks for execution."""
    shard_id: str
    shard_index: int
    total_shards: int
    task_ids: list[str] = Field(default_factory=list)
    target_files: list[str] = Field(default_factory=list)
    estimated_tokens: int = 0
    context_summary: str = ""  # Summary from previous shard
    
    model_config = ConfigDict(extra="forbid")


class ShardSummary(BaseModel):
    """Summary of completed shard for carry-forward."""
    shard_id: str
    completed_tasks: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    key_decisions: list[str] = Field(default_factory=list)
    remaining_work: list[str] = Field(default_factory=list)
    token_usage: int = 0
    
    def to_context_string(self, max_length: int = 2000) -> str:
        """Convert summary to context string for next shard."""
        parts = []
        
        if self.completed_tasks:
            parts.append(f"Completed: {', '.join(self.completed_tasks[:5])}")
            if len(self.completed_tasks) > 5:
                parts.append(f"  ... and {len(self.completed_tasks) - 5} more")
        
        if self.files_modified:
            parts.append(f"Modified files: {', '.join(self.files_modified[:10])}")
        
        if self.key_decisions:
            parts.append("Key decisions:")
            for decision in self.key_decisions[:5]:
                parts.append(f"  - {decision[:100]}")
        
        result = "\n".join(parts)
        if len(result) > max_length:
            result = result[:max_length - 3] + "..."
        
        return result


@dataclass  
class ShardConfig:
    """Configuration for task sharding."""
    max_files_per_shard: int = 20
    max_tasks_per_shard: int = 10
    max_tokens_per_shard: int = 30000
    min_shards: int = 1
    max_shards: int = 20
    context_carry_forward_tokens: int = 2000


class LargeProjectHandler:
    """
    Handles large project operations.
    
    Provides:
    - Project size analysis
    - Task sharding
    - Context summarization
    - Shard execution coordination
    """
    
    # File extensions to analyze
    CODE_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
        ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
        ".kt", ".scala", ".clj", ".ex", ".exs", ".hs", ".ml",
    }
    
    CONFIG_EXTENSIONS = {
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
        ".xml", ".env", ".properties",
    }
    
    DOC_EXTENSIONS = {
        ".md", ".rst", ".txt", ".adoc",
    }
    
    def __init__(
        self,
        workspace_root: Path,
        config: Optional[ShardConfig] = None,
    ) -> None:
        """
        Initialize the large project handler.
        
        Args:
            workspace_root: Root directory of the workspace
            config: Sharding configuration
        """
        self._workspace_root = workspace_root.resolve()
        self._config = config or ShardConfig()
        self._metrics: Optional[ProjectMetrics] = None
    
    def analyze_project(
        self,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> ProjectMetrics:
        """
        Analyze the project to determine size and complexity.
        
        Args:
            include_patterns: Glob patterns to include
            exclude_patterns: Glob patterns to exclude
            
        Returns:
            Project metrics
        """
        exclude_patterns = exclude_patterns or [
            "**/node_modules/**",
            "**/.git/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**",
            "**/.venv/**",
            "**/env/**",
        ]
        
        metrics = ProjectMetrics()
        seen_dirs = set()
        
        for file_path in self._workspace_root.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Check exclude patterns
            relative = file_path.relative_to(self._workspace_root)
            skip = False
            for pattern in exclude_patterns:
                import fnmatch
                if fnmatch.fnmatch(str(relative), pattern):
                    skip = True
                    break
            if skip:
                continue
            
            # Track directory
            seen_dirs.add(file_path.parent)
            
            # Get file extension
            ext = file_path.suffix.lower()
            metrics.file_types[ext] = metrics.file_types.get(ext, 0) + 1
            
            try:
                stat = file_path.stat()
                metrics.total_size_bytes += stat.st_size
                
                # Count lines for code files
                if ext in self.CODE_EXTENSIONS or ext in self.CONFIG_EXTENSIONS:
                    try:
                        content = file_path.read_text(errors="ignore")
                        metrics.total_lines += len(content.splitlines())
                    except (IOError, UnicodeDecodeError):
                        pass
                
                metrics.total_files += 1
                
            except (IOError, OSError):
                continue
        
        metrics.directories = len(seen_dirs)
        self._metrics = metrics
        
        return metrics
    
    def create_shards(
        self,
        subtasks: list[dict[str, Any]],
        file_token_estimates: Optional[dict[str, int]] = None,
    ) -> list[TaskShard]:
        """
        Create shards from subtasks.
        
        Args:
            subtasks: List of subtask dictionaries
            file_token_estimates: Optional token estimates per file
            
        Returns:
            List of task shards
        """
        if not subtasks:
            return []
        
        file_token_estimates = file_token_estimates or {}
        
        # Build dependency graph
        task_deps: dict[str, set[str]] = {}
        task_files: dict[str, list[str]] = {}
        
        for task in subtasks:
            task_id = task.get("id", "")
            task_deps[task_id] = set(task.get("dependencies", []))
            task_files[task_id] = task.get("target_files", [])
        
        # Topological sort respecting dependencies
        sorted_tasks = self._topological_sort(subtasks, task_deps)
        
        # Create shards
        shards: list[TaskShard] = []
        current_shard_tasks: list[str] = []
        current_shard_files: set[str] = set()
        current_shard_tokens: int = 0
        
        for task in sorted_tasks:
            task_id = task.get("id", "")
            task_target_files = task_files.get(task_id, [])
            task_tokens = sum(
                file_token_estimates.get(f, 500) for f in task_target_files
            )
            
            # Check if task fits in current shard
            should_split = (
                len(current_shard_tasks) >= self._config.max_tasks_per_shard or
                len(current_shard_files) + len(task_target_files) > self._config.max_files_per_shard or
                current_shard_tokens + task_tokens > self._config.max_tokens_per_shard
            )
            
            if should_split and current_shard_tasks:
                # Create shard
                shard = TaskShard(
                    shard_id=f"shard_{len(shards):03d}",
                    shard_index=len(shards),
                    total_shards=0,  # Updated later
                    task_ids=current_shard_tasks,
                    target_files=list(current_shard_files),
                    estimated_tokens=current_shard_tokens,
                )
                shards.append(shard)
                
                # Reset
                current_shard_tasks = []
                current_shard_files = set()
                current_shard_tokens = 0
            
            # Add task to current shard
            current_shard_tasks.append(task_id)
            current_shard_files.update(task_target_files)
            current_shard_tokens += task_tokens
        
        # Don't forget the last shard
        if current_shard_tasks:
            shard = TaskShard(
                shard_id=f"shard_{len(shards):03d}",
                shard_index=len(shards),
                total_shards=0,
                task_ids=current_shard_tasks,
                target_files=list(current_shard_files),
                estimated_tokens=current_shard_tokens,
            )
            shards.append(shard)
        
        # Update total_shards
        for shard in shards:
            shard.total_shards = len(shards)
        
        return shards
    
    def _topological_sort(
        self,
        subtasks: list[dict[str, Any]],
        deps: dict[str, set[str]],
    ) -> list[dict[str, Any]]:
        """Sort tasks respecting dependencies."""
        task_map = {t.get("id"): t for t in subtasks}
        visited: set[str] = set()
        result: list[dict[str, Any]] = []
        
        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            visited.add(task_id)
            
            for dep_id in deps.get(task_id, set()):
                if dep_id in task_map:
                    visit(dep_id)
            
            if task_id in task_map:
                result.append(task_map[task_id])
        
        for task_id in task_map:
            visit(task_id)
        
        return result
    
    def create_shard_summary(
        self,
        shard: TaskShard,
        completed_tasks: list[str],
        files_modified: list[str],
        key_decisions: list[str],
        remaining_tasks: list[str],
        token_usage: int,
    ) -> ShardSummary:
        """
        Create a summary of a completed shard.
        
        Args:
            shard: The completed shard
            completed_tasks: Tasks that were completed
            files_modified: Files that were modified
            key_decisions: Important decisions made
            remaining_tasks: Tasks that remain
            token_usage: Tokens used in shard
            
        Returns:
            Shard summary for carry-forward
        """
        return ShardSummary(
            shard_id=shard.shard_id,
            completed_tasks=completed_tasks,
            files_modified=files_modified,
            key_decisions=key_decisions,
            remaining_work=remaining_tasks,
            token_usage=token_usage,
        )
    
    def estimate_file_tokens(self, file_path: Path) -> int:
        """
        Estimate tokens needed to include a file in context.
        
        Args:
            file_path: Path to file
            
        Returns:
            Estimated token count
        """
        try:
            content = file_path.read_text(errors="ignore")
            # Rough estimate: ~4 chars per token
            return len(content) // 4
        except (IOError, OSError):
            return 500  # Default estimate
    
    def get_file_priority(self, file_path: Path, task_files: list[str]) -> int:
        """
        Get priority score for including a file in context.
        
        Higher score = higher priority.
        
        Args:
            file_path: Path to file
            task_files: Files targeted by current task
            
        Returns:
            Priority score (0-100)
        """
        relative = str(file_path.relative_to(self._workspace_root))
        
        # Direct target file
        if relative in task_files:
            return 100
        
        # Same directory as target
        for target in task_files:
            if Path(target).parent == file_path.parent:
                return 80
        
        # Config files
        if file_path.suffix.lower() in self.CONFIG_EXTENSIONS:
            return 60
        
        # Entry points
        if file_path.name in ("main.py", "index.js", "app.py", "server.py"):
            return 70
        
        # Test files (lower priority)
        if "test" in file_path.name.lower():
            return 30
        
        # Default
        return 40
    
    def summarize_for_context(
        self,
        file_path: Path,
        max_lines: int = 50,
    ) -> str:
        """
        Create a summarized version of a file for context.
        
        Args:
            file_path: Path to file
            max_lines: Maximum lines to include
            
        Returns:
            Summarized content
        """
        try:
            content = file_path.read_text(errors="ignore")
            lines = content.splitlines()
            
            if len(lines) <= max_lines:
                return content
            
            # Extract key parts
            result_lines = []
            
            # Imports/includes (first 10 lines)
            result_lines.extend(lines[:10])
            result_lines.append("# ... imports truncated ...")
            
            # Function/class signatures
            for i, line in enumerate(lines):
                stripped = line.strip()
                if (stripped.startswith("def ") or 
                    stripped.startswith("class ") or
                    stripped.startswith("async def ") or
                    stripped.startswith("function ") or
                    stripped.startswith("export ")):
                    result_lines.append(line)
                    # Include docstring if present
                    if i + 1 < len(lines) and '"""' in lines[i + 1]:
                        result_lines.append(lines[i + 1])
            
            # Limit total
            if len(result_lines) > max_lines:
                result_lines = result_lines[:max_lines]
                result_lines.append("# ... truncated ...")
            
            return "\n".join(result_lines)
            
        except (IOError, OSError):
            return f"# Unable to read {file_path}"
