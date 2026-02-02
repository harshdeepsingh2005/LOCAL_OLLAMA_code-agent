"""
Context Manager Module

Manages context windows for agents, handling token budgets,
content prioritization, and context compression.

Design Decisions:
- Strict token budget enforcement
- Hierarchical context importance
- Automatic truncation strategies
- File content caching
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import tiktoken
from pydantic import BaseModel, Field


class ContextPriority(Enum):
    """Priority levels for context items."""
    CRITICAL = auto()  # System prompts, task description
    HIGH = auto()      # Current file being edited
    MEDIUM = auto()    # Related files, recent history
    LOW = auto()       # Background context, examples


class ContextType(str, Enum):
    """Types of context content."""
    SYSTEM = "system"
    TASK = "task"
    FILE_CONTENT = "file_content"
    DIFF = "diff"
    REVIEW = "review"
    ERROR = "error"
    HISTORY = "history"
    EXAMPLE = "example"


class ContextItem(BaseModel):
    """A single item of context."""
    content: str
    context_type: ContextType
    priority: ContextPriority
    source: str | None = None  # e.g., file path
    tokens: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __lt__(self, other: "ContextItem") -> bool:
        """Compare by priority for sorting."""
        return self.priority.value < other.priority.value


@dataclass
class ContextBudget:
    """Token budget configuration."""
    total_tokens: int = 4096
    system_reserved: int = 500
    response_reserved: int = 1024
    
    @property
    def available_for_context(self) -> int:
        """Tokens available for context content."""
        return self.total_tokens - self.system_reserved - self.response_reserved


@dataclass
class ContextSnapshot:
    """A snapshot of context state for checkpointing."""
    items: list[ContextItem]
    total_tokens: int
    budget: ContextBudget
    timestamp: float


class ContextManager:
    """
    Manages context windows for agent interactions.
    
    Responsibilities:
    - Track and prioritize context items
    - Enforce token budgets
    - Compress and truncate as needed
    - Provide context for LLM calls
    
    Thread Safety: NOT thread-safe. Designed for sequential execution.
    """
    
    # Default truncation settings
    DEFAULT_TRUNCATION_STRATEGY = "priority"  # priority, fifo, or proportional
    FILE_PREVIEW_LINES = 50  # Lines to show when truncating files
    
    def __init__(
        self,
        budget: ContextBudget | None = None,
        model_context_length: int = 4096,
    ) -> None:
        """
        Initialize the context manager.
        
        Args:
            budget: Token budget configuration
            model_context_length: Model's context length for validation
        """
        self._budget = budget or ContextBudget(total_tokens=model_context_length)
        self._items: list[ContextItem] = []
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._file_cache: dict[str, str] = {}
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._tokenizer.encode(text))
    
    def add(
        self,
        content: str,
        context_type: ContextType,
        priority: ContextPriority,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Add a context item.
        
        Args:
            content: Content to add
            context_type: Type of context
            priority: Priority level
            source: Optional source identifier
            metadata: Optional metadata
            
        Returns:
            True if item was added, False if rejected
        """
        tokens = self.count_tokens(content)
        
        item = ContextItem(
            content=content,
            context_type=context_type,
            priority=priority,
            source=source,
            tokens=tokens,
            metadata=metadata or {},
        )
        
        self._items.append(item)
        return True
    
    def add_file(
        self,
        path: Path,
        content: str,
        priority: ContextPriority = ContextPriority.MEDIUM,
    ) -> bool:
        """
        Add file content to context.
        
        Args:
            path: File path
            content: File content
            priority: Priority level
            
        Returns:
            True if added
        """
        # Cache the content
        self._file_cache[str(path)] = content
        
        return self.add(
            content=f"File: {path}\n```\n{content}\n```",
            context_type=ContextType.FILE_CONTENT,
            priority=priority,
            source=str(path),
            metadata={"file_path": str(path), "line_count": len(content.splitlines())},
        )
    
    def add_diff(
        self,
        file_path: Path,
        diff_content: str,
        priority: ContextPriority = ContextPriority.HIGH,
    ) -> bool:
        """
        Add a diff to context.
        
        Args:
            file_path: File the diff applies to
            diff_content: Unified diff content
            priority: Priority level
            
        Returns:
            True if added
        """
        return self.add(
            content=f"Diff for {file_path}:\n```diff\n{diff_content}\n```",
            context_type=ContextType.DIFF,
            priority=priority,
            source=str(file_path),
            metadata={"file_path": str(file_path)},
        )
    
    def add_error(
        self,
        error: str,
        context: str | None = None,
        priority: ContextPriority = ContextPriority.HIGH,
    ) -> bool:
        """
        Add an error to context.
        
        Args:
            error: Error message
            context: Additional context
            priority: Priority level
            
        Returns:
            True if added
        """
        content = f"Error: {error}"
        if context:
            content += f"\nContext: {context}"
        
        return self.add(
            content=content,
            context_type=ContextType.ERROR,
            priority=priority,
            metadata={"error": error},
        )
    
    def get_total_tokens(self) -> int:
        """Get total tokens across all context items."""
        return sum(item.tokens for item in self._items)
    
    def get_items_by_type(self, context_type: ContextType) -> list[ContextItem]:
        """Get all items of a specific type."""
        return [item for item in self._items if item.context_type == context_type]
    
    def get_items_by_priority(self, priority: ContextPriority) -> list[ContextItem]:
        """Get all items of a specific priority."""
        return [item for item in self._items if item.priority == priority]
    
    def remove_by_source(self, source: str) -> int:
        """
        Remove items by source.
        
        Args:
            source: Source identifier to remove
            
        Returns:
            Number of items removed
        """
        original_count = len(self._items)
        self._items = [item for item in self._items if item.source != source]
        return original_count - len(self._items)
    
    def remove_by_type(self, context_type: ContextType) -> int:
        """
        Remove items by type.
        
        Args:
            context_type: Type to remove
            
        Returns:
            Number of items removed
        """
        original_count = len(self._items)
        self._items = [item for item in self._items if item.context_type != context_type]
        return original_count - len(self._items)
    
    def clear(self, keep_system: bool = True) -> None:
        """
        Clear all context items.
        
        Args:
            keep_system: Whether to keep system context
        """
        if keep_system:
            self._items = [
                item for item in self._items 
                if item.context_type == ContextType.SYSTEM
            ]
        else:
            self._items = []
        self._file_cache.clear()
    
    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within token limit."""
        tokens = self._tokenizer.encode(content)
        if len(tokens) <= max_tokens:
            return content
        
        # Decode truncated tokens
        truncated = self._tokenizer.decode(tokens[:max_tokens])
        return truncated + "\n... [truncated]"
    
    def _truncate_file_content(self, content: str, max_lines: int | None = None) -> str:
        """Truncate file content intelligently."""
        lines = content.splitlines()
        max_lines = max_lines or self.FILE_PREVIEW_LINES
        
        if len(lines) <= max_lines:
            return content
        
        # Show beginning and end
        half = max_lines // 2
        beginning = lines[:half]
        end = lines[-half:]
        
        return "\n".join(beginning) + f"\n\n... [{len(lines) - max_lines} lines omitted] ...\n\n" + "\n".join(end)
    
    def fit_to_budget(self, strategy: str = "priority") -> list[ContextItem]:
        """
        Fit context to token budget using specified strategy.
        
        Args:
            strategy: Truncation strategy (priority, fifo, proportional)
            
        Returns:
            List of context items that fit in budget
        """
        available = self._budget.available_for_context
        
        if strategy == "priority":
            return self._fit_by_priority(available)
        elif strategy == "fifo":
            return self._fit_fifo(available)
        elif strategy == "proportional":
            return self._fit_proportional(available)
        else:
            return self._fit_by_priority(available)
    
    def _fit_by_priority(self, available_tokens: int) -> list[ContextItem]:
        """Fit items by priority, keeping highest priority first."""
        # Sort by priority (highest first)
        sorted_items = sorted(self._items, key=lambda x: x.priority.value)
        
        result: list[ContextItem] = []
        used_tokens = 0
        
        for item in sorted_items:
            if used_tokens + item.tokens <= available_tokens:
                result.append(item)
                used_tokens += item.tokens
            elif item.priority == ContextPriority.CRITICAL:
                # Always include critical items, truncate if needed
                remaining = available_tokens - used_tokens
                if remaining > 100:  # Minimum useful size
                    truncated_content = self._truncate_content(item.content, remaining - 50)
                    truncated_item = ContextItem(
                        content=truncated_content,
                        context_type=item.context_type,
                        priority=item.priority,
                        source=item.source,
                        tokens=self.count_tokens(truncated_content),
                        metadata=item.metadata,
                    )
                    result.append(truncated_item)
                    used_tokens += truncated_item.tokens
        
        return result
    
    def _fit_fifo(self, available_tokens: int) -> list[ContextItem]:
        """Fit items in order they were added."""
        result: list[ContextItem] = []
        used_tokens = 0
        
        for item in self._items:
            if used_tokens + item.tokens <= available_tokens:
                result.append(item)
                used_tokens += item.tokens
        
        return result
    
    def _fit_proportional(self, available_tokens: int) -> list[ContextItem]:
        """Fit items by proportionally truncating each."""
        total_tokens = self.get_total_tokens()
        
        if total_tokens <= available_tokens:
            return self._items.copy()
        
        ratio = available_tokens / total_tokens
        result: list[ContextItem] = []
        
        for item in self._items:
            target_tokens = int(item.tokens * ratio)
            if target_tokens > 50:  # Minimum useful size
                truncated_content = self._truncate_content(item.content, target_tokens)
                truncated_item = ContextItem(
                    content=truncated_content,
                    context_type=item.context_type,
                    priority=item.priority,
                    source=item.source,
                    tokens=self.count_tokens(truncated_content),
                    metadata=item.metadata,
                )
                result.append(truncated_item)
        
        return result
    
    def build_messages(
        self,
        system_prompt: str,
        user_message: str,
    ) -> list[dict[str, str]]:
        """
        Build message list for LLM call.
        
        Args:
            system_prompt: System prompt content
            user_message: User message content
            
        Returns:
            List of message dictionaries
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add context items as part of the system context or as separate messages
        context_items = self.fit_to_budget()
        
        if context_items:
            context_text = "\n\n".join(item.content for item in context_items)
            messages.append({
                "role": "user",
                "content": f"Context:\n{context_text}\n\n{user_message}"
            })
        else:
            messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def create_snapshot(self) -> ContextSnapshot:
        """Create a snapshot of current context state."""
        import time
        return ContextSnapshot(
            items=self._items.copy(),
            total_tokens=self.get_total_tokens(),
            budget=self._budget,
            timestamp=time.time(),
        )
    
    def restore_snapshot(self, snapshot: ContextSnapshot) -> None:
        """Restore context from a snapshot."""
        self._items = snapshot.items.copy()
        self._budget = snapshot.budget
    
    @property
    def budget(self) -> ContextBudget:
        """Get current budget configuration."""
        return self._budget
    
    @budget.setter
    def budget(self, value: ContextBudget) -> None:
        """Set budget configuration."""
        self._budget = value
    
    @property
    def items(self) -> list[ContextItem]:
        """Get all context items."""
        return self._items.copy()
    
    def get_stats(self) -> dict[str, Any]:
        """Get context statistics."""
        return {
            "total_items": len(self._items),
            "total_tokens": self.get_total_tokens(),
            "available_tokens": self._budget.available_for_context,
            "over_budget": self.get_total_tokens() > self._budget.available_for_context,
            "items_by_type": {
                ct.value: len(self.get_items_by_type(ct))
                for ct in ContextType
            },
            "items_by_priority": {
                cp.name: len(self.get_items_by_priority(cp))
                for cp in ContextPriority
            },
        }
