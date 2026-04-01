"""
Agent Base Module

Defines the abstract base class and contracts for all agents.
Each agent has strict input/output schemas and behavioral constraints.

Design Decisions:
- Agents are stateless executors
- Input/output schemas are enforced via Pydantic
- Agents CANNOT directly access files or run commands
- All side effects go through mediated tools
"""

from __future__ import annotations

import time
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from src.config import AgentPolicy, Configuration
from src.core.context_manager import ContextManager
from src.core.llm_client import CompletionRequest, CompletionResponse, LLMClient, Message
from src.core.telemetry import TelemetryCollector
from src.core.memory import MemoryManager


# =============================================================================
# AGENT TYPE DEFINITIONS
# =============================================================================

class AgentType(str, Enum):
    """Types of agents in the system."""
    PLANNER = "planner"
    CODER = "coder"
    REVIEWER = "reviewer"
    FIXER = "fixer"


class AgentStatus(str, Enum):
    """Status of an agent execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REJECTED = "rejected"  # Policy violation


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class AgentInput(BaseModel):
    """Base class for agent inputs."""
    task_id: str = Field(..., description="Unique task identifier")
    run_id: str = Field(..., description="Run identifier for tracking")
    
    model_config = ConfigDict(extra="forbid")  # No extra fields allowed


class ToolCall(BaseModel):
    """A tool or skill call to be executed by the orchestrator."""
    tool_name: str = Field(..., description="Name of the tool/skill to call")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")
    
    model_config = ConfigDict(extra="allow")


class AgentOutput(BaseModel):
    """Base class for agent outputs."""
    agent_type: AgentType
    task_id: str
    status: AgentStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Execution metadata
    execution_time_ms: float = 0
    tokens_used: int = 0
    retries: int = 0
    
    # Error information
    error: str | None = None
    error_context: dict[str, Any] = Field(default_factory=dict)
    
    # Tools
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Optional tools to run")
    
    model_config = ConfigDict(extra="forbid")


# =============================================================================
# PLANNER AGENT CONTRACTS
# =============================================================================

class PlannerInput(AgentInput):
    """Input contract for Planner agent."""
    task_description: str = Field(..., min_length=10, max_length=5000)
    workspace_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Information about the workspace (file list, structure, etc.)"
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Explicit constraints on the planning"
    )
    previous_attempt: str | None = Field(
        None,
        description="Previous planning attempt if this is a retry"
    )


class Subtask(BaseModel):
    """A single subtask in the plan."""
    id: str = Field(..., description="Unique subtask identifier")
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=10, max_length=1000)
    acceptance_criteria: list[str] = Field(
        ...,
        min_length=1,
        description="Conditions for considering subtask complete"
    )
    target_files: list[str] = Field(
        default_factory=list,
        description="Files that will likely be modified"
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of subtasks this depends on"
    )
    estimated_complexity: str = Field(
        "medium",
        description="low, medium, or high"
    )
    
    model_config = ConfigDict(extra="forbid")


class PlannerOutput(AgentOutput):
    """Output contract for Planner agent."""
    agent_type: AgentType = AgentType.PLANNER
    
    # Planning results
    plan_summary: str = Field("", description="High-level summary of the approach")
    subtasks: list[Subtask] = Field(default_factory=list)
    
    # Analysis
    identified_risks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    
    # Metadata
    requires_clarification: bool = False
    clarification_questions: list[str] = Field(default_factory=list)


# =============================================================================
# CODER AGENT CONTRACTS
# =============================================================================

class CoderInput(AgentInput):
    """Input contract for Coder agent."""
    subtask: Subtask = Field(..., description="The subtask to implement")
    file_contents: dict[str, str] = Field(
        default_factory=dict,
        description="Current content of relevant files"
    )
    context: str = Field("", description="Additional context from planner/reviewer")
    constraints: list[str] = Field(default_factory=list)
    previous_attempt: str | None = Field(
        None,
        description="Previous coding attempt if this is a retry"
    )


class CodeChange(BaseModel):
    """A single code change."""
    file_path: str = Field(..., description="Path to the file")
    change_type: str = Field(..., description="create, modify, or delete")
    description: str = Field(..., description="What this change does")
    
    # For modifications
    original_content: str | None = None
    new_content: str = Field("", description="New file content")
    
    # Diff information
    diff: str | None = Field(None, description="Unified diff format")
    hunks: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Optional patch hunks for surgical edits. "
            "Each hunk supports start_line, end_line, original_content, new_content, "
            "context_before, context_after."
        ),
    )
    
    # Metadata
    lines_added: int = 0
    lines_removed: int = 0
    
    model_config = ConfigDict(extra="forbid")


class CoderOutput(AgentOutput):
    """Output contract for Coder agent."""
    agent_type: AgentType = AgentType.CODER
    
    # Code changes
    changes: list[CodeChange] = Field(default_factory=list)
    
    # Explanation
    implementation_notes: str = Field("", description="Explanation of the approach")
    
    # Self-assessment
    confidence: str = Field("medium", description="low, medium, or high")
    concerns: list[str] = Field(default_factory=list)
    
    # Testing suggestions
    suggested_tests: list[str] = Field(default_factory=list)


# =============================================================================
# REVIEWER AGENT CONTRACTS
# =============================================================================

class ReviewerInput(AgentInput):
    """Input contract for Reviewer agent."""
    subtask: Subtask = Field(..., description="The subtask that was implemented")
    code_changes: list[CodeChange] = Field(..., description="Changes to review")
    original_files: dict[str, str] = Field(
        default_factory=dict,
        description="Original file contents before changes"
    )
    implementation_notes: str = Field("", description="Coder's notes")


class ReviewIssue(BaseModel):
    """A single issue found in review."""
    severity: str = Field(..., description="critical, major, minor, suggestion")
    file_path: str = Field(..., description="File where issue was found")
    line_range: str | None = Field(None, description="e.g., '10-15'")
    description: str = Field(..., description="What the issue is")
    suggestion: str = Field("", description="How to fix it")
    
    model_config = ConfigDict(extra="forbid")


class ReviewVerdict(str, Enum):
    """Possible review verdicts."""
    APPROVE = "APPROVE"
    REQUEST_CHANGES = "REQUEST_CHANGES"
    REJECT = "REJECT"


class ReviewerOutput(AgentOutput):
    """Output contract for Reviewer agent."""
    agent_type: AgentType = AgentType.REVIEWER
    
    # Verdict
    verdict: ReviewVerdict = ReviewVerdict.REQUEST_CHANGES
    
    # EXPLICIT TERMINAL STATE: True means task is complete, stop all loops
    # This is the authoritative signal that no more coder/fixer runs are needed
    task_complete: bool = Field(
        default=False,
        description="TERMINAL STATE: True if task fully meets acceptance criteria. "
                    "When True, the orchestrator MUST stop and NOT invoke coder/fixer again."
    )
    
    # Issues found
    issues: list[ReviewIssue] = Field(default_factory=list)
    
    # Analysis
    summary: str = Field("", description="Overall assessment")
    strengths: list[str] = Field(default_factory=list)
    
    # Acceptance criteria check
    criteria_met: dict[str, bool] = Field(
        default_factory=dict,
        description="Which acceptance criteria were satisfied"
    )


# =============================================================================
# FIXER AGENT CONTRACTS
# =============================================================================

class FixerInput(AgentInput):
    """Input contract for Fixer agent."""
    original_changes: list[CodeChange] = Field(..., description="Original code changes")
    review_issues: list[ReviewIssue] = Field(..., description="Issues to fix")
    file_contents: dict[str, str] = Field(
        default_factory=dict,
        description="Current file contents"
    )


class FixerOutput(AgentOutput):
    """Output contract for Fixer agent."""
    agent_type: AgentType = AgentType.FIXER
    
    # Fixed changes
    fixed_changes: list[CodeChange] = Field(default_factory=list)
    
    # Issue resolution
    issues_addressed: list[str] = Field(
        default_factory=list,
        description="Which issues were addressed"
    )
    issues_not_addressed: list[str] = Field(
        default_factory=list,
        description="Issues that couldn't be addressed and why"
    )
    
    # Explanation
    fix_notes: str = Field("", description="Explanation of fixes")


# =============================================================================
# AGENT EXECUTION CONTEXT
# =============================================================================

@dataclass
class AgentContext:
    """
    Execution context provided to agents.
    
    Contains everything an agent needs to execute,
    but carefully controls what they can access.
    """
    run_id: str
    agent_type: AgentType
    config: Configuration
    llm_client: LLMClient
    context_manager: ContextManager
    telemetry: TelemetryCollector | None = None
    memory_manager: MemoryManager | None = None
    
    # Execution constraints (from policy)
    max_tokens: int = 8000
    max_retries: int = 3
    timeout_seconds: float = 600
    
    # Metadata
    start_time: float = field(default_factory=time.perf_counter)
    
    @property
    def elapsed_seconds(self) -> float:
        """Time elapsed since execution started."""
        return time.perf_counter() - self.start_time
    
    @property
    def policy(self) -> AgentPolicy:
        """Get policy for this agent type."""
        return self.config.get_agent_policy(self.agent_type.value)


# =============================================================================
# ABSTRACT BASE AGENT
# =============================================================================

InputT = TypeVar("InputT", bound=AgentInput)
OutputT = TypeVar("OutputT", bound=AgentOutput)


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for all agents.
    
    Agents are stateless executors that:
    - Take typed input
    - Produce typed output
    - Cannot directly access files or run commands
    - Are constrained by explicit policies
    
    Subclasses must implement:
    - agent_type: The type of this agent
    - system_prompt: The system prompt for the LLM
    - _execute_impl: The core execution logic
    """
    
    def __init__(self) -> None:
        """Initialize the agent."""
        self._execution_count = 0
    
    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        ...
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        ...
    
    @abstractmethod
    def _execute_impl(
        self,
        input_data: InputT,
        context: AgentContext,
    ) -> OutputT:
        """
        Core execution logic to be implemented by subclasses.
        
        Args:
            input_data: Validated input data
            context: Execution context
            
        Returns:
            Agent output
        """
        ...
    
    @abstractmethod
    def _parse_response(
        self,
        response: str,
        input_data: InputT,
        context: AgentContext,
    ) -> OutputT:
        """
        Parse LLM response into typed output.
        
        Args:
            response: Raw LLM response
            input_data: Original input
            context: Execution context
            
        Returns:
            Parsed and validated output
        """
        ...
    
    def _validate_input(
        self,
        input_data: InputT,
        context: AgentContext,
    ) -> list[str]:
        """
        Validate input against policy constraints.
        
        Args:
            input_data: Input to validate
            context: Execution context
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors: list[str] = []
        
        # Subclasses can override for additional validation
        return errors
    
    def _validate_output(
        self,
        output: OutputT,
        context: AgentContext,
    ) -> list[str]:
        """
        Validate output against policy constraints.
        
        Args:
            output: Output to validate
            context: Execution context
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors: list[str] = []
        
        # Check for sensitive data in output
        if context.config.policies.safety.output_validation.validate_outputs:
            output_str = str(output.model_dump())
            for pattern in context.config.policies.safety.output_validation.block_sensitive_patterns:
                if pattern.lower() in output_str.lower():
                    errors.append(f"Output contains blocked pattern: {pattern}")
        
        return errors
    
    def _sanitize_input(self, text: str, context: AgentContext) -> str:
        """
        Sanitize input text to prevent prompt injection.
        
        Args:
            text: Text to sanitize
            context: Execution context
            
        Returns:
            Sanitized text
        """
        if not context.config.policies.safety.prompt_injection.sanitize_inputs:
            return text
        
        sanitized = text
        for pattern in context.config.policies.safety.prompt_injection.block_patterns:
            if pattern and re.search(re.escape(pattern), sanitized, flags=re.IGNORECASE):
                # Case-insensitive replacement with consistent marker.
                sanitized = re.sub(re.escape(pattern), "[BLOCKED]", sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def execute(
        self,
        input_data: InputT,
        context: AgentContext,
    ) -> OutputT:
        """
        Execute the agent with full validation and safety checks.
        
        This is the main entry point for agent execution.
        
        Args:
            input_data: Typed input data
            context: Execution context
            
        Returns:
            Typed output data
        """
        start_time = time.perf_counter()
        self._execution_count += 1
        
        # Record start in telemetry
        if context.telemetry:
            context.telemetry.record_agent_start(
                agent_id=self.agent_type.value,
                input_summary=str(input_data.model_dump())[:500],
            )
        
        try:
            # Validate input
            input_errors = self._validate_input(input_data, context)
            if input_errors:
                return self._create_error_output(
                    input_data,
                    context,
                    AgentStatus.REJECTED,
                    f"Input validation failed: {'; '.join(input_errors)}",
                    start_time,
                )
            
            max_attempts = max(1, int(context.max_retries) + 1)
            last_output: OutputT | None = None

            for attempt in range(max_attempts):
                if context.elapsed_seconds > context.timeout_seconds:
                    if context.telemetry:
                        context.telemetry.record_warning(
                            "agent_timeout",
                            context={
                                "agent": self.agent_type.value,
                                "attempt": attempt,
                                "elapsed_seconds": round(context.elapsed_seconds, 3),
                                "timeout_seconds": context.timeout_seconds,
                            },
                        )
                    return self._create_error_output(
                        input_data,
                        context,
                        AgentStatus.TIMEOUT,
                        f"Agent timeout exceeded ({context.timeout_seconds}s)",
                        start_time,
                    )

                # Execute implementation
                output = self._execute_impl(input_data, context)

                # Validate output
                output_errors = self._validate_output(output, context)
                if output_errors:
                    output = self._create_error_output(
                        input_data,
                        context,
                        AgentStatus.REJECTED,
                        f"Output validation failed: {'; '.join(output_errors)}",
                        start_time,
                    )

                if context.elapsed_seconds > context.timeout_seconds:
                    if context.telemetry:
                        context.telemetry.record_warning(
                            "agent_timeout",
                            context={
                                "agent": self.agent_type.value,
                                "attempt": attempt,
                                "elapsed_seconds": round(context.elapsed_seconds, 3),
                                "timeout_seconds": context.timeout_seconds,
                            },
                        )
                    return self._create_error_output(
                        input_data,
                        context,
                        AgentStatus.TIMEOUT,
                        f"Agent timeout exceeded ({context.timeout_seconds}s)",
                        start_time,
                    )

                output.retries = attempt
                last_output = output

                if output.status == AgentStatus.SUCCESS:
                    break

                if attempt < max_attempts - 1 and context.telemetry:
                    context.telemetry.record_warning(
                        "agent_retry",
                        context={
                            "agent": self.agent_type.value,
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "status": output.status.value,
                            "error": output.error,
                        },
                    )

            assert last_output is not None

            # Update execution time
            last_output.execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Record completion in telemetry
            if context.telemetry:
                context.telemetry.record_agent_end(
                    agent_id=self.agent_type.value,
                    success=last_output.status == AgentStatus.SUCCESS,
                    output_summary=str(last_output.model_dump())[:500],
                )

            return last_output
            
        except Exception as e:
            # Record error
            if context.telemetry:
                context.telemetry.record_error(
                    error=str(e),
                    context={"agent": self.agent_type.value},
                )
                context.telemetry.record_agent_end(
                    agent_id=self.agent_type.value,
                    success=False,
                    output_summary=f"Error: {e}",
                )
            
            return self._create_error_output(
                input_data,
                context,
                AgentStatus.FAILED,
                str(e),
                start_time,
            )
    
    @abstractmethod
    def _create_error_output(
        self,
        input_data: InputT,
        context: AgentContext,
        status: AgentStatus,
        error: str,
        start_time: float,
    ) -> OutputT:
        """
        Create an error output when execution fails.
        
        Must be implemented by subclasses to return the correct output type.
        """
        ...
    
    def _call_llm(
        self,
        user_message: str,
        context: AgentContext,
        max_tokens: int | None = None,
    ) -> CompletionResponse:
        """
        Make an LLM call with proper context management.
        
        Args:
            user_message: The user message to send
            context: Execution context
            max_tokens: Optional token limit override
            
        Returns:
            LLM completion response
        """
        if context.elapsed_seconds > context.timeout_seconds:
            raise TimeoutError(f"Agent timeout exceeded ({context.timeout_seconds}s)")

        # Build messages
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_message),
        ]

        # Token management: cap generation by available budget after prompt size.
        prompt_text = f"{self.system_prompt}\n\n{user_message}"
        prompt_tokens = context.context_manager.count_tokens(prompt_text)
        requested_max = max_tokens if max_tokens is not None else context.max_tokens
        available_for_completion = max(64, int(context.max_tokens) - int(prompt_tokens))
        effective_max_tokens = max(1, min(int(requested_max), available_for_completion))

        if context.telemetry and effective_max_tokens < int(requested_max):
            context.telemetry.record_warning(
                "agent_token_budget_adjusted",
                context={
                    "agent": self.agent_type.value,
                    "requested_max_tokens": int(requested_max),
                    "effective_max_tokens": effective_max_tokens,
                    "prompt_tokens": int(prompt_tokens),
                    "context_max_tokens": int(context.max_tokens),
                },
            )
        
        # Create request
        request = CompletionRequest(
            messages=messages,
            max_tokens=effective_max_tokens,
        )
        
        # Make call
        response = context.llm_client.complete(
            request,
            token_budget=context.max_tokens,
        )
        
        return response
    
    @property
    def execution_count(self) -> int:
        """Number of times this agent has been executed."""
        return self._execution_count
