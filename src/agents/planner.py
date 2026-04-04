"""
Planner Agent

Responsible for task decomposition and planning.
Does NOT generate code - only creates structured plans.

Design Decisions:
- Output is always a list of subtasks
- Each subtask has acceptance criteria
- Cannot access file contents directly
- Limited to planning operations only
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.agents.base import (
    AgentContext,
    AgentStatus,
    AgentType,
    BaseAgent,
    PlannerInput,
    PlannerOutput,
    Subtask,
    SubtaskToolPlan,
    ToolPlanStep,
    ToolCall,
)
from src.agents.json_utils import parse_json_object
from src.core.agent_tools import TOOL_SCHEMAS, get_tools_system_prompt


ALLOWED_TOOL_NAMES: set[str] = {schema["name"] for schema in TOOL_SCHEMAS if "name" in schema}


class PlannerSubtaskSchema(BaseModel):
    """Schema for a planner-produced subtask before conversion to contracts."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=5, max_length=200)
    description: str = Field(min_length=10, max_length=1000)
    acceptance_criteria: list[str] = Field(default_factory=list, min_length=1)
    target_files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    tool_plan: PlannerSubtaskToolPlanSchema | None = None
    estimated_complexity: str = Field(default="medium")
    estimated_iterations: int | None = None
    fallback_strategy: str | None = None


class PlannerToolCallSchema(BaseModel):
    """Schema for planner tool calls."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class PlannerToolPlanStepSchema(BaseModel):
    """Schema for one deterministic tool-plan step."""

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    reason: str = Field(min_length=3)
    arguments: dict[str, Any] = Field(default_factory=dict)
    fallback: "PlannerToolPlanStepSchema | None" = None


class PlannerSubtaskToolPlanSchema(BaseModel):
    """Schema for bounded deterministic tool-plan emitted for a subtask."""

    model_config = ConfigDict(extra="forbid")

    steps: list[PlannerToolPlanStepSchema] = Field(default_factory=list, max_length=3)


class PlannerResponseSchema(BaseModel):
    """Schema for raw planner LLM response."""

    model_config = ConfigDict(extra="allow")

    plan_summary: str = Field(default="")
    subtasks: list[PlannerSubtaskSchema] = Field(default_factory=list)
    identified_risks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_questions: list[str] = Field(default_factory=list)
    tool_calls: list[PlannerToolCallSchema] = Field(default_factory=list)


class PlannerAgent(BaseAgent[PlannerInput, PlannerOutput]):
    """
    Plans and decomposes tasks into actionable subtasks.
    
    The Planner:
    - Analyzes the task description
    - Identifies what needs to be done
    - Creates ordered subtasks with acceptance criteria
    - Does NOT generate any code
    """
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.PLANNER
    
    @property
    def system_prompt(self) -> str:
        tools_prompt = get_tools_system_prompt()
        return """You are a senior software architect and technical planner.

Your role is to analyze tasks and create detailed implementation plans.
You do NOT write code - you only create plans.

## Your Responsibilities:
1. Understand the task requirements fully
2. Break down complex tasks into smaller, manageable subtasks
3. Define clear acceptance criteria for each subtask
4. Identify target files that will likely be modified
5. Order subtasks based on dependencies
6. Identify risks and assumptions
7. If needed, call tools to gather context or skills before finalizing the plan

## Available Tools:
You can call tools to inspect code, search context, run safe commands, and gather evidence before finalizing the plan.
When uncertain, use tools first; do not guess.

""" + tools_prompt + """

## Tool Usage Rules:
- Follow a two-pass planning approach: (1) gather evidence with minimal tools, (2) emit final executable plan.
- Use tool calls iteratively until you have enough evidence.
- Prefer lightweight reads/searches first, then targeted commands.
- Include tool calls in `tool_calls` when additional context is required.
- If all required context is already available, return `tool_calls: []`.
- Ground your plan in repository evidence: map each major step to real files/folders.
- If requested behavior does not match visible workspace context, state that mismatch explicitly in risks/assumptions.
- Do not invent file paths; prefer discovered files from workspace context.
- Prefer sectioned reasoning: separately consider architecture, files, constraints, and acceptance criteria.

## Output Format:
You MUST respond with a valid JSON object in this exact format:
```json
{
    "plan_summary": "Brief overview of the implementation approach",
    "subtasks": [
        {
            "id": "1",
            "title": "Short task title",
            "description": "Detailed description of what needs to be done",
            "acceptance_criteria": ["Criterion 1", "Criterion 2"],
            "target_files": ["path/to/file.py"],
            "dependencies": [],
            "tool_plan": {
                "steps": [
                    {
                        "tool": "read_file",
                        "reason": "Inspect existing implementation before edits",
                        "arguments": {"path": "src/main.py"},
                        "fallback": {
                            "tool": "grep_workspace",
                            "reason": "Fallback discovery if path moved",
                            "arguments": {"pattern": "def main"}
                        }
                    }
                ]
            },
            "estimated_complexity": "low|medium|high",
            "estimated_iterations": 1,
            "fallback_strategy": "targeted_replan|evidence_gather_then_retry|scope_reduce"
        }
    ],
    "identified_risks": ["Risk 1", "Risk 2"],
    "assumptions": ["Assumption 1", "Assumption 2"],
    "requires_clarification": false,
    "clarification_questions": [],
    "tool_calls": [
        {"tool_name": "read_memory", "arguments": {}}
    ]
}
```

## Rules:
- Maximum 10 subtasks
- If a subtask includes `tool_plan`, it must contain at most 3 ordered steps
- For simple, single-file edits (especially docs/readme wording changes), produce exactly 1 focused subtask.
- Each subtask must have at least 1 acceptance criterion
- If the task is unclear, set requires_clarification to true and list questions
- Do not include code in your response
- Be specific about file paths when possible
- Consider edge cases and error handling in your plan"""
    
    def _validate_input(
        self,
        input_data: PlannerInput,
        context: AgentContext,
    ) -> list[str]:
        """Validate planner input."""
        errors = super()._validate_input(input_data, context)
        
        # Check task description length
        if len(input_data.task_description) < 10:
            errors.append("Task description too short")
        
        if len(input_data.task_description) > 5000:
            errors.append("Task description too long")
        
        return errors
    
    def _execute_impl(
        self,
        input_data: PlannerInput,
        context: AgentContext,
    ) -> PlannerOutput:
        """Execute the planning process."""
        # Build the prompt
        prompt = self._build_prompt(input_data, context)
        
        # Sanitize for injection
        prompt = self._sanitize_input(prompt, context)
        
        # Call LLM
        response = self._call_llm(prompt, context)
        
        # Parse response
        output = self._parse_response(response.content, input_data, context)
        output.tokens_used = response.total_tokens
        
        return output
    
    def _build_prompt(
        self,
        input_data: PlannerInput,
        context: AgentContext,
    ) -> str:
        """Build the prompt for the LLM."""
        parts = [
            "## Task to Plan:",
            input_data.task_description,
            "",
        ]
        
        # Inject memory if available
        if context.memory_manager:
            parts.append(context.memory_manager.get_all_context())
            parts.append("")
        
        # Add workspace context if available
        if input_data.workspace_context:
            parts.append("## Workspace Context:")
            if "route_domain" in input_data.workspace_context:
                parts.append(f"Task domain: {input_data.workspace_context['route_domain']}")
            if "module_hints" in input_data.workspace_context and input_data.workspace_context["module_hints"]:
                parts.append(
                    "Module hints: " + ", ".join(input_data.workspace_context["module_hints"][:10])
                )
            if "relevant_files" in input_data.workspace_context and input_data.workspace_context["relevant_files"]:
                parts.append("Most relevant files for this request:")
                for f in input_data.workspace_context["relevant_files"][:30]:
                    parts.append(f"  - {f}")
                parts.append("")

            if "files" in input_data.workspace_context:
                parts.append("Available files:")
                for f in input_data.workspace_context["files"][:80]:
                    parts.append(f"  - {f}")
            if "top_directories" in input_data.workspace_context and input_data.workspace_context["top_directories"]:
                parts.append(
                    f"Top-level directories/files: {', '.join(input_data.workspace_context['top_directories'])}"
                )
            if "structure" in input_data.workspace_context:
                parts.append(f"Project structure: {input_data.workspace_context['structure']}")
            if "orchestration_context" in input_data.workspace_context:
                parts.append("Orchestration context:")
                parts.append(str(input_data.workspace_context["orchestration_context"]))
            parts.append("")
        
        # Add constraints
        if input_data.constraints:
            parts.append("## Constraints:")
            for c in input_data.constraints:
                parts.append(f"  - {c}")
            parts.append("")
        
        # Add previous attempt if retry
        if input_data.previous_attempt:
            parts.append("## Previous Attempt (needs improvement):")
            parts.append(input_data.previous_attempt)
            parts.append("")
        
        parts.append(
            "Please create a detailed plan for this task, explicitly connecting each subtask to files/folders from the workspace context."
        )
        
        return "\n".join(parts)
    
    def _parse_response(
        self,
        response: str,
        input_data: PlannerInput,
        context: AgentContext,
    ) -> PlannerOutput:
        """Parse LLM response into PlannerOutput."""
        try:
            data = parse_json_object(response)
            if not isinstance(data, dict):
                raise TypeError("Planner response root must be a JSON object")

            schema = PlannerResponseSchema.model_validate(data)

            # Normalize + cap
            max_subtasks = context.config.limits.tasks.max_subtasks
            normalized_subtasks = list(schema.subtasks)[:max_subtasks]

            # Validate and normalize tool calls
            tool_calls: list[ToolCall] = []
            invalid_tools: list[str] = []
            for tc in schema.tool_calls:
                normalized_name = tc.tool_name.strip()
                if not self._is_valid_tool_call(normalized_name):
                    invalid_tools.append(normalized_name)
                    continue
                tool_calls.append(ToolCall(tool_name=normalized_name, arguments=tc.arguments))

            if invalid_tools and context.telemetry:
                context.telemetry.record_warning(
                    "planner_invalid_tool_calls",
                    context={"invalid": invalid_tools[:10]},
                )

            subtasks = [
                Subtask(
                    id=str(st.id),
                    title=st.title,
                    description=st.description,
                    acceptance_criteria=list(st.acceptance_criteria),
                    target_files=[str(path) for path in st.target_files if str(path).strip()],
                    dependencies=list(st.dependencies),
                    tool_plan=self._convert_tool_plan(getattr(st, "tool_plan", None)),
                    estimated_complexity=st.estimated_complexity,
                    estimated_iterations=self._normalize_estimated_iterations(
                        st.estimated_iterations,
                        st.estimated_complexity,
                    ),
                    fallback_strategy=self._normalize_fallback_strategy(
                        st.fallback_strategy,
                        st.estimated_complexity,
                    ),
                )
                for st in normalized_subtasks
            ]

            # Plan validation pass
            plan_errors = self._validate_plan_semantics(
                subtasks=subtasks,
                tool_calls=tool_calls,
                requires_clarification=schema.requires_clarification,
            )
            if plan_errors:
                raise TypeError("; ".join(plan_errors))

            confidence = self._score_plan_confidence(
                subtasks=subtasks,
                tool_calls=tool_calls,
                workspace_context=input_data.workspace_context,
                requires_clarification=schema.requires_clarification,
            )
            confidence_note = f"Planner confidence: {confidence:.2f}"
            assumptions = list(schema.assumptions)
            assumptions.append(confidence_note)

            if context.telemetry:
                context.telemetry.record_warning(
                    "planner_plan_validated",
                    context={
                        "subtasks": len(subtasks),
                        "tool_calls": len(tool_calls),
                        "requires_clarification": schema.requires_clarification,
                        "confidence": round(confidence, 3),
                    },
                )

            return PlannerOutput(
                task_id=input_data.task_id,
                status=AgentStatus.SUCCESS,
                plan_summary=schema.plan_summary,
                subtasks=subtasks,
                identified_risks=list(schema.identified_risks),
                assumptions=assumptions,
                requires_clarification=schema.requires_clarification,
                clarification_questions=list(schema.clarification_questions),
                tool_calls=tool_calls,
            )
            
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as e:
            # Return with parsing error
            return PlannerOutput(
                task_id=input_data.task_id,
                status=AgentStatus.FAILED,
                error=f"Failed to parse planner response: {e}",
                error_context={"raw_response": response[:1000]},
            )
    
    def _create_error_output(
        self,
        input_data: PlannerInput,
        context: AgentContext,
        status: AgentStatus,
        error: str,
        start_time: float,
    ) -> PlannerOutput:
        """Create error output for planner."""
        return PlannerOutput(
            task_id=input_data.task_id,
            status=status,
            error=error,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    @staticmethod
    def _is_valid_tool_call(tool_name: str) -> bool:
        """Validate that requested tool exists in the exposed planner tool catalog."""
        if not tool_name:
            return False
        if tool_name in ALLOWED_TOOL_NAMES:
            return True
        # Keep MCP wildcard compatibility when server tool names are surfaced dynamically.
        return tool_name.startswith("mcp_")

    @staticmethod
    def _validate_plan_semantics(
        subtasks: list[Subtask],
        tool_calls: list[ToolCall],
        requires_clarification: bool,
    ) -> list[str]:
        """Semantic validation pass after schema parsing."""
        errors: list[str] = []

        if not subtasks and not tool_calls and not requires_clarification:
            errors.append("Planner returned no subtasks, no tool calls, and no clarification request")

        seen_ids: set[str] = set()
        for st in subtasks:
            sid = st.id.strip()
            if sid in seen_ids:
                errors.append(f"Duplicate subtask id: {sid}")
            seen_ids.add(sid)

            if not st.acceptance_criteria:
                errors.append(f"Subtask {sid} missing acceptance criteria")

            complexity = st.estimated_complexity.lower().strip()
            if complexity not in {"low", "medium", "high"}:
                errors.append(f"Subtask {sid} has invalid complexity: {st.estimated_complexity}")
            if st.estimated_iterations < 1 or st.estimated_iterations > 6:
                errors.append(f"Subtask {sid} has invalid estimated_iterations: {st.estimated_iterations}")
            if not st.fallback_strategy.strip():
                errors.append(f"Subtask {sid} missing fallback_strategy")

        return errors

    @staticmethod
    def _score_plan_confidence(
        subtasks: list[Subtask],
        tool_calls: list[ToolCall],
        workspace_context: dict[str, Any],
        requires_clarification: bool,
    ) -> float:
        """Compute lightweight confidence score for plan quality and evidence fit."""
        if requires_clarification:
            return 0.25

        score = 0.35
        if subtasks:
            score += min(0.35, len(subtasks) * 0.08)

        target_count = sum(len(st.target_files) for st in subtasks)
        if target_count:
            score += min(0.20, target_count * 0.04)

        if tool_calls:
            score += min(0.10, len(tool_calls) * 0.03)

        relevant = workspace_context.get("relevant_files", []) if isinstance(workspace_context, dict) else []
        if isinstance(relevant, list) and relevant:
            score += 0.05

        return max(0.0, min(1.0, score))

    @staticmethod
    def _normalize_estimated_iterations(value: int | None, complexity: str) -> int:
        """Normalize estimated iterations using explicit value or complexity-derived defaults."""
        if isinstance(value, int):
            return max(1, min(6, value))

        normalized = complexity.strip().lower()
        if normalized == "low":
            return 1
        if normalized == "high":
            return 3
        return 2

    @staticmethod
    def _normalize_fallback_strategy(value: str | None, complexity: str) -> str:
        """Normalize fallback strategy and infer deterministic defaults by complexity."""
        if value and value.strip():
            return value.strip()[:120]

        normalized = complexity.strip().lower()
        if normalized == "high":
            return "evidence_gather_then_retry"
        if normalized == "low":
            return "targeted_replan"
        return "scope_reduce"

    def _convert_tool_plan(
        self,
        schema_plan: PlannerSubtaskToolPlanSchema | None,
    ) -> SubtaskToolPlan | None:
        """Convert planner tool-plan schema into contract model with strict validation."""
        if schema_plan is None:
            return None

        steps = [self._convert_tool_plan_step(step) for step in list(schema_plan.steps)[:3]]
        if not steps:
            return None
        return SubtaskToolPlan(steps=steps)

    def _convert_tool_plan_step(self, step: PlannerToolPlanStepSchema) -> ToolPlanStep:
        """Convert one recursive planner tool-plan step into contract model."""
        fallback = self._convert_tool_plan_step(step.fallback) if step.fallback else None
        return ToolPlanStep(
            tool=step.tool.strip(),
            reason=step.reason.strip(),
            arguments=dict(step.arguments),
            fallback=fallback,
        )
