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

from src.agents.base import (
    AgentContext,
    AgentStatus,
    AgentType,
    BaseAgent,
    PlannerInput,
    PlannerOutput,
    Subtask,
)


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
            "estimated_complexity": "low|medium|high"
        }
    ],
    "identified_risks": ["Risk 1", "Risk 2"],
    "assumptions": ["Assumption 1", "Assumption 2"],
    "requires_clarification": false,
    "clarification_questions": []
}
```

## Rules:
- Maximum 10 subtasks
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
        
        # Add workspace context if available
        if input_data.workspace_context:
            parts.append("## Workspace Context:")
            if "files" in input_data.workspace_context:
                parts.append("Available files:")
                for f in input_data.workspace_context["files"][:50]:  # Limit
                    parts.append(f"  - {f}")
            if "structure" in input_data.workspace_context:
                parts.append(f"Project structure: {input_data.workspace_context['structure']}")
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
        
        parts.append("Please create a detailed plan for this task.")
        
        return "\n".join(parts)
    
    def _parse_response(
        self,
        response: str,
        input_data: PlannerInput,
        context: AgentContext,
    ) -> PlannerOutput:
        """Parse LLM response into PlannerOutput."""
        try:
            # Extract JSON from response
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_str = response.strip()
                if not json_str.startswith("{"):
                    # Try to extract anything that looks like JSON
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    if start != -1 and end > start:
                        json_str = response[start:end]
            
            data = json.loads(json_str)
            
            # Parse subtasks
            subtasks = []
            for st in data.get("subtasks", []):
                subtask = Subtask(
                    id=str(st.get("id", len(subtasks) + 1)),
                    title=st.get("title", "Untitled"),
                    description=st.get("description", ""),
                    acceptance_criteria=st.get("acceptance_criteria", ["Complete the task"]),
                    target_files=st.get("target_files", []),
                    dependencies=st.get("dependencies", []),
                    estimated_complexity=st.get("estimated_complexity", "medium"),
                )
                subtasks.append(subtask)
            
            # Limit subtasks to max
            max_subtasks = context.config.limits.tasks.max_subtasks
            if len(subtasks) > max_subtasks:
                subtasks = subtasks[:max_subtasks]
            
            return PlannerOutput(
                task_id=input_data.task_id,
                status=AgentStatus.SUCCESS,
                plan_summary=data.get("plan_summary", ""),
                subtasks=subtasks,
                identified_risks=data.get("identified_risks", []),
                assumptions=data.get("assumptions", []),
                requires_clarification=data.get("requires_clarification", False),
                clarification_questions=data.get("clarification_questions", []),
            )
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
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
