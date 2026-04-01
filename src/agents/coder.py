"""
Coder Agent

Responsible for generating code changes based on subtasks.
Outputs structured diffs, NOT raw file content.

Design Decisions:
- All output is in diff format for reviewability
- Cannot directly write files
- Must include implementation notes
- Self-assesses confidence
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
    CodeChange,
    CoderInput,
    CoderOutput,
    Subtask,
    ToolCall,
)
from src.agents.json_utils import is_safe_relative_path, parse_json_object
from src.agents import json_utils as _json_utils
from src.core.agent_tools import TOOL_SCHEMAS
from src.core.agent_tools import get_tools_system_prompt


ALLOWED_TOOL_NAMES: set[str] = {schema["name"] for schema in TOOL_SCHEMAS if "name" in schema}


class CoderAgent(BaseAgent[CoderInput, CoderOutput]):
    """
    Generates code implementations for subtasks.
    
    The Coder:
    - Analyzes the subtask requirements
    - Generates code changes as structured diffs
    - Provides implementation notes
    - Self-assesses confidence level
    - Does NOT directly modify files
    """
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.CODER
    
    @property
    def system_prompt(self) -> str:
        tools_prompt = get_tools_system_prompt()
        return """You are an expert software engineer implementing code changes.

Your role is to generate high-quality code that fulfills the given subtask requirements.

## Your Responsibilities:
1. Understand the subtask and acceptance criteria
2. Generate clean, well-documented code
3. Follow best practices for the language/framework
4. Include proper error handling
5. Provide implementation notes explaining your approach
6. If needed, call tools to gather context or skills before finalizing changes

## Available Tools:
Use tools to inspect repository state, search implementation patterns, run tests, and validate assumptions before finalizing changes.

""" + tools_prompt + """

## Tool Usage Rules:
- Prefer tool-grounded implementation over assumptions.
- Use search/read tools before writing code in unfamiliar files.
- Run relevant tests/checks where possible via tool calls.
- Return `tool_calls` when more evidence or validation is needed; otherwise use an empty list.
- If target files are missing or ambiguous, request tool calls instead of guessing paths.
- Align every change with acceptance criteria and existing repository patterns.
- Reason in explicit sections: request intent, relevant files, constraints, implementation plan, verification.

## Output Format:
You MUST respond with a valid JSON object in this exact format:
```json
{
    "changes": [
        {
            "file_path": "path/to/file.py",
            "change_type": "create|modify|delete",
            "description": "What this change does",
            "new_content": "Full file content here",
            "hunks": [
                {
                    "start_line": 10,
                    "end_line": 14,
                    "original_content": "old text",
                    "new_content": "new text",
                    "context_before": ["..."],
                    "context_after": ["..."]
                }
            ],
            "lines_added": 10,
            "lines_removed": 5
        }
    ],
    "implementation_notes": "Explanation of the approach taken",
    "confidence": "low|medium|high",
    "concerns": ["Any concerns about the implementation"],
    "suggested_tests": ["Test cases that should be written"],
    "tool_calls": [
        {"tool_name": "run_command", "arguments": {"command": "npm run test"}}
    ]
}
```

## Code Quality Rules:
- Include docstrings for all functions and classes
- Use type hints in Python code
- Keep functions under 50 lines
- Handle edge cases and errors
- Follow existing code style if modifying files
- Do not include secrets or credentials
- Use meaningful variable names

## Important:
- For "modify" changes, prefer patch-style `hunks` for surgical edits when practical
- If not using hunks, provide the COMPLETE new file content
- For "create" changes, provide the full new file
- For "delete" changes, set new_content to empty string
- You may also provide `hunks` for surgical patch-style edits when appropriate
- Always explain your implementation approach"""
    
    def _validate_input(
        self,
        input_data: CoderInput,
        context: AgentContext,
    ) -> list[str]:
        """Validate coder input."""
        errors = super()._validate_input(input_data, context)
        
        # Verify subtask has required fields
        if not input_data.subtask.title:
            errors.append("Subtask title is required")
        
        if not input_data.subtask.acceptance_criteria:
            errors.append("Subtask must have acceptance criteria")
        
        return errors
    
    def _execute_impl(
        self,
        input_data: CoderInput,
        context: AgentContext,
    ) -> CoderOutput:
        """Execute the coding process."""
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
        input_data: CoderInput,
        context: AgentContext,
    ) -> str:
        """Build the prompt for the LLM."""
        parts = [
            "## Subtask to Implement:",
            f"**Title:** {input_data.subtask.title}",
            f"**Description:** {input_data.subtask.description}",
            "",
            "**Acceptance Criteria:**",
        ]
        
        # Inject memory if available
        if context.memory_manager:
            parts.append(context.memory_manager.get_all_context())
            parts.append("")
        
        for i, criterion in enumerate(input_data.subtask.acceptance_criteria, 1):
            parts.append(f"  {i}. {criterion}")
        
        parts.append("")
        
        # Add target files
        if input_data.subtask.target_files:
            parts.append("**Target Files:**")
            for f in input_data.subtask.target_files:
                parts.append(f"  - {f}")
            parts.append("")
        
        # Add existing file contents
        if input_data.file_contents:
            parts.append("## Existing File Contents:")
            for file_path, content in input_data.file_contents.items():
                # Truncate very long files
                if len(content) > 5000:
                    content = content[:5000] + "\n... [truncated]"
                parts.append(f"\n### {file_path}")
                parts.append(f"```\n{content}\n```")
            parts.append("")
        
        # Add context
        if input_data.context:
            parts.append("## Additional Context:")
            parts.append(input_data.context)
            parts.append("")

        parts.append("## Grounding Requirement:")
        parts.append("Use the provided file contents and discovered paths as the source of truth.")
        parts.append("If information is insufficient, return tool_calls to gather evidence before proposing changes.")
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
        
        parts.append("Please implement the code changes for this subtask.")
        
        return "\n".join(parts)
    
    def _parse_response(
        self,
        response: str,
        input_data: CoderInput,
        context: AgentContext,
    ) -> CoderOutput:
        """Parse LLM response into CoderOutput."""
        try:
            data = parse_json_object(response)
            if not isinstance(data, dict):
                raise TypeError("Coder response root must be a JSON object")
            
            # Parse code changes
            changes: list[CodeChange] = []
            raw_changes = data.get("changes", [])
            if isinstance(raw_changes, dict):
                raw_changes = [raw_changes]
            if not isinstance(raw_changes, list):
                raw_changes = []

            for change_data in raw_changes:
                if not isinstance(change_data, dict):
                    continue

                # Get original content if modifying
                original = None
                if change_data.get("change_type") == "modify":
                    original = input_data.file_contents.get(change_data.get("file_path", ""))

                raw_hunks = change_data.get("hunks", [])
                if not isinstance(raw_hunks, list):
                    raw_hunks = []
                hunks = [h for h in raw_hunks if isinstance(h, dict)]
                
                change = CodeChange(
                    file_path=change_data.get("file_path", ""),
                    change_type=change_data.get("change_type", "modify"),
                    description=change_data.get("description", ""),
                    original_content=original,
                    new_content=change_data.get("new_content", ""),
                    hunks=hunks,
                    lines_added=change_data.get("lines_added", 0),
                    lines_removed=change_data.get("lines_removed", 0),
                )
                if not is_safe_relative_path(change.file_path):
                    raise TypeError(f"Unsafe file path emitted by coder: {change.file_path}")
                changes.append(change)
            
            # Parse tool calls
            tool_calls = []
            raw_tool_calls = data.get("tool_calls", [])
            if isinstance(raw_tool_calls, dict):
                raw_tool_calls = [raw_tool_calls]
            if not isinstance(raw_tool_calls, list):
                raw_tool_calls = []

            for tc in raw_tool_calls:
                if not isinstance(tc, dict):
                    continue
                tool_name = str(tc.get("tool_name", "")).strip()
                if not tool_name:
                    continue
                if tool_name not in ALLOWED_TOOL_NAMES and not tool_name.startswith("mcp_"):
                    continue
                tool_calls.append(ToolCall(tool_name=tool_name, arguments=tc.get("arguments", {})))
            
            # Validate changes against limits
            max_files = context.config.limits.files.max_per_task
            if len(changes) > max_files:
                changes = changes[:max_files]
            
            # Normalize suggested_tests to list of strings
            # (LLM sometimes returns dicts with description/code)
            raw_tests = data.get("suggested_tests", [])
            if isinstance(raw_tests, (str, dict)):
                raw_tests = [raw_tests]
            if not isinstance(raw_tests, list):
                raw_tests = []
            suggested_tests: list[str] = []
            for test in raw_tests:
                if isinstance(test, str):
                    suggested_tests.append(test)
                elif isinstance(test, dict):
                    # Extract description or code from dict
                    test_str = test.get("description") or test.get("code") or str(test)
                    suggested_tests.append(test_str)
            
            return CoderOutput.model_validate(
                {
                    "task_id": input_data.task_id,
                    "status": AgentStatus.SUCCESS,
                    "changes": changes,
                    "implementation_notes": data.get("implementation_notes", ""),
                    "confidence": data.get("confidence", "medium"),
                    "concerns": data.get("concerns", []),
                    "suggested_tests": suggested_tests,
                    "tool_calls": tool_calls,
                }
            )
            
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            # Return with parsing error
            return CoderOutput(
                task_id=input_data.task_id,
                status=AgentStatus.FAILED,
                error=f"Failed to parse coder response: {e}",
                error_context={"raw_response": response[:1000]},
            )

    @staticmethod
    def _escape_control_chars_in_json_strings(raw_json: str) -> str:
        """Backward-compatible wrapper for existing tests and callers."""
        return _json_utils._escape_control_chars_in_strings(raw_json)

    def _create_error_output(
        self,
        input_data: CoderInput,
        context: AgentContext,
        status: AgentStatus,
        error: str,
        start_time: float,
    ) -> CoderOutput:
        """Create error output for coder."""
        return CoderOutput(
            task_id=input_data.task_id,
            status=status,
            error=error,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )
