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
)


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
        return """You are an expert software engineer implementing code changes.

Your role is to generate high-quality code that fulfills the given subtask requirements.

## Your Responsibilities:
1. Understand the subtask and acceptance criteria
2. Generate clean, well-documented code
3. Follow best practices for the language/framework
4. Include proper error handling
5. Provide implementation notes explaining your approach

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
            "lines_added": 10,
            "lines_removed": 5
        }
    ],
    "implementation_notes": "Explanation of the approach taken",
    "confidence": "low|medium|high",
    "concerns": ["Any concerns about the implementation"],
    "suggested_tests": ["Test cases that should be written"]
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
- For "modify" changes, provide the COMPLETE new file content
- For "create" changes, provide the full new file
- For "delete" changes, set new_content to empty string
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
            # Extract JSON from response
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.strip()
                if not json_str.startswith("{"):
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    if start != -1 and end > start:
                        json_str = response[start:end]
            
            data = json.loads(json_str)
            
            # Parse code changes
            changes: list[CodeChange] = []
            for change_data in data.get("changes", []):
                # Get original content if modifying
                original = None
                if change_data.get("change_type") == "modify":
                    original = input_data.file_contents.get(change_data.get("file_path", ""))
                
                change = CodeChange(
                    file_path=change_data.get("file_path", ""),
                    change_type=change_data.get("change_type", "modify"),
                    description=change_data.get("description", ""),
                    original_content=original,
                    new_content=change_data.get("new_content", ""),
                    lines_added=change_data.get("lines_added", 0),
                    lines_removed=change_data.get("lines_removed", 0),
                )
                changes.append(change)
            
            # Validate changes against limits
            max_files = context.config.limits.files.max_per_task
            if len(changes) > max_files:
                changes = changes[:max_files]
            
            return CoderOutput(
                task_id=input_data.task_id,
                status=AgentStatus.SUCCESS,
                changes=changes,
                implementation_notes=data.get("implementation_notes", ""),
                confidence=data.get("confidence", "medium"),
                concerns=data.get("concerns", []),
                suggested_tests=data.get("suggested_tests", []),
            )
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Return with parsing error
            return CoderOutput(
                task_id=input_data.task_id,
                status=AgentStatus.FAILED,
                error=f"Failed to parse coder response: {e}",
                error_context={"raw_response": response[:1000]},
            )
    
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
