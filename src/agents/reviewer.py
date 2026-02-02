"""
Reviewer Agent

Responsible for reviewing code changes and providing feedback.
Does NOT generate code - only critiques and suggests improvements.

Design Decisions:
- Must cite specific line numbers
- Provides structured feedback
- Cannot modify files directly
- Renders clear verdict
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
    ReviewerInput,
    ReviewerOutput,
    ReviewIssue,
    ReviewVerdict,
)


class ReviewerAgent(BaseAgent[ReviewerInput, ReviewerOutput]):
    """
    Reviews code changes for quality and correctness.
    
    The Reviewer:
    - Analyzes code changes against acceptance criteria
    - Identifies issues with severity levels
    - Provides specific, actionable feedback
    - Does NOT generate new code
    """
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.REVIEWER
    
    @property
    def system_prompt(self) -> str:
        return """You are a senior code reviewer with expertise in software quality.

Your role is to review code changes and provide constructive feedback.

## Your Responsibilities:
1. Check if the code meets acceptance criteria
2. Identify bugs, issues, and improvements
3. Assess code quality (readability, maintainability)
4. Verify error handling and edge cases
5. Provide specific, actionable feedback

## Output Format:
You MUST respond with a valid JSON object in this exact format:
```json
{
    "verdict": "APPROVE|REQUEST_CHANGES|REJECT",
    "summary": "Overall assessment of the changes",
    "issues": [
        {
            "severity": "critical|major|minor|suggestion",
            "file_path": "path/to/file.py",
            "line_range": "10-15",
            "description": "What the issue is",
            "suggestion": "How to fix it"
        }
    ],
    "strengths": ["What was done well"],
    "criteria_met": {
        "Criterion 1": true,
        "Criterion 2": false
    }
}
```

## Review Guidelines:
- APPROVE: Code meets all acceptance criteria with no critical/major issues
- REQUEST_CHANGES: Issues that must be fixed before approval
- REJECT: Fundamental problems that require complete rework

## Severity Levels:
- critical: Security issues, data loss risk, crashes
- major: Bugs, missing functionality, significant design issues
- minor: Code style, minor inefficiencies, small improvements
- suggestion: Nice-to-have improvements, not required

## Rules:
- Always cite specific line numbers when possible
- Be specific about what needs to change
- Explain WHY something is an issue
- Do not write new code, only describe what should change
- Check all acceptance criteria"""
    
    def _validate_input(
        self,
        input_data: ReviewerInput,
        context: AgentContext,
    ) -> list[str]:
        """Validate reviewer input."""
        errors = super()._validate_input(input_data, context)
        
        # Must have changes to review
        if not input_data.code_changes:
            errors.append("No code changes to review")
        
        return errors
    
    def _execute_impl(
        self,
        input_data: ReviewerInput,
        context: AgentContext,
    ) -> ReviewerOutput:
        """Execute the review process."""
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
        input_data: ReviewerInput,
        context: AgentContext,
    ) -> str:
        """Build the prompt for the LLM."""
        parts = [
            "## Subtask Being Implemented:",
            f"**Title:** {input_data.subtask.title}",
            f"**Description:** {input_data.subtask.description}",
            "",
            "**Acceptance Criteria:**",
        ]
        
        for i, criterion in enumerate(input_data.subtask.acceptance_criteria, 1):
            parts.append(f"  {i}. {criterion}")
        
        parts.append("")
        
        # Add implementation notes
        if input_data.implementation_notes:
            parts.append("## Implementation Notes from Coder:")
            parts.append(input_data.implementation_notes)
            parts.append("")
        
        # Add code changes
        parts.append("## Code Changes to Review:")
        for i, change in enumerate(input_data.code_changes, 1):
            parts.append(f"\n### Change {i}: {change.file_path}")
            parts.append(f"**Type:** {change.change_type}")
            parts.append(f"**Description:** {change.description}")
            
            if change.change_type == "modify" and change.original_content:
                parts.append("\n**Original:**")
                parts.append(f"```\n{change.original_content[:3000]}\n```")
            
            if change.new_content:
                parts.append("\n**New:**")
                parts.append(f"```\n{change.new_content[:3000]}\n```")
            
            parts.append("")
        
        parts.append("Please review these changes and provide your assessment.")
        
        return "\n".join(parts)
    
    def _parse_response(
        self,
        response: str,
        input_data: ReviewerInput,
        context: AgentContext,
    ) -> ReviewerOutput:
        """Parse LLM response into ReviewerOutput."""
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
            
            # Parse verdict
            verdict_str = data.get("verdict", "REQUEST_CHANGES").upper()
            try:
                verdict = ReviewVerdict(verdict_str)
            except ValueError:
                verdict = ReviewVerdict.REQUEST_CHANGES
            
            # Parse issues
            issues: list[ReviewIssue] = []
            for issue_data in data.get("issues", []):
                issue = ReviewIssue(
                    severity=issue_data.get("severity", "minor"),
                    file_path=issue_data.get("file_path", ""),
                    line_range=issue_data.get("line_range"),
                    description=issue_data.get("description", ""),
                    suggestion=issue_data.get("suggestion", ""),
                )
                issues.append(issue)
            
            return ReviewerOutput(
                task_id=input_data.task_id,
                status=AgentStatus.SUCCESS,
                verdict=verdict,
                issues=issues,
                summary=data.get("summary", ""),
                strengths=data.get("strengths", []),
                criteria_met=data.get("criteria_met", {}),
            )
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Return with parsing error
            return ReviewerOutput(
                task_id=input_data.task_id,
                status=AgentStatus.FAILED,
                error=f"Failed to parse reviewer response: {e}",
                error_context={"raw_response": response[:1000]},
            )
    
    def _create_error_output(
        self,
        input_data: ReviewerInput,
        context: AgentContext,
        status: AgentStatus,
        error: str,
        start_time: float,
    ) -> ReviewerOutput:
        """Create error output for reviewer."""
        return ReviewerOutput(
            task_id=input_data.task_id,
            status=status,
            error=error,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )
