"""
Fixer Agent

Responsible for fixing issues identified by the Reviewer.
Makes minimal, targeted changes to address specific feedback.

Design Decisions:
- Only addresses cited issues
- Minimal changes - no new features
- Must track which issues were addressed
- Cannot ignore critical issues
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
    FixerInput,
    FixerOutput,
    ReviewIssue,
)


class FixerAgent(BaseAgent[FixerInput, FixerOutput]):
    """
    Fixes issues identified during code review.
    
    The Fixer:
    - Addresses specific issues from review
    - Makes minimal changes
    - Does NOT add new features
    - Tracks which issues were resolved
    """
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.FIXER
    
    @property
    def system_prompt(self) -> str:
        return """You are a precise code fixer specializing in targeted corrections.

Your role is to fix specific issues identified during code review.

## Your Responsibilities:
1. Address ONLY the issues cited in the review
2. Make minimal changes to fix each issue
3. Do NOT add new features or make unrelated changes
4. Track which issues you addressed
5. Explain any issues you couldn't address

## Output Format:
You MUST respond with a valid JSON object in this exact format:
```json
{
    "fixed_changes": [
        {
            "file_path": "path/to/file.py",
            "change_type": "modify",
            "description": "Fixed issue X by doing Y",
            "new_content": "Complete fixed file content",
            "lines_added": 2,
            "lines_removed": 1
        }
    ],
    "issues_addressed": ["Issue 1 description", "Issue 2 description"],
    "issues_not_addressed": ["Issue that couldn't be fixed: reason"],
    "fix_notes": "Explanation of the fixes made"
}
```

## Rules:
- Address critical and major issues first
- Make the smallest change that fixes the issue
- Do NOT refactor unrelated code
- Do NOT add new functionality
- Preserve existing code style
- Keep track of ALL issues from the review
- If an issue cannot be fixed, explain why

## Important:
- Provide COMPLETE file content for each changed file
- Only include files that actually changed
- Maximum 50 lines changed per fix"""
    
    def _validate_input(
        self,
        input_data: FixerInput,
        context: AgentContext,
    ) -> list[str]:
        """Validate fixer input."""
        errors = super()._validate_input(input_data, context)
        
        # Must have issues to fix
        if not input_data.review_issues:
            errors.append("No issues to fix")
        
        # Must have original changes
        if not input_data.original_changes:
            errors.append("No original changes provided")
        
        return errors
    
    def _execute_impl(
        self,
        input_data: FixerInput,
        context: AgentContext,
    ) -> FixerOutput:
        """Execute the fixing process."""
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
        input_data: FixerInput,
        context: AgentContext,
    ) -> str:
        """Build the prompt for the LLM."""
        parts = [
            "## Issues to Fix:",
            "",
        ]
        
        # Group issues by severity
        critical_issues = [i for i in input_data.review_issues if i.severity == "critical"]
        major_issues = [i for i in input_data.review_issues if i.severity == "major"]
        minor_issues = [i for i in input_data.review_issues if i.severity == "minor"]
        suggestions = [i for i in input_data.review_issues if i.severity == "suggestion"]
        
        if critical_issues:
            parts.append("### Critical Issues (MUST FIX):")
            for i, issue in enumerate(critical_issues, 1):
                parts.append(f"{i}. **{issue.file_path}**")
                if issue.line_range:
                    parts.append(f"   Lines: {issue.line_range}")
                parts.append(f"   Issue: {issue.description}")
                if issue.suggestion:
                    parts.append(f"   Suggestion: {issue.suggestion}")
            parts.append("")
        
        if major_issues:
            parts.append("### Major Issues (MUST FIX):")
            for i, issue in enumerate(major_issues, 1):
                parts.append(f"{i}. **{issue.file_path}**")
                if issue.line_range:
                    parts.append(f"   Lines: {issue.line_range}")
                parts.append(f"   Issue: {issue.description}")
                if issue.suggestion:
                    parts.append(f"   Suggestion: {issue.suggestion}")
            parts.append("")
        
        if minor_issues:
            parts.append("### Minor Issues (SHOULD FIX):")
            for i, issue in enumerate(minor_issues, 1):
                parts.append(f"{i}. **{issue.file_path}**: {issue.description}")
                if issue.suggestion:
                    parts.append(f"   Suggestion: {issue.suggestion}")
            parts.append("")
        
        if suggestions:
            parts.append("### Suggestions (OPTIONAL):")
            for i, issue in enumerate(suggestions, 1):
                parts.append(f"{i}. **{issue.file_path}**: {issue.description}")
            parts.append("")
        
        # Add current file contents
        parts.append("## Current Code (with issues):")
        for change in input_data.original_changes:
            parts.append(f"\n### {change.file_path}")
            if change.new_content:
                content = change.new_content
                # Truncate if very long
                if len(content) > 4000:
                    content = content[:4000] + "\n... [truncated]"
                parts.append(f"```\n{content}\n```")
        
        parts.append("")
        parts.append("Please fix the issues. Focus on critical and major issues first.")
        
        return "\n".join(parts)
    
    def _parse_response(
        self,
        response: str,
        input_data: FixerInput,
        context: AgentContext,
    ) -> FixerOutput:
        """Parse LLM response into FixerOutput."""
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
            
            # Parse fixed changes
            fixed_changes: list[CodeChange] = []
            for change_data in data.get("fixed_changes", []):
                # Get original content for reference
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
                fixed_changes.append(change)
            
            # Validate line changes against policy
            max_lines = context.policy.max_lines_per_fix or 50
            total_lines_changed = sum(
                c.lines_added + c.lines_removed for c in fixed_changes
            )
            
            if total_lines_changed > max_lines:
                return FixerOutput(
                    task_id=input_data.task_id,
                    status=AgentStatus.REJECTED,
                    error=f"Fix exceeds line limit: {total_lines_changed} > {max_lines}",
                )
            
            return FixerOutput(
                task_id=input_data.task_id,
                status=AgentStatus.SUCCESS,
                fixed_changes=fixed_changes,
                issues_addressed=data.get("issues_addressed", []),
                issues_not_addressed=data.get("issues_not_addressed", []),
                fix_notes=data.get("fix_notes", ""),
            )
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Return with parsing error
            return FixerOutput(
                task_id=input_data.task_id,
                status=AgentStatus.FAILED,
                error=f"Failed to parse fixer response: {e}",
                error_context={"raw_response": response[:1000]},
            )
    
    def _create_error_output(
        self,
        input_data: FixerInput,
        context: AgentContext,
        status: AgentStatus,
        error: str,
        start_time: float,
    ) -> FixerOutput:
        """Create error output for fixer."""
        return FixerOutput(
            task_id=input_data.task_id,
            status=status,
            error=error,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )
