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

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
from src.agents.json_utils import is_safe_relative_path, parse_json_object


class ReviewerIssueSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: str = Field(default="minor")
    file_path: str = Field(default="")
    line_range: str | None = None
    description: str = Field(default="")
    suggestion: str = Field(default="")
    issue_code: str = Field(default="")
    acceptance_criterion_ref: str = Field(default="")
    evidence: str = Field(default="")
    blocking: bool | None = None


class ReviewerResponseSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    verdict: str = Field(default="REQUEST_CHANGES")
    task_complete: bool | None = None
    summary: str = Field(default="")
    issues: list[ReviewerIssueSchema] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    criteria_met: dict[str, bool] = Field(default_factory=dict)
    correctness_score: float | None = None
    maintainability_score: float | None = None
    risk_score: float | None = None
    confidence_score: float | None = None
    potential_breakages: list[str] = Field(default_factory=list)


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
    "task_complete": true|false,
    "summary": "Overall assessment of the changes",
    "issues": [
        {
            "severity": "critical|major|minor|suggestion",
            "file_path": "path/to/file.py",
            "line_range": "10-15",
            "description": "What the issue is",
            "suggestion": "How to fix it",
            "issue_code": "REVIEW_001",
            "acceptance_criterion_ref": "Criterion 1",
            "evidence": "Function divide() does not handle zero divisor on lines 12-15",
            "blocking": true
        }
    ],
    "strengths": ["What was done well"],
    "correctness_score": 0.0,
    "maintainability_score": 0.0,
    "risk_score": 0.0,
    "confidence_score": 0.0,
    "potential_breakages": [
        "What could break later in edge conditions"
    ],
    "criteria_met": {
        "Criterion 1": true,
        "Criterion 2": false
    }
}
```

## CRITICAL: task_complete Field
- Set task_complete=true ONLY when ALL acceptance criteria are met
- When task_complete=true, the system will STOP and not make any more changes
- When task_complete=false, the system will continue trying to fix issues

## Review Guidelines:
- APPROVE + task_complete=true: Code meets ALL acceptance criteria. STOPS the run.
- REQUEST_CHANGES + task_complete=false: Fixable issues exist, fixer will address them.
- REJECT + task_complete=false: Fundamental problems requiring complete rework. ABORTS the run.

## IMPORTANT: When to set task_complete=true
- ALL acceptance criteria are satisfied
- No critical or major issues remain
- The code is ready for production
- Even if there are minor suggestions, if criteria are met, set task_complete=true

## Severity Levels:
- critical: Security issues, data loss risk, crashes
- major: Bugs, missing functionality, significant design issues
- minor: Code style, minor inefficiencies, small improvements
- suggestion: Nice-to-have improvements, not required

## Required Scoring:
- correctness_score: 0.0-1.0 (higher is better)
- maintainability_score: 0.0-1.0 (higher is better)
- risk_score: 0.0-1.0 (higher means more likely breakage)
- confidence_score: 0.0-1.0 (confidence in your assessment)
- potential_breakages: list at least one realistic edge-case regression when risk_score > 0.4

## Rules:
- Always cite specific line numbers when possible
- Be specific about what needs to change
- Explain WHY something is an issue
- Every issue MUST include: issue_code, acceptance_criterion_ref, evidence, suggestion
- Link each issue to at least one acceptance criterion from the subtask
- Mark blocking=true for issues that prevent task completion
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
            data = parse_json_object(response)
            schema = ReviewerResponseSchema.model_validate(data)
            
            # Parse verdict
            verdict_str = schema.verdict.upper()
            try:
                verdict = ReviewVerdict(verdict_str)
            except ValueError:
                verdict = ReviewVerdict.REQUEST_CHANGES
            
            # Parse task_complete - EXPLICIT TERMINAL STATE
            # If verdict is APPROVE, default task_complete to True for backward compatibility
            # This ensures approval always terminates the loop
            task_complete = schema.task_complete if schema.task_complete is not None else (verdict == ReviewVerdict.APPROVE)
            
            # Parse issues
            issues: list[ReviewIssue] = []
            allowed_paths = {c.file_path for c in input_data.code_changes if c.file_path}
            for issue_data in schema.issues:
                if issue_data.file_path and not is_safe_relative_path(issue_data.file_path):
                    continue
                if issue_data.file_path and allowed_paths and issue_data.file_path not in allowed_paths:
                    continue
                issue = ReviewIssue(
                    severity=issue_data.severity,
                    file_path=issue_data.file_path,
                    line_range=issue_data.line_range,
                    description=issue_data.description,
                    suggestion=issue_data.suggestion,
                    issue_code=(issue_data.issue_code or "").strip() or "REVIEW_GENERIC",
                    acceptance_criterion_ref=(issue_data.acceptance_criterion_ref or "").strip(),
                    evidence=(issue_data.evidence or "").strip() or issue_data.description,
                    blocking=(
                        bool(issue_data.blocking)
                        if issue_data.blocking is not None
                        else issue_data.severity.lower() in {"critical", "major"}
                    ),
                )
                issues.append(issue)
            
            # SAFETY: If verdict is APPROVE, force task_complete=True
            # This prevents any scenario where APPROVE doesn't terminate
            if verdict == ReviewVerdict.APPROVE:
                task_complete = True

            severities = [i.severity.lower() for i in issues]
            if any(sev in {"critical", "major"} for sev in severities):
                task_complete = False
                if verdict == ReviewVerdict.APPROVE:
                    verdict = ReviewVerdict.REQUEST_CHANGES

            if any(i.blocking for i in issues):
                task_complete = False
                if verdict == ReviewVerdict.APPROVE:
                    verdict = ReviewVerdict.REQUEST_CHANGES

            confidence = self._score_review_confidence(issues=issues, criteria_met=schema.criteria_met)
            score_bundle = self._derive_quality_scores(
                issues=issues,
                criteria_met=schema.criteria_met,
                correctness_score=schema.correctness_score,
                maintainability_score=schema.maintainability_score,
                risk_score=schema.risk_score,
                confidence_score=schema.confidence_score,
                confidence_fallback=confidence,
            )
            potential_breakages = [b.strip() for b in schema.potential_breakages if b and b.strip()][:8]
            if context.telemetry:
                context.telemetry.record_warning(
                    "reviewer_review_validated",
                    context={
                        "issues": len(issues),
                        "verdict": verdict.value,
                        "task_complete": bool(task_complete),
                        "confidence": round(score_bundle["confidence_score"], 3),
                        "correctness_score": round(score_bundle["correctness_score"], 3),
                        "maintainability_score": round(score_bundle["maintainability_score"], 3),
                        "risk_score": round(score_bundle["risk_score"], 3),
                        "potential_breakages": potential_breakages[:3],
                    },
                )
            
            return ReviewerOutput(
                task_id=input_data.task_id,
                status=AgentStatus.SUCCESS,
                verdict=verdict,
                task_complete=task_complete,
                issues=issues,
                summary=schema.summary,
                strengths=list(schema.strengths),
                correctness_score=score_bundle["correctness_score"],
                maintainability_score=score_bundle["maintainability_score"],
                risk_score=score_bundle["risk_score"],
                confidence_score=score_bundle["confidence_score"],
                potential_breakages=potential_breakages,
                criteria_met=dict(schema.criteria_met),
            )
            
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as e:
            # Return with parsing error
            return ReviewerOutput(
                task_id=input_data.task_id,
                status=AgentStatus.FAILED,
                error=f"Failed to parse reviewer response: {e}",
                error_context={"raw_response": response[:1000]},
            )

    @staticmethod
    def _score_review_confidence(issues: list[ReviewIssue], criteria_met: dict[str, bool]) -> float:
        """Lightweight review confidence score for telemetry."""
        score = 0.4
        if criteria_met:
            passed = sum(1 for v in criteria_met.values() if v)
            score += min(0.35, passed * 0.08)

        severe = sum(1 for i in issues if i.severity.lower() in {"critical", "major"})
        score -= min(0.35, severe * 0.12)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _derive_quality_scores(
        *,
        issues: list[ReviewIssue],
        criteria_met: dict[str, bool],
        correctness_score: float | None,
        maintainability_score: float | None,
        risk_score: float | None,
        confidence_score: float | None,
        confidence_fallback: float,
    ) -> dict[str, float]:
        """Normalize reviewer quality scores with deterministic fallbacks."""
        severe = sum(1 for i in issues if i.severity.lower() in {"critical", "major"})
        minor = sum(1 for i in issues if i.severity.lower() in {"minor", "suggestion"})
        passed = sum(1 for v in criteria_met.values() if v)
        total = max(1, len(criteria_met))
        criteria_ratio = passed / total

        default_correctness = max(0.0, min(1.0, 0.35 + (0.5 * criteria_ratio) - (0.15 * severe)))
        default_maintainability = max(0.0, min(1.0, 0.45 + (0.25 * criteria_ratio) - (0.08 * minor) - (0.12 * severe)))
        default_risk = max(0.0, min(1.0, 0.15 + (0.22 * severe) + (0.08 * minor) + (0.22 * (1 - criteria_ratio))))
        default_confidence = max(0.0, min(1.0, confidence_fallback))

        def _normalize(value: float | None, fallback: float) -> float:
            if value is None:
                return round(fallback, 4)
            return round(max(0.0, min(1.0, float(value))), 4)

        return {
            "correctness_score": _normalize(correctness_score, default_correctness),
            "maintainability_score": _normalize(maintainability_score, default_maintainability),
            "risk_score": _normalize(risk_score, default_risk),
            "confidence_score": _normalize(confidence_score, default_confidence),
        }
    
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
