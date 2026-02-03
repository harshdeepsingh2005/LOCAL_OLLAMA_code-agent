"""
Contract Enforcement Module

Runtime enforcement of agent contracts and output validation.
Ensures agents cannot bypass policies or produce invalid outputs.

This module avoids importing from agents to prevent circular dependencies.
It works with dict-based outputs instead of typed Pydantic models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ViolationType(str, Enum):
    """Types of contract violations."""
    SCHEMA_VIOLATION = "schema_violation"
    POLICY_VIOLATION = "policy_violation"
    LIMIT_EXCEEDED = "limit_exceeded"
    INJECTION_ATTEMPT = "injection_attempt"
    PATH_TRAVERSAL = "path_traversal"
    DANGEROUS_OPERATION = "dangerous_operation"
    RESOURCE_ABUSE = "resource_abuse"


@dataclass
class ContractViolation:
    """A contract violation record."""
    violation_type: ViolationType
    message: str
    agent_type: str
    field_path: Optional[str] = None
    value: Optional[Any] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ValidationResult:
    """Result of contract validation."""
    valid: bool
    violations: list[ContractViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def add_violation(
        self,
        violation_type: ViolationType,
        message: str,
        agent_type: str,
        field_path: Optional[str] = None,
        value: Any = None,
    ) -> None:
        """Add a violation."""
        self.violations.append(ContractViolation(
            violation_type=violation_type,
            message=message,
            agent_type=agent_type,
            field_path=field_path,
            value=value,
        ))
        self.valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning (doesn't fail validation)."""
        self.warnings.append(message)


class ContractEnforcer:
    """
    Enforces agent contracts at runtime.
    
    Validates:
    - Input prompts for injection
    - Output paths for traversal
    - Agent outputs for policy compliance
    
    Works with dict-based outputs to avoid circular imports.
    """
    
    # Patterns that indicate prompt injection attempts
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+instructions",
        r"disregard\s+(previous|above|all)",
        r"forget\s+(everything|all|previous)",
        r"new\s+instructions\s*:",
        r"system\s*:\s*you\s+are",
        r"<\|.*?\|>",  # Special tokens
        r"```system",
        r"\[INST\]",
        r"\[/INST\]",
        r"\$\(.*\)",   # Command substitution
        r"`[^`]+`",    # Backtick command execution
        r";\s*rm\s+-",  # Shell injection
        r"&&\s*cat\s+",  # Command chaining
        r"\|\s*nc\s+",   # Pipe to netcat
    ]
    
    # Dangerous patterns in file paths
    DANGEROUS_PATH_PATTERNS = [
        r"\.\./",  # Directory traversal
        r"^/etc/",
        r"^/usr/",
        r"^/var/",
        r"^/root/",
        r"^/home/[^/]+/\.",  # Hidden files in home dirs
        r"^~",
        r"\$\{",  # Variable expansion
        r"\$\(",  # Command substitution
    ]
    
    # Dangerous code patterns
    DANGEROUS_CODE_PATTERNS = [
        r"os\.system\s*\(",
        r"subprocess\.(run|call|Popen|check_output)\s*\(",
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__\s*\(",
        r"shutil\.rmtree\s*\(['\"]?/",  # rmtree on root paths
        r"open\s*\(['\"]?/etc/",  # Opening system files
    ]
    
    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        max_subtasks: int = 10,
        max_files_per_task: int = 5,
        max_lines_per_edit: int = 500,
    ) -> None:
        """
        Initialize the contract enforcer.
        
        Args:
            workspace_root: Root directory for path validation
            max_subtasks: Maximum subtasks allowed from planner
            max_files_per_task: Maximum files per coder task
            max_lines_per_edit: Maximum lines per edit
        """
        self._workspace_root = workspace_root.resolve() if workspace_root else Path.cwd()
        self._max_subtasks = max_subtasks
        self._max_files_per_task = max_files_per_task
        self._max_lines_per_edit = max_lines_per_edit
        self._violation_history: list[ContractViolation] = []
    
    def validate_input(self, user_input: str) -> ValidationResult:
        """
        Validate user input for injection attempts.
        
        Args:
            user_input: Raw user input string
            
        Returns:
            Validation result
        """
        result = ValidationResult(valid=True)
        
        # Check for injection patterns
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                result.add_violation(
                    ViolationType.INJECTION_ATTEMPT,
                    f"Potential injection pattern detected in input",
                    "input",
                    value=pattern,
                )
        
        # Check for path traversal in input
        for pattern in self.DANGEROUS_PATH_PATTERNS:
            if re.search(pattern, user_input):
                result.add_violation(
                    ViolationType.PATH_TRAVERSAL,
                    f"Potential path traversal in input",
                    "input",
                    value=pattern,
                )
        
        self._violation_history.extend(result.violations)
        return result
    
    def validate_path(self, path: Path) -> ValidationResult:
        """
        Validate a file path is within workspace.
        
        Args:
            path: Path to validate
            
        Returns:
            Validation result
        """
        result = ValidationResult(valid=True)
        
        try:
            resolved = path.resolve()
            
            # Check if within workspace
            try:
                resolved.relative_to(self._workspace_root)
            except ValueError:
                result.add_violation(
                    ViolationType.PATH_TRAVERSAL,
                    f"Path outside workspace: {path}",
                    "path",
                    value=str(path),
                )
        except (OSError, ValueError) as e:
            result.add_violation(
                ViolationType.PATH_TRAVERSAL,
                f"Invalid path: {e}",
                "path",
                value=str(path),
            )
        
        self._violation_history.extend(result.violations)
        return result
    
    def validate_output(
        self,
        agent_type: str,
        output: dict[str, Any],
    ) -> ValidationResult:
        """
        Validate agent output dict.
        
        Args:
            agent_type: Type of agent ("planner", "coder", "reviewer", "fixer")
            output: Agent output as dictionary
            
        Returns:
            Validation result
        """
        result = ValidationResult(valid=True)
        
        # Common validation: check for injection in all string fields
        self._check_injection_recursive(output, agent_type, result, "")
        
        # Agent-specific validation
        if agent_type == "planner":
            self._validate_planner_output(output, result)
        elif agent_type == "coder":
            self._validate_coder_output(output, result)
        elif agent_type == "reviewer":
            self._validate_reviewer_output(output, result)
        elif agent_type == "fixer":
            self._validate_fixer_output(output, result)
        
        self._violation_history.extend(result.violations)
        return result
    
    def _check_injection_recursive(
        self,
        data: Any,
        agent_type: str,
        result: ValidationResult,
        path: str,
    ) -> None:
        """Recursively check for injection patterns."""
        if isinstance(data, str):
            for pattern in self.INJECTION_PATTERNS:
                if re.search(pattern, data, re.IGNORECASE):
                    result.add_violation(
                        ViolationType.INJECTION_ATTEMPT,
                        f"Injection pattern in output",
                        agent_type,
                        field_path=path,
                        value=pattern,
                    )
                    break  # One violation per field is enough
        elif isinstance(data, dict):
            for key, value in data.items():
                self._check_injection_recursive(
                    value, agent_type, result, f"{path}.{key}" if path else key
                )
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._check_injection_recursive(
                    item, agent_type, result, f"{path}[{i}]"
                )
    
    def _validate_planner_output(
        self,
        output: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate planner-specific contracts."""
        subtasks = output.get("subtasks", [])
        
        # Check subtask count
        if len(subtasks) > self._max_subtasks:
            result.add_violation(
                ViolationType.LIMIT_EXCEEDED,
                f"Too many subtasks: {len(subtasks)} > {self._max_subtasks}",
                "planner",
                field_path="subtasks",
                value=len(subtasks),
            )
        
        # Check each subtask
        subtask_ids = {s.get("id") for s in subtasks if isinstance(s, dict)}
        
        for i, subtask in enumerate(subtasks):
            if not isinstance(subtask, dict):
                continue
            
            # Check dependencies
            deps = subtask.get("dependencies", [])
            for dep in deps:
                if dep not in subtask_ids:
                    result.add_warning(f"Unknown dependency: {dep}")
                if dep == subtask.get("id"):
                    result.add_violation(
                        ViolationType.SCHEMA_VIOLATION,
                        f"Self-referential dependency",
                        "planner",
                        field_path=f"subtasks[{i}].dependencies",
                    )
            
            # Check target files
            for file_path in subtask.get("target_files", []):
                self._validate_file_path_str(file_path, "planner", result)
    
    def _validate_coder_output(
        self,
        output: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate coder-specific contracts."""
        # Check file_path if present
        file_path = output.get("file_path")
        if file_path:
            self._validate_file_path_str(file_path, "coder", result)
        
        # Check code for dangerous patterns
        code = output.get("code", "")
        if code:
            for pattern in self.DANGEROUS_CODE_PATTERNS:
                if re.search(pattern, code):
                    result.add_violation(
                        ViolationType.DANGEROUS_OPERATION,
                        f"Dangerous code pattern detected",
                        "coder",
                        field_path="code",
                        value=pattern,
                    )
        
        # Check changes if present
        changes = output.get("changes", [])
        if len(changes) > self._max_files_per_task:
            result.add_violation(
                ViolationType.LIMIT_EXCEEDED,
                f"Too many file changes: {len(changes)} > {self._max_files_per_task}",
                "coder",
            )
        
        for i, change in enumerate(changes):
            if isinstance(change, dict):
                fp = change.get("file_path")
                if fp:
                    self._validate_file_path_str(fp, "coder", result)
    
    def _validate_reviewer_output(
        self,
        output: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate reviewer-specific contracts."""
        # Reviewer outputs are generally safe (just text)
        # Check for extreme issue counts
        issues = output.get("issues", [])
        if len(issues) > 50:
            result.add_warning(f"Very high issue count: {len(issues)}")
    
    def _validate_fixer_output(
        self,
        output: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate fixer-specific contracts."""
        # Similar to coder
        changes = output.get("fixed_changes", output.get("changes", []))
        for change in changes:
            if isinstance(change, dict):
                fp = change.get("file_path")
                if fp:
                    self._validate_file_path_str(fp, "fixer", result)
    
    def _validate_file_path_str(
        self,
        file_path: str,
        agent_type: str,
        result: ValidationResult,
    ) -> None:
        """Validate a file path string."""
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATH_PATTERNS:
            if re.search(pattern, file_path):
                result.add_violation(
                    ViolationType.PATH_TRAVERSAL,
                    f"Dangerous path pattern: {file_path}",
                    agent_type,
                    value=file_path,
                )
                return
        
        # Check if absolute path outside workspace
        if file_path.startswith("/"):
            try:
                path = Path(file_path).resolve()
                path.relative_to(self._workspace_root)
            except ValueError:
                result.add_violation(
                    ViolationType.PATH_TRAVERSAL,
                    f"Absolute path outside workspace: {file_path}",
                    agent_type,
                    value=file_path,
                )
    
    def get_violation_summary(self) -> dict[str, Any]:
        """Get summary of all violations."""
        by_type: dict[str, int] = {}
        by_agent: dict[str, int] = {}
        
        for v in self._violation_history:
            by_type[v.violation_type.value] = by_type.get(v.violation_type.value, 0) + 1
            by_agent[v.agent_type] = by_agent.get(v.agent_type, 0) + 1
        
        return {
            "total_violations": len(self._violation_history),
            "by_type": by_type,
            "by_agent": by_agent,
            "recent": [
                {
                    "type": v.violation_type.value,
                    "message": v.message,
                    "agent": v.agent_type,
                    "timestamp": v.timestamp.isoformat(),
                }
                for v in self._violation_history[-10:]
            ],
        }
    
    def clear_history(self) -> None:
        """Clear violation history."""
        self._violation_history.clear()
