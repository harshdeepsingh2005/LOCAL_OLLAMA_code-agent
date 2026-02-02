"""
Shell Tools Module

Provides controlled shell command execution.
All commands are validated, logged, and sandboxed.

Design Decisions:
- Command whitelist only
- Working directory restricted to workspace
- Timeout enforcement
- No network commands by default
- Full audit trail
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.core import TelemetryCollector


class CommandCategory(str, Enum):
    """Categories of shell commands."""
    FILE_SYSTEM = "file_system"
    GIT = "git"
    BUILD = "build"
    PYTHON = "python"
    CUSTOM = "custom"


class CommandStatus(str, Enum):
    """Status of command execution."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    ERROR = "error"


class CommandResult(BaseModel):
    """Result of a shell command execution."""
    command: str
    status: CommandStatus
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0
    error: str | None = None


# Default allowed command prefixes
DEFAULT_ALLOWED_COMMANDS = {
    # File system (read-only)
    "ls", "find", "cat", "head", "tail", "wc", "grep", "awk", "sed",
    "tree", "file", "stat", "du", "df",
    
    # Git operations
    "git status", "git diff", "git log", "git show", "git branch",
    "git remote", "git fetch", "git pull",
    
    # Python
    "python", "python3", "pip", "pip3",
    
    # Build tools
    "make", "npm", "yarn", "pnpm", "cargo", "go",
    
    # Utilities
    "echo", "which", "env", "pwd", "date",
}

# Commands that are always blocked
BLOCKED_COMMANDS = {
    "rm -rf /", "rm -rf /*", "rm -rf ~",
    "mkfs", "fdisk", "dd",
    "curl", "wget", "nc", "netcat",
    "ssh", "scp", "rsync",
    "sudo", "su", "chmod 777",
    "eval", "exec",
    "> /dev/sd", "| sh", "| bash",
}


class ShellExecutor:
    """
    Controlled shell command executor.
    
    Safety features:
    - Command validation against whitelist
    - Blocked command patterns
    - Working directory restriction
    - Timeout enforcement
    - Environment sanitization
    """
    
    def __init__(
        self,
        workspace_root: Path,
        telemetry: "TelemetryCollector | None" = None,
        allowed_commands: set[str] | None = None,
        blocked_patterns: set[str] | None = None,
        default_timeout: float = 60.0,
        max_output_size: int = 500_000,  # 500KB
        allow_network: bool = False,
    ) -> None:
        """
        Initialize shell executor.
        
        Args:
            workspace_root: Allowed working directory
            telemetry: Optional telemetry collector
            allowed_commands: Set of allowed command prefixes
            blocked_patterns: Set of blocked command patterns
            default_timeout: Default command timeout
            max_output_size: Maximum output size to capture
            allow_network: Allow network commands
        """
        self._workspace_root = workspace_root.resolve()
        self._telemetry = telemetry
        self._allowed_commands = allowed_commands or DEFAULT_ALLOWED_COMMANDS
        self._blocked_patterns = blocked_patterns or BLOCKED_COMMANDS
        self._default_timeout = default_timeout
        self._max_output_size = max_output_size
        self._allow_network = allow_network
        
        # Add network commands to blocked if not allowed
        if not allow_network:
            self._blocked_patterns = self._blocked_patterns | {
                "curl", "wget", "nc", "netcat", "ssh", "scp", "rsync",
                "http://", "https://", "ftp://",
            }
    
    def _log(self, command: str, status: CommandStatus, duration_ms: float) -> None:
        """Log command execution to telemetry."""
        if self._telemetry:
            self._telemetry.record_tool_call(
                tool_name="shell.execute",
                inputs={"command": command},
                outputs={"status": status.value, "duration_ms": duration_ms},
                success=status == CommandStatus.SUCCESS,
            )
    
    def _is_command_allowed(self, command: str) -> tuple[bool, str | None]:
        """
        Check if a command is allowed.
        
        Returns:
            Tuple of (is_allowed, reason_if_blocked)
        """
        cmd_lower = command.lower().strip()
        
        # Check blocked patterns first
        for pattern in self._blocked_patterns:
            if pattern.lower() in cmd_lower:
                return False, f"Blocked pattern: {pattern}"
        
        # Check if command starts with allowed prefix
        cmd_parts = shlex.split(command)
        if not cmd_parts:
            return False, "Empty command"
        
        base_cmd = cmd_parts[0]
        
        # Check direct match or prefix match
        for allowed in self._allowed_commands:
            allowed_parts = allowed.split()
            if base_cmd == allowed_parts[0]:
                # For multi-word allowed commands (like "git status"),
                # check if the actual command matches
                if len(allowed_parts) == 1:
                    return True, None
                elif command.startswith(allowed):
                    return True, None
        
        return False, f"Command not in allowlist: {base_cmd}"
    
    def _sanitize_env(self) -> dict[str, str]:
        """Create sanitized environment for command execution."""
        env = os.environ.copy()
        
        # Remove sensitive variables
        sensitive_vars = [
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
            "GITHUB_TOKEN", "GH_TOKEN",
            "API_KEY", "SECRET_KEY", "PASSWORD",
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        ]
        
        for var in sensitive_vars:
            env.pop(var, None)
        
        # Set safe defaults
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        
        if not self._allow_network:
            env["NO_PROXY"] = "*"
            env["no_proxy"] = "*"
        
        return env
    
    def execute(
        self,
        command: str,
        timeout: float | None = None,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """
        Execute a shell command.
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds (default: 60)
            working_dir: Working directory (must be within workspace)
            env: Additional environment variables
            
        Returns:
            CommandResult with execution details
        """
        start_time = datetime.now(timezone.utc)
        
        # Validate command
        is_allowed, reason = self._is_command_allowed(command)
        if not is_allowed:
            self._log(command, CommandStatus.BLOCKED, 0)
            return CommandResult(
                command=command,
                status=CommandStatus.BLOCKED,
                error=reason,
            )
        
        # Validate working directory
        if working_dir:
            work_path = Path(working_dir).resolve()
            try:
                work_path.relative_to(self._workspace_root)
            except ValueError:
                return CommandResult(
                    command=command,
                    status=CommandStatus.BLOCKED,
                    error=f"Working directory outside workspace: {working_dir}",
                )
        else:
            work_path = self._workspace_root
        
        # Prepare environment
        run_env = self._sanitize_env()
        if env:
            run_env.update(env)
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(work_path),
                capture_output=True,
                timeout=timeout or self._default_timeout,
                env=run_env,
                text=True,
            )
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            stdout = result.stdout[:self._max_output_size]
            stderr = result.stderr[:self._max_output_size]
            
            status = CommandStatus.SUCCESS if result.returncode == 0 else CommandStatus.FAILED
            
            self._log(command, status, duration_ms)
            
            return CommandResult(
                command=command,
                status=status,
                exit_code=result.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration_ms,
            )
            
        except subprocess.TimeoutExpired:
            duration_ms = (timeout or self._default_timeout) * 1000
            self._log(command, CommandStatus.TIMEOUT, duration_ms)
            
            return CommandResult(
                command=command,
                status=CommandStatus.TIMEOUT,
                duration_ms=duration_ms,
                error=f"Command timed out after {timeout or self._default_timeout}s",
            )
            
        except Exception as e:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._log(command, CommandStatus.ERROR, duration_ms)
            
            return CommandResult(
                command=command,
                status=CommandStatus.ERROR,
                duration_ms=duration_ms,
                error=str(e),
            )
    
    def run_git(
        self,
        subcommand: str,
        args: list[str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """
        Run a git command.
        
        Args:
            subcommand: Git subcommand (status, diff, log, etc.)
            args: Additional arguments
            timeout: Timeout in seconds
            
        Returns:
            CommandResult with execution details
        """
        # Only allow safe git operations
        safe_subcommands = {
            "status", "diff", "log", "show", "branch", "remote",
            "fetch", "pull", "tag", "stash list", "rev-parse",
        }
        
        if subcommand not in safe_subcommands:
            return CommandResult(
                command=f"git {subcommand}",
                status=CommandStatus.BLOCKED,
                error=f"Git subcommand not allowed: {subcommand}",
            )
        
        cmd_parts = ["git", subcommand]
        if args:
            cmd_parts.extend(args)
        
        command = " ".join(shlex.quote(p) for p in cmd_parts)
        return self.execute(command, timeout=timeout)
    
    def run_python(
        self,
        script: str,
        args: list[str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """
        Run a Python script or module.
        
        Args:
            script: Script path or module name (with -m)
            args: Additional arguments
            timeout: Timeout in seconds
            
        Returns:
            CommandResult with execution details
        """
        if script.startswith("-m "):
            cmd_parts = ["python3", "-m", script[3:]]
        else:
            # Validate script path is within workspace
            script_path = self._workspace_root / script
            try:
                script_path.relative_to(self._workspace_root)
            except ValueError:
                return CommandResult(
                    command=f"python {script}",
                    status=CommandStatus.BLOCKED,
                    error="Script path outside workspace",
                )
            cmd_parts = ["python3", str(script_path)]
        
        if args:
            cmd_parts.extend(args)
        
        command = " ".join(shlex.quote(p) for p in cmd_parts)
        return self.execute(command, timeout=timeout)
    
    def check_command_allowed(self, command: str) -> dict:
        """
        Check if a command would be allowed without executing it.
        
        Args:
            command: Command to check
            
        Returns:
            Dict with allowed status and reason
        """
        is_allowed, reason = self._is_command_allowed(command)
        return {
            "command": command,
            "allowed": is_allowed,
            "reason": reason,
        }
    
    def get_allowed_commands(self) -> list[str]:
        """Get list of allowed command prefixes."""
        return sorted(self._allowed_commands)
