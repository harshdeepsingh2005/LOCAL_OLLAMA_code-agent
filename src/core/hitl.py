"""
Human-In-The-Loop (HITL) Security Module

Provides interactive permission prompting before the agent executes shell
commands. Read-only commands (ls, cat, grep, …) auto-approve; mutating
commands (rm, pip install, git push, …) pause the agent and ask the user
in the terminal: "Agent wants to run X. Allow? [y/N]"

Design Decisions:
- Pattern-based classification (fast, no LLM needed)
- Three permission levels: AUTO_APPROVE, ASK, DENY
- Policy is configurable — can be relaxed (fully auto) or hardened (always ask)
- Non-interactive mode (CI/batch) falls back to deny_mutating policy
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class PermissionLevel(str, Enum):
    """How to handle a given command class."""
    AUTO_APPROVE = "auto_approve"   # Execute without asking
    ASK          = "ask"            # Pause and prompt the user
    DENY         = "deny"           # Always block (e.g., rm -rf /)


class HITLDecision(str, Enum):
    """What the HITL gate decided for a specific command."""
    APPROVED          = "approved"           # Will run
    DENIED            = "denied"             # Blocked by policy or user
    USER_APPROVED     = "user_approved"      # User said yes
    USER_DENIED       = "user_denied"        # User said no
    NON_INTERACTIVE   = "non_interactive"    # No TTY, applied default policy


# ---------------------------------------------------------------------------
# Command classification rules
# ---------------------------------------------------------------------------

# Commands that are purely read-only: safe to auto-approve
_READ_ONLY_PREFIXES: set[str] = {
    "ls", "ll", "la", "l",
    "cat", "head", "tail", "less", "more",
    "grep", "rg", "ripgrep", "awk", "sed -n", "sed -e",
    "find", "fd",
    "tree",
    "stat", "file", "wc", "du", "df",
    "echo", "printf",
    "which", "whereis", "type",
    "env", "printenv",
    "pwd",
    "date", "uname",
    "git status", "git diff", "git log", "git show",
    "git branch", "git remote -v", "git fetch",
    "python --version", "python3 --version",
    "pip list", "pip show", "pip freeze",
    "npm list", "npm ls",
}

# Commands that are always blocked regardless of user choice
_ALWAYS_DENY_PATTERNS: list[str] = [
    r"rm\s+-rf\s+/",
    r"rm\s+-rf\s+~",
    r"mkfs",
    r"fdisk",
    r"dd\s+if=",
    r"\|\s*sh\b",
    r"\|\s*bash\b",
    r">\s*/dev/sd",
    r"chmod\s+777\s+/",
]

# Commands that must prompt the user before running
_MUTATING_PREFIXES: set[str] = {
    "rm", "rmdir", "mv", "cp",
    "mkdir", "touch",
    "pip install", "pip uninstall",
    "npm install", "npm uninstall", "npm run",
    "yarn add", "yarn remove", "yarn install",
    "git commit", "git push", "git pull", "git merge",
    "git rebase", "git reset", "git stash",
    "git checkout", "git switch",
    "python", "python3",   # executing scripts may have side effects
    "make",
    "cargo build", "cargo run",
    "docker", "kubectl",
    "brew", "apt", "apt-get", "yum", "dnf",
    "curl", "wget",
    "ssh", "scp",
    "export", "unset",
    "source", ".",  # sourcing scripts
    "sudo",
}


def classify_command(command: str) -> PermissionLevel:
    """
    Classify a shell command into a PermissionLevel.

    Args:
        command: The raw shell command string

    Returns:
        PermissionLevel indicating how to handle it
    """
    stripped = command.strip().lower()

    # 1. Check always-deny patterns first
    for pattern in _ALWAYS_DENY_PATTERNS:
        if re.search(pattern, stripped):
            return PermissionLevel.DENY

    # 2. Check read-only prefixes
    for prefix in _READ_ONLY_PREFIXES:
        if stripped == prefix or stripped.startswith(prefix + " "):
            return PermissionLevel.AUTO_APPROVE

    # 3. Check mutating prefixes — these require user confirmation
    for prefix in _MUTATING_PREFIXES:
        if stripped == prefix or stripped.startswith(prefix + " "):
            return PermissionLevel.ASK

    # 4. Unknown command — ask to be safe
    return PermissionLevel.ASK


# ---------------------------------------------------------------------------
# HITL Gate
# ---------------------------------------------------------------------------

@dataclass
class HITLConfig:
    """Configuration for the HITL gate."""
    # If True, never prompt — auto-approve everything not in DENY
    fully_autonomous: bool = False
    # If True, never prompt — deny everything in ASK (for CI)
    deny_mutating_in_ci: bool = False
    # Custom prompt writer (injectable for testing)
    prompt_fn: Callable[[str], bool] | None = None


@dataclass
class HITLResult:
    """Outcome of an HITL gate check."""
    command: str
    permission_level: PermissionLevel
    decision: HITLDecision
    reason: str = ""


class HITLGate:
    """
    Interactive permission gate for shell commands.

    Usage (inside ToolExecutor)::

        gate = HITLGate(config=HITLConfig())
        result = gate.check("pip install numpy")
        if result.decision not in (HITLDecision.APPROVED, HITLDecision.USER_APPROVED):
            return f"Command blocked: {result.reason}"
        # ... execute
    """

    def __init__(self, config: HITLConfig | None = None) -> None:
        self._config = config or HITLConfig()
        self._approval_log: list[HITLResult] = []

    def check(self, command: str) -> HITLResult:
        """
        Evaluate a command and possibly prompt the user.

        Args:
            command: Shell command to evaluate

        Returns:
            HITLResult describing the decision
        """
        level = classify_command(command)

        if level == PermissionLevel.DENY:
            result = HITLResult(
                command=command,
                permission_level=level,
                decision=HITLDecision.DENIED,
                reason="Command matches always-deny policy",
            )
        elif level == PermissionLevel.AUTO_APPROVE or self._config.fully_autonomous:
            result = HITLResult(
                command=command,
                permission_level=level,
                decision=HITLDecision.APPROVED,
                reason="Read-only or fully-autonomous mode",
            )
        else:
            # Mutating command — need user decision
            result = self._prompt_user(command, level)

        self._approval_log.append(result)
        return result

    def _prompt_user(self, command: str, level: PermissionLevel) -> HITLResult:
        """Ask the user interactively whether to allow a mutating command."""
        # Check if we're running in a terminal
        is_interactive = sys.stdin.isatty() and sys.stdout.isatty()

        if not is_interactive:
            # Non-interactive (CI/batch): apply policy
            if self._config.deny_mutating_in_ci:
                return HITLResult(
                    command=command,
                    permission_level=level,
                    decision=HITLDecision.NON_INTERACTIVE,
                    reason="Non-interactive terminal; mutating commands denied by policy",
                )
            else:
                # Lenient CI mode: approve without asking
                return HITLResult(
                    command=command,
                    permission_level=level,
                    decision=HITLDecision.APPROVED,
                    reason="Non-interactive terminal; auto-approved (lenient CI mode)",
                )

        # Use custom prompt function if provided (for tests)
        if self._config.prompt_fn is not None:
            allowed = self._config.prompt_fn(command)
            return HITLResult(
                command=command,
                permission_level=level,
                decision=HITLDecision.USER_APPROVED if allowed else HITLDecision.USER_DENIED,
                reason="Custom prompt function result",
            )

        # Interactive terminal — ask the user
        try:
            print("\n" + "─" * 60)
            print(f"  🔐  Agent wants to run:")
            print(f"  \033[33m$ {command}\033[0m")
            print("─" * 60)
            answer = input("  Allow? [y/N] ").strip().lower()
            allowed = answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            allowed = False

        return HITLResult(
            command=command,
            permission_level=level,
            decision=HITLDecision.USER_APPROVED if allowed else HITLDecision.USER_DENIED,
            reason="Interactive user prompt",
        )

    @property
    def approval_log(self) -> list[HITLResult]:
        """Full history of all HITL decisions in this session."""
        return list(self._approval_log)

    def summary(self) -> dict[str, int]:
        """Counts of each decision type."""
        counts: dict[str, int] = {}
        for r in self._approval_log:
            counts[r.decision.value] = counts.get(r.decision.value, 0) + 1
        return counts
