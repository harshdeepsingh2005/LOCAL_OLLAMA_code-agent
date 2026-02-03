"""
CLI Package

Interactive command-line interface for the local coding agent.
Provides a Claude Code–like experience with human-in-the-loop safety.
"""

from src.cli.app import AgentCLI, main
from src.cli.session import Session, SessionState
from src.cli.commands import CommandHandler
from src.cli.display import Display

__all__ = [
    "AgentCLI",
    "main",
    "Session",
    "SessionState",
    "CommandHandler",
    "Display",
]
