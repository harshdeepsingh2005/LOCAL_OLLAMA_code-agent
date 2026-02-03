"""
Local Coding Agents

A production-grade local AI coding agent system.
Runs fully offline on Apple Silicon (M-series, 16 GB RAM).
Uses Ollama for local LLM inference.
"""

from src.config import Configuration, get_config
from src.orchestration import ExecutionResult, Executor

__version__ = "0.1.0"
__all__ = [
    "Configuration",
    "get_config",
    "Executor",
    "ExecutionResult",
]
