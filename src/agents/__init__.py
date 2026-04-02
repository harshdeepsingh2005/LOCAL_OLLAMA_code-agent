"""
Agents Package

Exports all agent implementations and their contracts.
"""

from src.agents.base import (
    AgentContext,
    AgentInput,
    AgentOutput,
    AgentStatus,
    AgentType,
    BaseAgent,
    CodeChange,
    CoderInput,
    CoderOutput,
    FixerInput,
    FixerOutput,
    PlannerInput,
    PlannerOutput,
    ReviewerInput,
    ReviewerOutput,
    ReviewIssue,
    ReviewVerdict,
    SubtaskToolPlan,
    Subtask,
    ToolCall,
    ToolPlanStep,
)
from src.agents.coder import CoderAgent
from src.agents.fixer import FixerAgent
from src.agents.planner import PlannerAgent
from src.agents.reviewer import ReviewerAgent

__all__ = [
    # Base types
    "AgentContext",
    "AgentInput",
    "AgentOutput",
    "AgentStatus",
    "AgentType",
    "BaseAgent",
    # Planner
    "PlannerAgent",
    "PlannerInput",
    "PlannerOutput",
    "Subtask",
    "SubtaskToolPlan",
    "ToolCall",
    "ToolPlanStep",
    # Coder
    "CoderAgent",
    "CoderInput",
    "CoderOutput",
    "CodeChange",
    # Reviewer
    "ReviewerAgent",
    "ReviewerInput",
    "ReviewerOutput",
    "ReviewIssue",
    "ReviewVerdict",
    # Fixer
    "FixerAgent",
    "FixerInput",
    "FixerOutput",
]
