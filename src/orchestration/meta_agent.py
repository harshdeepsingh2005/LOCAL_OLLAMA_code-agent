"""Meta-agent reflection loop for post-run strategy adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetaReflection:
    """Structured post-run reflection output."""

    diagnosis: str
    priority: str
    strategy_updates: list[str]
    confidence: float


class MetaAgentReflector:
    """Derive strategy-level guidance from run outcomes and telemetry."""

    def reflect(
        self,
        *,
        task_description: str,
        success: bool,
        termination_reason: str,
        iterations: int,
        telemetry_summary: dict[str, Any],
        policy_profile_name: str,
    ) -> MetaReflection:
        insights = telemetry_summary.get("actionable_insights", {}) if isinstance(telemetry_summary, dict) else {}
        stagnation_hits = int(insights.get("stagnation_hits", 0))
        tool_plan_violations = int(insights.get("tool_plan_violations", 0))
        fallback_invocations = int(insights.get("fallback_invocations", 0))

        strategy_updates: list[str] = []
        if stagnation_hits > 0:
            strategy_updates.append("increase early strategy-shift sensitivity for fixer loop")
        if tool_plan_violations > 0:
            strategy_updates.append("tighten planner tool-plan precision with stronger evidence requirements")
        if fallback_invocations > 0:
            strategy_updates.append("prefer historically reliable tools before fallback-prone tools")
        if iterations >= 12:
            strategy_updates.append("decompose future tasks into smaller subtasks with lower iteration budgets")
        if not success:
            strategy_updates.append("escalate planning rigor and add explicit rollback checkpoints")

        if not strategy_updates:
            strategy_updates.append("retain current strategy profile; no major regressions detected")

        if not success:
            diagnosis = f"Run failed ({termination_reason}) for task: {task_description[:140]}"
            priority = "high"
            confidence = 0.82
        elif stagnation_hits or tool_plan_violations:
            diagnosis = "Run succeeded with orchestration friction requiring optimization"
            priority = "medium"
            confidence = 0.74
        else:
            diagnosis = "Run succeeded with stable execution dynamics"
            priority = "low"
            confidence = 0.68

        if policy_profile_name == "strict" and fallback_invocations > 0:
            strategy_updates.append("review strict-profile constraints against required tool pathways")

        return MetaReflection(
            diagnosis=diagnosis,
            priority=priority,
            strategy_updates=strategy_updates[:6],
            confidence=max(0.0, min(1.0, confidence)),
        )
