from src.orchestration.meta_agent import MetaAgentReflector
from src.core.memory import MemoryManager


def test_meta_reflector_emits_strategy_updates_on_friction():
    reflector = MetaAgentReflector()
    reflection = reflector.reflect(
        task_description="refactor planner fallback behavior",
        success=False,
        termination_reason="fatal_error",
        iterations=14,
        telemetry_summary={
            "actionable_insights": {
                "stagnation_hits": 2,
                "tool_plan_violations": 1,
                "fallback_invocations": 3,
            }
        },
        policy_profile_name="strict",
    )

    assert reflection.priority == "high"
    assert len(reflection.strategy_updates) >= 3
    assert any("strategy-shift" in item for item in reflection.strategy_updates)


def test_memory_stores_meta_reflections(tmp_path):
    memory = MemoryManager(workspace_root=tmp_path)

    status = memory.record_meta_reflection(
        task_description="improve context ranking",
        success=True,
        termination_reason="success",
        diagnosis="Run succeeded with stable execution dynamics",
        priority="low",
        strategy_updates=["retain current strategy profile; no major regressions detected"],
        confidence=0.72,
    )

    assert status == "success"
    rendered = memory.format_meta_reflections("context ranking optimization")
    assert "## Meta Reflections" in rendered
    assert "retain current strategy profile" in rendered
