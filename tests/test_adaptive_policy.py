from src.config import get_config
from src.core.memory import MemoryManager
from src.orchestration.executor import Executor


def test_memory_derives_tightening_hints(tmp_path):
    memory = MemoryManager(workspace_root=tmp_path)
    for _ in range(2):
        memory.record_meta_reflection(
            task_description="stabilize tool plan",
            success=False,
            termination_reason="unrecoverable_failure",
            diagnosis="Run failed with stagnation and tool plan violations",
            priority="high",
            strategy_updates=[
                "tighten planner tool-plan precision with stronger evidence requirements",
            ],
            confidence=0.85,
        )

    hints = memory.derive_policy_hints("tool plan evidence")
    assert hints["max_tool_steps_adjustment"] <= 0
    assert hints["require_reason_evidence"] is True


def test_memory_ignores_low_signal_reflections_for_hints(tmp_path):
    memory = MemoryManager(workspace_root=tmp_path)
    for _ in range(6):
        memory.record_meta_reflection(
            task_description="minor formatting",
            success=True,
            termination_reason="success",
            diagnosis="ok",
            priority="low",
            strategy_updates=[],
            confidence=0.02,
        )

    hints = memory.derive_policy_hints("tool plan evidence")
    assert hints["max_tool_steps_adjustment"] == 0
    assert hints["require_reason_evidence"] is False


def test_executor_applies_adaptive_policy_constraints(tmp_path):
    cfg = get_config()
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    executor = Executor(config=cfg, workspace_root=tmp_path, log_dir=log_dir)
    executor._initialize_run("run_adaptive_policy")
    assert executor._memory_manager is not None

    for _ in range(2):
        executor._memory_manager.record_meta_reflection(
            task_description="api auth task",
            success=False,
            termination_reason="fatal_error",
            diagnosis="Need stronger evidence anchors in tool plans",
            priority="high",
            strategy_updates=["tighten planner tool-plan precision with stronger evidence requirements"],
            confidence=0.9,
        )

    executor._refresh_adaptive_policy("api auth refactor")
    constraints = executor._build_plan_constraints({"confidence_score": 0.6})

    assert any("Tool-plan depth cap" in item for item in constraints)
    assert any("evidence anchor" in item for item in constraints)


def test_executor_logs_adaptive_policy_application(tmp_path):
    cfg = get_config()
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    executor = Executor(config=cfg, workspace_root=tmp_path, log_dir=log_dir)
    executor._initialize_run("run_adaptive_policy_logging")
    assert executor._memory_manager is not None
    assert executor._telemetry is not None

    executor._memory_manager.record_meta_reflection(
        task_description="api migration",
        success=False,
        termination_reason="fatal_error",
        diagnosis="Need stronger evidence anchors in tool plans",
        priority="high",
        strategy_updates=["tighten planner tool-plan precision with stronger evidence requirements"],
        confidence=0.8,
    )

    executor._refresh_adaptive_policy("api migration follow-up")

    warning_events = [
        e
        for e in executor._telemetry.events
        if str(e.data.get("warning", "")) == "adaptive_policy_applied"
    ]
    assert warning_events
    context = warning_events[-1].data.get("context", {})
    assert context.get("source") == "memory_hints"
    assert "effective_max_tool_steps" in context
