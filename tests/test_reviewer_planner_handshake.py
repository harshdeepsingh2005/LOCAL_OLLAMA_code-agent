from src.agents import AgentStatus, ReviewerOutput, ReviewVerdict
from src.config import get_config
from src.orchestration.executor import Executor


def test_reviewer_risk_signals_feed_planner_constraints(tmp_path):
    cfg = get_config()
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    executor = Executor(config=cfg, workspace_root=tmp_path, log_dir=log_dir)
    executor._initialize_run("run_reviewer_planner_handshake")

    reviewer_output = ReviewerOutput(
        task_id="task_1",
        status=AgentStatus.SUCCESS,
        verdict=ReviewVerdict.REQUEST_CHANGES,
        task_complete=False,
        issues=[],
        summary="Potential auth/session regression risk",
        risk_score=0.72,
        potential_breakages=["Session token refresh may break for expired cookies"],
        criteria_met={"auth": False},
    )

    executor._ingest_reviewer_risk_signals(reviewer_output, "task_1")
    constraints = executor._build_plan_constraints({"confidence_score": 0.7})

    assert any("reviewer-identified breakage risks" in item for item in constraints)
    assert any("Session token refresh" in item for item in constraints)


def test_reviewer_risk_signal_emits_telemetry_warning(tmp_path):
    cfg = get_config()
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    executor = Executor(config=cfg, workspace_root=tmp_path, log_dir=log_dir)
    executor._initialize_run("run_reviewer_risk_telemetry")
    assert executor._telemetry is not None

    reviewer_output = ReviewerOutput(
        task_id="task_2",
        status=AgentStatus.SUCCESS,
        verdict=ReviewVerdict.REQUEST_CHANGES,
        task_complete=False,
        issues=[],
        summary="Data migration order can break foreign keys",
        risk_score=0.65,
        potential_breakages=["FK dependency order failure during rollback"],
        criteria_met={},
    )

    executor._ingest_reviewer_risk_signals(reviewer_output, "task_2")
    warnings = [
        event
        for event in executor._telemetry.events
        if str(event.data.get("warning", "")) == "reviewer_risk_signal_captured"
    ]
    assert warnings
