"""Tests for loop controller task reset behavior."""

from src.config import get_config
from src.orchestration.loop_controller import LoopController, LoopState


def test_reset_for_new_task_clears_terminal_state() -> None:
    """reset_for_new_task should provide a clean per-task loop state."""
    config = get_config()
    loop = LoopController(config=config, run_id="run_test")

    loop.start()
    first = loop.begin_iteration("planner")
    assert first is not None
    loop.end_iteration(first, success=True, tokens_used=42)
    loop.complete_success("done")

    assert loop.is_terminal
    assert loop.iteration_count == 1

    loop.reset_for_new_task()

    assert loop.state == LoopState.PLANNING
    assert loop.iteration_count == 0
    assert loop.termination_reason is None
    assert not loop.needs_user_continue

    second = loop.begin_iteration("planner")
    assert second is not None
    assert second.iteration_number == 1


def test_reset_for_new_task_clears_paused_state() -> None:
    """reset_for_new_task should recover from paused max-iteration state."""
    config = get_config()
    config.limits.iterations.max_loop_iterations = 1

    loop = LoopController(config=config, run_id="run_test")
    loop.start()

    first = loop.begin_iteration("planner")
    assert first is not None

    paused = loop.begin_iteration("planner")
    assert paused is None
    assert loop.is_paused
    assert loop.needs_user_continue

    loop.reset_for_new_task()

    assert loop.state == LoopState.PLANNING
    assert not loop.needs_user_continue

    retry = loop.begin_iteration("planner")
    assert retry is not None
