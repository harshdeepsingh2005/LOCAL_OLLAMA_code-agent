"""
End-to-End Agent Loop Tests

Tests the complete agent execution flow from input to output.
Uses mocked LLM responses to test the pipeline without network calls.
"""

import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Any


class MockLLMResponse:
    """Mock LLM response for testing."""
    
    def __init__(self, content: str):
        self.content = content
    
    def to_dict(self) -> dict:
        return {"content": self.content}


class MockLLMClient:
    """Mock LLM client that returns predefined responses."""
    
    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.calls: list[dict] = []
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        **kwargs
    ) -> str:
        """Return mock response based on agent type hint in system prompt."""
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "kwargs": kwargs,
        })
        
        # Determine agent type from system prompt
        if "planner" in system_prompt.lower():
            return self.responses.get("planner", "{}")
        elif "coder" in system_prompt.lower():
            return self.responses.get("coder", "{}")
        elif "reviewer" in system_prompt.lower():
            return self.responses.get("reviewer", "{}")
        elif "fixer" in system_prompt.lower():
            return self.responses.get("fixer", "{}")
        
        return "{}"


@pytest.fixture
def workspace(tmp_path):
    """Create a test workspace with initial files."""
    # Create source file
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    
    (src_dir / "calculator.py").write_text('''
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b  # Bug: should be a - b but let's pretend there's an issue

def multiply(a, b):
    return a * b
''')
    
    # Create test file
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    
    (tests_dir / "test_calculator.py").write_text('''
from src.calculator import add, subtract

def test_add():
    assert add(2, 3) == 5

def test_subtract():
    assert subtract(5, 3) == 2
''')
    
    return tmp_path


@pytest.fixture
def mock_responses():
    """Mock LLM responses for each agent."""
    return {
        "planner": json.dumps({
            "analysis": "The calculator needs a divide function.",
            "subtasks": [
                {
                    "id": "task_001",
                    "title": "Add divide function",
                    "description": "Add a divide function to calculator.py",
                    "target_files": ["src/calculator.py"],
                    "dependencies": [],
                    "complexity": "low",
                }
            ],
        }),
        "coder": json.dumps({
            "success": True,
            "code": '''def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b''',
            "file_path": "src/calculator.py",
            "diff": '''@@ -9,3 +9,8 @@ def subtract(a, b):
 
 def multiply(a, b):
     return a * b
+
+def divide(a, b):
+    if b == 0:
+        raise ValueError("Cannot divide by zero")
+    return a / b''',
            "reasoning": "Added divide function with zero-check",
            "suggested_tests": ["test_divide", "test_divide_by_zero"],
        }),
        "reviewer": json.dumps({
            "approved": True,
            "issues": [],
            "suggestions": ["Consider adding type hints"],
            "security_concerns": [],
            "summary": "Code looks good. Added proper zero division check.",
        }),
    }


class TestAgentLoop:
    """Test the complete agent loop."""
    
    def test_planner_agent_contract_exists(self):
        """Planner agent should expose contract-level system prompt and type."""
        from src.agents.planner import PlannerAgent

        planner = PlannerAgent()
        assert planner.agent_type is not None
        assert isinstance(planner.system_prompt, str)
        assert len(planner.system_prompt) > 10
    
    def test_coder_agent_contract_exists(self):
        """Coder agent should expose contract-level system prompt and type."""
        from src.agents.coder import CoderAgent

        coder = CoderAgent()
        assert coder.agent_type is not None
        assert isinstance(coder.system_prompt, str)
        assert len(coder.system_prompt) > 10
    
    def test_reviewer_agent_contract_exists(self):
        """Reviewer agent should expose contract-level system prompt and type."""
        from src.agents.reviewer import ReviewerAgent

        reviewer = ReviewerAgent()
        assert reviewer.agent_type is not None
        assert isinstance(reviewer.system_prompt, str)
        assert len(reviewer.system_prompt) > 10


class TestExecutorFlow:
    """Test the executor orchestration."""
    
    def test_executor_requires_config_and_paths(self, workspace):
        """Executor should be constructible with current required dependencies."""
        from src.orchestration.executor import Executor
        from src.config import get_config

        cfg = get_config()
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)
        assert executor is not None

    def test_fast_map_returns_confidence(self, workspace):
        """Fast map should return bounded context with confidence score."""
        from src.orchestration.executor import Executor
        from src.config import get_config

        cfg = get_config()
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)

        executor._initialize_run("run_fast_map")
        executor._active_route = executor._task_router.route("add divide function")

        fast_map = executor._run_fast_map("add divide function")

        assert "confidence_score" in fast_map
        assert 0.0 <= fast_map["confidence_score"] <= 1.0
        assert "relevant_files" in fast_map

    def test_build_plan_constraints_low_confidence_adds_probe_limit(self, workspace):
        """Low-confidence scenarios should force targeted probing constraints."""
        from src.orchestration.executor import Executor
        from src.config import get_config

        cfg = get_config()
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)

        constraints = executor._build_plan_constraints({"confidence_score": 0.05})

        assert any("Low confidence" in c for c in constraints)
        assert any("Probe depth capped" in c for c in constraints)

    def test_verification_gate_passes_for_clean_review(self, workspace):
        """Verification gate should pass when there are no major/critical issues."""
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import ReviewerOutput, AgentStatus

        cfg = get_config()
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)

        reviewer_output = ReviewerOutput(
            task_id="t1",
            status=AgentStatus.SUCCESS,
            task_complete=True,
            issues=[],
            criteria_met={"criterion": True},
        )

        allowed, gate = executor._verification_gate(reviewer_output)

        assert allowed is True
        assert gate["no_errors"] is True

    def test_apply_changes_rejects_excessive_file_surface(self, workspace):
        """Apply changes should reject modifications beyond per-cycle file limits."""
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import CodeChange

        cfg = get_config()
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)
        executor._initialize_run("run_surface_files")

        limit = executor._execution_policy.max_files_per_cycle
        changes = [
            CodeChange(
                file_path=f"src/file_{i}.py",
                change_type="modify",
                description="test",
                new_content="print('x')\n",
            )
            for i in range(limit + 1)
        ]

        assert executor._apply_changes(changes) is False

    def test_apply_changes_rejects_excessive_line_surface(self, workspace):
        """Apply changes should reject modifications beyond per-cycle line limits."""
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import CodeChange

        cfg = get_config()
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)
        executor._initialize_run("run_surface_lines")

        too_many_lines = "\n".join(["line"] * (executor._execution_policy.max_lines_per_cycle + 50))
        change = CodeChange(
            file_path="src/huge_change.py",
            change_type="modify",
            description="test large change",
            new_content=too_many_lines,
        )

        assert executor._apply_changes([change]) is False

    def test_fallback_plan_avoids_placeholder_targets(self, workspace):
        """Fallback planning should avoid hidden/non-code placeholders and infer practical targets."""
        from src.orchestration.executor import Executor
        from src.config import get_config

        cfg = get_config()
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)

        plan = executor._build_fallback_plan(
            task_description="Create a Python function that calculates fibonacci numbers",
            workspace_context={"relevant_files": ["src/.gitkeep", "tests/.gitkeep", ".agent_memory.json"]},
            reason="test",
        )

        assert plan.subtasks
        targets = plan.subtasks[0].target_files
        assert "src/fibonacci.py" in targets
        assert "tests/test_fibonacci.py" in targets
        assert all(not t.endswith(".gitkeep") for t in targets)
        assert all(not t.startswith(".") for t in targets)


class TestToolMemoryOrchestrationIntegration:
    """Integration checks for tools, memory, and orchestration wiring."""

    def test_tool_executor_memory_and_search_dispatch(self, workspace):
        """Tool executor should dispatch memory and workspace-learning tools."""
        from src.core.memory import MemoryManager
        from src.core.agent_tools import ToolExecutor
        from src.agents.base import ToolCall

        memory = MemoryManager(workspace)
        tool_executor = ToolExecutor(memory_manager=memory, workspace_root=str(workspace), run_id="run_tools")

        write_result = tool_executor.execute_call(
            ToolCall(tool_name="write_memory", arguments={"fact": "workspace uses pytest"})
        )
        assert "Successfully" in write_result or "already exists" in write_result

        read_result = tool_executor.execute_call(ToolCall(tool_name="read_memory", arguments={}))
        assert "pytest" in read_result

        grep_result = tool_executor.execute_call(
            ToolCall(tool_name="grep_search", arguments={"pattern": "def add", "glob": "**/*.py"})
        )
        assert "def add" in grep_result

    def test_tool_executor_mcp_entrypoints_fail_gracefully_without_server(self, workspace):
        """MCP tool entrypoints should be wired and return safe errors when no server is connected."""
        from src.core.memory import MemoryManager
        from src.core.agent_tools import ToolExecutor
        from src.agents.base import ToolCall

        memory = MemoryManager(workspace)
        tool_executor = ToolExecutor(memory_manager=memory, workspace_root=str(workspace), run_id="run_mcp")

        list_result = tool_executor.execute_call(ToolCall(tool_name="mcp_list_tools", arguments={}))
        assert "No MCP servers" in list_result

        call_result = tool_executor.execute_call(
            ToolCall(tool_name="mcp_call", arguments={"tool_name": "nonexistent_tool", "arguments": {}})
        )
        assert "MCP tool error" in call_result

    def test_executor_initialize_wires_memory_tools_and_context_pipeline(self, workspace):
        """Executor run initialization should wire memory manager, tools, and context builder."""
        from src.orchestration.executor import Executor
        from src.config import get_config

        cfg = get_config()
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)

        executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)
        run_id = executor._initialize_run("run_integration")

        assert run_id == "run_integration"
        assert executor._memory_manager is not None
        assert executor._tool_executor is not None
        assert executor._context_builder is not None
        assert executor._task_router is not None
        assert executor._validation_layer is not None

    def test_planning_tool_calls_execute_before_final_plan(self, workspace):
        """Planner tool calls should be executed and planning should continue to subtasks."""
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import AgentStatus, PlannerOutput, Subtask
        from src.agents.base import ToolCall

        cfg = get_config()
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)
        executor._initialize_run("run_planner_tools")
        assert executor._loop is not None
        executor._loop.start()

        subtask = Subtask(
            id="task_1",
            title="Add divide helper",
            description="Add a divide helper to calculator module",
            acceptance_criteria=["division helper exists"],
            target_files=["src/calculator.py"],
            dependencies=[],
        )

        outputs = [
            PlannerOutput(
                task_id="planning",
                status=AgentStatus.SUCCESS,
                tool_calls=[ToolCall(tool_name="read_memory", arguments={})],
                subtasks=[],
            ),
            PlannerOutput(
                task_id="planning",
                status=AgentStatus.SUCCESS,
                tool_calls=[],
                subtasks=[subtask],
            ),
        ]

        original_execute = executor._planner.execute

        def fake_execute(*args, **kwargs):
            return outputs.pop(0)

        executor._planner.execute = fake_execute  # type: ignore[assignment]
        try:
            result = executor._execute_planning(
                "add divide helper",
                fast_map={"confidence_score": 0.9},
            )
        finally:
            executor._planner.execute = original_execute  # type: ignore[assignment]

        assert result is not None
        assert len(result.subtasks) == 1
        assert result.subtasks[0].id == "task_1"

    def test_planning_tool_only_loop_uses_fallback_plan(self, workspace):
        """Planner should not stall in tool-only refinement loops; fallback plan must unblock execution."""
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import AgentStatus, PlannerOutput
        from src.agents.base import ToolCall

        cfg = get_config()
        cfg.limits.iterations.max_planning_cycles = 2
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)

        executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)
        executor._initialize_run("run_planner_fallback")
        assert executor._loop is not None
        executor._loop.start()

        def _tool_only_planner(*_args, **_kwargs):
            return PlannerOutput(
                task_id="planning",
                status=AgentStatus.SUCCESS,
                tool_calls=[ToolCall(tool_name="read_memory", arguments={})],
                subtasks=[],
            )

        original_execute = executor._planner.execute
        executor._planner.execute = _tool_only_planner  # type: ignore[assignment]
        try:
            plan = executor._execute_planning("Add divide helper")
        finally:
            executor._planner.execute = original_execute  # type: ignore[assignment]

        assert plan is not None
        assert plan.status == AgentStatus.SUCCESS
        assert len(plan.subtasks) == 1
        assert "fallback" in plan.plan_summary.lower()

    def test_planning_parse_failure_uses_fallback_plan(self, workspace):
        """Planner parse failures should degrade to fallback planning instead of aborting the run."""
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import AgentStatus, PlannerOutput

        cfg = get_config()
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)

        executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)
        executor._initialize_run("run_planner_parse_fallback")
        assert executor._loop is not None
        executor._loop.start()

        def _failed_planner(*_args, **_kwargs):
            return PlannerOutput(
                task_id="planning",
                status=AgentStatus.FAILED,
                error="Failed to parse planner response: malformed JSON",
            )

        original_execute = executor._planner.execute
        executor._planner.execute = _failed_planner  # type: ignore[assignment]
        try:
            plan = executor._execute_planning("Create fibonacci function")
        finally:
            executor._planner.execute = original_execute  # type: ignore[assignment]

        assert plan is not None
        assert plan.status == AgentStatus.SUCCESS
        assert len(plan.subtasks) == 1
        assert "fallback" in plan.plan_summary.lower()

    def test_coder_parse_repairs_unescaped_control_chars(self):
        """Coder parser should repair unescaped control characters inside JSON strings."""
        from src.agents.coder import CoderAgent

        raw = '{"implementation_notes":"line1\nline2\x0bline3"}'
        repaired = CoderAgent._escape_control_chars_in_json_strings(raw)

        assert "\\n" in repaired
        assert "\\u000b" in repaired

    def test_execute_coder_uses_fallback_on_parse_failure_for_fibonacci(self, workspace):
        """Executor should recover from coder parse failures using deterministic fibonacci fallback."""
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import AgentStatus, CoderOutput, Subtask

        cfg = get_config()
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)
        executor._initialize_run("run_coder_fallback")
        assert executor._loop is not None
        executor._loop.start()

        subtask = Subtask(
            id="1",
            title="Create fibonacci function",
            description="Create a Python function that calculates fibonacci numbers",
            acceptance_criteria=["function exists"],
            target_files=["src/fibonacci.py", "tests/test_fibonacci.py"],
            dependencies=[],
        )

        def _failed_coder(*_args, **_kwargs):
            return CoderOutput(
                task_id="1",
                status=AgentStatus.FAILED,
                error="Failed to parse coder response: malformed JSON",
            )

        original_execute = executor._coder.execute
        executor._coder.execute = _failed_coder  # type: ignore[assignment]
        try:
            out = executor._execute_coder(subtask, file_contents={})
        finally:
            executor._coder.execute = original_execute  # type: ignore[assignment]

        assert out is not None
        assert out.status == AgentStatus.SUCCESS
        assert len(out.changes) >= 1
        assert any(change.file_path == "src/fibonacci.py" for change in out.changes)

    def test_execute_task_completes_on_coder_noop(self, workspace):
        """Executor should complete a task when coder returns success with no required changes."""
        from src.orchestration.executor import Executor
        from src.orchestration.task_graph import TaskNode, TaskStatus
        from src.config import get_config
        from src.agents import AgentStatus, CoderOutput, Subtask

        cfg = get_config()
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)
        executor._initialize_run("run_noop_task")
        assert executor._loop is not None
        executor._loop.start()

        subtask = Subtask(
            id="task_noop",
            title="No-op task",
            description="Confirm behavior already implemented",
            acceptance_criteria=["Behavior exists"],
            target_files=["src/calculator.py"],
            dependencies=[],
        )
        node = TaskNode(id="task_noop", subtask=subtask)

        def _noop_coder(*_args, **_kwargs):
            return CoderOutput(
                task_id="task_noop",
                status=AgentStatus.SUCCESS,
                changes=[],
                implementation_notes="Behavior already exists; no file edits required.",
            )

        original = executor._execute_coder
        executor._execute_coder = _noop_coder  # type: ignore[assignment]
        try:
            ok = executor._execute_task(node)
        finally:
            executor._execute_coder = original  # type: ignore[assignment]

        assert ok is True
        assert node.status == TaskStatus.COMPLETED


class TestPlannerHardening:
        """Planner parsing and validation hardening tests."""

        def _planner_context(self, workspace):
                from src.orchestration.executor import Executor
                from src.config import get_config
                from src.agents import AgentType

                cfg = get_config()
                log_dir = workspace / "logs"
                log_dir.mkdir(exist_ok=True)
                executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)
                executor._initialize_run("run_planner_hardening")
                ctx = executor._create_agent_context(AgentType.PLANNER)
                return executor._planner, ctx

        def test_planner_no_masking_on_empty_plan(self, workspace):
                """Planner should fail (not mask) when model returns no subtasks/tool_calls/clarification."""
                from src.agents import AgentStatus, PlannerInput

                planner, ctx = self._planner_context(workspace)
                planner_input = PlannerInput(
                        task_id="planning",
                        run_id="run_planner_hardening",
                        task_description="Implement task",
                        workspace_context={"relevant_files": ["src/app.py"]},
                )

                output = planner._parse_response(
                        """```json
                        {"plan_summary": "x", "subtasks": [], "tool_calls": [], "requires_clarification": false}
                        ```""",
                        planner_input,
                        ctx,
                )

                assert output.status == AgentStatus.FAILED
                assert "no subtasks" in (output.error or "").lower()

        def test_planner_does_not_merge_simple_task_subtasks(self, workspace):
                """Planner should preserve model subtasks and avoid implicit merge masking."""
                from src.agents import AgentStatus, PlannerInput

                planner, ctx = self._planner_context(workspace)
                planner_input = PlannerInput(
                        task_id="planning",
                        run_id="run_planner_hardening",
                        task_description="Small README wording update",
                        workspace_context={"relevant_files": ["README.md"]},
                )

                output = planner._parse_response(
                        """```json
                        {
                            "plan_summary": "update docs",
                            "subtasks": [
                                {
                                    "id": "1",
                                    "title": "Update sentence",
                                    "description": "Update intro sentence in README.",
                                    "acceptance_criteria": ["Sentence updated"],
                                    "target_files": ["README.md"],
                                    "dependencies": [],
                                    "estimated_complexity": "low"
                                },
                                {
                                    "id": "2",
                                    "title": "Verify readability",
                                    "description": "Review wording and ensure tone remains consistent.",
                                    "acceptance_criteria": ["Tone is consistent"],
                                    "target_files": ["README.md"],
                                    "dependencies": ["1"],
                                    "estimated_complexity": "low"
                                }
                            ],
                            "tool_calls": []
                        }
                        ```""",
                        planner_input,
                        ctx,
                )

                assert output.status == AgentStatus.SUCCESS
                assert len(output.subtasks) == 2

        def test_planner_filters_invalid_tool_calls(self, workspace):
                """Planner should validate tool calls and discard unknown tool names."""
                from src.agents import AgentStatus, PlannerInput

                planner, ctx = self._planner_context(workspace)
                planner_input = PlannerInput(
                        task_id="planning",
                        run_id="run_planner_hardening",
                        task_description="Investigate repository structure",
                        workspace_context={"relevant_files": ["src/main.py"]},
                )

                output = planner._parse_response(
                        """```json
                        {
                            "plan_summary": "collect evidence",
                            "subtasks": [],
                            "requires_clarification": false,
                            "tool_calls": [
                                {"tool_name": "list_dir", "arguments": {"path": "src"}},
                                {"tool_name": "not_a_real_tool", "arguments": {}}
                            ]
                        }
                        ```""",
                        planner_input,
                        ctx,
                )

                assert output.status == AgentStatus.SUCCESS
                assert len(output.tool_calls) == 1
                assert output.tool_calls[0].tool_name == "list_dir"


class TestBaseAgentStability:
    """Base agent reliability controls: retries, timeout, sanitization, token management."""

    def _build_context(self, workspace):
        from src.config import get_config
        from src.agents.base import AgentContext, AgentType
        from src.core.context_manager import ContextManager

        class DummyLLMClient:
            def __init__(self):
                self.last_request = None

            def complete(self, request, token_budget=None):
                from src.core.llm_client import CompletionResponse

                self.last_request = request
                return CompletionResponse(
                    content='{"ok": true}',
                    model="dummy",
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    latency_ms=1.0,
                    finish_reason="stop",
                )

        cfg = get_config()
        llm = DummyLLMClient()
        ctx = AgentContext(
            run_id="run_base_agent",
            agent_type=AgentType.PLANNER,
            config=cfg,
            llm_client=llm,  # type: ignore[arg-type]
            context_manager=ContextManager(),
            max_tokens=128,
            max_retries=2,
            timeout_seconds=0.5,
        )
        return ctx, llm

    def test_base_agent_retries_until_success(self, workspace):
        from src.agents.base import AgentStatus, PlannerInput, PlannerOutput
        from src.agents.planner import PlannerAgent

        class FlakyPlanner(PlannerAgent):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def _execute_impl(self, input_data, context):
                self.calls += 1
                if self.calls < 2:
                    return PlannerOutput(task_id=input_data.task_id, status=AgentStatus.FAILED, error="transient")
                return PlannerOutput(
                    task_id=input_data.task_id,
                    status=AgentStatus.SUCCESS,
                    plan_summary="ok",
                    subtasks=[],
                    tool_calls=[{"tool_name": "read_memory", "arguments": {}}],
                )

        ctx, _ = self._build_context(workspace)
        agent = FlakyPlanner()
        out = agent.execute(
            PlannerInput(task_id="planning", run_id="run_base_agent", task_description="Investigate repo"),
            ctx,
        )

        assert out.status == AgentStatus.SUCCESS
        assert out.retries == 1

    def test_base_agent_enforces_timeout(self, workspace):
        from src.agents.base import AgentStatus, PlannerInput, PlannerOutput
        from src.agents.planner import PlannerAgent

        class SlowPlanner(PlannerAgent):
            def _execute_impl(self, input_data, context):
                time.sleep(0.03)
                return PlannerOutput(task_id=input_data.task_id, status=AgentStatus.SUCCESS, plan_summary="ok")

        ctx, _ = self._build_context(workspace)
        ctx.timeout_seconds = 0.01
        out = SlowPlanner().execute(
            PlannerInput(task_id="planning", run_id="run_base_agent", task_description="Investigate repo"),
            ctx,
        )

        assert out.status == AgentStatus.TIMEOUT

    def test_sanitize_input_case_insensitive(self, workspace):
        from src.agents.planner import PlannerAgent

        ctx, _ = self._build_context(workspace)
        ctx.config.policies.safety.prompt_injection.block_patterns = ["DROP TABLE"]

        sanitized = PlannerAgent()._sanitize_input("please drop table users", ctx)
        assert "[BLOCKED]" in sanitized

    def test_call_llm_adjusts_token_budget(self, workspace):
        from src.agents.planner import PlannerAgent

        ctx, llm = self._build_context(workspace)
        ctx.max_tokens = 80
        prompt = "x " * 300

        PlannerAgent()._call_llm(prompt, ctx)

        assert llm.last_request is not None
        assert llm.last_request.max_tokens <= 80
        assert llm.last_request.max_tokens >= 1


class TestAgentSharedParsingAndConsistency:
    """Cross-agent parsing and consistency checks."""

    def test_shared_parser_extracts_json_without_regex_fragility(self):
        from src.agents.json_utils import parse_json_object

        raw = "analysis...```json\n{\"a\": 1,}\n``` trailing"
        parsed = parse_json_object(raw)
        assert parsed["a"] == 1

    def test_coder_filters_unknown_tool_calls(self, workspace):
        from src.agents import CoderInput, Subtask, AgentStatus, AgentType
        from src.agents.coder import CoderAgent
        from src.orchestration.executor import Executor
        from src.config import get_config

        cfg = get_config()
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=workspace / "logs")
        executor._initialize_run("run_coder_tools_filter")
        ctx = executor._create_agent_context(AgentType.CODER)

        subtask = Subtask(
            id="1",
            title="Add helper",
            description="Add helper function",
            acceptance_criteria=["helper exists"],
            target_files=["src/calculator.py"],
            dependencies=[],
        )
        input_data = CoderInput(task_id="1", run_id="run", subtask=subtask)

        output = CoderAgent()._parse_response(
            """```json
            {"changes": [], "tool_calls": [{"tool_name": "bad_tool", "arguments": {}}, {"tool_name": "read_file", "arguments": {"path": "src/calculator.py"}}]}
            ```""",
            input_data,
            ctx,
        )
        assert output.status == AgentStatus.SUCCESS
        assert len(output.tool_calls) == 1
        assert output.tool_calls[0].tool_name == "read_file"

    def test_reviewer_filters_external_issue_paths(self, workspace):
        from src.agents import ReviewerInput, Subtask, CodeChange, AgentStatus, AgentType
        from src.agents.reviewer import ReviewerAgent
        from src.orchestration.executor import Executor
        from src.config import get_config

        cfg = get_config()
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=workspace / "logs")
        executor._initialize_run("run_reviewer_path_filter")
        ctx = executor._create_agent_context(AgentType.REVIEWER)

        subtask = Subtask(
            id="1",
            title="Review",
            description="Review change",
            acceptance_criteria=["ok"],
            target_files=["src/calculator.py"],
            dependencies=[],
        )
        change = CodeChange(file_path="src/calculator.py", change_type="modify", description="x", new_content="x")
        input_data = ReviewerInput(task_id="1", run_id="run", subtask=subtask, code_changes=[change])

        output = ReviewerAgent()._parse_response(
            """```json
            {
              "verdict": "REQUEST_CHANGES",
              "issues": [
                {"severity": "major", "file_path": "/etc/passwd", "description": "bad", "suggestion": "x"},
                {"severity": "minor", "file_path": "src/calculator.py", "description": "style", "suggestion": "x"}
              ]
            }
            ```""",
            input_data,
            ctx,
        )
        assert output.status == AgentStatus.SUCCESS
        assert len(output.issues) == 1
        assert output.issues[0].file_path == "src/calculator.py"

    def test_fixer_rejects_unsafe_paths(self, workspace):
        from src.agents import FixerInput, CodeChange, ReviewIssue, AgentStatus, AgentType
        from src.agents.fixer import FixerAgent
        from src.orchestration.executor import Executor
        from src.config import get_config

        cfg = get_config()
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=workspace / "logs")
        executor._initialize_run("run_fixer_path_reject")
        ctx = executor._create_agent_context(AgentType.FIXER)

        input_data = FixerInput(
            task_id="fix",
            run_id="run",
            original_changes=[CodeChange(file_path="src/calculator.py", change_type="modify", description="x", new_content="x")],
            review_issues=[ReviewIssue(severity="major", file_path="src/calculator.py", description="issue")],
            file_contents={"src/calculator.py": "x"},
        )

        output = FixerAgent()._parse_response(
            """```json
            {"fixed_changes":[{"file_path":"../../etc/passwd","change_type":"modify","description":"x","new_content":"x"}]}
            ```""",
            input_data,
            ctx,
        )
        assert output.status == AgentStatus.FAILED


class TestMalformedOutputResilience:
    """Inject malformed LLM outputs and verify graceful degradation (no unrecoverable exits)."""

    def test_reviewer_parse_failure_fallback(self, workspace):
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import Subtask, CodeChange, AgentStatus, ReviewerOutput

        cfg = get_config()
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=workspace / "logs")
        executor._initialize_run("run_reviewer_malformed")
        executor._loop.start()  # type: ignore[union-attr]

        subtask = Subtask(
            id="t1",
            title="Test reviewer fallback",
            description="Test reviewer malformed output handling",
            acceptance_criteria=["criterion met"],
            target_files=["src/calculator.py"],
            dependencies=[],
        )
        changes = [CodeChange(file_path="src/calculator.py", change_type="modify", description="x", new_content="x")]

        def _bad_reviewer(*_args, **_kwargs):
            return ReviewerOutput(task_id="t1", status=AgentStatus.FAILED, error="Failed to parse reviewer response: malformed JSON")

        original = executor._reviewer.execute
        executor._reviewer.execute = _bad_reviewer  # type: ignore[assignment]
        try:
            out = executor._execute_reviewer(subtask, changes, "notes")
        finally:
            executor._reviewer.execute = original  # type: ignore[assignment]

        assert out is not None
        assert out.status == AgentStatus.SUCCESS
        assert out.task_complete is True

    def test_fixer_parse_failure_fallback(self, workspace):
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import CodeChange, ReviewIssue, AgentStatus, FixerOutput

        cfg = get_config()
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=workspace / "logs")
        executor._initialize_run("run_fixer_malformed")
        executor._loop.start()  # type: ignore[union-attr]

        changes = [CodeChange(file_path="src/calculator.py", change_type="modify", description="x", new_content="x")]
        issues = [ReviewIssue(severity="major", file_path="src/calculator.py", description="fix needed")]

        def _bad_fixer(*_args, **_kwargs):
            return FixerOutput(task_id="fix", status=AgentStatus.FAILED, error="Failed to parse fixer response: malformed JSON")

        original = executor._fixer.execute
        executor._fixer.execute = _bad_fixer  # type: ignore[assignment]
        try:
            out = executor._execute_fixer(changes, issues, {"src/calculator.py": "x"})
        finally:
            executor._fixer.execute = original  # type: ignore[assignment]

        assert out is not None
        assert out.status == AgentStatus.SUCCESS
        assert len(out.fixed_changes) == 1

    def test_pipeline_recovers_from_malformed_planner_coder_reviewer(self, workspace):
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import AgentStatus, PlannerOutput, CoderOutput, ReviewerOutput

        cfg = get_config()
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=workspace / "logs")

        def _bad_planner(*_args, **_kwargs):
            return PlannerOutput(task_id="planning", status=AgentStatus.FAILED, error="Failed to parse planner response: malformed JSON")

        def _bad_coder(*_args, **_kwargs):
            return CoderOutput(task_id="1", status=AgentStatus.FAILED, error="Failed to parse coder response: malformed JSON")

        def _bad_reviewer(*_args, **_kwargs):
            return ReviewerOutput(task_id="1", status=AgentStatus.FAILED, error="Failed to parse reviewer response: malformed JSON")

        p0, c0, r0 = executor._planner.execute, executor._coder.execute, executor._reviewer.execute
        executor._planner.execute = _bad_planner  # type: ignore[assignment]
        executor._coder.execute = _bad_coder  # type: ignore[assignment]
        executor._reviewer.execute = _bad_reviewer  # type: ignore[assignment]
        try:
            with patch("src.orchestration.executor.LLMClient.health_check", return_value=True):
                result = executor.execute("Create a Python function that calculates fibonacci numbers", run_id="run_malformed_triplet")
        finally:
            executor._planner.execute = p0  # type: ignore[assignment]
            executor._coder.execute = c0  # type: ignore[assignment]
            executor._reviewer.execute = r0  # type: ignore[assignment]

        assert result.success is True

    def test_pipeline_recover_with_malformed_fixer(self, workspace):
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import AgentStatus, PlannerOutput, CoderOutput, ReviewerOutput, FixerOutput, Subtask, CodeChange, ReviewIssue, ReviewVerdict

        cfg = get_config()
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=workspace / "logs")

        subtask = Subtask(
            id="1",
            title="Modify calculator",
            description="Modify calculator implementation",
            acceptance_criteria=["change applied"],
            target_files=["src/calculator.py"],
            dependencies=[],
        )

        def _ok_planner(*_args, **_kwargs):
            return PlannerOutput(task_id="planning", status=AgentStatus.SUCCESS, subtasks=[subtask])

        def _ok_coder(*_args, **_kwargs):
            return CoderOutput(
                task_id="1",
                status=AgentStatus.SUCCESS,
                changes=[CodeChange(file_path="src/calculator.py", change_type="modify", description="x", new_content="def add(a,b):\n    return a+b\n")],
                implementation_notes="implemented",
            )

        review_outputs = [
            ReviewerOutput(
                task_id="1",
                status=AgentStatus.SUCCESS,
                verdict=ReviewVerdict.REQUEST_CHANGES,
                task_complete=False,
                issues=[ReviewIssue(severity="major", file_path="src/calculator.py", description="needs fix")],
                criteria_met={"change applied": False},
            ),
            ReviewerOutput(
                task_id="1",
                status=AgentStatus.SUCCESS,
                verdict=ReviewVerdict.APPROVE,
                task_complete=True,
                issues=[],
                criteria_met={"change applied": True},
            ),
        ]

        def _seq_reviewer(*_args, **_kwargs):
            return review_outputs.pop(0)

        def _bad_fixer(*_args, **_kwargs):
            return FixerOutput(task_id="fix", status=AgentStatus.FAILED, error="Failed to parse fixer response: malformed JSON")

        p0, c0, r0, f0 = executor._planner.execute, executor._coder.execute, executor._reviewer.execute, executor._fixer.execute
        executor._planner.execute = _ok_planner  # type: ignore[assignment]
        executor._coder.execute = _ok_coder  # type: ignore[assignment]
        executor._reviewer.execute = _seq_reviewer  # type: ignore[assignment]
        executor._fixer.execute = _bad_fixer  # type: ignore[assignment]
        try:
            with patch("src.orchestration.executor.LLMClient.health_check", return_value=True):
                result = executor.execute("Modify calculator implementation", run_id="run_malformed_fixer")
        finally:
            executor._planner.execute = p0  # type: ignore[assignment]
            executor._coder.execute = c0  # type: ignore[assignment]
            executor._reviewer.execute = r0  # type: ignore[assignment]
            executor._fixer.execute = f0  # type: ignore[assignment]

        assert result.success is True

    def test_shared_parser_handles_control_char_malformed_payload(self):
        from src.agents.json_utils import parse_json_object

        raw = "```json\n{\"summary\":\"line1\nline2\x0bline3\",\"ok\":true}\n```"
        out = parse_json_object(raw)
        assert out["ok"] is True


class TestRollbackFlow:
    """Test rollback functionality in agent loop."""
    
    def test_rollback_after_bad_code(self, workspace):
        """Test that rollback restores files after bad code."""
        from src.orchestration.rollback import RollbackManager
        from src.core.file_guard import FileGuard, FileGuardPolicy
        
        calc_file = workspace / "src" / "calculator.py"
        original_content = calc_file.read_text()

        policy = FileGuardPolicy(
            allowed_roots=[workspace],
            blocked_patterns=[],
            allowed_extensions=[".py", ".txt", ".md", ".json"],
            max_file_size_bytes=5 * 1024 * 1024,
            max_files_per_run=100,
        )
        file_guard = FileGuard(workspace_root=workspace, policy=policy)
        
        rollback = RollbackManager(
            workspace_root=workspace,
            checkpoint_dir=workspace / ".checkpoints",
            file_guard=file_guard,
        )
        
        # Create checkpoint
        cp = rollback.create_checkpoint(
            run_id="run_test",
            description="Before changes",
            task_graph_state={},
            modified_files=[calc_file],
        )
        
        # Simulate bad code write
        calc_file.write_text("CORRUPTED CODE")
        assert calc_file.read_text() == "CORRUPTED CODE"
        
        # Rollback
        success = rollback.rollback_to(cp.id)
        assert success
        
        # Verify restoration
        assert calc_file.read_text() == original_content


class TestContractEnforcementInLoop:
    """Test that contracts are enforced during execution."""
    
    def test_malicious_output_blocked(self, workspace):
        """Malicious agent outputs should be blocked."""
        from src.core.contracts import ContractEnforcer
        
        enforcer = ContractEnforcer(workspace_root=workspace)
        
        # Simulate malicious coder output
        malicious_output = {
            "success": True,
            "code": "import os; os.system('rm -rf /')",
            "file_path": "/etc/passwd",  # Outside workspace!
            "diff": "...",
            "reasoning": "...",
        }
        
        result = enforcer.validate_output("coder", malicious_output)
        
        # Should be blocked
        assert not result.valid
        assert len(result.violations) > 0


class TestLargeProjectMode:
    """Test large project handling."""
    
    def test_project_analysis(self, workspace):
        """Test project size analysis."""
        from src.orchestration.large_project import LargeProjectHandler
        
        handler = LargeProjectHandler(workspace_root=workspace)
        metrics = handler.analyze_project()
        
        assert metrics.total_files > 0
        assert metrics.total_lines >= 0
    
    def test_task_sharding(self, workspace):
        """Test task sharding for large projects."""
        from src.orchestration.large_project import LargeProjectHandler, ShardConfig
        
        handler = LargeProjectHandler(
            workspace_root=workspace,
            config=ShardConfig(max_tasks_per_shard=2),
        )
        
        subtasks = [
            {"id": "t1", "target_files": ["a.py"], "dependencies": []},
            {"id": "t2", "target_files": ["b.py"], "dependencies": []},
            {"id": "t3", "target_files": ["c.py"], "dependencies": ["t1"]},
            {"id": "t4", "target_files": ["d.py"], "dependencies": ["t2"]},
            {"id": "t5", "target_files": ["e.py"], "dependencies": []},
        ]
        
        shards = handler.create_shards(subtasks)
        
        assert len(shards) > 1  # Should split into multiple shards
        
        # All tasks should be in some shard
        all_task_ids = set()
        for shard in shards:
            all_task_ids.update(shard.task_ids)
        
        assert all_task_ids == {"t1", "t2", "t3", "t4", "t5"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
