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
            risk_score=0.2,
            criteria_met={"criterion": True},
        )

        allowed, gate = executor._verification_gate(reviewer_output)

        assert allowed is True
        assert gate["no_errors"] is True

    def test_verification_gate_blocks_high_risk_review(self, workspace):
        """Verification gate should block completion when residual risk is too high."""
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import ReviewerOutput, AgentStatus

        cfg = get_config()
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)

        reviewer_output = ReviewerOutput(
            task_id="t2",
            status=AgentStatus.SUCCESS,
            task_complete=True,
            issues=[],
            risk_score=0.9,
            criteria_met={"criterion": True},
        )

        allowed, gate = executor._verification_gate(reviewer_output)

        assert allowed is False
        assert gate["risk_score_ok"] is False

    def test_reviewer_parser_populates_intelligence_scores(self, workspace):
        """Reviewer parser should map explicit scoring fields and potential breakages."""
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import AgentType, ReviewerInput, Subtask, AgentStatus

        cfg = get_config()
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)
        executor._initialize_run("run_reviewer_scoring")
        context = executor._create_agent_context(AgentType.REVIEWER)

        subtask = Subtask(
            id="r1",
            title="Review scoring",
            description="Validate reviewer scores",
            acceptance_criteria=["scores present"],
            target_files=["src/calculator.py"],
            dependencies=[],
        )
        reviewer_input = ReviewerInput(
            task_id="r1",
            run_id="run_reviewer_scoring",
            subtask=subtask,
            code_changes=[],
        )

        raw = """{
            "verdict": "REQUEST_CHANGES",
            "task_complete": false,
            "summary": "Needs work",
            "issues": [
              {
                "severity": "major",
                "file_path": "src/calculator.py",
                "description": "Edge case missing",
                "suggestion": "Handle zero",
                "issue_code": "REVIEW_001",
                "acceptance_criterion_ref": "scores present",
                "evidence": "No zero guard",
                "blocking": true
              }
            ],
            "correctness_score": 0.42,
            "maintainability_score": 0.61,
            "risk_score": 0.77,
            "confidence_score": 0.83,
            "potential_breakages": ["Division by zero in production"],
            "criteria_met": {"scores present": false}
        }"""

        output = executor._reviewer._parse_response(raw, reviewer_input, context)

        assert output.status == AgentStatus.SUCCESS
        assert output.correctness_score == pytest.approx(0.42)
        assert output.maintainability_score == pytest.approx(0.61)
        assert output.risk_score == pytest.approx(0.77)
        assert output.confidence_score == pytest.approx(0.83)
        assert output.potential_breakages

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


class TestPhase5PluginArchitecture:
    """Phase 5.1 plugin registry and dispatch checks."""

    def test_tool_registration_exposes_allowlist(self, workspace):
        from src.core.memory import MemoryManager
        from src.core.agent_tools import ToolExecutor

        memory = MemoryManager(workspace)
        tool_executor = ToolExecutor(memory_manager=memory, workspace_root=str(workspace), run_id="run_plugin_allowlist")

        allowlist = tool_executor.tool_allowlist
        assert "read_memory" in allowlist
        assert "run_command" in allowlist
        assert "mcp_call" in allowlist

    def test_unregistered_tool_is_rejected(self, workspace):
        from src.core.memory import MemoryManager
        from src.core.agent_tools import ToolExecutor
        from src.agents.base import ToolCall

        memory = MemoryManager(workspace)
        tool_executor = ToolExecutor(memory_manager=memory, workspace_root=str(workspace), run_id="run_plugin_reject")
        result = tool_executor.execute_call(ToolCall(tool_name="not_registered_tool", arguments={}))

        assert "Unknown tool" in result


class TestPhase5PolicyProfiles:
    """Phase 5.2 policy profile enforcement checks."""

    def test_strict_profile_blocks_unsafe_tools(self, workspace):
        from src.core.memory import MemoryManager
        from src.core.agent_tools import ToolExecutor
        from src.core.policy import get_policy_profile
        from src.agents.base import ToolCall

        memory = MemoryManager(workspace)
        tool_executor = ToolExecutor(
            memory_manager=memory,
            workspace_root=str(workspace),
            policy_profile=get_policy_profile("strict"),
            run_id="run_policy_strict",
        )

        blocked = tool_executor.execute_call(ToolCall(tool_name="run_command", arguments={"command": "pwd"}))
        assert "Policy blocked" in blocked

    def test_balanced_profile_allows_controlled_execution(self, workspace):
        from src.core.memory import MemoryManager
        from src.core.agent_tools import ToolExecutor
        from src.core.policy import get_policy_profile
        from src.agents.base import ToolCall

        memory = MemoryManager(workspace)
        tool_executor = ToolExecutor(
            memory_manager=memory,
            workspace_root=str(workspace),
            policy_profile=get_policy_profile("balanced"),
            run_id="run_policy_balanced",
        )

        output = tool_executor.execute_call(ToolCall(tool_name="read_memory", arguments={}))
        assert isinstance(output, str)

    def test_strict_profile_sets_deterministic_llm_parameters(self, workspace):
        from src.config import get_config
        from src.orchestration.executor import Executor

        cfg = get_config()
        executor = Executor(
            config=cfg,
            workspace_root=workspace,
            log_dir=workspace / "logs",
            policy_profile="strict",
        )
        executor._initialize_run("run_policy_determinism")
        assert executor._llm_client is not None
        assert executor._llm_client._default_config.temperature == 0.0
        assert executor._llm_client._default_config.top_p == 1.0


class TestPhase5FormalGuarantees:
    """Phase 5.3 invariants and failure normalization checks."""

    def test_invalid_task_state_transition_is_caught(self, workspace):
        from src.agents import Subtask
        from src.orchestration.task_graph import InvalidTaskTransitionError, TaskNode

        node = TaskNode(
            id="inv-1",
            subtask=Subtask(
                id="inv-1",
                title="Invariant",
                description="Ensure transitions are guarded",
                acceptance_criteria=["must guard invalid transition"],
            ),
        )

        with pytest.raises(InvalidTaskTransitionError):
            node.mark_completed({"ok": False})

    def test_failure_classification_is_normalized(self, workspace):
        from src.config import get_config
        from src.orchestration.executor import Executor

        executor = Executor(config=get_config(), workspace_root=workspace, log_dir=workspace / "logs")
        assert executor._classify_failure("Planning failed due to malformed JSON") == "planning_error"
        assert executor._classify_failure("Tool command blocked by policy") == "tool_error"
        assert executor._classify_failure("contract violation on output schema") == "contract_violation"


class TestPhase5MultiWorkspace:
    """Phase 5.4 multi-workspace orchestration checks."""

    def test_workspace_contexts_are_isolated(self, workspace):
        from src.orchestration.workspace_manager import WorkspaceManager

        ws_a = workspace / "workspace_a"
        ws_b = workspace / "workspace_b"
        ws_a.mkdir(exist_ok=True)
        ws_b.mkdir(exist_ok=True)

        manager = WorkspaceManager([ws_b, ws_a])
        manager.get_workspace("workspace_a").memory.add_fact("workspace_a_fact")

        ctx_a = manager.build_workspace_context("workspace_a", "add API endpoint")
        ctx_b = manager.build_workspace_context("workspace_b", "add API endpoint")

        assert "workspace_a_fact" in ctx_a
        assert "workspace_a_fact" not in ctx_b

    def test_multi_workspace_execution_order_is_deterministic(self, workspace):
        from src.orchestration.workspace_manager import WorkspaceManager

        ws_a = workspace / "a"
        ws_b = workspace / "b"
        ws_a.mkdir(exist_ok=True)
        ws_b.mkdir(exist_ok=True)

        manager = WorkspaceManager([ws_b, ws_a])
        assignments = {
            "b": ["task-b1", "task-b2"],
            "a": ["task-a1"],
        }

        order: list[tuple[str, str]] = []

        def _runner(workspace_name: str, item: str) -> str:
            order.append((workspace_name, item))
            return item

        manager.execute_sequential(assignments, _runner)
        assert order == [("a", "task-a1"), ("b", "task-b1"), ("b", "task-b2")]

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

    def test_memory_learning_normalizes_error_category(self, workspace):
        from src.core.memory import MemoryManager

        memory = MemoryManager(workspace)
        memory.record_failure_pattern(
            task_description="Add API endpoint",
            error_message="ImportError: cannot import name APIRouter",
        )

        data = memory._load_memory(memory._project_memory_file)
        assert data["failure_patterns"]
        assert data["failure_patterns"][0]["category"] == "missing_import"

    def test_memory_learning_deduplicates_similar_failures(self, workspace):
        from src.core.memory import MemoryManager

        memory = MemoryManager(workspace)
        memory.record_failure_pattern(
            task_description="Fix imports in service",
            error_message="ImportError: module missing in service",
        )
        memory.record_failure_pattern(
            task_description="Fix imports in service module",
            error_message="ImportError: missing module in service",
        )

        data = memory._load_memory(memory._project_memory_file)
        patterns = data["failure_patterns"]
        assert len(patterns) == 1
        assert patterns[0]["frequency"] >= 2

    def test_memory_learning_deduplicates_failure_alias_categories(self, workspace):
        from src.core.memory import MemoryManager

        memory = MemoryManager(workspace)
        data = memory._load_memory(memory._project_memory_file)
        data["failure_patterns"] = [
            {
                "pattern_id": "fp_a",
                "category": "import_error",
                "summary": "import error in api router",
                "root_cause": "ImportError",
                "resolution_hint": "fix import path",
                "tags": ["api", "import"],
                "frequency": 1,
                "confidence": 0.5,
                "created_at": memory._now_iso(),
                "last_used_at": memory._now_iso(),
            },
            {
                "pattern_id": "fp_b",
                "category": "missing_import",
                "summary": "missing import in router api",
                "root_cause": "ModuleNotFoundError",
                "resolution_hint": "fix import path",
                "tags": ["router", "import"],
                "frequency": 1,
                "confidence": 0.55,
                "created_at": memory._now_iso(),
                "last_used_at": memory._now_iso(),
            },
        ]
        memory._save_memory(memory._project_memory_file, data)

        memory.record_failure_pattern(
            task_description="Fix router imports",
            error_message="ModuleNotFoundError: missing import path",
        )

        refreshed = memory._load_memory(memory._project_memory_file)
        patterns = refreshed["failure_patterns"]
        assert len(patterns) == 1
        assert patterns[0]["category"] == "missing_import"
        assert patterns[0]["frequency"] >= 3

    def test_memory_learning_ranked_split_retrieval(self, workspace):
        from src.core.memory import MemoryManager

        memory = MemoryManager(workspace)
        memory.record_failure_pattern(
            task_description="Fix import in api router",
            error_message="ImportError: APIRouter missing",
        )
        memory.record_success_patterns_from_changes(
            changes=[
                {
                    "file_path": "src/routes/api.py",
                    "change_type": "modify",
                    "description": "add login route",
                    "new_content": "from fastapi import APIRouter\nrouter = APIRouter()\n@router.post('/login')\ndef login():\n    return {}\n",
                }
            ],
            task_description="Add fastapi login endpoint",
        )

        retrieved = memory.retrieve_relevant_patterns("Add fastapi router endpoint with imports")
        assert "failures" in retrieved and "successes" in retrieved
        assert len(retrieved["failures"]) >= 1
        assert len(retrieved["successes"]) >= 1

    def test_memory_learning_deduplicates_success_pattern_aliases(self, workspace):
        from src.core.memory import MemoryManager

        memory = MemoryManager(workspace)
        data = memory._load_memory(memory._project_memory_file)
        data["success_patterns"] = [
            {
                "pattern_id": "sp_a",
                "pattern_type": "endpoint_pattern",
                "summary": "endpoint pattern in routes",
                "reusable_snippet": "@router.get('/x')",
                "tags": ["api", "router"],
                "success_rate": 1.0,
                "frequency": 1,
                "confidence": 0.6,
                "created_at": memory._now_iso(),
                "last_used_at": memory._now_iso(),
            },
            {
                "pattern_id": "sp_b",
                "pattern_type": "api_endpoint_pattern",
                "summary": "api endpoint pattern routes",
                "reusable_snippet": "@router.post('/x')",
                "tags": ["router", "fastapi"],
                "success_rate": 1.0,
                "frequency": 1,
                "confidence": 0.6,
                "created_at": memory._now_iso(),
                "last_used_at": memory._now_iso(),
            },
        ]
        memory._save_memory(memory._project_memory_file, data)

        memory.record_success_patterns_from_changes(
            changes=[
                {
                    "file_path": "src/routes/users.py",
                    "change_type": "modify",
                    "description": "add users endpoint",
                    "new_content": "from fastapi import APIRouter\nrouter = APIRouter()\n@router.get('/users')\ndef users():\n    return []\n",
                }
            ],
            task_description="Add users endpoint",
        )

        refreshed = memory._load_memory(memory._project_memory_file)
        patterns = refreshed["success_patterns"]
        assert len(patterns) == 1
        assert patterns[0]["pattern_type"] == "api_endpoint_pattern"
        assert patterns[0]["frequency"] >= 3

    def test_memory_learning_structured_injection_budget(self, workspace):
        from src.core.memory import MemoryManager
        from src.orchestration.context_pipeline import ContextBuilder, TaskRoute, TaskDomain

        memory = MemoryManager(workspace)
        for i in range(20):
            memory.record_failure_pattern(
                task_description=f"Fix import issue {i}",
                error_message="ImportError: missing module path",
            )

        block = memory.format_learned_patterns("Fix import issues in api", max_chars=300)
        assert len(block) <= 300
        assert "Learned Patterns" in block

        builder = ContextBuilder(workspace_root=workspace, memory_manager=memory)
        packet = builder.build("Fix import issues in api", TaskRoute(domain=TaskDomain.BACKEND))
        prompt = packet.to_prompt_context(max_chars=1200)
        assert "Learned Patterns" in prompt

    def test_memory_semantic_graph_links_task_error_fix_file(self, workspace):
        from src.core.memory import MemoryManager

        memory = MemoryManager(workspace)
        memory.record_failure_pattern(
            task_description="Fix import error in api router",
            error_message="ImportError: cannot import APIRouter",
        )
        memory.record_success_patterns_from_changes(
            changes=[
                {
                    "file_path": "src/routes/api.py",
                    "change_type": "modify",
                    "description": "fix import and add endpoint",
                    "new_content": "from fastapi import APIRouter\nrouter = APIRouter()\n",
                }
            ],
            task_description="Fix import error in api router",
        )
        memory.record_task_outcome(
            task_description="Fix import error in api router",
            success=True,
            error=None,
        )

        data = memory._load_memory(memory._project_memory_file)
        graph = data.get("semantic_graph", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        node_types = {str(n.get("type", "")) for n in nodes}
        relations = {str(e.get("relation", "")) for e in edges}

        assert "task" in node_types
        assert "failure" in node_types
        assert "file" in node_types
        assert "outcome" in node_types
        assert "encountered" in relations
        assert "modified" in relations
        assert "resulted_in" in relations

    def test_context_builder_includes_semantic_links(self, workspace):
        from src.core.memory import MemoryManager
        from src.orchestration.context_pipeline import ContextBuilder, TaskRoute, TaskDomain

        memory = MemoryManager(workspace)
        memory.record_failure_pattern(
            task_description="Fix api router import",
            error_message="ImportError: APIRouter missing",
        )

        builder = ContextBuilder(workspace_root=workspace, memory_manager=memory)
        packet = builder.build("Fix api router import", TaskRoute(domain=TaskDomain.BACKEND))
        prompt = packet.to_prompt_context(max_chars=1400)

        assert "## Semantic Links" in prompt

    def test_memory_records_and_formats_test_signals(self, workspace):
        from src.core.memory import MemoryManager

        memory = MemoryManager(workspace)
        memory.record_test_signal(
            task_description="Fix divide behavior",
            test_name="tests/test_calculator.py::test_divide_by_zero",
            status="failed",
            message="AssertionError: ValueError not raised",
            file_path="tests/test_calculator.py",
        )
        memory.record_test_signal(
            task_description="Fix divide behavior",
            test_name="tests/test_calculator.py::test_divide_by_zero",
            status="failed",
            message="AssertionError: ValueError not raised",
            file_path="tests/test_calculator.py",
        )

        data = memory._load_memory(memory._project_memory_file)
        signals = data.get("test_signals", [])
        assert len(signals) == 1
        assert signals[0]["frequency"] >= 2

        block = memory.format_recent_test_signals("Fix divide behavior", max_chars=300)
        assert "FAILED" in block
        assert "test_divide_by_zero" in block

    def test_context_builder_includes_test_signals(self, workspace):
        from src.core.memory import MemoryManager
        from src.orchestration.context_pipeline import ContextBuilder, TaskRoute, TaskDomain

        memory = MemoryManager(workspace)
        memory.record_test_signal(
            task_description="Fix router auth regression",
            test_name="tests/test_auth.py::test_login_requires_token",
            status="failed",
            message="Expected 401 got 200",
            file_path="tests/test_auth.py",
        )

        builder = ContextBuilder(workspace_root=workspace, memory_manager=memory)
        packet = builder.build("Fix router auth regression", TaskRoute(domain=TaskDomain.BACKEND))
        prompt = packet.to_prompt_context(max_chars=1500)

        assert "## Test Signals" in prompt
        assert "test_login_requires_token" in prompt

    def test_executor_records_test_learning_from_reviewer_output(self, workspace):
        from src.config import get_config
        from src.orchestration.executor import Executor
        from src.agents import ReviewerOutput, ReviewIssue, AgentStatus, ReviewVerdict

        cfg = get_config()
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=workspace / "logs")
        executor._initialize_run("run_test_learning")
        assert executor._memory_manager is not None

        reviewer_output = ReviewerOutput(
            task_id="t1",
            status=AgentStatus.SUCCESS,
            verdict=ReviewVerdict.REQUEST_CHANGES,
            task_complete=False,
            criteria_met={"tests pass": False, "api contract": True},
            issues=[
                ReviewIssue(
                    severity="major",
                    file_path="tests/test_api.py",
                    description="failing test due to changed response shape",
                    suggestion="update serializer",
                    issue_code="TEST_FAIL_SHAPE",
                    evidence="pytest failure in test_api.py",
                    blocking=True,
                )
            ],
        )

        executor._record_test_learning("Fix API response shape", reviewer_output)

        data = executor._memory_manager._load_memory(executor._memory_manager._project_memory_file)
        signals = data.get("test_signals", [])
        assert len(signals) >= 2
        joined = "\n".join(str(s.get("test_name", "")) for s in signals)
        assert "criterion::tests pass" in joined
        assert "TEST_FAIL_SHAPE" in joined

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

    def test_execute_task_detects_stagnation_and_shifts_strategy(self, workspace):
        """Repeated no-progress fix cycles should trigger coder strategy shift and allow completion."""
        from src.orchestration.executor import Executor
        from src.orchestration.task_graph import TaskNode, TaskStatus
        from src.config import get_config
        from src.agents import (
            AgentStatus,
            CoderOutput,
            CodeChange,
            FixerOutput,
            ReviewIssue,
            ReviewerOutput,
            ReviewVerdict,
            Subtask,
        )

        cfg = get_config()
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=log_dir)
        executor._initialize_run("run_stagnation_strategy_shift")
        assert executor._loop is not None
        executor._loop.start()

        subtask = Subtask(
            id="task_stagnation",
            title="Improve calculator",
            description="Fix calculator with safe divide implementation",
            acceptance_criteria=["divide function exists", "tests pass"],
            target_files=["src/calculator.py"],
            dependencies=[],
        )
        node = TaskNode(id="task_stagnation", subtask=subtask)

        change_a = CodeChange(
            file_path="src/calculator.py",
            change_type="modify",
            description="first draft",
            new_content="def divide(a, b):\n    return a / b\n",
        )
        change_b = CodeChange(
            file_path="src/calculator.py",
            change_type="modify",
            description="strategy shifted draft",
            new_content="def divide(a, b):\n    if b == 0:\n        raise ValueError('zero')\n    return a / b\n",
        )

        coder_calls = {"count": 0}

        def _mock_execute_coder(_subtask, _file_contents):
            coder_calls["count"] += 1
            if coder_calls["count"] == 1:
                return CoderOutput(task_id=subtask.id, status=AgentStatus.SUCCESS, changes=[change_a])
            return CoderOutput(task_id=subtask.id, status=AgentStatus.SUCCESS, changes=[change_b])

        reviewer_calls = {"count": 0}
        issue = ReviewIssue(
            severity="major",
            file_path="src/calculator.py",
            description="Division by zero not handled",
            suggestion="Add guard for b == 0",
        )

        def _mock_execute_reviewer(_subtask, current_changes, _notes):
            reviewer_calls["count"] += 1
            if reviewer_calls["count"] <= 2:
                return ReviewerOutput(
                    task_id=subtask.id,
                    status=AgentStatus.SUCCESS,
                    verdict=ReviewVerdict.REQUEST_CHANGES,
                    task_complete=False,
                    issues=[issue],
                    summary="Needs zero-division guard",
                )
            assert current_changes[0].description == "strategy shifted draft"
            return ReviewerOutput(
                task_id=subtask.id,
                status=AgentStatus.SUCCESS,
                verdict=ReviewVerdict.APPROVE,
                task_complete=True,
                issues=[
                    ReviewIssue(
                        severity="minor",
                        file_path="src/calculator.py",
                        description="Looks good",
                        suggestion="Optional type hints",
                    )
                ],
                summary="Approved after strategy shift",
                criteria_met={"divide function exists": True, "tests pass": True},
            )

        def _mock_execute_fixer(current_changes, _issues, _file_contents):
            return FixerOutput(
                task_id=subtask.id,
                status=AgentStatus.SUCCESS,
                fixed_changes=current_changes,
            )

        original_coder = executor._execute_coder
        original_reviewer = executor._execute_reviewer
        original_fixer = executor._execute_fixer
        original_apply = executor._apply_changes

        executor._execute_coder = _mock_execute_coder  # type: ignore[assignment]
        executor._execute_reviewer = _mock_execute_reviewer  # type: ignore[assignment]
        executor._execute_fixer = _mock_execute_fixer  # type: ignore[assignment]
        executor._apply_changes = lambda _changes: True  # type: ignore[assignment]
        try:
            ok = executor._execute_task(node)
        finally:
            executor._execute_coder = original_coder  # type: ignore[assignment]
            executor._execute_reviewer = original_reviewer  # type: ignore[assignment]
            executor._execute_fixer = original_fixer  # type: ignore[assignment]
            executor._apply_changes = original_apply  # type: ignore[assignment]

        assert ok is True
        assert node.status == TaskStatus.COMPLETED
        assert coder_calls["count"] >= 2


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


class TestDeterministicPlanningExplainabilityAndContext:
    """Coverage for deterministic tool plans, telemetry, reviewer explainability, and context quality."""

    def test_tool_plan_respected_and_executed_in_order(self, workspace):
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import (
            AgentStatus,
            CoderOutput,
            Subtask,
            SubtaskToolPlan,
            ToolPlanStep,
        )

        cfg = get_config()
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=workspace / "logs")
        executor._initialize_run("run_tool_plan_respected")
        assert executor._loop is not None
        executor._loop.start()

        subtask = Subtask(
            id="1",
            title="Read memory first",
            description="Use planned tools before coding",
            acceptance_criteria=["planned tool executed"],
            target_files=["src/calculator.py"],
            tool_plan=SubtaskToolPlan(
                steps=[
                    ToolPlanStep(
                        tool="read_memory",
                        reason="Load persistent context",
                        arguments={},
                    )
                ]
            ),
        )

        assert executor._tool_executor is not None
        executed: list[str] = []

        def _tool_spy(call):
            executed.append(call.tool_name)
            return "memory loaded"

        executor._tool_executor.execute_call = _tool_spy  # type: ignore[assignment]

        def _coder_ok(*_args, **_kwargs):
            return CoderOutput(task_id="1", status=AgentStatus.SUCCESS, changes=[])

        executor._coder.execute = _coder_ok  # type: ignore[assignment]

        output = executor._execute_coder(subtask, {"src/calculator.py": "def add(a, b):\n    return a + b\n"})
        assert output is not None
        assert output.status == AgentStatus.SUCCESS
        assert executed[:1] == ["read_memory"]

    def test_invalid_tool_plan_rejected_gracefully(self, workspace):
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import Subtask, SubtaskToolPlan, ToolPlanStep, AgentStatus

        cfg = get_config()
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=workspace / "logs")
        executor._initialize_run("run_invalid_tool_plan")
        assert executor._loop is not None
        executor._loop.start()

        subtask = Subtask(
            id="2",
            title="Invalid plan",
            description="Should fail safely",
            acceptance_criteria=["error handled"],
            target_files=["src/calculator.py"],
            tool_plan=SubtaskToolPlan(
                steps=[
                    ToolPlanStep(
                        tool="totally_unknown_tool",
                        reason="invalid",
                        arguments={},
                    )
                ]
            ),
        )

        output = executor._execute_coder(subtask, {"src/calculator.py": "def add(a,b):\n    return a+b\n"})
        assert output is not None
        assert output.status == AgentStatus.FAILED
        assert "unknown tool" in (output.error or "")

        assert executor._telemetry is not None
        violations = [
            e
            for e in executor._telemetry.events
            if e.event_type.value == "warning"
            and e.data.get("warning") == "tool_plan_violation"
        ]
        assert violations

    def test_tool_plan_fallback_execution_works(self, workspace):
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import Subtask, SubtaskToolPlan, ToolPlanStep

        cfg = get_config()
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=workspace / "logs")
        executor._initialize_run("run_tool_plan_fallback")

        subtask = Subtask(
            id="3",
            title="Fallback step",
            description="Primary tool fails then fallback succeeds",
            acceptance_criteria=["fallback executed"],
            target_files=["src/calculator.py"],
            tool_plan=SubtaskToolPlan(
                steps=[
                    ToolPlanStep(
                        tool="read_memory",
                        reason="attempt primary",
                        arguments={},
                        fallback=ToolPlanStep(
                            tool="grep_search",
                            reason="fallback search",
                            arguments={"pattern": "def add"},
                        ),
                    )
                ]
            ),
        )

        assert executor._tool_executor is not None
        calls = {"n": 0}

        def _tool_spy(call):
            calls["n"] += 1
            if calls["n"] == 1:
                return "Error: primary failed"
            return "fallback ok"

        executor._tool_executor.execute_call = _tool_spy  # type: ignore[assignment]

        ok, _ctx, executed, fallback_count, err = executor._execute_subtask_tool_plan(subtask)
        assert ok is True
        assert err is None
        assert executed == ["read_memory", "grep_search"]
        assert fallback_count == 1

    def test_telemetry_records_plan_metrics_and_violations(self, workspace):
        from src.orchestration.executor import Executor
        from src.config import get_config
        from src.agents import Subtask, SubtaskToolPlan, ToolPlanStep

        cfg = get_config()
        executor = Executor(config=cfg, workspace_root=workspace, log_dir=workspace / "logs")
        executor._initialize_run("run_tool_plan_metrics")
        assert executor._telemetry is not None

        subtask = Subtask(
            id="4",
            title="Metrics",
            description="Check planned vs executed metrics",
            acceptance_criteria=["metrics available"],
            target_files=["src/calculator.py"],
            tool_plan=SubtaskToolPlan(steps=[ToolPlanStep(tool="read_memory", reason="seed", arguments={})]),
        )

        executor._record_tool_plan_adherence(
            subtask=subtask,
            planned_tools=["read_memory"],
            executed_tools=["read_memory", "grep_search"],
            fallback_count=1,
        )

        metrics = executor._telemetry.run_metrics
        assert metrics.planned_tools >= 1
        assert metrics.executed_tools >= 2
        assert metrics.fallback_count >= 1
        assert metrics.plan_adherence_samples >= 1

        violation_events = [
            e
            for e in executor._telemetry.events
            if e.event_type.value == "warning"
            and e.data.get("warning") == "tool_plan_violation"
        ]
        assert violation_events

    def test_reviewer_schema_includes_issue_code_and_criterion_ref(self, workspace):
        from src.agents.reviewer import ReviewerAgent
        from src.agents import ReviewerInput, Subtask, CodeChange

        reviewer = ReviewerAgent()
        reviewer_input = ReviewerInput(
            task_id="r1",
            run_id="run_r1",
            subtask=Subtask(
                id="r1",
                title="Review",
                description="Review for explainability",
                acceptance_criteria=["Criterion 1"],
                target_files=["src/calculator.py"],
            ),
            code_changes=[
                CodeChange(
                    file_path="src/calculator.py",
                    change_type="modify",
                    description="change",
                    new_content="def add(a, b):\n    return a + b\n",
                )
            ],
            original_files={},
            implementation_notes="",
        )

        payload = json.dumps(
            {
                "verdict": "REQUEST_CHANGES",
                "task_complete": False,
                "summary": "needs fixes",
                "issues": [
                    {
                        "severity": "major",
                        "file_path": "src/calculator.py",
                        "line_range": "1-1",
                        "description": "No input validation",
                        "suggestion": "Validate numeric inputs",
                        "issue_code": "VAL_001",
                        "acceptance_criterion_ref": "Criterion 1",
                        "evidence": "Function accepts arbitrary non-numeric types",
                        "blocking": True,
                    }
                ],
                "criteria_met": {"Criterion 1": False},
            }
        )

        out = reviewer._parse_response(payload, reviewer_input, MagicMock())
        assert out.issues
        issue = out.issues[0]
        assert issue.issue_code == "VAL_001"
        assert issue.acceptance_criterion_ref == "Criterion 1"
        assert issue.blocking is True

    def test_reviewer_blocking_issue_forces_non_terminal_state(self, workspace):
        from src.agents.reviewer import ReviewerAgent
        from src.agents import ReviewerInput, Subtask, CodeChange, ReviewVerdict

        reviewer = ReviewerAgent()
        reviewer_input = ReviewerInput(
            task_id="r2",
            run_id="run_r2",
            subtask=Subtask(
                id="r2",
                title="Review",
                description="Blocking issue must prevent completion",
                acceptance_criteria=["Criterion 1"],
                target_files=["src/calculator.py"],
            ),
            code_changes=[
                CodeChange(
                    file_path="src/calculator.py",
                    change_type="modify",
                    description="change",
                    new_content="def add(a, b):\n    return a + b\n",
                )
            ],
            original_files={},
            implementation_notes="",
        )

        payload = json.dumps(
            {
                "verdict": "APPROVE",
                "task_complete": True,
                "summary": "looks good",
                "issues": [
                    {
                        "severity": "minor",
                        "file_path": "src/calculator.py",
                        "description": "Actually blocking",
                        "suggestion": "Fix",
                        "issue_code": "BLOCK_001",
                        "acceptance_criterion_ref": "Criterion 1",
                        "evidence": "Missing acceptance criterion behavior",
                        "blocking": True,
                    }
                ],
                "criteria_met": {"Criterion 1": False},
            }
        )

        out = reviewer._parse_response(payload, reviewer_input, MagicMock())
        assert out.task_complete is False
        assert out.verdict == ReviewVerdict.REQUEST_CHANGES

    def test_context_packet_caps_and_order_are_deterministic(self):
        from src.orchestration.context_pipeline import ContextPacket

        packet = ContextPacket(
            route_summary="domain=backend",
            interfaces_context="I\n" * 120,
            dependencies_context="D\n" * 120,
            recent_failures_context="F\n" * 120,
            recent_successes_context="S\n" * 120,
            documentation_context="DOC\n" * 300,
        )

        rendered_a = packet.to_prompt_context(max_chars=900)
        rendered_b = packet.to_prompt_context(max_chars=900)

        assert rendered_a == rendered_b
        assert "## Interfaces" in rendered_a
        assert "## Dependencies" in rendered_a
        assert rendered_a.index("## Interfaces") < rendered_a.index("## Dependencies")
        assert len(rendered_a) <= 900


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
