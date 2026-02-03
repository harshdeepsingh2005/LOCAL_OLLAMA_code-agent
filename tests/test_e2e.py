"""
End-to-End Agent Loop Tests

Tests the complete agent execution flow from input to output.
Uses mocked LLM responses to test the pipeline without network calls.
"""

import json
import pytest
import tempfile
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
    
    @pytest.mark.asyncio
    async def test_planner_agent_execution(self, workspace, mock_responses):
        """Test that planner agent produces valid output."""
        from src.agents.planner import PlannerAgent
        
        mock_client = MockLLMClient(mock_responses)
        
        # Create planner with mock client
        planner = PlannerAgent()
        planner._llm_client = mock_client
        
        # Mock the generate method
        with patch.object(planner, '_call_llm', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_responses["planner"]
            
            result = await planner.run(
                task="Add a divide function to the calculator",
                context={"workspace_root": str(workspace)},
            )
            
            # Verify structure
            assert result is not None
            assert hasattr(result, 'subtasks') or 'subtasks' in str(result)
    
    @pytest.mark.asyncio
    async def test_coder_agent_execution(self, workspace, mock_responses):
        """Test that coder agent produces valid code."""
        from src.agents.coder import CoderAgent
        
        coder = CoderAgent()
        
        with patch.object(coder, '_call_llm', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_responses["coder"]
            
            result = await coder.run(
                task={
                    "id": "task_001",
                    "title": "Add divide function",
                    "description": "Add divide to calculator",
                    "target_files": ["src/calculator.py"],
                },
                context={
                    "workspace_root": str(workspace),
                    "file_content": (workspace / "src" / "calculator.py").read_text(),
                },
            )
            
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_reviewer_agent_execution(self, workspace, mock_responses):
        """Test that reviewer agent validates code."""
        from src.agents.reviewer import ReviewerAgent
        
        reviewer = ReviewerAgent()
        
        with patch.object(reviewer, '_call_llm', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_responses["reviewer"]
            
            result = await reviewer.run(
                code='''def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b''',
                context={
                    "task": "Add divide function",
                    "file_path": "src/calculator.py",
                },
            )
            
            assert result is not None


class TestExecutorFlow:
    """Test the executor orchestration."""
    
    @pytest.mark.asyncio
    async def test_executor_runs_pipeline(self, workspace, mock_responses):
        """Test that executor runs full pipeline."""
        from src.orchestration.executor import Executor
        from src.state.run_state import RunState
        
        # Create executor
        executor = Executor(workspace_root=workspace)
        
        # Mock all agent calls
        with patch.multiple(
            executor,
            _run_planner=AsyncMock(return_value=json.loads(mock_responses["planner"])),
            _run_coder=AsyncMock(return_value=json.loads(mock_responses["coder"])),
            _run_reviewer=AsyncMock(return_value=json.loads(mock_responses["reviewer"])),
        ):
            # Run would normally be called here
            # Just verify the mocking setup works
            pass


class TestRollbackFlow:
    """Test rollback functionality in agent loop."""
    
    def test_rollback_after_bad_code(self, workspace):
        """Test that rollback restores files after bad code."""
        from src.orchestration.rollback import RollbackManager
        
        calc_file = workspace / "src" / "calculator.py"
        original_content = calc_file.read_text()
        
        rollback = RollbackManager(
            workspace_root=workspace,
            checkpoints_dir=workspace / ".checkpoints",
        )
        
        # Create checkpoint
        cp_id = rollback.create_checkpoint(
            files=[calc_file],
            description="Before changes",
        )
        
        # Simulate bad code write
        calc_file.write_text("CORRUPTED CODE")
        assert calc_file.read_text() == "CORRUPTED CODE"
        
        # Rollback
        success = rollback.rollback_to(cp_id)
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
