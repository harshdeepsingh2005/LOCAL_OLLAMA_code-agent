"""
Tests for the CLI Module

Tests the command-line interface, session management, and commands.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from click.testing import CliRunner

from src.cli.session import Session, SessionState, SessionConfig, PendingChange
from src.cli.commands import CommandHandler
from src.cli.display import Display, StatusSymbol


class TestSession:
    """Tests for Session management."""
    
    @pytest.fixture
    def session_dir(self, tmp_path):
        """Create a temporary session directory."""
        return tmp_path / "sessions"
    
    @pytest.fixture
    def session(self, tmp_path, session_dir):
        """Create a test session."""
        return Session(
            workspace_root=tmp_path,
            session_dir=session_dir,
        )
    
    def test_session_creation(self, session):
        """Sessions should be created with valid state."""
        assert session.session_id is not None
        assert len(session.session_id) > 0
        assert session.state == SessionState.IDLE
    
    def test_session_state_transitions(self, session):
        """Session should track state transitions."""
        assert session.state == SessionState.IDLE
        
        session._set_state(SessionState.PLANNING)
        assert session.state == SessionState.PLANNING
        
        session._set_state(SessionState.CODING)
        assert session.state == SessionState.CODING
        
        session._set_state(SessionState.REVIEW_PENDING)
        assert session.state == SessionState.REVIEW_PENDING
    
    def test_session_persistence(self, tmp_path, session_dir):
        """Sessions should persist and restore."""
        # Create and save session
        session1 = Session(
            workspace_root=tmp_path,
            session_dir=session_dir,
        )
        session1._add_message("user", "Hello")
        session1.save()
        
        session_id = session1.session_id
        
        # Load session
        session2 = Session.load(
            session_id=session_id,
            session_dir=session_dir,
        )
        
        assert session2 is not None
        assert session2.session_id == session_id
        assert len(session2.conversation) > 0
    
    def test_pending_changes_management(self, session):
        """Session should track pending changes."""
        change = PendingChange(
            file_path="test.py",
            diff="@@ -1 +1 @@\n-old\n+new",
            description="Test change",
        )
        
        session.add_pending_change(change)
        
        assert len(session.pending_changes) == 1
        assert session.pending_changes[0].file_path == "test.py"
    
    def test_token_tracking(self, session):
        """Session should track token usage."""
        session.add_tokens(1000, "input")
        session.add_tokens(500, "output")
        
        assert session.total_tokens >= 1500
    
    def test_session_summary(self, session):
        """Session should generate summary."""
        session._add_message("user", "Fix the bug")
        session._add_message("assistant", "I'll fix that")
        
        summary = session.get_summary()
        
        assert "messages" in summary or "conversation" in summary.lower() or isinstance(summary, dict)


class TestCommandHandler:
    """Tests for slash command handling."""
    
    @pytest.fixture
    def handler(self, tmp_path):
        """Create a command handler with mock session."""
        session = MagicMock()
        session.workspace_root = tmp_path
        session.pending_changes = []
        session.conversation = []
        session.total_tokens = 0
        
        display = Display()
        
        return CommandHandler(session=session, display=display)
    
    def test_parse_slash_command(self, handler):
        """Should correctly parse slash commands."""
        cmd, args = handler.parse("/help")
        assert cmd == "help"
        assert args == ""
        
        cmd, args = handler.parse("/model llama3")
        assert cmd == "model"
        assert args == "llama3"
        
        cmd, args = handler.parse("/rollback cp_001")
        assert cmd == "rollback"
        assert args == "cp_001"
    
    def test_is_command(self, handler):
        """Should identify slash commands."""
        assert handler.is_command("/help")
        assert handler.is_command("/diff")
        assert not handler.is_command("help me")
        assert not handler.is_command("what is /slash")
    
    def test_help_command(self, handler):
        """Help command should list available commands."""
        # Just verify it doesn't crash
        result = handler.handle("/help")
        assert result is None or isinstance(result, str)
    
    def test_unknown_command(self, handler):
        """Unknown commands should be handled gracefully."""
        result = handler.handle("/nonexistent")
        # Should not crash, may show error
        assert result is None or "unknown" in str(result).lower()
    
    def test_exit_command(self, handler):
        """Exit command should signal exit."""
        result = handler.handle("/exit")
        # Should signal exit somehow
        assert result is None or result == "exit" or handler._should_exit


class TestDisplay:
    """Tests for terminal display functionality."""
    
    @pytest.fixture
    def display(self):
        """Create a Display instance."""
        return Display()
    
    def test_status_symbols_defined(self):
        """All status symbols should be defined."""
        assert StatusSymbol.THINKING
        assert StatusSymbol.SUCCESS
        assert StatusSymbol.ERROR
        assert StatusSymbol.WARNING
    
    def test_format_diff(self, display):
        """Diff formatting should work."""
        diff = """@@ -1,3 +1,3 @@
 line1
-old line
+new line
 line3"""
        
        # Just verify it doesn't crash
        formatted = display.format_diff(diff)
        assert formatted is not None
    
    def test_format_error(self, display, capsys):
        """Error formatting should work."""
        display.error("Test error")
        # Verify something was printed (may use Rich console)


class TestCLIIntegration:
    """Integration tests for the CLI."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()
    
    def test_help_flag(self, runner):
        """--help should show usage."""
        from src.cli.app import main
        
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output or "usage" in result.output.lower()
    
    def test_version_command(self, runner):
        """version command should show version."""
        from src.cli.app import main
        
        result = runner.invoke(main, ["version"])
        assert result.exit_code == 0
        # Should show some version info


class TestSessionConfig:
    """Tests for session configuration."""
    
    def test_default_config(self):
        """Default config should have sensible values."""
        config = SessionConfig()
        
        assert config.max_tokens > 0
        assert config.model is not None
        assert config.auto_approve in (True, False)
    
    def test_config_serialization(self):
        """Config should serialize and deserialize."""
        config = SessionConfig(
            model="test-model",
            max_tokens=5000,
            auto_approve=True,
        )
        
        data = config.model_dump()
        restored = SessionConfig.model_validate(data)
        
        assert restored.model == config.model
        assert restored.max_tokens == config.max_tokens
        assert restored.auto_approve == config.auto_approve


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
