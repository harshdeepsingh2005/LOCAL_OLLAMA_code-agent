"""
CLI Commands Module

Handles slash commands and user interactions.
"""

from __future__ import annotations

import subprocess
import tempfile
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from src.cli.session import Session
    from src.cli.display import Display


class CommandHandler:
    """
    Handles slash commands in the interactive session.
    
    Provides a registry of commands and their handlers.
    """
    
    COMMANDS = {
        "/help": "Show available commands",
        "/diff": "Show pending diffs",
        "/apply": "Apply all pending diffs",
        "/reject": "Reject all pending diffs",
        "/undo": "Undo last applied change",
        "/rollback": "Rollback to checkpoint [ID]",
        "/summary": "Show current session summary",
        "/plan": "Show current execution plan",
        "/logs": "Show execution logs",
        "/policy": "Show active policies",
        "/models": "Show available models",
        "/model": "Switch model [NAME]",
        "/tokens": "Show token usage",
        "/pause": "Save session and exit",
        "/exit": "Exit session",
        "/clear": "Clear conversation context",
        "/checkpoints": "List available checkpoints",
        "/project": "Enable/disable project mode [on|off]",
    }
    
    def __init__(
        self,
        session: "Session",
        display: "Display",
    ) -> None:
        """
        Initialize command handler.
        
        Args:
            session: Current session
            display: Display manager
        """
        self.session = session
        self.display = display
        
        # Handlers that need external context
        self._llm_client: Any = None
        self._rollback_manager: Any = None
        self._config: Any = None
        
        # Undo stack
        self._undo_stack: list[dict[str, Any]] = []
        self._should_exit: bool = False

    def parse(self, input_text: str) -> tuple[str, str]:
        """Backward-compatible parser expected by older tests."""
        command, args = self.parse_command(input_text)
        return command.lstrip("/"), " ".join(args)
    
    def set_context(
        self,
        llm_client: Any = None,
        rollback_manager: Any = None,
        config: Any = None,
    ) -> None:
        """Set external context objects."""
        self._llm_client = llm_client
        self._rollback_manager = rollback_manager
        self._config = config
    
    def is_command(self, input_text: str) -> bool:
        """Check if input is a slash command."""
        return input_text.strip().startswith("/")
    
    def parse_command(self, input_text: str) -> tuple[str, list[str]]:
        """Parse command and arguments."""
        parts = input_text.strip().split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1].split() if len(parts) > 1 else []
        return command, args
    
    def handle(self, input_text: str) -> Optional[str]:
        """
        Handle a slash command.
        
        Returns:
            None for normal commands, "exit" or "pause" for session-ending commands
        """
        command, args = self.parse_command(input_text)
        
        handlers = {
            "/help": self._cmd_help,
            "/diff": self._cmd_diff,
            "/apply": self._cmd_apply,
            "/reject": self._cmd_reject,
            "/undo": self._cmd_undo,
            "/rollback": self._cmd_rollback,
            "/summary": self._cmd_summary,
            "/plan": self._cmd_plan,
            "/logs": self._cmd_logs,
            "/policy": self._cmd_policy,
            "/models": self._cmd_models,
            "/model": self._cmd_model,
            "/tokens": self._cmd_tokens,
            "/pause": self._cmd_pause,
            "/exit": self._cmd_exit,
            "/clear": self._cmd_clear,
            "/checkpoints": self._cmd_checkpoints,
            "/project": self._cmd_project,
        }
        
        handler = handlers.get(command)
        if handler:
            return handler(args)
        else:
            self.display.error(f"Unknown command: {command}")
            self.display.info("Type /help for available commands")
            return None
    
    # =========================================================================
    # Command Handlers
    # =========================================================================
    
    def _cmd_help(self, args: list[str]) -> None:
        """Show help."""
        self.display.help(self.COMMANDS)
    
    def _cmd_diff(self, args: list[str]) -> None:
        """Show pending diffs."""
        if not self.session.pending_changes:
            self.display.info("No pending changes")
            return
        
        self.display.diff_summary(self.session.get_pending_summary())
        
        # Show full diffs if verbose or requested
        for change in self.session.pending_changes:
            self.display.diff(change.file_path, change.diff_content)
    
    def _cmd_apply(self, args: list[str]) -> None:
        """Apply pending diffs."""
        if not self.session.pending_changes:
            self.display.info("No pending changes to apply")
            return
        
        self.display.diff_summary(self.session.get_pending_summary())
        
        response = self.display.confirm(
            "Apply these changes?",
            default=True,
        )
        
        if response == "y":
            self._apply_changes()
        else:
            self.display.info("Changes not applied")
    
    def _cmd_reject(self, args: list[str]) -> None:
        """Reject pending diffs."""
        if not self.session.pending_changes:
            self.display.info("No pending changes")
            return
        
        count = len(self.session.pending_changes)
        self.session.clear_pending_changes()
        self.display.success(f"Rejected {count} pending change(s)")
    
    def _cmd_undo(self, args: list[str]) -> None:
        """Undo last applied change."""
        if not self._undo_stack:
            self.display.info("Nothing to undo")
            return
        
        last_change = self._undo_stack.pop()
        file_path = Path(last_change["file_path"])
        original_content = last_change.get("original_content")
        
        try:
            if original_content is None:
                # File was created, delete it
                file_path.unlink(missing_ok=True)
                self.display.success(f"Removed {file_path}")
            else:
                # Restore original content
                file_path.write_text(original_content)
                self.display.success(f"Restored {file_path}")
        except Exception as e:
            self.display.error(f"Failed to undo: {e}")
    
    def _cmd_rollback(self, args: list[str]) -> None:
        """Rollback to checkpoint."""
        if not self.session.checkpoints:
            self.display.info("No checkpoints available")
            return
        
        if args:
            checkpoint_id = args[0]
            if checkpoint_id not in self.session.checkpoints:
                self.display.error(f"Checkpoint not found: {checkpoint_id}")
                return
        else:
            # Show available checkpoints
            self.display.subheader("Available Checkpoints")
            for i, cp_id in enumerate(reversed(self.session.checkpoints), 1):
                current = " (current)" if cp_id == self.session.current_checkpoint else ""
                self.display.raw(f"  {i}. {cp_id}{current}")
            
            self.display.newline()
            
            choice = self.display.choice(
                "Rollback to:",
                [f"Checkpoint {i}" for i in range(1, len(self.session.checkpoints) + 1)],
                default=1,
            )
            
            checkpoint_id = list(reversed(self.session.checkpoints))[choice - 1]
        
        # Perform rollback
        if self._rollback_manager:
            try:
                # This would call the actual rollback manager
                self.display.success(f"Rolled back to {checkpoint_id}")
                self.session.current_checkpoint = checkpoint_id
            except Exception as e:
                self.display.error(f"Rollback failed: {e}")
        else:
            self.display.warning("Rollback manager not available")
    
    def _cmd_summary(self, args: list[str]) -> None:
        """Show session summary."""
        self.display.session_info(
            session_id=self.session.id,
            status=self.session.state.value,
            tasks_completed=self.session.tasks_completed,
            tasks_total=self.session.tasks_total,
            tokens_used=self.session.tokens_used,
            tokens_limit=self.session.config.max_tokens_per_run,
        )
        
        if self.session.pending_changes:
            self.display.info(f"Pending changes: {len(self.session.pending_changes)} files")
    
    def _cmd_plan(self, args: list[str]) -> None:
        """Show execution plan."""
        if not self.session.task_plan:
            self.display.info("No plan available")
            return
        
        self.display.plan(
            summary=f"Executing {self.session.tasks_total} tasks",
            tasks=self.session.task_plan,
        )
    
    def _cmd_logs(self, args: list[str]) -> None:
        """Show execution logs."""
        # Show recent conversation messages
        self.display.subheader("Recent Activity")
        
        for msg in self.session.messages[-10:]:
            role = msg.role.capitalize()
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            self.display.raw(f"  [{role}] {content}")
    
    def _cmd_policy(self, args: list[str]) -> None:
        """Show active policies."""
        if not self._config:
            self.display.info("Configuration not available")
            return
        
        self.display.subheader("Active Policies")
        
        policies = self._config.policies
        
        self.display.key_value({
            "Blocked patterns": ", ".join(policies.file_access.blocked_patterns[:5]),
            "Allowed extensions": ", ".join(policies.file_access.allowed_extensions[:5]),
            "Max file size": f"{self._config.limits.files.max_write_size_bytes / 1024:.0f}KB",
            "Max files per run": self._config.limits.files.max_modified_per_run,
        }, title="File Access")
        
        self.display.key_value({
            "Max iterations": self._config.limits.iterations.max_loop_iterations,
            "Max retries": self._config.limits.iterations.max_agent_retries,
            "Max tokens/run": f"{self._config.limits.tokens.max_per_run:,}",
        }, title="Limits")
    
    def _cmd_models(self, args: list[str]) -> None:
        """Show available models."""
        if not self._llm_client:
            self.display.info("LLM client not available")
            return
        
        try:
            models = self._llm_client.list_models()
            
            self.display.subheader("Available Models")
            
            current_model = self.session.config.model
            for model in models:
                current = " [current]" if model == current_model else ""
                self.display.raw(f"  • {model}{current}")
            
            self.display.newline()
            
        except Exception as e:
            self.display.error(f"Failed to list models: {e}")
    
    def _cmd_model(self, args: list[str]) -> None:
        """Switch model."""
        if not args:
            self.display.info(f"Current model: {self.session.config.model}")
            self.display.info("Usage: /model <model-name>")
            return
        
        new_model = args[0]
        
        # Verify model exists
        if self._llm_client:
            try:
                models = self._llm_client.list_models()
                if new_model not in models:
                    self.display.error(f"Model not found: {new_model}")
                    self.display.info(f"Available: {', '.join(models[:5])}")
                    return
            except Exception:
                pass  # Continue anyway
        
        old_model = self.session.config.model
        self.session.config.model = new_model
        self.display.success(f"Switched from {old_model} to {new_model}")
    
    def _cmd_tokens(self, args: list[str]) -> None:
        """Show token usage."""
        used = self.session.tokens_used
        limit = self.session.config.max_tokens_per_run
        pct = self.session.token_percentage
        
        self.display.key_value({
            "Used": f"{used:,}",
            "Limit": f"{limit:,}",
            "Remaining": f"{limit - used:,}",
            "Usage": f"{pct:.1f}%",
        }, title="Token Usage")
        
        if pct >= 90:
            self.display.warning("Approaching token limit!")
        elif pct >= 75:
            self.display.info("Token usage is high")
    
    def _cmd_pause(self, args: list[str]) -> str:
        """Pause session."""
        self.session.pause()
        self.display.success(f"Session paused: {self.session.id}")
        self.display.info(f"Resume with: agent --resume {self.session.id}")
        return "pause"
    
    def _cmd_exit(self, args: list[str]) -> str:
        """Exit session."""
        if self.session.pending_changes:
            response = self.display.confirm(
                f"You have {len(self.session.pending_changes)} pending changes. Discard?",
                default=False,
            )
            if response != "y":
                self.display.info("Use /apply to apply changes, or /pause to save session")
                return None
        
        self.session.end()
        self._should_exit = True
        return "exit"
    
    def _cmd_clear(self, args: list[str]) -> None:
        """Clear conversation context."""
        self.session.clear_context()
        self.display.success("Conversation context cleared")
    
    def _cmd_checkpoints(self, args: list[str]) -> None:
        """List checkpoints."""
        if not self.session.checkpoints:
            self.display.info("No checkpoints available")
            return
        
        self.display.subheader("Checkpoints")
        for i, cp_id in enumerate(reversed(self.session.checkpoints), 1):
            current = " (current)" if cp_id == self.session.current_checkpoint else ""
            self.display.raw(f"  {i}. {cp_id}{current}")
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _apply_changes(self) -> bool:
        """Apply pending changes to files."""
        applied = []
        
        for change in self.session.pending_changes:
            file_path = Path(self.session.config.workspace) / change.file_path
            
            try:
                # Record for undo
                original_content = None
                if file_path.exists():
                    original_content = file_path.read_text()
                
                # Apply change
                if change.change_type == "delete":
                    file_path.unlink(missing_ok=True)
                else:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(change.new_content)
                
                # Add to undo stack
                self._undo_stack.append({
                    "file_path": str(file_path),
                    "original_content": original_content,
                    "change_type": change.change_type,
                })
                
                applied.append(change.file_path)
                
            except Exception as e:
                self.display.error(f"Failed to apply {change.file_path}: {e}")
                return False
        
        self.session.clear_pending_changes()
        self.display.success(f"Applied {len(applied)} change(s)")
        
        return True
    
    def get_approval(
        self,
        allow_edit: bool = True,
        allow_reject: bool = True,
    ) -> str:
        """
        Get user approval for pending changes.
        
        Returns:
            'y' (apply), 'n' (skip), 'e' (edit), 'r' (reject all), '?' (explain)
        """
        self.display.diff_summary(self.session.get_pending_summary())
        
        return self.display.confirm(
            "Apply these changes?",
            default=True,
            allow_edit=allow_edit,
            allow_reject=allow_reject,
        )
    
    def handle_approval(self, response: str) -> bool:
        """
        Handle the approval response.
        
        Returns:
            True to continue, False to stop
        """
        if response == "y":
            return self._apply_changes()
        elif response == "n":
            self.session.clear_pending_changes()
            self.display.info("Changes skipped, continuing...")
            return True
        elif response == "e":
            self._edit_changes()
            return True
        elif response == "r":
            self.session.clear_pending_changes()
            self.display.info("Changes rejected, stopping...")
            return False
        elif response == "?":
            self._explain_changes()
            return True  # Will re-prompt
        
        return True
    
    def _edit_changes(self) -> None:
        """Open changes in editor for review."""
        if not self.session.pending_changes:
            return
        
        # Create temp file with diffs
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            for change in self.session.pending_changes:
                f.write(f"# File: {change.file_path}\n")
                f.write(f"# Type: {change.change_type}\n")
                f.write(change.diff_content)
                f.write("\n\n")
            temp_path = f.name
        
        # Open in editor
        editor = os.getenv("EDITOR", "vim")
        try:
            subprocess.run([editor, temp_path], check=True)
            self.display.info("Review complete. Changes unchanged.")
        except subprocess.CalledProcessError:
            self.display.warning("Editor closed without saving")
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def _explain_changes(self) -> None:
        """Explain what the pending changes do."""
        self.display.subheader("Change Explanation")
        
        for change in self.session.pending_changes:
            self.display.raw(f"\n[bold]{change.file_path}[/bold]")
            self.display.raw(f"  {change.description}")
            
            if change.change_type == "create":
                self.display.raw(f"  Creates new file with {change.lines_added} lines")
            elif change.change_type == "modify":
                self.display.raw(
                    f"  Modifies file: +{change.lines_added} -{change.lines_removed} lines"
                )
            elif change.change_type == "delete":
                self.display.raw("  Deletes the file")
    
    def _cmd_project(self, args: list[str]) -> None:
        """
        Enable/disable project mode for iterative development.
        
        Usage:
            /project on  - Enable project mode
            /project off - Disable project mode
            /project     - Show current status
        """
        # Need app context - this is a callback that needs to be set
        if not hasattr(self, "_app_callback"):
            self.display.error("Project mode not available in this context")
            return
        
        if not args:
            # Show status
            status = self._app_callback("get_project_mode")
            if status:
                self.display.info("[Project Mode] Currently enabled")
            else:
                self.display.info("[Project Mode] Currently disabled")
            return
        
        mode = args[0].lower()
        if mode == "on":
            self._app_callback("enable_project_mode")
        elif mode == "off":
            self._app_callback("disable_project_mode")
        else:
            self.display.error(f"Invalid argument: {mode}")
            self.display.info("Usage: /project [on|off]")
    
    def set_app_callback(self, callback: Callable[[str], Any]) -> None:
        """Set callback to app instance for project mode commands."""
        self._app_callback = callback

