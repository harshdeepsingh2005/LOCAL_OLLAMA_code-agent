"""
CLI Application Module

Main entry point for the interactive agent CLI.
Provides Claude Code–like experience with human-in-the-loop safety.
"""

from __future__ import annotations

import re
import signal
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import click

from src.cli.commands import CommandHandler
from src.cli.display import Display
from src.cli.session import PendingChange, Session, SessionConfig, SessionState
from src.config import Configuration, get_config
from src.core import DiffEngine, FileGuard, FileGuardPolicy, LLMClient
from src.orchestration import Executor, RollbackManager
from src.orchestration.workspace_manager import WorkspaceManager


VERSION = "1.0.0"


class AgentCLI:
    """
    Main interactive CLI application.
    
    Provides a conversational interface for the coding agent
    with human-in-the-loop safety guarantees.
    """
    
    def __init__(
        self,
        workspace: Path,
        config: Optional[Configuration] = None,
        session_id: Optional[str] = None,
        quiet: bool = False,
        verbose: bool = False,
        model: Optional[str] = None,
        policy_profile: str = "balanced",
        dry_run: bool = False,
    ) -> None:
        """
        Initialize the CLI.
        
        Args:
            workspace: Workspace directory
            config: Configuration (loaded from defaults if not provided)
            session_id: Session ID to resume
            quiet: Minimal output mode
            verbose: Detailed output mode
            model: Override model name
            dry_run: Don't apply changes
        """
        self.workspace = workspace.resolve()
        self.config = config or get_config()
        self.dry_run = dry_run
        
        # Display manager
        self.display = Display(quiet=quiet, verbose=verbose)
        
        # Session management
        if session_id:
            self.session = Session.load(session_id)
            if not self.session:
                self.display.error(f"Session not found: {session_id}")
                sys.exit(1)
        else:
            session_config = SessionConfig(
                workspace=self.workspace,
                model=model or self.config.models.default_model,
                policy_profile=policy_profile,
                max_tokens_per_run=self.config.limits.tokens.max_per_run,
            )
            self.session = Session(config=session_config)

        self._policy_profile = self.session.config.policy_profile
        self._workspace_roots: list[Path] = [self.workspace]
        
        # Command handler
        self.commands = CommandHandler(self.session, self.display)
        
        # Connect project mode callback to commands
        self.commands.set_app_callback(self._handle_project_callback)
        
        # Core components (initialized lazily)
        self._llm_client: Optional[LLMClient] = None
        self._executor: Optional[Executor] = None
        self._file_guard: Optional[FileGuard] = None
        self._diff_engine: Optional[DiffEngine] = None
        self._rollback: Optional[RollbackManager] = None
        
        # Project mode: keeps executor alive for iterative development
        self._project_mode: bool = False
        
        # Running state
        self._running = False
        self._interrupted = False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame) -> None:
        """Handle Ctrl+C interrupt."""
        self._interrupted = True
        if self._running:
            self.display.newline()
            self.display.warning("Interrupted")
    
    def _initialize_components(self) -> bool:
        """Initialize core components."""
        try:
            # Initialize LLM client
            self._llm_client = LLMClient(
                base_url=self.config.models.ollama.base_url,
                timeout=float(self.config.models.ollama.timeout_seconds),
            )
            
            # Check LLM health
            if not self._llm_client.health_check():
                self._show_ollama_error()
                return False
            
            # Verify model exists
            models = self._llm_client.list_models()
            if self.session.config.model not in models:
                self._show_model_error(models)
                return False
            
            # Initialize file guard
            file_policy = FileGuardPolicy(
                allowed_roots=[self.workspace],
                blocked_patterns=self.config.policies.file_access.blocked_patterns,
                allowed_extensions=self.config.policies.file_access.allowed_extensions,
                max_file_size_bytes=self.config.limits.files.max_read_size_bytes,
                max_files_per_run=self.config.limits.files.max_modified_per_run,
            )
            self._file_guard = FileGuard(
                workspace_root=self.workspace,
                policy=file_policy,
            )
            
            # Set command handler context
            self.commands.set_context(
                llm_client=self._llm_client,
                config=self.config,
            )
            
            return True
            
        except Exception as e:
            self.display.error(f"Failed to initialize: {e}")
            return False
    
    def _show_ollama_error(self) -> None:
        """Show Ollama connection error with recovery options."""
        self.display.error_panel(
            title="Cannot connect to Ollama",
            message=f"Failed to connect to {self.config.models.ollama.base_url}",
            causes=[
                "Ollama is not running",
                "A different port is configured",
                "Network/firewall issue",
            ],
            solutions=[
                "Run: ollama serve",
                "Check: ollama list",
                f"Verify: curl {self.config.models.ollama.base_url}/api/tags",
            ],
        )
    
    def _show_model_error(self, available_models: list[str]) -> None:
        """Show model not found error with recovery options."""
        self.display.error_panel(
            title=f"Model '{self.session.config.model}' not found",
            message="The configured model is not available in Ollama.",
            solutions=[
                f"Pull the model: ollama pull {self.session.config.model}",
                f"Use an available model: {', '.join(available_models[:3])}",
            ],
        )
        
        if available_models:
            choice = self.display.choice(
                "Options:",
                [f"Use {m} instead" for m in available_models[:3]] + ["Exit"],
                default=1,
            )
            
            if choice <= len(available_models[:3]):
                self.session.config.model = available_models[choice - 1]
                self.display.success(f"Switched to {self.session.config.model}")
    
    def run_interactive(self) -> int:
        """
        Run interactive session.
        
        Returns:
            Exit code
        """
        self._running = True
        
        # Initialize
        if not self._initialize_components():
            return 1
        
        # Show welcome
        self.display.welcome(
            version=VERSION,
            model=self.session.config.model,
            workspace=self.workspace,
        )
        
        # Show project mode hint
        self.display.info("💡 Tip: Use /project on for continuous iterative development")
        self.display.info(f"💡 Active policy profile: {self._policy_profile}")
        
        # Start session
        self.session.start()
        
        try:
            # Main interaction loop
            while self._running and not self._interrupted:
                # Check for pending approvals
                if self.session.state == SessionState.PENDING_APPROVAL:
                    response = self.commands.get_approval()
                    if not self.commands.handle_approval(response):
                        break
                    continue

                self.display.session_status_line(
                    model=self.session.config.model,
                    project_mode=self._project_mode,
                    tokens_used=self.session.tokens_used,
                    tokens_limit=self.session.config.max_tokens_per_run,
                    pending_changes=len(self.session.pending_changes),
                )
                
                # Get user input
                try:
                    prompt_msg = "🔄 [Project]" if self._project_mode else ""
                    user_input = self.display.prompt(prompt_msg)
                except (EOFError, KeyboardInterrupt):
                    break
                
                if not user_input.strip():
                    continue
                
                # Handle slash commands
                if self.commands.is_command(user_input):
                    result = self.commands.handle(user_input)
                    if result in ("exit", "pause"):
                        break
                    continue
                
                # Process task
                self._process_task(user_input)
                
        except Exception as e:
            self.display.error(f"Unexpected error: {e}")
            self.session.set_error(str(e))
            return 1
        
        finally:
            self._cleanup()
        
        return 0
    
    def run_oneshot(self, task: str) -> int:
        """
        Run one-shot task execution.
        
        Args:
            task: Task description
            
        Returns:
            Exit code
        """
        self._running = True
        
        # Initialize
        if not self._initialize_components():
            return 1
        
        # Process task
        self.session.start()
        success = self._process_task(task)
        
        # Handle pending changes
        if self.session.pending_changes:
            response = self.commands.get_approval()
            if not self.commands.handle_approval(response):
                self._cleanup()
                return 1
        
        self._cleanup()
        return 0 if success else 1
    
    def _process_task(self, task: str) -> bool:
        """
        Process a user task.
        
        In project mode, reuses the same executor for iterative development.
        In one-shot mode, creates a new executor for each task.
        
        Args:
            task: Task description
            
        Returns:
            True if successful
        """
        self._maybe_show_scope_shift_hint(task)
        self.session.add_user_message(task)
        
        try:
            log_dir = Path.home() / ".local" / "share" / "agent" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            if len(self._workspace_roots) > 1:
                return self._process_task_multi_workspace(task, log_dir)

            if not self._preview_and_confirm_plan(task, log_dir):
                self.display.info("Task cancelled before implementation")
                return False
            
            # In project mode, reuse executor for iterative development
            if self._project_mode and self._executor is not None:
                self.display.info(f"[Project Mode] Continuing on same project...")
                result = self._execute_with_continuation(
                    execute_fn=lambda: self._executor.execute_additional_task(task),
                    continue_fn=lambda: self._executor.continue_execution(),
                )
            else:
                # Create new executor for this task
                if self._project_mode:
                    self.display.info("[Project Mode] Starting new project...")
                
                executor = Executor(
                    config=self.config,
                    workspace_root=self.workspace,
                    log_dir=log_dir,
                    policy_profile=self._policy_profile,
                )
                
                # Store executor if in project mode
                if self._project_mode:
                    self._executor = executor
                
                result = self._execute_with_continuation(
                    execute_fn=lambda: executor.execute(task, run_id=self.session.id),
                    continue_fn=lambda: executor.continue_execution(),
                )
            
            # Update session
            self.session.add_tokens(result.tokens_delta)
            
            if result.success:
                shown_tokens = result.tokens_delta if result.tokens_delta > 0 else result.total_tokens
                self.display.success(
                    f"Done - {result.subtasks_completed}/{result.subtasks_total} tasks • "
                    f"{result.total_duration_ms / 1000:.1f}s • "
                    f"{shown_tokens:,} tokens"
                )
                self.display.raw(f"Run ID: {result.run_id}")
                
                # Show file changes
                if result.files_created or result.files_modified:
                    self.display.subheader("Files Changed")
                    for f in result.files_created:
                        self.display.raw(f"  [file.add]+[/file.add] {f}")
                    for f in result.files_modified:
                        self.display.raw(f"  [file.modify]~[/file.modify] {f}")
                
                return True
            else:
                self.display.error(
                    f"Failed - {result.subtasks_completed}/{result.subtasks_total} tasks • "
                    f"{result.termination_reason}"
                )
                if result.error:
                    self.display.raw(f"Error: {result.error}")
                
                return False
                
        except Exception as e:
            self.display.error(f"Task execution failed: {e}")
            self.session.set_error(str(e))
            return False

    def _maybe_show_scope_shift_hint(self, new_task: str) -> None:
        """Show a lightweight hint when task scope appears to shift significantly."""
        previous_user_tasks = [
            m.content
            for m in self.session.messages
            if m.role == "user" and m.content.strip()
        ]
        if not previous_user_tasks:
            return

        previous = previous_user_tasks[-1]
        prev_tokens = self._extract_scope_tokens(previous)
        new_tokens = self._extract_scope_tokens(new_task)
        if not prev_tokens or not new_tokens:
            return

        overlap = len(prev_tokens & new_tokens)
        union = len(prev_tokens | new_tokens)
        similarity = overlap / union if union else 1.0

        prev_paths = self._extract_path_hints(previous)
        new_paths = self._extract_path_hints(new_task)
        path_overlap = bool(prev_paths & new_paths)

        if similarity < 0.18 and not path_overlap:
            self.display.warning(
                "Possible scope shift detected. Consider starting a fresh context for better reasoning."
            )
            self.display.info("Tip: use /clear to reset conversation context before continuing.")

    @staticmethod
    def _extract_scope_tokens(text: str) -> set[str]:
        """Extract lexical scope tokens from a task prompt."""
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "into", "about",
            "please", "need", "want", "make", "add", "fix", "update", "change",
            "file", "files", "code", "task", "agent", "project", "run", "test",
        }
        raw_tokens = re.findall(r"[a-zA-Z0-9_\-/\.]+", text.lower())
        tokens: set[str] = set()
        for token in raw_tokens:
            clean = token.strip("._-/")
            if len(clean) < 4:
                continue
            if clean in stopwords:
                continue
            tokens.add(clean)
        return tokens

    @staticmethod
    def _extract_path_hints(text: str) -> set[str]:
        """Extract likely file/module path hints from user text."""
        matches = re.findall(r"[a-zA-Z0-9_\-/]+\.[a-zA-Z0-9]+", text)
        return {m.lower() for m in matches}

    def _preview_and_confirm_plan(self, task: str, log_dir: Path) -> bool:
        """Preview plan and request approval before implementation begins."""
        self.display.info("Planning...")

        preview_executor = Executor(
            config=self.config,
            workspace_root=self.workspace,
            log_dir=log_dir,
            policy_profile=self._policy_profile,
        )

        try:
            plan = preview_executor.preview_plan(task, run_id=self.session.id)
        except Exception as e:
            self.display.warning(f"Plan preview failed: {e}")
            fallback = self.display.confirm(
                "Continue without plan preview?",
                default=True,
            )
            return fallback == "y"

        plan_tasks = [
            {
                "title": subtask.title,
                "description": subtask.description,
                "status": "not_started",
            }
            for subtask in plan.subtasks
        ]
        self.display.plan(
            summary=plan.plan_summary or f"Planned {len(plan.subtasks)} task(s)",
            tasks=plan_tasks,
        )

        if plan.identified_risks:
            self.display.subheader("Identified Risks")
            for risk in plan.identified_risks[:5]:
                self.display.raw(f"  - {risk}")

        response = self.display.confirm(
            "Proceed with implementation using this plan?",
            default=True,
        )
        return response == "y"
    
    def _execute_with_continuation(
        self,
        execute_fn: Callable[[], ExecutionResult],
        continue_fn: Callable[[], ExecutionResult],
    ) -> ExecutionResult:
        """
        Execute a function that returns ExecutionResult, handling continuation prompts.
        
        Args:
            execute_fn: Callable that executes and returns ExecutionResult
            
        Returns:
            Final ExecutionResult after all continuations
        """
        with self.display.spinner("Executing...") as progress:
            progress.add_task(description="Working on task...", total=None)
            result = execute_fn()

        total_token_delta = result.tokens_delta if result.tokens_delta else result.total_tokens
        created_files = set(result.files_created)
        modified_files = set(result.files_modified)
        
        # Handle continuation loop
        while result.needs_continuation:
            self.display.warning(
                f"Max iterations reached ({result.iterations} iterations)"
            )
            self.display.info(
                f"Progress: {result.subtasks_completed}/{result.subtasks_total} tasks completed"
            )
            
            if not self._ask_continue():
                self.display.info("Stopping at user request")
                break
            
            # Continue execution
            self.display.info("Continuing with 10 more iterations...")
            with self.display.spinner("Continuing...") as progress:
                progress.add_task(description="Continuing task...", total=None)
                result = continue_fn()

            step_delta = result.tokens_delta if result.tokens_delta else result.total_tokens
            total_token_delta += step_delta
            created_files.update(result.files_created)
            modified_files.update(result.files_modified)

        modified_files -= created_files
        result.tokens_delta = total_token_delta
        result.files_created = sorted(created_files)
        result.files_modified = sorted(modified_files)
        
        return result
    
    def enable_project_mode(self) -> None:
        """Enable project mode for iterative development."""
        self._project_mode = True
        self.display.success("[Project Mode] Enabled - executor will persist across tasks")
    
    def disable_project_mode(self) -> None:
        """Disable project mode and clean up executor."""
        self._project_mode = False
        if self._executor:
            # Executor cleanup happens automatically
            self._executor = None
        self.display.info("[Project Mode] Disabled")
    
    def set_policy_profile(self, profile: str) -> None:
        """Set active policy profile for subsequent executor runs."""
        profile = profile.lower().strip()
        if profile not in {"strict", "balanced", "permissive"}:
            raise ValueError(f"Unsupported policy profile: {profile}")
        self._policy_profile = profile
        self.session.config.policy_profile = profile
        if self._executor is not None:
            # Force rebuild to ensure policy is applied deterministically.
            self._executor = None
        self.display.success(f"[Policy Profile] Set to {profile}")

    def add_workspace(self, workspace_path: str) -> None:
        """Add workspace to sequential multi-workspace list."""
        path = Path(workspace_path).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            self.display.error(f"Workspace path does not exist: {path}")
            return
        if path in self._workspace_roots:
            self.display.info(f"Workspace already configured: {path}")
            return
        self._workspace_roots.append(path)
        self._workspace_roots = sorted(set(self._workspace_roots), key=lambda p: str(p))
        self.display.success(f"Added workspace: {path}")

    def clear_workspaces(self) -> None:
        """Reset to primary workspace only."""
        self._workspace_roots = [self.workspace]
        self.display.info("Workspace list reset to primary workspace")

    def _process_task_multi_workspace(self, task: str, log_dir: Path) -> bool:
        """Execute a task sequentially across configured workspaces (v1)."""
        manager = WorkspaceManager(self._workspace_roots)
        assignments = {name: [task] for name in manager.list_workspaces()}

        self.display.info(
            "[Multi-Workspace] Sequential execution order: "
            + ", ".join(manager.list_workspaces())
        )

        failures = 0

        def _runner(workspace_name: str, item: str) -> bool:
            workspace_ctx = manager.get_workspace(workspace_name)
            executor = Executor(
                config=self.config,
                workspace_root=workspace_ctx.root,
                log_dir=log_dir / workspace_name,
                policy_profile=self._policy_profile,
            )
            result = executor.execute(item, run_id=f"{self.session.id}_{workspace_name}")
            self.display.info(
                f"[Multi-Workspace:{workspace_name}] "
                f"{result.subtasks_completed}/{result.subtasks_total} complete"
            )
            return result.success

        for workspace_name, success in manager.execute_sequential(assignments, _runner):
            if not success:
                failures += 1
                self.display.warning(f"[Multi-Workspace:{workspace_name}] execution failed")

        return failures == 0

    def _handle_project_callback(self, action: str, payload: Any = None) -> Any:
        """
        Handle project mode callbacks from CommandHandler.
        
        Args:
            action: Action to perform
            
        Returns:
            Result of the action
        """
        if action == "get_project_mode":
            return self._project_mode
        elif action == "enable_project_mode":
            self.enable_project_mode()
        elif action == "disable_project_mode":
            self.disable_project_mode()
        elif action == "get_policy_profile":
            return self._policy_profile
        elif action == "set_policy_profile":
            self.set_policy_profile(str(payload))
        elif action == "get_workspaces":
            return [str(p) for p in self._workspace_roots]
        elif action == "add_workspace":
            self.add_workspace(str(payload))
        elif action == "clear_workspaces":
            self.clear_workspaces()
        else:
            raise ValueError(f"Unknown project callback action: {action}")
    
    def _ask_continue(self) -> bool:
        """
        Ask the user if they want to continue execution.
        
        Returns:
            True if user wants to continue, False otherwise
        """
        try:
            response = self.display.prompt(
                "Continue with 10 more iterations? [y/N]"
            ).strip().lower()
            return response in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            return False
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        self._running = False
        
        if self._llm_client:
            self._llm_client.close()
        
        # Save session if not ended
        if self.session.state not in (SessionState.ENDED, SessionState.PAUSED):
            self.session.end()


# =============================================================================
# CLI Entry Points
# =============================================================================

@click.group(invoke_without_command=True)
@click.option(
    "--workspace", "-w",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Workspace directory",
)
@click.option(
    "--resume", "-r",
    type=str,
    default=None,
    help="Resume a previous session",
)
@click.option(
    "--model", "-m",
    type=str,
    default=None,
    help="Model to use",
)
@click.option(
    "--policy-profile",
    type=click.Choice(["strict", "balanced", "permissive"], case_sensitive=False),
    default="balanced",
    show_default=True,
    help="Execution policy profile",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbose output",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Minimal output",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without applying changes",
)
@click.option(
    "--task", "-t",
    type=str,
    default=None,
    help="Run a one-shot task (alternative to interactive mode)",
)
@click.version_option(version=VERSION, prog_name="Agent")
@click.pass_context
def main(
    ctx: click.Context,
    workspace: Optional[Path],
    resume: Optional[str],
    model: Optional[str],
    policy_profile: str,
    verbose: bool,
    quiet: bool,
    dry_run: bool,
    task: Optional[str],
) -> None:
    """
    Agent - Local AI Coding Assistant
    
    Run fully offline on Apple Silicon. Uses Ollama for local LLM inference.
    
    \b
    Examples:
        agent                                   Start interactive session
        agent -t "fix the bug in auth.py"       One-shot task
        agent --resume abc123                   Resume session
        agent doctor                            Check prerequisites
    """
    # If a subcommand is being invoked, let it handle things
    if ctx.invoked_subcommand is not None:
        return
    
    # Determine workspace
    ws = workspace or Path.cwd()
    
    # Create CLI instance
    cli = AgentCLI(
        workspace=ws,
        session_id=resume,
        quiet=quiet,
        verbose=verbose,
        model=model,
        policy_profile=policy_profile,
        dry_run=dry_run,
    )
    
    # Run in appropriate mode
    if task:
        exit_code = cli.run_oneshot(task)
    else:
        exit_code = cli.run_interactive()
    
    sys.exit(exit_code)


@main.command()
def doctor() -> None:
    """Check system prerequisites."""
    display = Display()
    
    display.raw("\n[bold]Agent Doctor[/bold] - System Check\n")
    
    all_ok = True
    
    # Check Python version
    import platform
    py_version = platform.python_version()
    if tuple(map(int, py_version.split(".")[:2])) >= (3, 10):
        display.success(f"Python {py_version}")
    else:
        display.error(f"Python {py_version} (need 3.10+)")
        all_ok = False
    
    # Check Ollama
    display.raw("\nChecking Ollama...")
    try:
        config = get_config()
        client = LLMClient(
            base_url=config.models.ollama.base_url,
            timeout=10,
        )
        
        if client.health_check():
            display.success(f"Ollama running at {config.models.ollama.base_url}")
            
            # List models
            models = client.list_models()
            if models:
                display.success(f"Models available: {len(models)}")
                for m in models[:5]:
                    display.raw(f"    • {m}")
                if len(models) > 5:
                    display.raw(f"    ... and {len(models) - 5} more")
            else:
                display.warning("No models installed")
                display.info("  Run: ollama pull qwen2.5-coder:7b-instruct-q4_K_M")
                all_ok = False
            
            client.close()
        else:
            display.error("Ollama not responding")
            all_ok = False
            
    except Exception as e:
        display.error(f"Ollama check failed: {e}")
        all_ok = False
    
    # Check workspace
    display.raw("\nChecking workspace...")
    cwd = Path.cwd()
    display.success(f"Current directory: {cwd}")
    
    # Check config
    display.raw("\nChecking configuration...")
    try:
        config = get_config()
        display.success(f"Default model: {config.models.default_model}")
        display.success(f"Token limit: {config.limits.tokens.max_per_run:,}")
    except Exception as e:
        display.error(f"Config error: {e}")
        all_ok = False
    
    # Summary
    display.raw("")
    if all_ok:
        display.success("All checks passed! Ready to use.")
    else:
        display.error("Some checks failed. Please fix the issues above.")
    
    sys.exit(0 if all_ok else 1)


@main.command()
def version() -> None:
    """Show version information."""
    display = Display()
    
    display.raw(f"\nAgent v{VERSION}")
    display.raw("Local AI Coding Assistant")
    display.raw("")
    
    try:
        config = get_config()
        display.raw(f"Default model: {config.models.default_model}")
        display.raw(f"Ollama URL: {config.models.ollama.base_url}")
    except Exception:
        pass


@main.command()
@click.option("--limit", "-n", type=int, default=10, help="Number of sessions to show")
def sessions(limit: int) -> None:
    """List recent sessions."""
    display = Display()
    
    sessions_list = Session.list_sessions(limit=limit)
    
    if not sessions_list:
        display.info("No sessions found")
        return
    
    display.subheader("Recent Sessions")
    
    for s in sessions_list:
        state_style = "success" if s["state"] == "ended" else "warning" if s["state"] == "paused" else "info"
        display.raw(
            f"  [{state_style}]{s['state']:10}[/{state_style}] "
            f"{s['id']} "
            f"({s['tasks_completed']}/{s['tasks_total']} tasks)"
        )


@main.command()
@click.argument("key", required=False)
@click.argument("value", required=False)
def config(key: Optional[str], value: Optional[str]) -> None:
    """View or set configuration."""
    display = Display()
    
    if key and value:
        display.warning("Config modification not yet implemented")
        display.info("Edit ~/.config/agent/config.yaml directly")
        return
    
    try:
        cfg = get_config()
        
        display.subheader("Current Configuration")
        
        display.key_value({
            "Default model": cfg.models.default_model,
            "Ollama URL": cfg.models.ollama.base_url,
            "Timeout": f"{cfg.models.ollama.timeout_seconds}s",
        }, title="Models")
        
        display.key_value({
            "Max tokens/run": f"{cfg.limits.tokens.max_per_run:,}",
            "Max iterations": cfg.limits.iterations.max_loop_iterations,
            "Max files/run": cfg.limits.files.max_modified_per_run,
        }, title="Limits")
        
    except Exception as e:
        display.error(f"Failed to load config: {e}")


@main.command()
def stats() -> None:
    """Show usage statistics."""
    display = Display()
    
    sessions_list = Session.list_sessions(limit=100)
    
    if not sessions_list:
        display.info("No usage data")
        return
    
    total_sessions = len(sessions_list)
    completed = sum(1 for s in sessions_list if s["state"] == "ended")
    total_tasks = sum(s["tasks_completed"] for s in sessions_list)
    
    display.key_value({
        "Total sessions": total_sessions,
        "Completed": completed,
        "Tasks completed": total_tasks,
    }, title="Usage Statistics")


if __name__ == "__main__":
    main()
