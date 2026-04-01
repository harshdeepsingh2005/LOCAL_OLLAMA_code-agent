"""
Conversational REPL Module

Implements Feature 5: a true interactive REPL that wraps the Executor.

Key capabilities:
- Start a task, watch it run, interrupt at any time with Ctrl+C
- After interruption, type a correction/addendum and the agent re-plans
- Session history is preserved across multiple tasks
- Rich terminal output with live status updates

Design Decisions:
- Uses threading so the executor runs in the background while the
  main thread listens for keyboard interrupts
- Uses an asyncio-friendly signal handler approach on macOS/Linux
- Falls back to simple prompt_toolkit polling on Windows
- The REPL stores an interrupt_event (threading.Event) that the
  executor's LoopController checks at the start of each iteration
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.text import Text
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


@dataclass
class REPLSession:
    """Tracks state across multiple REPL turns."""
    session_id: str
    workspace_root: Path
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    turns: list[dict] = field(default_factory=list)
    interrupt_requested: bool = False
    interrupt_message: str = ""

    def record_turn(
        self,
        task: str,
        result_summary: str,
        interrupted: bool = False,
    ) -> None:
        """Append a completed turn to the session history."""
        self.turns.append({
            "task": task,
            "result": result_summary,
            "interrupted": interrupted,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


class ConversationalREPL:
    """
    Interactive conversational REPL wrapping the Executor.

    Usage (standalone)::

        repl = ConversationalREPL(config, workspace_root, log_dir)
        repl.run()

    The REPL:
    1. Prompts the user for a task
    2. Runs the executor in a background thread
    3. Listens for Ctrl+C — on interrupt, pauses and asks for a correction
    4. Injects the correction into a new execution cycle
    5. Loops until the user types 'exit' or 'quit'
    """

    _BANNER = """
╔══════════════════════════════════════════════════════════╗
║          LOCAL CODING AGENTS  ·  Conversational REPL     ║
║     Press Ctrl+C at any time to interrupt & redirect     ║
╚══════════════════════════════════════════════════════════╝
"""

    def __init__(self, config: Any, workspace_root: Path, log_dir: Path) -> None:
        self._config = config
        self._workspace_root = workspace_root
        self._log_dir = log_dir
        self._console = Console() if _RICH_AVAILABLE else None
        self._executor: Any = None      # Lazy import to avoid circular
        self._interrupt_event = threading.Event()
        self._result_holder: dict = {}
        self._session: REPLSession | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive REPL loop."""
        import uuid
        from src.orchestration.executor import Executor

        self._session = REPLSession(
            session_id=uuid.uuid4().hex[:8],
            workspace_root=self._workspace_root,
        )

        self._print_banner()

        while True:
            try:
                task = self._prompt_task()
            except (KeyboardInterrupt, EOFError):
                self._print("\n👋  Session ended.")
                break

            if task.strip().lower() in ("exit", "quit", "q"):
                self._print("👋  Goodbye.")
                break

            if not task.strip():
                continue

            # Initialize or reuse executor
            if self._executor is None:
                self._executor = Executor(
                    config=self._config,
                    workspace_root=self._workspace_root,
                    log_dir=self._log_dir,
                )

            self._run_task(task)

    # ------------------------------------------------------------------
    # Task execution with interrupt support
    # ------------------------------------------------------------------

    def _run_task(self, task: str) -> None:
        """Run a task in a background thread; handle Ctrl+C interrupts."""
        self._interrupt_event.clear()
        self._result_holder.clear()

        thread = threading.Thread(
            target=self._executor_thread,
            args=(task,),
            daemon=True,
        )

        self._print(f"\n🚀  Starting: {task[:80]}")
        self._print("   (Press Ctrl+C to interrupt and redirect)\n")

        thread.start()

        interrupted = False
        try:
            while thread.is_alive():
                thread.join(timeout=0.25)   # poll interval
        except KeyboardInterrupt:
            interrupted = True
            self._interrupt_event.set()
            self._print("\n\n⚡  Interrupted! Waiting for agent to pause…")
            thread.join(timeout=10)

        result = self._result_holder.get("result")
        session_turn = self._session

        if interrupted:
            self._handle_interrupt(task)
        else:
            if result:
                success = getattr(result, "success", False)
                summary = f"{'✅ Done' if success else '❌ Failed'}"
                self._print(f"\n{summary}  —  task: {task[:60]}")
                if session_turn:
                    session_turn.record_turn(task, summary)

    def _executor_thread(self, task: str) -> None:
        """Target function for background execution thread."""
        try:
            if self._executor is None:
                return
            # Patch LoopController with our interrupt event
            self._patch_loop_interrupt()
            result = self._executor.execute(task)
            self._result_holder["result"] = result
        except Exception as e:
            self._result_holder["error"] = str(e)

    def _patch_loop_interrupt(self) -> None:
        """
        Inject our interrupt_event into LoopController so it stops cleanly
        when the user presses Ctrl+C.
        """
        try:
            if self._executor and self._executor._loop:
                self._executor._loop._interrupt_event = self._interrupt_event
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Interrupt handling
    # ------------------------------------------------------------------

    def _handle_interrupt(self, original_task: str) -> None:
        """
        After Ctrl+C, ask the user what they want to change and
        re-run the agent with the amended task.
        """
        self._print("\n" + "─" * 60)
        self._print("  What would you like to change?")
        self._print(f"  (Original task: {original_task[:60]})")
        self._print("  (Press Enter to abandon, or type a correction)")
        self._print("─" * 60)

        try:
            correction = input("  ✏️  Correction: ").strip()
        except (EOFError, KeyboardInterrupt):
            correction = ""

        if not correction:
            self._print("  ↩  Abandoning interrupted task.\n")
            if self._session:
                self._session.record_turn(original_task, "abandoned", interrupted=True)
            return

        # Compose amended task
        amended = (
            f"{original_task}\n\n"
            f"[CORRECTION from user]: {correction}"
        )
        self._print(f"\n🔄  Re-running with updated goal…")

        # Reset executor state for fresh run
        self._executor = None
        from src.orchestration.executor import Executor
        self._executor = Executor(
            config=self._config,
            workspace_root=self._workspace_root,
            log_dir=self._log_dir,
        )

        if self._session:
            self._session.record_turn(original_task, "interrupted→redirected", interrupted=True)

        self._run_task(amended)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        if _RICH_AVAILABLE and self._console:
            self._console.print(self._BANNER, style="bold cyan")
        else:
            print(self._BANNER)

    def _print(self, msg: str) -> None:
        if _RICH_AVAILABLE and self._console:
            self._console.print(msg)
        else:
            print(msg, flush=True)

    def _prompt_task(self) -> str:
        turn_no = len(self._session.turns) + 1 if self._session else 1
        prompt_str = f"\n[{turn_no}] 💬  Task"
        if _RICH_AVAILABLE and self._console:
            return Prompt.ask(prompt_str)
        else:
            return input(f"{prompt_str}: ")

    # ------------------------------------------------------------------
    # Session info
    # ------------------------------------------------------------------

    @property
    def session(self) -> REPLSession | None:
        """Current session state."""
        return self._session
