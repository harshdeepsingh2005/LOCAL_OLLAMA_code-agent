"""
Persistent PTY Shell Module

Implements a persistent pseudo-terminal (PTY) shell that maintains state
across multiple tool calls. Directory changes (cd), environment variables,
and background processes all persist between commands — matching Claude
Code's "hands" abstraction.

Design Decisions:
- Single long-lived /bin/bash process per session
- Uses pexpect for reliable PTY interaction
- Unique sentinel token to detect command completion
- Thread-safe command dispatch with locks
- Graceful teardown
"""

from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path
from typing import Any


_PEXPECT_AVAILABLE = False
try:
    import pexpect  # type: ignore
    _PEXPECT_AVAILABLE = True
except ImportError:
    pass


class PTYShellError(Exception):
    """Raised when the PTY shell encounters an unrecoverable error."""
    pass


class PTYSession:
    """
    A persistent pseudo-terminal session.

    Wraps a long-lived bash process so the agent can issue multiple
    commands that share environment state (cwd, exported vars, etc.).

    Usage::

        session = PTYSession(workspace_root=Path("/my/project"))
        session.start()
        out = session.run("cd src && ls")
        out2 = session.run("pwd")   # still in /my/project/src
        session.close()
    """

    _SENTINEL_BASE = "__LCA_CMD_DONE__"
    _TIMEOUT = 120  # seconds per command

    def __init__(
        self,
        workspace_root: Path,
        default_timeout: float = 120.0,
        max_output_chars: int = 200_000,
    ) -> None:
        self._workspace_root = workspace_root.resolve()
        self._default_timeout = default_timeout
        self._max_output_chars = max_output_chars
        self._child: Any = None   # pexpect.spawn
        self._lock = threading.Lock()
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background bash process."""
        if not _PEXPECT_AVAILABLE:
            raise PTYShellError(
                "pexpect is not installed. Run: pip install pexpect"
            )
        if self._started:
            return

        import pexpect  # type: ignore

        self._child = pexpect.spawn(
            "/bin/bash",
            ["--norc", "--noprofile"],  # clean environment
            cwd=str(self._workspace_root),
            encoding="utf-8",
            timeout=self._default_timeout,
            dimensions=(50, 220),
        )
        # Disable command echoing so output is clean
        self._child.sendline("stty -echo")
        # Use a simple, predictable PS1 so we can reliably detect prompt
        self._child.sendline("export PS1=''")
        self._flush()
        self._started = True

    def close(self) -> None:
        """Terminate the underlying bash process."""
        if self._child and self._started:
            try:
                self._child.sendline("exit")
                self._child.close()
            except Exception:
                pass
        self._started = False
        self._child = None

    def is_alive(self) -> bool:
        """Return True if the shell process is still running."""
        if not self._child or not self._started:
            return False
        return self._child.isalive()

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def run(
        self,
        command: str,
        timeout: float | None = None,
    ) -> str:
        """
        Execute a shell command in the persistent session.

        The command runs inside the already-live bash process, so previous
        ``cd``, ``export``, and background processes are visible.

        Args:
            command: Shell command (may include pipes, redirects, etc.)
            timeout: Per-command timeout in seconds

        Returns:
            Combined stdout+stderr of the command

        Raises:
            PTYShellError: If the shell is not started or times out badly
        """
        if not self._started:
            raise PTYShellError("PTY session not started. Call start() first.")

        sentinel = f"{self._SENTINEL_BASE}_{uuid.uuid4().hex[:8]}"
        timeout = timeout or self._default_timeout

        with self._lock:
            # Send the command, then echo the sentinel on exit
            full_cmd = f"{command.strip()}; echo {sentinel}"
            self._child.sendline(full_cmd)

            output_parts: list[str] = []
            try:
                while True:
                    idx = self._child.expect(
                        [re.escape(sentinel), pexpect.TIMEOUT, pexpect.EOF],
                        timeout=timeout,
                    )
                    chunk: str = self._child.before or ""
                    output_parts.append(chunk)

                    if idx == 0:
                        # Sentinel found — command finished
                        break
                    elif idx == 1:
                        raise PTYShellError(
                            f"Command timed out after {timeout}s: {command[:80]}"
                        )
                    else:
                        # EOF — process died
                        raise PTYShellError("PTY process died unexpectedly.")
            except pexpect.TIMEOUT:
                raise PTYShellError(
                    f"Command timed out after {timeout}s: {command[:80]}"
                )

        raw = "".join(output_parts)
        # Strip the echoed command itself from the front (bash may echo it)
        raw = raw.replace(full_cmd, "").replace(sentinel, "").strip()
        return raw[: self._max_output_chars]

    def get_cwd(self) -> str:
        """Return the current working directory of the persistent shell."""
        return self.run("pwd").strip()

    def get_env(self, var: str) -> str:
        """Return the value of an environment variable."""
        return self.run(f"echo ${var}").strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush(self, timeout: float = 2.0) -> None:
        """Drain any pending output after startup commands."""
        try:
            self._child.expect(pexpect.TIMEOUT, timeout=timeout)
        except Exception:
            pass


class PTYShellManager:
    """
    Registry that manages multiple named PTY sessions.

    The ToolExecutor uses this to maintain one shell per agent run,
    so commands accumulate state across tool calls within a single task.
    """

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root
        self._sessions: dict[str, PTYSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> PTYSession:
        """Return an existing session or create a new one."""
        with self._lock:
            if session_id not in self._sessions:
                session = PTYSession(self._workspace_root)
                session.start()
                self._sessions[session_id] = session
            return self._sessions[session_id]

    def close(self, session_id: str) -> None:
        """Close and remove a session."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].close()
                del self._sessions[session_id]

    def close_all(self) -> None:
        """Close all active sessions."""
        with self._lock:
            for session in self._sessions.values():
                try:
                    session.close()
                except Exception:
                    pass
            self._sessions.clear()

    @property
    def active_sessions(self) -> list[str]:
        """List of active session IDs."""
        with self._lock:
            return list(self._sessions.keys())
