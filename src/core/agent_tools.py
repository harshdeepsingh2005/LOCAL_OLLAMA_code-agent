"""
Agent Tools Module  (upgraded — Claude Code parity)

Central dispatcher that exposes ALL agent-callable tools to the LLM:

  Shell / Environment
  ──────────────────
  • run_command          — execute in a persistent PTY session (Feature 1)

  Memory
  ──────
  • read_memory          — retrieve persistent agent context
  • write_memory         — persist a fact across sessions

  File Editing  (Feature 4 — granular iterative editing)
  ────────────
  • read_file            — read a file with line numbers
  • list_dir             — list directory contents
  • replace_string       — targeted string replace (not whole-file overwrite)
  • write_file           — create / overwrite a file
  • delete_file          — remove a file
  • grep_file            — regex search within a file
  • grep_workspace       — regex search across workspace files

  Semantic Navigation  (Feature 6 — RAG / AST search)
  ───────────────────
  • grep_search          — fast pattern search across the workspace
  • semantic_search      — embedding / TF-IDF natural-language code search
  • reindex_codebase     — rebuild the semantic search index

  MCP  (Feature 2 — native MCP client)
  ───
  • mcp_call             — call a tool on a registered MCP server
  • mcp_list_tools       — list all tools from registered MCP servers

HITL / safety (Feature 3) is enforced transparently inside run_command.
"""

from __future__ import annotations

import structlog
import time
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from src.core.memory import MemoryManager
from src.core.hitl import HITLGate, HITLConfig, HITLDecision
from src.core.pty_shell import PTYShellManager
from src.core.file_editing_tools import FileEditingTools
from src.core.semantic_search import CodebaseNavigator
from src.core.mcp_client import MCPClient, MCPServerConfig
from src.core.policy import PolicyProfile, get_policy_profile
from src.tools.base import ToolExecutionContext
from src.tools.plugins import FilesystemPlugin, MCPPlugin, MemoryPlugin, ShellPlugin
from src.tools.registry import ToolRegistry, ToolResolutionError

if TYPE_CHECKING:
    from src.agents.base import ToolCall

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool schema definitions (injected into agent system prompts)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[Dict[str, Any]] = [
    {
        "name": "run_command",
        "description": (
            "Run a shell command in a PERSISTENT terminal session. "
            "Directory changes (cd), environment variables, and background "
            "processes persist across calls. Mutating commands require user approval."
        ),
        "parameters": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {"type": "number", "description": "Timeout in seconds (default: 120)"},
        },
        "required": ["command"],
    },
    {
        "name": "read_memory",
        "description": "Read all facts stored in the agent's persistent memory.",
        "parameters": {},
        "required": [],
    },
    {
        "name": "write_memory",
        "description": "Persist a fact to the agent's cross-session memory.",
        "parameters": {
            "fact": {"type": "string", "description": "Fact to remember"},
            "global_scope": {"type": "boolean", "description": "Make visible across all runs"},
        },
        "required": ["fact"],
    },
    # ── Granular file editing ──
    {
        "name": "read_file",
        "description": "Read a file from the workspace with line numbers.",
        "parameters": {
            "path": {"type": "string", "description": "File path (relative to workspace root)"},
            "start_line": {"type": "integer", "description": "First line (1-indexed, default: 1)"},
            "end_line": {"type": "integer", "description": "Last line (inclusive, optional)"},
        },
        "required": ["path"],
    },
    {
        "name": "list_dir",
        "description": "List the contents of a directory in the workspace.",
        "parameters": {
            "path": {"type": "string", "description": "Directory path (default: '.')"},
        },
        "required": [],
    },
    {
        "name": "replace_string",
        "description": (
            "Replace an exact occurrence of old_string with new_string in a file. "
            "Fails if old_string is not found or appears multiple times (safer than whole-file overwrite)."
        ),
        "parameters": {
            "path": {"type": "string", "description": "Target file path"},
            "old_string": {"type": "string", "description": "Exact text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "expect_count": {"type": "integer", "description": "Number of replacements expected (default: 1)"},
        },
        "required": ["path", "old_string", "new_string"],
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file in the workspace.",
        "parameters": {
            "path": {"type": "string", "description": "Target file path"},
            "content": {"type": "string", "description": "Full file content"},
            "overwrite": {"type": "boolean", "description": "Allow overwriting existing files (default: true)"},
        },
        "required": ["path", "content"],
    },
    {
        "name": "delete_file",
        "description": "Delete a file from the workspace.",
        "parameters": {
            "path": {"type": "string", "description": "File path to delete"},
            "require_confirmation": {"type": "boolean", "description": "Ask for confirmation first (default: false)"},
        },
        "required": ["path"],
    },
    {
        "name": "grep_file",
        "description": "Search for a regex pattern inside a single file.",
        "parameters": {
            "path": {"type": "string", "description": "File path"},
            "pattern": {"type": "string", "description": "Regex pattern"},
            "case_sensitive": {"type": "boolean", "description": "Default: true"},
            "max_results": {"type": "integer", "description": "Max matching lines (default: 100)"},
        },
        "required": ["path", "pattern"],
    },
    {
        "name": "grep_workspace",
        "description": "Search for a regex pattern across all workspace source files.",
        "parameters": {
            "pattern": {"type": "string", "description": "Regex pattern"},
            "glob": {"type": "string", "description": "File glob filter, e.g. '**/*.py' (default)"},
            "case_sensitive": {"type": "boolean", "description": "Default: true"},
            "max_results": {"type": "integer", "description": "Max total matches (default: 200)"},
        },
        "required": ["pattern"],
    },
    # ── Semantic navigation ──
    {
        "name": "grep_search",
        "description": (
            "Fast pattern-based search across all workspace source files. "
            "Returns matching lines with surrounding context."
        ),
        "parameters": {
            "pattern": {"type": "string", "description": "Regex or plain-text pattern"},
            "glob": {"type": "string", "description": "File type filter (e.g. '**/*.ts')"},
            "max_results": {"type": "integer", "description": "Max results (default: 30)"},
        },
        "required": ["pattern"],
    },
    {
        "name": "semantic_search",
        "description": (
            "Natural-language code search using TF-IDF / embedding similarity. "
            "Useful for finding 'where are HTTP errors handled' or 'all database calls'."
        ),
        "parameters": {
            "query": {"type": "string", "description": "Natural-language search query"},
            "top_k": {"type": "integer", "description": "Number of results (default: 8)"},
        },
        "required": ["query"],
    },
    {
        "name": "reindex_codebase",
        "description": "Rebuild the semantic search index after major file changes.",
        "parameters": {},
        "required": [],
    },
    # ── MCP ──
    {
        "name": "mcp_list_tools",
        "description": "List all tools available from connected MCP servers.",
        "parameters": {},
        "required": [],
    },
    {
        "name": "mcp_call",
        "description": "Call a tool provided by a connected MCP server.",
        "parameters": {
            "tool_name": {"type": "string", "description": "Name of the MCP tool"},
            "arguments": {"type": "object", "description": "Arguments for the tool"},
        },
        "required": ["tool_name"],
    },
]


def get_tools_system_prompt() -> str:
    """Format all built-in tool schemas for injection into agent system prompts."""
    import json
    lines = ["## Available Tools\n",
             "Call a tool by emitting a JSON block like:\n",
             '```json\n{"tool_name": "<name>", "arguments": {...}}\n```\n',
             "### Tool Catalog\n"]
    for schema in TOOL_SCHEMAS:
        params = json.dumps(schema.get("parameters", {}), indent=2)
        req = schema.get("required", [])
        lines.append(f"**{schema['name']}** — {schema['description']}")
        lines.append(f"  Parameters: {params}")
        if req:
            lines.append(f"  Required: {req}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """
    Central tool dispatcher for all agent tool calls.

    Instantiated once per run and shared across planner / coder / reviewer.
    Each new run_id gets its own PTY session so shell state is isolated
    between runs but persistent within a run.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        workspace_root: str,
        hitl_config: HITLConfig | None = None,
        mcp_client: MCPClient | None = None,
        policy_profile: PolicyProfile | None = None,
        run_id: str = "default",
    ) -> None:
        self.memory = memory_manager
        self.workspace_root = workspace_root
        self._run_id = run_id

        # Feature 1: Persistent PTY shell
        self._pty_manager = PTYShellManager(Path(workspace_root))

        # Feature 3: HITL security gate
        self._hitl = HITLGate(hitl_config or HITLConfig())

        # Feature 4: Granular file editing
        self._files = FileEditingTools(Path(workspace_root))

        # Feature 6: Semantic codebase navigation
        self._nav = CodebaseNavigator(Path(workspace_root))

        # Feature 2: MCP client (optional)
        self._mcp = mcp_client or MCPClient()
        self._policy_profile = policy_profile or get_policy_profile("balanced")
        self._registry = ToolRegistry()
        self._register_builtin_plugins()

    def _register_builtin_plugins(self) -> None:
        """Register all built-in tools as explicit plugins."""
        plugins = [
            ShellPlugin(self._execute_run_command),
            MemoryPlugin("read_memory", self._execute_read_memory),
            MemoryPlugin("write_memory", self._execute_write_memory, required_args=["fact"]),
            FilesystemPlugin("read_file", "filesystem-read", self._execute_read_file, required_args=["path"]),
            FilesystemPlugin("list_dir", "filesystem-list", self._execute_list_dir),
            FilesystemPlugin(
                "replace_string",
                "filesystem-edit",
                self._execute_replace_string,
                required_args=["path", "old_string", "new_string"],
            ),
            FilesystemPlugin("write_file", "filesystem-write", self._execute_write_file, required_args=["path", "content"]),
            FilesystemPlugin("delete_file", "filesystem-delete", self._execute_delete_file, required_args=["path"]),
            FilesystemPlugin("grep_file", "filesystem-search", self._execute_grep_file, required_args=["path", "pattern"]),
            FilesystemPlugin("grep_workspace", "filesystem-search", self._execute_grep_workspace, required_args=["pattern"]),
            FilesystemPlugin("grep_search", "semantic-search", self._execute_grep_search, required_args=["pattern"]),
            FilesystemPlugin("semantic_search", "semantic-search", self._execute_semantic_search, required_args=["query"]),
            FilesystemPlugin("reindex_codebase", "semantic-search", self._execute_reindex_codebase),
            MCPPlugin("mcp_list_tools", self._execute_mcp_list_tools),
            MCPPlugin("mcp_call", self._execute_mcp_call, required_args=["tool_name"]),
        ]
        self._registry.register_many(plugins)

    @property
    def tool_allowlist(self) -> set[str]:
        """Return registered tool names."""
        return self._registry.allowlist()

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    def execute_call(self, call: "ToolCall") -> str:  # type: ignore[override]
        """Execute a tool call and return its string result."""
        logger.info("tool_call", tool=call.tool_name)
        args = call.arguments or {}
        started = time.perf_counter()

        def _record(success: bool, error_message: str = "") -> None:
            try:
                duration_ms = (time.perf_counter() - started) * 1000.0
                self.memory.record_tool_signal(
                    task_description=str(args.get("task_description", "")) or f"tool:{call.tool_name}",
                    tool_name=call.tool_name,
                    success=success,
                    duration_ms=duration_ms,
                    error_message=error_message,
                )
            except Exception:
                # Tool telemetry should never block execution.
                pass

        try:
            plugin = self._registry.resolve(call.tool_name)
        except ToolResolutionError:
            _record(False, "unknown_tool")
            return f"Error: Unknown tool '{call.tool_name}'."

        valid, validation_error = plugin.validate(args)
        if not valid:
            _record(False, validation_error or "validation_error")
            return f"Error: {validation_error or 'Invalid arguments'}"

        profile_allowed, profile_reason = self._policy_profile.validate_tool_call(call.tool_name)
        if not profile_allowed:
            _record(False, profile_reason or "policy_blocked")
            return f"Error: Policy blocked tool '{call.tool_name}': {profile_reason}"

        if not self._policy_profile.file_write_permissions and call.tool_name in {
            "write_file",
            "replace_string",
            "delete_file",
        }:
            _record(False, "read_only_policy")
            return (
                f"Error: Policy blocked tool '{call.tool_name}': "
                f"profile '{self._policy_profile.name}' is read-only"
            )

        context = ToolExecutionContext(
            run_id=self._run_id,
            workspace_root=self.workspace_root,
            metadata={"policy_profile": self._policy_profile.name},
        )
        allowed, policy_error = plugin.policy_check(context, args)
        if not allowed:
            _record(False, policy_error or "policy_check_failed")
            return f"Error: Policy blocked tool '{call.tool_name}': {policy_error}"

        try:
            result = plugin.execute(args)
            if not isinstance(result, str):
                _record(False, "contract_violation_non_string")
                return (
                    f"Error: contract_violation for tool '{call.tool_name}': "
                    "plugin execute() must return a string"
                )
            _record(True)
            return result
        except KeyError as e:
            _record(False, f"missing_required_argument:{e}")
            return f"Error: Missing required argument {e} for tool '{call.tool_name}'."
        except Exception as e:
            logger.error("tool_error", tool=call.tool_name, error=str(e))
            _record(False, str(e))
            return f"Error executing '{call.tool_name}': {e}"

    def _execute_read_file(self, arguments: Dict[str, Any]) -> str:
        return self._files.read_file(
            arguments["path"],
            arguments.get("start_line", 1),
            arguments.get("end_line"),
        )

    def _execute_list_dir(self, arguments: Dict[str, Any]) -> str:
        return self._files.list_dir(arguments.get("path", "."))

    def _execute_replace_string(self, arguments: Dict[str, Any]) -> str:
        return self._files.replace_string(
            arguments["path"],
            arguments["old_string"],
            arguments["new_string"],
            arguments.get("expect_count", 1),
        )

    def _execute_write_file(self, arguments: Dict[str, Any]) -> str:
        return self._files.write_file(
            arguments["path"],
            arguments["content"],
            arguments.get("overwrite", True),
        )

    def _execute_delete_file(self, arguments: Dict[str, Any]) -> str:
        return self._files.delete_file(
            arguments["path"],
            arguments.get("require_confirmation", False),
        )

    def _execute_grep_file(self, arguments: Dict[str, Any]) -> str:
        return self._files.grep_file(
            arguments["path"],
            arguments["pattern"],
            arguments.get("case_sensitive", True),
            arguments.get("max_results", 100),
        )

    def _execute_grep_workspace(self, arguments: Dict[str, Any]) -> str:
        return self._files.grep_workspace(
            arguments["pattern"],
            arguments.get("glob", "**/*.py"),
            arguments.get("case_sensitive", True),
            arguments.get("max_results", 200),
        )

    def _execute_grep_search(self, arguments: Dict[str, Any]) -> str:
        return self._nav.grep_search(
            arguments["pattern"],
            arguments.get("glob"),
            arguments.get("case_sensitive", True),
            arguments.get("max_results", 30),
        )

    def _execute_semantic_search(self, arguments: Dict[str, Any]) -> str:
        return self._nav.semantic_search(
            arguments["query"],
            arguments.get("top_k", 8),
        )

    def _execute_reindex_codebase(self, arguments: Dict[str, Any]) -> str:
        _ = arguments
        return self._nav.reindex()

    # ------------------------------------------------------------------
    # Shell (Feature 1 + Feature 3)
    # ------------------------------------------------------------------

    def _execute_run_command(self, arguments: Dict[str, Any]) -> str:
        command = arguments.get("command", "").strip()
        if not command:
            return "Error: missing 'command' argument."

        # Feature 3: HITL permission check
        result = self._hitl.check(command)
        if result.decision not in (
            HITLDecision.APPROVED,
            HITLDecision.USER_APPROVED,
        ):
            return (
                f"Command blocked: {result.reason}\n"
                f"Command was: {command}"
            )

        # Feature 1: Run in persistent PTY session
        try:
            session = self._pty_manager.get_or_create(self._run_id)
            timeout = float(arguments.get("timeout", 120))
            output = session.run(command, timeout=timeout)
            return output if output else "Command executed successfully with no output."
        except Exception as pty_err:
            # Graceful fallback to subprocess if pexpect unavailable
            logger.warning("pty_fallback", reason=str(pty_err))
            return self._subprocess_fallback(command, arguments)

    def _subprocess_fallback(self, command: str, arguments: Dict[str, Any]) -> str:
        """Fallback to subprocess.run when PTY is unavailable."""
        import subprocess
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=float(arguments.get("timeout", 60)),
            )
            stdout = r.stdout.strip()
            stderr = r.stderr.strip()
            if r.returncode != 0:
                return (
                    f"Command failed (exit {r.returncode}).\n"
                    f"STDOUT: {stdout}\nSTDERR: {stderr}"
                )
            return stdout or "Command executed successfully with no output."
        except subprocess.TimeoutExpired:
            return "Error: Command timed out."
        except Exception as e:
            return f"Error executing command: {e}"

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def _execute_read_memory(self, arguments: Dict[str, Any]) -> str:
        return self.memory.get_all_context()

    def _execute_write_memory(self, arguments: Dict[str, Any]) -> str:
        fact = arguments.get("fact")
        if not fact:
            return "Error: missing 'fact' argument."
        global_scope = arguments.get("global_scope", False)
        return self.memory.add_fact(fact, global_scope)

    # ------------------------------------------------------------------
    # MCP (Feature 2)
    # ------------------------------------------------------------------

    def _execute_mcp_list_tools(self, arguments: Dict[str, Any] | None = None) -> str:
        _ = arguments
        schemas = self._mcp.get_all_tool_schemas()
        if not schemas:
            return "No MCP servers connected or no tools available."
        lines = [f"MCP Tools ({len(schemas)} available):\n"]
        for s in schemas:
            lines.append(f"  • {s.name} [{s.server_id}] — {s.description[:80]}")
        return "\n".join(lines)

    def _execute_mcp_call(self, arguments: Dict[str, Any]) -> str:
        tool_name = arguments.get("tool_name")
        if not tool_name:
            return "Error: missing 'tool_name' argument."
        args = arguments.get("arguments", {})
        result = self._mcp.call_tool(tool_name, args)
        if result.success:
            return str(result.content)
        return f"MCP tool error: {result.error}"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release all held resources (PTY sessions, MCP connections)."""
        self._pty_manager.close_all()
        self._mcp.close_all()

    def register_mcp_server(self, config: MCPServerConfig) -> str:
        """Register and connect to an MCP server at runtime."""
        try:
            self._mcp.register_server(config)
            tools = self._mcp.fetch_tools(config.server_id)
            return (
                f"Connected to MCP server '{config.server_id}' "
                f"({len(tools)} tools available)."
            )
        except Exception as e:
            return f"Failed to connect to MCP server '{config.server_id}': {e}"
