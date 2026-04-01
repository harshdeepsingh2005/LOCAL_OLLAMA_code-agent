"""
Native MCP Client Module

Implements a native Model Context Protocol (MCP) client that connects to
MCP servers via JSON-RPC 2.0 (stdio transport), dynamically fetches
tool schemas, and exposes them as ToolCall-compatible definitions that
the orchestrator can inject directly into the agent's system prompt.

Design Decisions:
- Uses stdio transport (most compatible, no networking required)
- Caches tool schemas per server to avoid re-fetching every iteration
- Async-first with a sync shim for use inside the synchronous executor
- Per-server isolation: one subprocess per server
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class MCPToolSchema:
    """Schema of a single tool exposed by an MCP server."""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    server_id: str = ""

    def to_prompt_snippet(self) -> str:
        """Format this tool for injection into a system prompt."""
        params_str = json.dumps(self.parameters, indent=2)
        return (
            f"Tool: {self.name} (from MCP server '{self.server_id}')\n"
            f"Description: {self.description}\n"
            f"Parameters:\n{params_str}"
        )


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    server_id: str
    command: str                       # e.g. "npx"
    args: list[str] = field(default_factory=list)  # e.g. ["-y", "@modelcontextprotocol/server-github"]
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class MCPCallResult:
    """Result of an MCP tool invocation."""
    tool_name: str
    server_id: str
    success: bool
    content: Any = None
    error: str = ""


# ---------------------------------------------------------------------------
# Low-level JSON-RPC over stdio
# ---------------------------------------------------------------------------

class MCPStdioTransport:
    """
    Manages a stdio pipe to a running MCP server process.

    MCP over stdio sends line-delimited JSON-RPC 2.0 requests to the
    server's stdin and reads responses from stdout.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._process: subprocess.Popen | None = None  # type: ignore
        self._req_id = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Launch the MCP server subprocess."""
        import os
        env = os.environ.copy()
        env.update(self._config.env)

        self._process = subprocess.Popen(
            [self._config.command] + self._config.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )

    def stop(self) -> None:
        """Terminate the MCP server subprocess."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    def is_alive(self) -> bool:
        """Return True if the subprocess is still running."""
        if not self._process:
            return False
        return self._process.poll() is None

    def send_request(self, method: str, params: dict | None = None) -> dict:
        """
        Send a JSON-RPC request and return the parsed response.

        Raises:
            RuntimeError: If the server is not running or returns an error
        """
        if not self._process or not self.is_alive():
            raise RuntimeError(f"MCP server '{self._config.server_id}' is not running")

        with self._lock:
            self._req_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._req_id,
                "method": method,
                "params": params or {},
            }
            line = json.dumps(request) + "\n"

            assert self._process.stdin is not None
            self._process.stdin.write(line)
            self._process.stdin.flush()

            assert self._process.stdout is not None
            response_line = self._process.stdout.readline()
            if not response_line:
                raise RuntimeError("MCP server returned empty response")

            response = json.loads(response_line)

        if "error" in response:
            err = response["error"]
            raise RuntimeError(f"MCP error {err.get('code')}: {err.get('message')}")

        return response.get("result", {})


# ---------------------------------------------------------------------------
# High-level MCP client
# ---------------------------------------------------------------------------

class MCPClient:
    """
    Native MCP client for the orchestrator.

    Manages connections to one or more MCP servers, fetches their tool
    schemas, and dispatches tool calls. The orchestrator uses this to:

    1. ``get_all_tool_schemas()``   → inject into the LLM system prompt
    2. ``call_tool(name, args)``    → execute a tool when the LLM requests it
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPStdioTransport] = {}
        self._schema_cache: dict[str, list[MCPToolSchema]] = {}

    # ------------------------------------------------------------------
    # Server management
    # ------------------------------------------------------------------

    def register_server(self, config: MCPServerConfig) -> None:
        """
        Register and start an MCP server.

        Args:
            config: Server configuration
        """
        transport = MCPStdioTransport(config)
        transport.start()

        # Initialize the MCP session
        try:
            transport.send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "local-coding-agents", "version": "0.1.0"},
            })
        except Exception:
            # Some servers are lenient about initialize
            pass

        self._servers[config.server_id] = transport

    def remove_server(self, server_id: str) -> None:
        """Stop and remove an MCP server."""
        if server_id in self._servers:
            self._servers[server_id].stop()
            del self._servers[server_id]
        self._schema_cache.pop(server_id, None)

    def close_all(self) -> None:
        """Stop all registered servers."""
        for transport in self._servers.values():
            try:
                transport.stop()
            except Exception:
                pass
        self._servers.clear()
        self._schema_cache.clear()

    # ------------------------------------------------------------------
    # Tool schema discovery
    # ------------------------------------------------------------------

    def fetch_tools(self, server_id: str, force_refresh: bool = False) -> list[MCPToolSchema]:
        """
        Fetch tool schemas from a specific server.

        Results are cached so repeated calls don't spawn extra RPC round-trips.

        Args:
            server_id: ID of the registered server
            force_refresh: Bypass the cache

        Returns:
            List of tool schemas
        """
        if server_id not in self._servers:
            raise ValueError(f"Unknown MCP server: {server_id}")

        if not force_refresh and server_id in self._schema_cache:
            return self._schema_cache[server_id]

        transport = self._servers[server_id]
        result = transport.send_request("tools/list")
        raw_tools: list[dict] = result.get("tools", [])

        schemas = [
            MCPToolSchema(
                name=t.get("name", ""),
                description=t.get("description", ""),
                parameters=t.get("inputSchema", {}),
                server_id=server_id,
            )
            for t in raw_tools
        ]

        self._schema_cache[server_id] = schemas
        return schemas

    def get_all_tool_schemas(self) -> list[MCPToolSchema]:
        """
        Fetch and merge tool schemas from all registered servers.

        Returns:
            Flat list of all available MCP tool schemas
        """
        all_schemas: list[MCPToolSchema] = []
        for server_id in self._servers:
            try:
                schemas = self.fetch_tools(server_id)
                all_schemas.extend(schemas)
            except Exception:
                pass  # Server may be temporarily unavailable
        return all_schemas

    def build_system_prompt_section(self) -> str:
        """
        Build the MCP tool section to inject into agent system prompts.

        Returns:
            Formatted string listing all available MCP tools
        """
        schemas = self.get_all_tool_schemas()
        if not schemas:
            return ""

        lines = ["## Available MCP Tools\n"]
        for schema in schemas:
            lines.append(schema.to_prompt_snippet())
            lines.append("")

        lines.append(
            "To use an MCP tool, emit a tool_call JSON block with "
            "\"tool_name\": \"<tool>\", \"arguments\": {...}."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPCallResult:
        """
        Invoke a named tool across all registered servers.

        Searches servers in registration order, calls the first match.

        Args:
            tool_name: Name of the tool
            arguments: Arguments to pass

        Returns:
            MCPCallResult with content or error
        """
        # Find which server owns this tool
        for server_id, transport in self._servers.items():
            schemas = self._schema_cache.get(server_id, [])
            if any(s.name == tool_name for s in schemas):
                try:
                    result = transport.send_request("tools/call", {
                        "name": tool_name,
                        "arguments": arguments,
                    })
                    # MCP returns content as a list of content blocks
                    content_blocks: list[dict] = result.get("content", [])
                    text = "\n".join(
                        b.get("text", str(b))
                        for b in content_blocks
                        if isinstance(b, dict)
                    )
                    return MCPCallResult(
                        tool_name=tool_name,
                        server_id=server_id,
                        success=True,
                        content=text or result,
                    )
                except Exception as e:
                    return MCPCallResult(
                        tool_name=tool_name,
                        server_id=server_id,
                        success=False,
                        error=str(e),
                    )

        return MCPCallResult(
            tool_name=tool_name,
            server_id="",
            success=False,
            error=f"No MCP server provides tool: {tool_name}",
        )

    @property
    def registered_server_ids(self) -> list[str]:
        """List of registered server IDs."""
        return list(self._servers.keys())
