"""Deterministic policy profiles for platform execution modes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyProfile:
    """Execution policy profile."""

    name: str
    allowed_tools: frozenset[str]
    max_tool_steps: int
    fallback_allowed: bool
    file_write_permissions: bool
    llm_temperature: float
    llm_top_p: float

    def validate_tool_call(self, tool_name: str) -> tuple[bool, str | None]:
        if tool_name not in self.allowed_tools:
            return False, f"Tool '{tool_name}' is not allowed in profile '{self.name}'"
        return True, None


_ALL_TOOLS = frozenset(
    {
        "run_command",
        "read_memory",
        "write_memory",
        "read_file",
        "list_dir",
        "replace_string",
        "write_file",
        "delete_file",
        "grep_file",
        "grep_workspace",
        "grep_search",
        "semantic_search",
        "reindex_codebase",
        "mcp_list_tools",
        "mcp_call",
    }
)

_STRICT_TOOLS = frozenset(
    {
        "read_memory",
        "write_memory",
        "read_file",
        "list_dir",
        "grep_file",
        "grep_workspace",
        "grep_search",
        "semantic_search",
        "reindex_codebase",
        "mcp_list_tools",
    }
)

_BALANCED_TOOLS = frozenset(
    {
        "run_command",
        "read_memory",
        "write_memory",
        "read_file",
        "list_dir",
        "replace_string",
        "write_file",
        "grep_file",
        "grep_workspace",
        "grep_search",
        "semantic_search",
        "reindex_codebase",
        "mcp_list_tools",
        "mcp_call",
    }
)


POLICY_PROFILES: dict[str, PolicyProfile] = {
    "strict": PolicyProfile(
        name="strict",
        allowed_tools=_STRICT_TOOLS,
        max_tool_steps=1,
        fallback_allowed=False,
        file_write_permissions=False,
        llm_temperature=0.0,
        llm_top_p=1.0,
    ),
    "balanced": PolicyProfile(
        name="balanced",
        allowed_tools=_BALANCED_TOOLS,
        max_tool_steps=3,
        fallback_allowed=True,
        file_write_permissions=True,
        llm_temperature=0.1,
        llm_top_p=0.9,
    ),
    "permissive": PolicyProfile(
        name="permissive",
        allowed_tools=_ALL_TOOLS,
        max_tool_steps=5,
        fallback_allowed=True,
        file_write_permissions=True,
        llm_temperature=0.2,
        llm_top_p=0.95,
    ),
}


def get_policy_profile(name: str = "balanced") -> PolicyProfile:
    """Resolve policy profile by name with deterministic fallback to balanced."""
    return POLICY_PROFILES.get(name, POLICY_PROFILES["balanced"])
