"""
Persistent Memory Module

Provides functionality to store and retrieve facts, preferences, 
and context across sessions, similar to Claude Mem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import structlog
from datetime import datetime, timezone

logger = structlog.get_logger(__name__)

class ActionStatus:
    SUCCESS = "success"
    ERROR = "error"

class MemoryManager:
    """
    Manages persistent memory storage for the agent.
    Memory is stored locally to maintain privacy.
    """
    
    def __init__(self, workspace_root: Path):
        self._workspace_root = workspace_root
        # Store global memory in home folder, project specific in workspace
        self._global_memory_file = Path.home() / ".local_coding_agent" / "memory.json"
        self._project_memory_file = self._workspace_root / ".agent_memory.json"
        self._evergreen_candidates = [
            self._workspace_root / "core_context.md",
            self._workspace_root / ".agent_evergreen.md",
            self._workspace_root / "docs" / "agent-evergreen-context.md",
            self._workspace_root / "docs" / "evergreen-context.md",
        ]
        
        self._ensure_files()
        
    def _ensure_files(self) -> None:
        """Ensure memory files exist."""
        for file in [self._global_memory_file, self._project_memory_file]:
            if not file.exists():
                file.parent.mkdir(parents=True, exist_ok=True)
                with open(file, 'w') as f:
                    json.dump({"facts": [], "preferences": {}}, f)
                    
    def _load_memory(self, file_path: Path) -> Dict[str, Any]:
        """Load memory from a specific file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load memory from {file_path}: {e}")
            return {"facts": [], "preferences": {}}
            
    def _save_memory(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Save memory to a specific file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory to {file_path}: {e}")

    def get_all_context(self) -> str:
        """Format all memory context as a string for the system prompt."""
        global_mem = self._load_memory(self._global_memory_file)
        project_mem = self._load_memory(self._project_memory_file)
        evergreen = self._load_evergreen_context(max_chars=4000)
        
        parts = []
        if global_mem.get("facts"):
            parts.append("## Global Memory Facts:\n" + "\n".join(f"- {f}" for f in global_mem["facts"]))
        if list(global_mem.get("preferences", {}).keys()):
            parts.append("## User Preferences:\n" + "\n".join(f"- {k}: {v}" for k, v in global_mem["preferences"].items()))
            
        if project_mem.get("facts"):
            parts.append("## Project-Specific Details:\n" + "\n".join(f"- {f}" for f in project_mem["facts"]))

        if evergreen:
            parts.append("## Evergreen Project Context:\n" + evergreen)
            
        return "\n\n".join(parts) if parts else "No persistent memory found."

    def _load_evergreen_context(self, max_chars: int = 4000) -> str:
        """Load evergreen context document if present in the workspace."""
        for path in self._evergreen_candidates:
            try:
                if path.exists() and path.is_file():
                    content = path.read_text(encoding="utf-8").strip()
                    if len(content) <= max_chars:
                        return content
                    return content[: max_chars - 20].rstrip() + "\n... [truncated]"
            except Exception as e:
                logger.warning(f"Failed to read evergreen context from {path}: {e}")
        return ""

    def add_fact(self, fact: str, global_scope: bool = False) -> str:
        """Add a persistent fact. Returns status string."""
        target_file = self._global_memory_file if global_scope else self._project_memory_file
        data = self._load_memory(target_file)
        
        if fact not in data["facts"]:
            data["facts"].append(fact)
            self._save_memory(target_file, data)
            return f"Successfully saved fact to {'global' if global_scope else 'project'} memory."
        return "Fact already exists in memory."
        
    def update_preference(self, key: str, value: Any) -> str:
        """Update a global user preference."""
        data = self._load_memory(self._global_memory_file)
        data["preferences"][key] = value
        self._save_memory(self._global_memory_file, data)
        return f"Successfully updated preference: {key}"

    def remember_decision(self, decision: str) -> str:
        """Store an architecture or implementation decision in project memory."""
        data = self._load_memory(self._project_memory_file)
        decisions = data.setdefault("decisions", [])
        entry = {
            "decision": decision,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        decisions.append(entry)
        # Keep only latest 50 decisions to bound prompt size
        data["decisions"] = decisions[-50:]
        self._save_memory(self._project_memory_file, data)
        return "Successfully stored project decision."

    def remember_pattern(self, name: str, pattern: str) -> str:
        """Store a reusable successful pattern in project memory."""
        data = self._load_memory(self._project_memory_file)
        patterns = data.setdefault("patterns", {})
        patterns[name] = {
            "pattern": pattern,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_memory(self._project_memory_file, data)
        return f"Successfully stored reusable pattern: {name}"
