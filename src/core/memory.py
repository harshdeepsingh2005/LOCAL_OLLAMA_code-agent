"""
Persistent Memory Module

Provides functionality to store and retrieve facts, preferences, 
and context across sessions, similar to Claude Mem.
"""

from __future__ import annotations

import json
import math
import hashlib
import re
from pathlib import Path
from typing import Any
import structlog
from datetime import datetime, timezone
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

class ActionStatus:
    SUCCESS = "success"
    ERROR = "error"


class FailurePattern(BaseModel):
    pattern_id: str
    category: str
    summary: str
    root_cause: str
    resolution_hint: str
    tags: list[str] = Field(default_factory=list)
    frequency: int = 1
    confidence: float = 0.5
    created_at: str
    last_used_at: str


class SuccessPattern(BaseModel):
    pattern_id: str
    pattern_type: str
    summary: str
    reusable_snippet: str
    tags: list[str] = Field(default_factory=list)
    success_rate: float = 1.0
    frequency: int = 1
    confidence: float = 0.6
    created_at: str
    last_used_at: str


ERROR_CATEGORY_MAP: dict[str, str] = {
    "filenotfounderror": "missing_file",
    "importerror": "missing_import",
    "modulenotfounderror": "missing_import",
    "keyerror": "dict_key_missing",
    "typeerror": "type_mismatch",
    "valueerror": "value_error",
}

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
                    json.dump(self._default_memory(), f, indent=2)
                    
    def _default_memory(self) -> dict[str, Any]:
        return {
            "facts": [],
            "preferences": {},
            "failure_patterns": [],
            "success_patterns": [],
            "task_outcomes": [],
            "semantic_graph": {
                "nodes": [],
                "edges": [],
            },
        }

    def _ensure_schema(self, data: dict[str, Any]) -> dict[str, Any]:
        default = self._default_memory()
        for key, value in default.items():
            data.setdefault(key, value if not isinstance(value, list) else list(value))

        if not isinstance(data.get("semantic_graph"), dict):
            data["semantic_graph"] = {"nodes": [], "edges": []}
        graph = data["semantic_graph"]
        if not isinstance(graph.get("nodes"), list):
            graph["nodes"] = []
        if not isinstance(graph.get("edges"), list):
            graph["edges"] = []
        return data

    def _load_memory(self, file_path: Path) -> dict[str, Any]:
        """Load memory from a specific file."""
        try:
            with open(file_path, 'r') as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    return self._ensure_schema(loaded)
                return self._default_memory()
        except Exception as e:
            logger.error(f"Failed to load memory from {file_path}: {e}")
            return self._default_memory()
            
    def _save_memory(self, file_path: Path, data: dict[str, Any]) -> None:
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

        failure_count = len(project_mem.get("failure_patterns", []))
        success_count = len(project_mem.get("success_patterns", []))
        if failure_count or success_count:
            parts.append(
                "## Learning Summary:\n"
                f"- failure_patterns: {failure_count}\n"
                f"- success_patterns: {success_count}"
            )

        if evergreen:
            parts.append("## Evergreen Project Context:\n" + evergreen)
            
        return "\n\n".join(parts) if parts else "No persistent memory found."

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z_]{3,}", text.lower()))

    @staticmethod
    def _stable_hash(value: str) -> str:
        return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]

    def _text_vector(self, text: str, dims: int = 64) -> list[float]:
        vec = [0.0] * dims
        tokens = sorted(self._tokenize(text))
        for token in tokens:
            idx = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % dims
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 1e-9:
            return vec
        return [v / norm for v in vec]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def _summary_overlap(self, a: str, b: str) -> float:
        return self._jaccard(self._tokenize(a), self._tokenize(b))

    def _similarity_score(self, tags_a: list[str], tags_b: list[str], summary_a: str, summary_b: str) -> float:
        tag_jaccard = self._jaccard(set(tags_a), set(tags_b))
        summary_overlap = self._summary_overlap(summary_a, summary_b)
        return 0.7 * tag_jaccard + 0.3 * summary_overlap

    def _find_similar_pattern(
        self,
        existing_patterns: list[dict[str, Any]],
        new_kind: str,
        new_tags: list[str],
        new_summary: str,
        kind_key: str,
        threshold: float = 0.72,
    ) -> int | None:
        best_idx: int | None = None
        best_score = 0.0
        new_primary = new_tags[0] if new_tags else ""

        for idx, pattern in enumerate(existing_patterns):
            if pattern.get(kind_key) != new_kind:
                continue
            existing_tags = [str(t) for t in pattern.get("tags", [])]
            existing_primary = existing_tags[0] if existing_tags else ""
            if new_primary and existing_primary and new_primary != existing_primary:
                continue
            score = self._similarity_score(existing_tags, new_tags, str(pattern.get("summary", "")), new_summary)
            if score >= threshold and score > best_score:
                best_idx = idx
                best_score = score

        return best_idx

    @staticmethod
    def _normalize_confidence(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _frequency_norm(self, freq: int) -> float:
        return min(1.0, math.log1p(max(0, freq)) / math.log1p(20))

    def _recency_decay(self, last_used_at: str, half_life_days: float) -> float:
        try:
            dt = datetime.fromisoformat(last_used_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
        except Exception:
            age_days = half_life_days
        return math.exp(-math.log(2) * age_days / half_life_days)

    @staticmethod
    def _normalize_error_category(error_message: str) -> str:
        text = error_message.lower()
        for key, category in ERROR_CATEGORY_MAP.items():
            if key in text:
                return category
        if "import" in text:
            return "missing_import"
        if "file" in text and ("not found" in text or "no such" in text):
            return "missing_file"
        if "schema" in text:
            return "schema_mismatch"
        return "unknown_error"

    @staticmethod
    def _canonical_failure_category(category: str) -> str:
        normalized = category.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "missing_import": "missing_import",
            "import_error": "missing_import",
            "modulenotfound": "missing_import",
            "module_not_found": "missing_import",
            "undefined_module": "missing_import",
            "missing_module": "missing_import",
            "missing_file": "missing_file",
            "file_not_found": "missing_file",
            "type_mismatch": "type_mismatch",
            "type_error": "type_mismatch",
        }
        return aliases.get(normalized, normalized or "unknown_error")

    @staticmethod
    def _canonical_pattern_type(pattern_type: str) -> str:
        normalized = pattern_type.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "api_endpoint_pattern": "api_endpoint_pattern",
            "endpoint_pattern": "api_endpoint_pattern",
            "validation_pattern": "validation_pattern",
            "schema_validation_pattern": "validation_pattern",
            "retry_pattern": "retry_pattern",
            "parser_hardening_pattern": "parser_hardening_pattern",
            "json_parser_hardening": "parser_hardening_pattern",
        }
        return aliases.get(normalized, normalized or "code_structure_pattern")

    def _upsert_graph_node(
        self,
        data: dict[str, Any],
        node_type: str,
        label: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        graph = data.setdefault("semantic_graph", {"nodes": [], "edges": []})
        nodes = graph.setdefault("nodes", [])
        normalized_label = " ".join(label.strip().split())[:280]
        node_id = f"{node_type}_{self._stable_hash(node_type + ':' + normalized_label.lower())}"

        existing = None
        for node in nodes:
            if str(node.get("id", "")) == node_id:
                existing = node
                break

        now = self._now_iso()
        base_tags = sorted(set(str(t) for t in (tags or []) if str(t).strip()))
        vector = self._text_vector(f"{normalized_label} {' '.join(base_tags)}")

        if existing is None:
            existing = {
                "id": node_id,
                "type": node_type,
                "label": normalized_label,
                "tags": base_tags,
                "vector": vector,
                "frequency": 1,
                "created_at": now,
                "last_used_at": now,
                "metadata": metadata or {},
            }
            nodes.append(existing)
        else:
            existing["frequency"] = int(existing.get("frequency", 1)) + 1
            existing["last_used_at"] = now
            existing["label"] = normalized_label
            merged_tags = set(str(t) for t in existing.get("tags", [])) | set(base_tags)
            existing["tags"] = sorted(t for t in merged_tags if t)
            existing["vector"] = self._text_vector(f"{normalized_label} {' '.join(existing['tags'])}")
            if metadata:
                merged_meta = dict(existing.get("metadata", {}))
                merged_meta.update(metadata)
                existing["metadata"] = merged_meta

        return node_id

    def _upsert_graph_edge(
        self,
        data: dict[str, Any],
        source_id: str,
        target_id: str,
        relation: str,
    ) -> None:
        graph = data.setdefault("semantic_graph", {"nodes": [], "edges": []})
        edges = graph.setdefault("edges", [])
        edge_id = f"edge_{self._stable_hash(source_id + ':' + relation + ':' + target_id)}"

        for edge in edges:
            if str(edge.get("id", "")) == edge_id:
                edge["weight"] = int(edge.get("weight", 1)) + 1
                edge["last_used_at"] = self._now_iso()
                return

        edges.append(
            {
                "id": edge_id,
                "source": source_id,
                "target": target_id,
                "relation": relation,
                "weight": 1,
                "created_at": self._now_iso(),
                "last_used_at": self._now_iso(),
            }
        )

    def _record_semantic_failure_link(
        self,
        data: dict[str, Any],
        task_description: str,
        error_message: str,
        category: str,
        pattern_id: str,
        tags: list[str],
    ) -> None:
        task_id = self._upsert_graph_node(
            data,
            "task",
            task_description[:240],
            tags=tags,
            metadata={"source": "record_failure_pattern"},
        )
        failure_id = self._upsert_graph_node(
            data,
            "failure",
            f"{category}: {error_message[:220]}",
            tags=[category, *tags],
            metadata={"pattern_id": pattern_id},
        )
        pattern_node_id = self._upsert_graph_node(
            data,
            "failure_pattern",
            pattern_id,
            tags=[category],
            metadata={"category": category},
        )

        self._upsert_graph_edge(data, task_id, failure_id, "encountered")
        self._upsert_graph_edge(data, failure_id, pattern_node_id, "mapped_to")

    def _record_semantic_success_link(
        self,
        data: dict[str, Any],
        task_description: str,
        file_path: str,
        pattern_type: str,
        pattern_id: str,
        tags: list[str],
    ) -> None:
        task_id = self._upsert_graph_node(
            data,
            "task",
            task_description[:240] or "unspecified task",
            tags=tags,
            metadata={"source": "record_success_patterns_from_changes"},
        )
        file_id = self._upsert_graph_node(
            data,
            "file",
            file_path or "unknown_file",
            tags=["file", *tags],
            metadata={"path": file_path},
        )
        pattern_node_id = self._upsert_graph_node(
            data,
            "success_pattern",
            pattern_id,
            tags=[pattern_type, *tags],
            metadata={"pattern_type": pattern_type},
        )

        self._upsert_graph_edge(data, task_id, file_id, "modified")
        self._upsert_graph_edge(data, file_id, pattern_node_id, "resolved_with")

    def _extract_failure_tags(self, task_description: str, error_message: str) -> list[str]:
        tags = self._tokenize(task_description) | self._tokenize(error_message)
        priority = [
            "fastapi",
            "router",
            "import",
            "schema",
            "json",
            "parser",
            "test",
            "api",
            "python",
        ]
        ordered = [p for p in priority if p in tags]
        remaining = sorted(tags - set(ordered))[:6]
        result = ordered + remaining
        return result[:8] if result else ["general"]

    def _extract_success_structure(self, change: dict[str, Any]) -> tuple[str, list[str], str, str]:
        content = str(change.get("new_content", ""))
        description = str(change.get("description", ""))
        file_path = str(change.get("file_path", ""))
        lower = content.lower()

        pattern_type = "code_structure_pattern"
        tags: list[str] = []
        hint_lines: list[str] = []

        if "@router." in lower or "apirouter" in lower:
            pattern_type = "api_endpoint_pattern"
            tags = ["api", "router", "fastapi"]
            hint_lines = [line.strip() for line in content.splitlines() if "@router." in line or "APIRouter" in line][:2]
        elif "basemodel" in lower or "pydantic" in lower or "validate" in lower:
            pattern_type = "validation_pattern"
            tags = ["validation", "schema", "pydantic"]
            hint_lines = [line.strip() for line in content.splitlines() if "BaseModel" in line or "validate" in line][:2]
        elif "tenacity" in lower or "retry" in lower or "backoff" in lower:
            pattern_type = "retry_pattern"
            tags = ["retry", "resilience"]
            hint_lines = [line.strip() for line in content.splitlines() if "retry" in line.lower() or "tenacity" in line.lower()][:2]
        elif "json.loads" in lower and ("try:" in lower or "except" in lower):
            pattern_type = "parser_hardening_pattern"
            tags = ["parser", "fallback", "json"]
            hint_lines = [line.strip() for line in content.splitlines() if "json.loads" in line or "except" in line][:2]
        else:
            tags = self._extract_failure_tags(file_path + " " + description, content[:300])[:5]
            hint_lines = [line.strip() for line in content.splitlines() if line.strip()][:2]

        summary = f"{pattern_type} in {file_path or 'unknown_file'}"
        reusable = "\n".join(hint_lines).strip() or (description[:200] if description else "Reusable structural pattern")
        reusable = reusable[:300]
        return pattern_type, tags, summary, reusable

    def record_failure_pattern(
        self,
        task_description: str,
        error_message: str,
        resolution_hint: str = "Verify preconditions and apply smallest safe fix.",
    ) -> str:
        data = self._load_memory(self._project_memory_file)
        category = self._normalize_error_category(error_message)
        category = self._canonical_failure_category(category)
        tags = self._extract_failure_tags(task_description, error_message)
        summary = f"{category}: {task_description[:120]}"
        root_cause = error_message[:240]

        patterns: list[dict[str, Any]] = list(data.get("failure_patterns", []))
        similar_idx = self._find_similar_pattern(
            existing_patterns=patterns,
            new_kind=category,
            new_tags=tags,
            new_summary=summary,
            kind_key="category",
        )

        now = self._now_iso()
        pattern_id = ""
        if similar_idx is not None:
            item = patterns[similar_idx]
            item["frequency"] = int(item.get("frequency", 1)) + 1
            item["last_used_at"] = now
            item["confidence"] = self._normalize_confidence(float(item.get("confidence", 0.5)) + 0.02)
            patterns[similar_idx] = item
            pattern_id = str(item.get("pattern_id", ""))
        else:
            pattern_id = "fp_" + hashlib.sha1(f"{category}:{summary}".encode("utf-8")).hexdigest()[:12]
            pattern = FailurePattern(
                pattern_id=pattern_id,
                category=category,
                summary=summary,
                root_cause=root_cause,
                resolution_hint=resolution_hint,
                tags=tags,
                frequency=1,
                confidence=0.5,
                created_at=now,
                last_used_at=now,
            )
            patterns.append(pattern.model_dump())

        self._record_semantic_failure_link(
            data=data,
            task_description=task_description,
            error_message=error_message,
            category=category,
            pattern_id=pattern_id or "fp_unknown",
            tags=tags,
        )

        data["failure_patterns"] = patterns[-200:]
        data["semantic_graph"]["nodes"] = data["semantic_graph"].get("nodes", [])[-600:]
        data["semantic_graph"]["edges"] = data["semantic_graph"].get("edges", [])[-1200:]
        self._save_memory(self._project_memory_file, data)
        return ActionStatus.SUCCESS

    def record_success_patterns_from_changes(self, changes: list[dict[str, Any]], task_description: str = "") -> str:
        data = self._load_memory(self._project_memory_file)
        patterns: list[dict[str, Any]] = list(data.get("success_patterns", []))
        now = self._now_iso()

        for change in changes:
            pattern_type, tags, summary, reusable = self._extract_success_structure(change)
            pattern_type = self._canonical_pattern_type(pattern_type)
            if task_description:
                tags = list(dict.fromkeys(tags + self._extract_failure_tags(task_description, "")[:2]))

            similar_idx = self._find_similar_pattern(
                existing_patterns=patterns,
                new_kind=pattern_type,
                new_tags=tags,
                new_summary=summary,
                kind_key="pattern_type",
            )

            if similar_idx is not None:
                item = patterns[similar_idx]
                item["frequency"] = int(item.get("frequency", 1)) + 1
                freq = int(item["frequency"])
                old_rate = float(item.get("success_rate", 1.0))
                item["success_rate"] = ((old_rate * (freq - 1)) + 1.0) / freq
                item["last_used_at"] = now
                item["confidence"] = self._normalize_confidence(float(item.get("confidence", 0.6)) + 0.03)
                patterns[similar_idx] = item
                pattern_id = str(item.get("pattern_id", "sp_unknown"))
            else:
                pattern_id = "sp_" + hashlib.sha1(f"{pattern_type}:{summary}".encode("utf-8")).hexdigest()[:12]
                pattern = SuccessPattern(
                    pattern_id=pattern_id,
                    pattern_type=pattern_type,
                    summary=summary,
                    reusable_snippet=reusable,
                    tags=tags,
                    success_rate=1.0,
                    frequency=1,
                    confidence=0.6,
                    created_at=now,
                    last_used_at=now,
                )
                patterns.append(pattern.model_dump())

            self._record_semantic_success_link(
                data=data,
                task_description=task_description,
                file_path=str(change.get("file_path", "")),
                pattern_type=pattern_type,
                pattern_id=pattern_id,
                tags=tags,
            )

        data["success_patterns"] = patterns[-200:]
        data["semantic_graph"]["nodes"] = data["semantic_graph"].get("nodes", [])[-600:]
        data["semantic_graph"]["edges"] = data["semantic_graph"].get("edges", [])[-1200:]
        self._save_memory(self._project_memory_file, data)
        return ActionStatus.SUCCESS

    def retrieve_semantic_links(self, task_description: str, limit: int = 6) -> list[dict[str, Any]]:
        data = self._load_memory(self._project_memory_file)
        graph = data.get("semantic_graph", {})
        nodes = list(graph.get("nodes", []))
        edges = list(graph.get("edges", []))
        if not nodes:
            return []

        query_vec = self._text_vector(task_description)
        scored_nodes: list[tuple[float, dict[str, Any]]] = []
        for node in nodes:
            node_vec = [float(v) for v in node.get("vector", []) if isinstance(v, (int, float))]
            if not node_vec:
                node_vec = self._text_vector(str(node.get("label", "")) + " " + " ".join(node.get("tags", [])))
            score = self._cosine_similarity(query_vec, node_vec)
            freq_boost = self._frequency_norm(int(node.get("frequency", 1)))
            total = (0.75 * score) + (0.25 * freq_boost)
            scored_nodes.append((total, node))

        scored_nodes.sort(key=lambda item: item[0], reverse=True)
        top_nodes = scored_nodes[: max(1, limit)]
        top_ids = {str(node.get("id", "")) for _, node in top_nodes}

        related_edges = [
            edge
            for edge in edges
            if str(edge.get("source", "")) in top_ids or str(edge.get("target", "")) in top_ids
        ][: max(2, limit * 2)]

        rendered: list[dict[str, Any]] = []
        for score, node in top_nodes:
            node_id = str(node.get("id", ""))
            node_edges = [e for e in related_edges if e.get("source") == node_id or e.get("target") == node_id]
            rendered.append(
                {
                    "score": round(score, 4),
                    "node": {
                        "id": node_id,
                        "type": str(node.get("type", "unknown")),
                        "label": str(node.get("label", "")),
                    },
                    "edges": [
                        {
                            "relation": str(edge.get("relation", "related_to")),
                            "source": str(edge.get("source", "")),
                            "target": str(edge.get("target", "")),
                        }
                        for edge in node_edges[:3]
                    ],
                }
            )
        return rendered

    def format_semantic_links(self, task_description: str, max_chars: int = 900) -> str:
        links = self.retrieve_semantic_links(task_description, limit=6)
        if not links:
            return ""

        lines: list[str] = []
        for item in links:
            node = item.get("node", {})
            label = str(node.get("label", "")).strip()
            node_type = str(node.get("type", "unknown"))
            score = float(item.get("score", 0.0))
            if not label:
                continue
            lines.append(f"- [{node_type}] {label} (score={score:.2f})")
            for edge in item.get("edges", [])[:2]:
                relation = str(edge.get("relation", "related_to"))
                lines.append(f"  - relation: {relation}")

        block = "\n".join(lines).strip()
        if len(block) <= max_chars:
            return block
        return block[: max_chars - 20].rstrip() + "\n... [truncated]"

    def retrieve_relevant_patterns(
        self,
        task_description: str,
        k_failures: int = 3,
        k_successes: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        data = self._load_memory(self._project_memory_file)
        task_tokens = self._tokenize(task_description)

        def score_pattern(pattern: dict[str, Any], kind: str) -> float:
            p_tokens = set(pattern.get("tags", [])) | self._tokenize(str(pattern.get("summary", "")))
            similarity = self._jaccard(task_tokens, p_tokens)
            frequency_norm = self._frequency_norm(int(pattern.get("frequency", 1)))
            half_life = 14.0 if kind == "failure" else 21.0
            recency = self._recency_decay(str(pattern.get("last_used_at", "")), half_life)
            confidence = self._normalize_confidence(float(pattern.get("confidence", 0.5)))
            base = (0.6 * similarity) + (0.2 * frequency_norm) + (0.2 * recency)
            return base * max(0.2, confidence)

        failures = [dict(p) for p in data.get("failure_patterns", [])]
        successes = [dict(p) for p in data.get("success_patterns", [])]

        failures.sort(key=lambda p: score_pattern(p, "failure"), reverse=True)
        successes.sort(key=lambda p: score_pattern(p, "success"), reverse=True)

        return {
            "failures": failures[: max(1, k_failures)],
            "successes": successes[: max(1, k_successes)],
        }

    def format_learned_patterns(self, task_description: str, max_chars: int = 1500) -> str:
        retrieved = self.retrieve_relevant_patterns(task_description)
        failures = retrieved.get("failures", [])
        successes = retrieved.get("successes", [])

        if not failures and not successes:
            return ""

        lines: list[str] = ["## Learned Patterns"]
        if failures:
            lines.append("\n### Avoid These Failures:")
            for item in failures:
                lines.append(
                    f"- {item.get('category', 'unknown_error')}: {item.get('resolution_hint', 'Validate assumptions first')}"
                )
        if successes:
            lines.append("\n### Reuse These Patterns:")
            for item in successes:
                lines.append(
                    f"- {item.get('pattern_type', 'general_pattern')}: {item.get('summary', '')}"
                )

        block = "\n".join(lines).strip()
        if len(block) <= max_chars:
            return block
        return block[: max_chars - 20].rstrip() + "\n... [truncated]"

    def record_task_outcome(
        self,
        task_description: str,
        success: bool,
        error: str | None = None,
        patterns_used: list[str] | None = None,
    ) -> str:
        data = self._load_memory(self._project_memory_file)
        outcomes = list(data.get("task_outcomes", []))
        outcomes.append(
            {
                "task": task_description[:200],
                "success": success,
                "error": (error or "")[:240],
                "category": self._normalize_error_category(error or "") if error else "",
                "patterns_used": patterns_used or [],
                "timestamp": self._now_iso(),
            }
        )

        task_node_id = self._upsert_graph_node(
            data,
            "task",
            task_description[:240],
            tags=self._extract_failure_tags(task_description, error or "")[:6],
            metadata={"source": "record_task_outcome"},
        )
        outcome_label = "success" if success else "failure"
        outcome_node_id = self._upsert_graph_node(
            data,
            "outcome",
            outcome_label,
            tags=[outcome_label],
            metadata={"error": (error or "")[:180]},
        )
        self._upsert_graph_edge(data, task_node_id, outcome_node_id, "resulted_in")

        data["task_outcomes"] = outcomes[-300:]
        data["semantic_graph"]["nodes"] = data["semantic_graph"].get("nodes", [])[-600:]
        data["semantic_graph"]["edges"] = data["semantic_graph"].get("edges", [])[-1200:]
        self._save_memory(self._project_memory_file, data)
        return ActionStatus.SUCCESS

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
