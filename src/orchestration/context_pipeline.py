"""
Context Pipeline Module

Provides orchestration primitives for:
- task routing
- deterministic context assembly
- rule validation

This keeps prompt construction modular and architecture-aware.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.core.memory import MemoryManager
from src.core.semantic_search import CodebaseNavigator


class TaskDomain(str, Enum):
    """High-level task categories for routing and policy selection."""

    FRONTEND = "frontend"
    BACKEND = "backend"
    DATA_ML = "data_ml"
    INFRASTRUCTURE = "infrastructure"
    GENERAL = "general"


@dataclass
class TaskRoute:
    """Routing decision for a task."""

    domain: TaskDomain
    module_hints: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)


@dataclass
class ContextPacket:
    """Assembled context sections used by planner/coder."""

    evergreen_context: str = ""
    documentation_context: str = ""
    retrieved_code_context: str = ""
    interfaces_context: str = ""
    dependencies_context: str = ""
    recent_failures_context: str = ""
    recent_successes_context: str = ""
    learned_patterns_context: str = ""
    route_summary: str = ""
    constraints: list[str] = field(default_factory=list)

    def to_prompt_context(self, max_chars: int = 3200) -> str:
        """Serialize packet to deterministic markdown sections."""
        sections: list[str] = []

        if self.route_summary:
            sections.append("## Route\n" + self.route_summary)
        if self.evergreen_context:
            sections.append("## Evergreen Context\n" + self.evergreen_context)

        prioritized = [
            ("## Interfaces", self.interfaces_context),
            ("## Dependencies", self.dependencies_context),
            ("## Recent Failures", self.recent_failures_context),
            ("## Recent Successes", self.recent_successes_context),
            ("## Documentation", self.documentation_context),
            ("## Retrieved Code", self.retrieved_code_context),
        ]

        for header, body in prioritized:
            if body:
                sections.append(f"{header}\n{body}")

        if self.learned_patterns_context:
            sections.append(self.learned_patterns_context)
        if self.constraints:
            sections.append("## Constraints\n" + "\n".join(f"- {c}" for c in self.constraints))

        composed: list[str] = []
        current = 0
        for section in sections:
            chunk = section.strip()
            if not chunk:
                continue
            separator = 2 if composed else 0
            proposed = current + len(chunk) + separator
            if proposed > max_chars:
                break
            composed.append(chunk)
            current = proposed

        text = "\n\n".join(composed).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 20].rstrip() + "\n... [truncated]"


class TaskRouter:
    """Classify user tasks and emit domain-specific constraints."""

    _DOMAIN_KEYWORDS: dict[TaskDomain, tuple[str, ...]] = {
        TaskDomain.FRONTEND: (
            "ui",
            "frontend",
            "react",
            "vue",
            "css",
            "component",
            "page",
            "tailwind",
        ),
        TaskDomain.BACKEND: (
            "api",
            "backend",
            "endpoint",
            "service",
            "controller",
            "auth",
            "database",
        ),
        TaskDomain.DATA_ML: (
            "model",
            "dataset",
            "training",
            "inference",
            "feature",
            "ml",
            "rag",
            "embedding",
        ),
        TaskDomain.INFRASTRUCTURE: (
            "docker",
            "kubernetes",
            "ci",
            "pipeline",
            "deploy",
            "terraform",
            "infra",
            "monitoring",
        ),
    }

    def route(self, task_description: str) -> TaskRoute:
        """Return a routing decision based on lexical signals."""
        text = task_description.lower()
        scores: dict[TaskDomain, int] = {d: 0 for d in TaskDomain}

        for domain, keywords in self._DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    scores[domain] += 1

        best_domain = max(scores, key=scores.get)
        if scores.get(best_domain, 0) == 0:
            best_domain = TaskDomain.GENERAL

        module_hints = self._extract_module_hints(task_description)
        constraints = self._domain_constraints(best_domain)

        return TaskRoute(domain=best_domain, module_hints=module_hints, constraints=constraints)

    @staticmethod
    def _extract_module_hints(task_description: str) -> list[str]:
        """Extract path-like hints for retrieval focus."""
        paths = re.findall(r"[a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+", task_description)
        return sorted({p for p in paths})[:10]

    @staticmethod
    def _domain_constraints(domain: TaskDomain) -> list[str]:
        base = [
            "Use structured prompting sections: <task>, <context>, <constraints>, <examples>, <output_format>",
            "Keep scope single-responsibility and avoid monolithic changes.",
            "Use retrieved code and docs as grounding before implementation.",
            "Design subtasks to be parallelizable when module interfaces are independent.",
        ]
        domain_specific: dict[TaskDomain, list[str]] = {
            TaskDomain.FRONTEND: ["Preserve UI component boundaries and avoid backend side effects."],
            TaskDomain.BACKEND: ["Preserve service/controller/model separation and API contracts."],
            TaskDomain.DATA_ML: ["Track data assumptions and feature/schema consistency explicitly."],
            TaskDomain.INFRASTRUCTURE: ["Preserve deployment safety, idempotency, and rollback paths."],
            TaskDomain.GENERAL: ["Prefer minimal interfaces and explicit module ownership."],
        }
        return base + domain_specific[domain]


class ContextBuilder:
    """Build compact, relevant context packets for each task."""

    def __init__(self, workspace_root: Path, memory_manager: MemoryManager) -> None:
        self._root = workspace_root
        self._memory = memory_manager
        self._navigator = CodebaseNavigator(workspace_root)

    def build(self, task_description: str, route: TaskRoute) -> ContextPacket:
        """Assemble a task-specific context packet."""
        docs = self._load_docs_context(task_description=task_description, max_chars=2000)
        retrieved = self._retrieve_code(task_description, route.module_hints)
        learned = self._memory.format_learned_patterns(task_description, max_chars=900)

        patterns = self._memory.retrieve_relevant_patterns(task_description, k_failures=3, k_successes=3)
        failures = self._format_pattern_lines(patterns.get("failures", []), kind="failure", max_chars=700)
        successes = self._format_pattern_lines(patterns.get("successes", []), kind="success", max_chars=700)
        interfaces = self._build_interfaces_context(route.module_hints, max_chars=800)
        dependencies = self._build_dependencies_context(max_chars=700)

        route_summary = (
            f"domain={route.domain.value}; module_hints={route.module_hints or ['none']}"
        )

        return ContextPacket(
            evergreen_context=self._memory.get_all_context(),
            documentation_context=docs,
            retrieved_code_context=retrieved,
            interfaces_context=interfaces,
            dependencies_context=dependencies,
            recent_failures_context=failures,
            recent_successes_context=successes,
            learned_patterns_context=learned,
            route_summary=route_summary,
            constraints=list(route.constraints),
        )

    def build_structured_task(self, task_description: str, packet: ContextPacket) -> str:
        """Wrap raw task into deterministic structured format."""
        return (
            "<task>\n"
            f"{task_description.strip()}\n"
            "</task>\n\n"
            "<context>\n"
            f"{packet.to_prompt_context(max_chars=1800)}\n"
            "</context>\n\n"
            "<constraints>\n"
            + "\n".join(f"- {c}" for c in packet.constraints)
            + "\n</constraints>\n\n"
            "<examples>\n"
            "- Keep changes modular and file-local when possible.\n"
            "- Prefer interface-safe edits over broad rewrites.\n"
            "</examples>\n\n"
            "<output_format>\n"
            "- Return strict JSON according to the active agent contract.\n"
            "</output_format>"
        )

    def _retrieve_code(self, task_description: str, module_hints: list[str]) -> str:
        """Retrieve minimal relevant code snippets using semantic and pattern search."""
        snippets: list[str] = []

        # Priority 1: exact module/path hints (deterministic sorted order)
        for hint in sorted(set(module_hints))[:3]:
            try:
                snippets.append(self._navigator.grep_search(hint, max_results=3))
            except Exception:
                continue

        # Priority 2: semantic matches
        try:
            snippets.append(self._navigator.semantic_search(task_description, top_k=4))
        except Exception:
            pass

        text = "\n\n".join(s for s in snippets if s).strip()
        if len(text) <= 2400:
            return text
        return text[:2380].rstrip() + "\n... [truncated]"

    def _load_docs_context(self, task_description: str, max_chars: int = 2000) -> str:
        """Load canonical architecture/docs snippets for grounding."""
        task_lower = task_description.lower()
        architecture_weighted = any(
            token in task_lower
            for token in ("architecture", "contract", "orchestration", "design")
        )

        candidates = [
            self._root / "docs" / "architecture.md",
            self._root / "docs" / "agent_contracts.md",
            self._root / "docs" / "execution_flow.md",
            self._root / "core_context.md",
            self._root / "docs" / "agent-evergreen-context.md",
        ]

        if architecture_weighted:
            candidates = [
                self._root / "docs" / "architecture.md",
                self._root / "docs" / "agent_contracts.md",
                self._root / "core_context.md",
                self._root / "docs" / "execution_flow.md",
                self._root / "docs" / "agent-evergreen-context.md",
            ]

        sections: list[str] = []
        for path in candidates:
            if not path.exists() or not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if not content:
                continue
            sections.append(f"[{path.name}]\n{content[:700]}")

        text = "\n\n".join(sections)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 20].rstrip() + "\n... [truncated]"

    def _build_interfaces_context(self, module_hints: list[str], max_chars: int = 800) -> str:
        """Build deterministic interface signatures section from likely relevant files."""
        candidate_files = [
            p for p in sorted(self._root.rglob("*.py")) if "__pycache__" not in str(p)
        ]

        prioritized: list[Path] = []
        for hint in sorted(set(module_hints)):
            for path in candidate_files:
                rel = path.relative_to(self._root).as_posix()
                if hint in rel and path not in prioritized:
                    prioritized.append(path)

        if not prioritized:
            prioritized = candidate_files[:4]

        lines: list[str] = []
        pattern = re.compile(r"^\s*(def|class)\s+[A-Za-z_][A-Za-z0-9_]*")
        for path in prioritized[:4]:
            rel = path.relative_to(self._root).as_posix()
            try:
                content = path.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for line in content[:220]:
                if pattern.match(line):
                    lines.append(f"{rel}: {line.strip()}")
                if len("\n".join(lines)) > max_chars:
                    break
            if len("\n".join(lines)) > max_chars:
                break

        text = "\n".join(lines).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 20].rstrip() + "\n... [truncated]"

    def _build_dependencies_context(self, max_chars: int = 700) -> str:
        """Build deterministic dependency summary from pyproject and import signals."""
        lines: list[str] = []

        pyproject = self._root / "pyproject.toml"
        if pyproject.exists() and pyproject.is_file():
            try:
                content = pyproject.read_text(encoding="utf-8")
                dep_lines = [ln.strip() for ln in content.splitlines() if "dependencies" in ln.lower()]
                lines.extend(dep_lines[:8])
            except Exception:
                pass

        import_pattern = re.compile(r"^\s*(from\s+[A-Za-z0-9_\.]+\s+import|import\s+[A-Za-z0-9_\.]+)")
        for path in sorted((self._root / "src").rglob("*.py"))[:8]:
            try:
                content = path.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            rel = path.relative_to(self._root).as_posix()
            for line in content[:120]:
                if import_pattern.match(line):
                    lines.append(f"{rel}: {line.strip()}")
                if len("\n".join(lines)) > max_chars:
                    break
            if len("\n".join(lines)) > max_chars:
                break

        text = "\n".join(lines).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 20].rstrip() + "\n... [truncated]"

    @staticmethod
    def _format_pattern_lines(
        items: list[dict[str, object]],
        kind: str,
        max_chars: int,
    ) -> str:
        """Format recent failure/success lines deterministically with bounded size."""
        lines: list[str] = []
        if kind == "failure":
            for item in items:
                category = str(item.get("category", "unknown"))
                hint = str(item.get("resolution_hint", ""))
                lines.append(f"{category}: {hint}")
        else:
            for item in items:
                ptype = str(item.get("pattern_type", "pattern"))
                summary = str(item.get("summary", ""))
                lines.append(f"{ptype}: {summary}")

        text = "\n".join(lines).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 20].rstrip() + "\n... [truncated]"


class ValidationLayer:
    """Validate tasks and outputs against architecture rules."""

    _ACTION_VERBS = {
        "add",
        "update",
        "fix",
        "refactor",
        "implement",
        "remove",
        "create",
        "optimize",
        "document",
    }

    def validate_task(self, task_description: str) -> list[str]:
        """Return validation errors for an input task."""
        errors: list[str] = []
        text = task_description.strip()

        if len(text) < 15:
            errors.append("Task is too short; provide explicit intent and scope.")

        lower = text.lower()
        if not any(v in lower for v in self._ACTION_VERBS):
            errors.append("Task is ambiguous; include a clear action verb (add/fix/update/refactor/etc.).")

        return errors

    def validate_plan_targets(self, target_files: list[str], known_files: list[str]) -> list[str]:
        """Warn when planned files are outside known workspace files."""
        known = set(known_files)
        unknown = [f for f in target_files if f and f not in known]
        return [f"Unknown target file: {f}" for f in unknown[:10]]
