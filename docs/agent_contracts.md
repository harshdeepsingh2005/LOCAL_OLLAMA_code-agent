# Agent Contracts

This document defines how data moves safely and predictably between agents and the orchestrator.

## At a glance

- Contracts are the main safety boundary between autonomous agents and execution logic.
- Every agent output is parsed, validated, sanitized, and either accepted or replaced with a typed fallback.
- Reviewer terminal semantics are execution-critical and must remain stable.
- Backward-compatible schema evolution is required for safe iteration.

## Why contracts matter in this project

The runtime is built on autonomous loops. Without strict contracts, malformed or ambiguous agent output would quickly destabilize execution. Contracts provide:

- schema-level correctness,
- deterministic failure behavior,
- clear compatibility guarantees,
- safety filtering boundaries.

Canonical models are in `src/agents/base.py`.

## Contract architecture

Each agent interaction follows a consistent chain:

1. typed input object is created,
2. prompt is generated from input + context,
3. raw model response is captured,
4. response is parsed into structured data,
5. schema validation is applied,
6. safety sanitization is applied,
7. typed output is returned or fallback output is synthesized.

## Shared model primitives

## `AgentStatus`

Global outcome state:

- `SUCCESS`
- `FAILED`
- `TIMEOUT`
- `ERROR`

## `Subtask`

Planner’s executable work unit:

- stable `id`,
- clear intent (`title` + `description`),
- acceptance criteria,
- dependency list,
- target file hints.

## `CodeChange`

Canonical file mutation representation:

- `file_path`,
- `change_type` (`create` / `modify` / `delete`),
- `description`,
- `new_content`,
- optional `diff`.

All code-modifying agents should emit change intent through this structure.

## Agent-specific contracts

## Planner contract

### Planner inputs

- task description,
- workspace context summary,
- optional constraints,
- prior attempt context (if retry/continuation).

### Planner outputs (`PlannerOutput`)

- `plan_summary`,
- ordered subtasks,
- optional planning tool calls,
- confidence / notes.

### Planner critical invariants

- subtasks must be structurally valid,
- dependency references must be resolvable,
- planner fallback must remain executable.

## Coder contract

### Coder inputs

- one `Subtask`,
- relevant source context,
- optional reviewer feedback on retries.

### Coder outputs (`CoderOutput`)

- normalized `CodeChange` list,
- implementation notes,
- optional tool calls,
- confidence signal.

### Coder critical invariants

- no path escape,
- no unsupported change shape,
- no silent empty success unless explicit no-op intent is encoded.

## Reviewer contract

### Reviewer inputs

- subtask,
- proposed changes,
- acceptance criteria,
- implementation notes/context.

### Reviewer outputs (`ReviewerOutput`)

- verdict: `APPROVE` / `REQUEST_CHANGES` / `REJECT`,
- issue list (severity/path/message),
- criteria status mapping,
- terminal-state indicator (`task_complete`).

### Reviewer critical invariants

- `APPROVE` must imply terminal completion,
- issue payloads must be structured and path-safe,
- verdict semantics must remain stable for loop control.

## Fixer contract

### Fixer inputs

- failing reviewer issues,
- prior `CodeChange` values,
- file context around affected areas.

### Fixer outputs (`FixerOutput`)

- corrected `CodeChange` list,
- fix notes,
- optional unresolved-item annotations.

### Fixer critical invariants

- fixer output must remain schema-compliant coder-like changes,
- unresolved states must be explicit,
- repeated non-progress should surface to orchestrator limits.

## Validation and fallback policy

- invalid JSON never crashes the run,
- parser failures return typed failure/fallback objects,
- unsafe paths/tool calls are dropped or rejected,
- status + error context must always be explicit.

## Compatibility and versioning rules

- additive fields preferred,
- breaking renames require migration support,
- defaults should preserve old behavior,
- tests must pin both old and new parsing paths during migrations.

## Contract anti-patterns

- partial objects missing required fields,
- free-form prose instead of structured fields,
- hidden operational signals in non-operational text,
- unvalidated passthrough of unknown tool calls,
- ambiguous reviewer terminal signals.

## Recommended contract test matrix

- valid payload acceptance per agent,
- malformed JSON fallback per agent,
- missing-field rejection,
- unknown enum/value rejection,
- unsafe path sanitization,
- reviewer terminal consistency,
- backward-compat parser behavior for transitional payloads.
