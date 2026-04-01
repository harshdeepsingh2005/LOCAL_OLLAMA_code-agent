# Execution Flow

This document explains the runtime lifecycle from task intake to final result.

## At a glance

- Execution is phase-based: initialize, plan, execute, finalize.
- Subtasks run through coder/reviewer/fixer with bounded retries and strict terminal logic.
- State is recoverable via checkpoints and continuation semantics.
- Failures are intentionally explicit and categorized.

## End-to-end lifecycle

1. CLI/session receives a task request.
2. `Executor` initializes run id, telemetry, and controller state.
3. Preconditions are checked (model health, task validity, limits).
4. Planner creates a structured subtask plan.
5. Task graph is built and ordered by dependencies.
6. Each eligible subtask enters the coder/reviewer/fixer loop.
7. Approved changes are applied through guarded file operations.
8. Run finalization writes metrics, summary, and terminal reason.

## Phase model

### Phase A — Initialization

- generate run id,
- initialize loop controller,
- initialize telemetry/checkpoint scope,
- prepare route/context pipeline if enabled,
- verify LLM health and task constraints.

### Phase B — Planning

- send planning prompt with bounded context,
- parse and validate planner output,
- sanitize unsafe/invalid plan segments,
- fallback to deterministic plan if needed,
- create task graph and initial checkpoint.

### Phase C — Subtask execution

For each ready task node:

1. Coder proposes changes.
2. Contract + safety validation runs on output.
3. Reviewer evaluates changes against criteria.
4. If `REQUEST_CHANGES`, fixer loop iterates until:
   - reviewer approves,
   - fix iteration cap is reached,
   - unrecoverable failure occurs.
5. Node marked success/failure; dependency effects propagate.

### Phase D — Finalization

- aggregate completion stats,
- set termination reason,
- collect file changes,
- record final telemetry,
- close or preserve runtime resources depending on continuation state.

## Control-loop semantics

### Iteration accounting

- planning consumes loop budget,
- execution cycles consume loop budget,
- fix loops have additional bounded counters,
- limit exhaustion should pause safely (not corrupt state).

### Terminal states

- `success`: required subtasks completed and approved,
- `paused`: continuation required due to limits,
- `partial_failure`: failed/blocked task graph nodes remain,
- `fatal_error`: unrecoverable run-level failure.

### Failure handling hierarchy

1. **Agent-level recoverable failure** → fallback object and continue when safe.
2. **Task-level failure** → mark node failed and propagate blocked dependents.
3. **Run-level fatal failure** → terminate early with explicit error.

### Continue/resume behavior

- completed subtasks are not re-run,
- graph state is preserved across continuation,
- additional iterations can be granted,
- resumed run keeps audit continuity (same run context lineage).

### Determinism requirements

- explicit monotonic state transitions,
- reproducible termination behavior under same inputs/config,
- stable dependency ordering semantics,
- no hidden side effects outside tracked files/tools.

### Observability checkpoints

Recommended events for diagnostics:

- run start/end,
- planning start/success/fallback/failure,
- per-subtask coder/reviewer/fixer outputs,
- iteration-limit and timeout events,
- final termination and result summary.

### Practical debugging sequence

When a run behaves unexpectedly:

1. inspect termination reason,
2. inspect last reviewer verdict and issues,
3. inspect fixer iteration count and fallback usage,
4. inspect relevant telemetry warnings,
5. inspect checkpoint state before failure point.
