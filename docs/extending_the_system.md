# Extending the System

This guide explains how to add capabilities without breaking safety, contracts, or orchestration behavior.

## At a glance

- Extend through contracts first, implementation second.
- Treat safety checks and tests as part of the feature, not follow-up work.
- Keep orchestration semantics explicit and deterministic.
- Ship schema + parser + tests + docs together for any contract change.

## Extension philosophy

Extensions should use existing seams rather than bypassing core abstractions.

Preferred order of work:

1. define/extend schema contracts,
2. implement behavior,
3. wire orchestration and tool dispatch,
4. add tests,
5. update docs.

## Add a new tool

1. Implement a plugin that conforms to `ToolPlugin` in `src/tools/base.py`.
2. Register it in `ToolRegistry` via `ToolExecutor` setup.
3. Ensure deterministic `validate(args)` and `policy_check(context, args)` behavior.
4. Emit telemetry for success/failure and policy blocks.
5. Add integration/e2e tests for registered and unregistered behavior.

### Plugin checklist (Phase 5)

- define `name`, `version`, and `capabilities`,
- reject malformed args in `validate`,
- enforce policy in `policy_check`,
- never bypass registry lookup,
- return string outputs compatible with tool-call contracts.

## Add or evolve agent behavior

1. Update shared schemas in `src/agents/base.py`.
2. Update parsing + fallback logic in the target agent.
3. Validate safety sanitization logic.
4. Update executor behavior only when loop semantics require it.
5. Add regression tests for new/changed behavior.

## Add orchestration rules

When changing `Executor` or `LoopController`:

- preserve explicit terminal transitions,
- avoid hidden side effects,
- keep iteration accounting deterministic,
- include tests for pause/continue/failure boundaries.

## Add new config

1. Add typed config model fields in `src/config`.
2. Add defaults in YAML.
3. Validate at startup.
4. Add tests for default and override behavior.

## Policy profiles (strict / balanced / permissive)

Policy profile logic now lives in `src/core/policy/profiles.py`.

- `strict`: deterministic-first, read-oriented tool surface, `temperature=0.0`.
- `balanced`: default profile with bounded flexibility.
- `permissive`: widest tool surface and looser generation settings.

When extending behavior:

1. update profile constraints first,
2. enforce profile in tool execution and planner/coder tool-step validation,
3. test blocked + allowed paths for each relevant profile.

## Formal guarantees and invariants

The runtime now includes explicit invariant checks:

- task state transitions are validated (`pending/ready -> running -> completed|failed`),
- invalid transitions raise explicit errors,
- failure categories are normalized (`tool_error`, `contract_violation`, `planning_error`, `execution_error`),
- tool plugins must satisfy output postconditions.

When adding orchestration logic, preserve these invariants and add tests for invalid transitions.

## Multi-workspace orchestration (v1)

`WorkspaceManager` (`src/orchestration/workspace_manager.py`) provides:

- isolated per-workspace memory/context,
- deterministic workspace ordering,
- sequential execution only (no parallel workspace execution in v1).

CLI support includes workspace list/add/clear controls, and sequential task execution when multiple workspaces are configured.

Extension rule: do not allow context leakage between workspaces.

## Deep extension examples

### Example A: static analysis tool integration

1. Add adapter returning structured findings.
2. Wire tool dispatch.
3. Restrict to allowed files/extensions.
4. Add tests for:
   - valid invocation,
   - unsafe path rejection,
   - timeout behavior,
   - telemetry assertions.
5. Update `docs/tools.md` and `docs/testing.md`.

### Example B: reviewer enrichment

1. Extend reviewer issue metadata.
2. Maintain parser backward compatibility.
3. Preserve `APPROVE` terminal semantics.
4. Add e2e tests for `REQUEST_CHANGES` → `APPROVE` transition.

## Migration strategy for breaking changes

- introduce transitional parser support,
- mark deprecated fields in docs,
- add migration tests for old/new payload shapes,
- remove deprecated path only after all call sites are updated.

## Extension governance checklist

- architecture impact documented,
- contract deltas documented,
- safety implications documented,
- tests added for both happy and failure paths,
- roadmap and known-issues updated where relevant.
