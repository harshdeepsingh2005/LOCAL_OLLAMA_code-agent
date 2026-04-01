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

1. Implement adapter in `src/tools` with a clear input/output contract.
2. Register dispatch mapping in `src/core/agent_tools.py`.
3. Apply safety checks (paths/timeouts/input validation).
4. Emit telemetry for success and failure.
5. Add integration/e2e tests.

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
