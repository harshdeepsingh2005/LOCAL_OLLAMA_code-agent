# Tools

This document describes the tool subsystem used by agents during execution.

## At a glance

- Tools are capability adapters, not unrestricted command surfaces.
- Every call should be validated, bounded, and telemetry-instrumented.
- File and shell tools are safety-sensitive and should be treated as privileged operations.
- Tool error normalization is required for reliable orchestrator behavior.

## Tooling model

Tools are adapters invoked through orchestrator-controlled dispatch, not direct arbitrary execution. The model is intentionally constrained to keep behavior safe, testable, and auditable.

Primary goals:

- controlled capability surface,
- explicit auditing of calls and outputs,
- workspace safety boundaries,
- deterministic error behavior.

## Tool categories

### Filesystem tools

Typical operations:

- read file,
- write file,
- create file,
- list/search directories/files.

Primary guardrails:

- path must remain within workspace roots,
- blocked patterns/extensions are denied,
- operation volume and file size limits are enforced.

### Shell tools

Typical operations:

- build/test/format commands,
- diagnostics and environment checks.

Primary guardrails:

- controlled working directory,
- bounded timeout,
- captured stdout/stderr,
- explicit failure signaling and return codes.

### Testing tools

Typical operations:

- focused test selection,
- full-suite execution,
- output and failure summary normalization.

Primary guardrails:

- timeout and output bounds,
- deterministic result parsing,
- graceful handling of partial failures.

### Memory tools

Typical operations:

- write durable run facts,
- retrieve prior context snippets.

Current behavior:

- strong persistence semantics,
- limited semantic-linking sophistication.

### MCP tools

Typical operations:

- connect to configured MCP servers,
- execute server-provided capabilities.

Primary guardrails:

- explicit endpoint/config requirement,
- robust failure mapping when unavailable/misconfigured.

## Tool execution contract

Each invocation should include:

- explicit `tool_name`,
- validated argument payload,
- bounded execution window,
- normalized structured output,
- deterministic exception-to-error mapping.

## Error taxonomy

Recommended stable error classes:

- `validation_error`,
- `permission_error`,
- `timeout_error`,
- `execution_error`,
- `unavailable_error`.

This stable taxonomy simplifies retry/fallback decisions in orchestrator logic.

## Integration points

- dispatch layer: `src/core/agent_tools.py`,
- file policy enforcement: `src/core/file_guard.py`,
- telemetry capture: `src/core/telemetry.py`,
- orchestration usage: `src/orchestration/executor.py`.

## Hardening checklist for new tools

- deny-by-default for risky arguments,
- strict workspace/path constraints,
- sanitize command/input surfaces,
- enforce output size limits,
- emit telemetry for both success and failure,
- add tests for valid, malformed, and malicious inputs.

## Performance and context hygiene

- prefer narrow scoped reads over wide scans,
- avoid redundant repeated shell invocations,
- cache reusable expensive lookups during run scope,
- return concise outputs to reduce prompt bloat.

## Operational best practices

- keep tool interfaces stable and explicit,
- avoid hidden side effects,
- preserve deterministic behavior for equivalent inputs,
- document tool contracts before broad usage.
