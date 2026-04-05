# LOCAL_OLLAMA_code-agent

Local-first, contract-driven coding agent runtime with explicit orchestration, safety controls, and resumable execution.

## Why this project exists

Most coding-agent stacks optimize for speed first and safety second. This project intentionally prioritizes reliability and controlled autonomy:

- strict typed contracts between all major components,
- guarded file and tool operations,
- deterministic execution boundaries (iterations/time/tokens),
- robust failure handling and recoverable run state.

The objective is practical autonomy with clear operational guardrails.

## Core capabilities

- Multi-agent execution loop: `Planner -> Coder -> Reviewer -> Fixer`
- Subtask decomposition with dependency-aware task graph execution
- Persistent run telemetry, checkpoints, and rollback support
- Controlled tool execution (filesystem, shell, testing, memory, MCP)
- Session-aware CLI with continuation support after bounded pauses
- Plugin-style tool registry with explicit tool registration
- Policy profiles: `strict`, `balanced`, `permissive`
- Formal execution guarantees (state invariants + normalized failures)
- Deterministic sequential multi-workspace orchestration (v1)
- Meta-agent reflection + adaptive policy tuning loops
- Query-aware context optimization with budgeted summarization

## Repository structure

- `src/agents`: agent definitions, prompt/parse logic, shared schemas
- `src/orchestration`: executor, loop control, rollback, task graph
- `src/core`: context, safety, telemetry, llm client, memory, mcp
- `src/cli`: command handlers, session model, display logic
- `src/tools`: operational tool adapters
- `tests`: unit/integration/e2e safety and orchestration tests
- `docs`: architecture, contracts, operations, extension guidance

## Quick start

### 1) Install dependencies

`pip install -e ".[dev]"`

### 2) Ensure model runtime is available

The project expects Ollama (or configured equivalent) to be reachable from the configured endpoint.

### 3) Run tests

`pytest tests/ -v`

### 4) Launch CLI

`agent`

### Policy profiles

- `agent --policy-profile strict` (deterministic-first, read-oriented)
- `agent --policy-profile balanced` (default)
- `agent --policy-profile permissive` (widest tool surface)

Interactive commands:

- `/profile` (show active profile)
- `/profile strict|balanced|permissive` (switch profile)
- `/workspaces list|add <path>|clear` (configure multi-workspace list)

## Typical workflow

1. Define a clear task objective.
2. Run a focused test baseline.
3. Execute the task with the CLI.
4. Review generated changes and telemetry.
5. Re-run affected tests and finalize.

## Documentation index

- [Architecture](docs/architecture.md)
- [Agent contracts](docs/agent_contracts.md)
- [Execution flow](docs/execution_flow.md)
- [Tools](docs/tools.md)
- [Safety and constraints](docs/safety_and_constraints.md)
- [Extending the system](docs/extending_the_system.md)
- [Testing](docs/testing.md)
- [Release notes](docs/release-notes.md)
- [Known issues](docs/known_issues.md)
- [Roadmap](docs/roadmap.md)

## Operational notes

- Keep policy and limit files under version control.
- Prefer focused, incremental changes over broad rewrites.
- Add tests for every bug fix touching orchestration/safety logic.
- Treat shell and file tools as privileged surfaces.
- MCP tools are explicit plugin registrations; there is no implicit catch-all dispatch.

## License

MIT
