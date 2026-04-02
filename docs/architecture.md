# Architecture

## Overview

`LOCAL_OLLAMA_code-agent` is a local-first, contract-driven multi-agent coding runtime. It coordinates planning, implementation, review, and fix loops through strict schemas, bounded execution limits, and guarded file operations.

This document describes the current architecture. For the full documentation set, see:

- `docs/agent_contracts.md`
- `docs/execution_flow.md`
- `docs/tools.md`
- `docs/safety_and_constraints.md`
- `docs/extending_the_system.md`
- `docs/testing.md`
- `docs/known_issues.md`
- `docs/roadmap.md`

## Design Principles

### 1. Local-First
- No external API calls
- No telemetry leaves the machine
- All state stored locally
- Works offline

### 2. Safety by Default
- All file operations mediated through FileGuard
- Whitelist-based path validation
- Rate limiting on operations
- Automatic backups before modifications

### 3. Deterministic Execution
- Sequential agent execution (no parallelism)
- Reproducible with same inputs
- Full audit trail
- Checkpoint-based recovery
- Profile-controlled determinism (`strict` sets deterministic LLM params)

### 4. Resource Consciousness
- Hard limits on tokens, time, iterations
- Single model resident in memory
- Progressive context management
- Efficient file handling

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                          CLI (main.py)                       │
│  Commands: run | resume | rollback | config | logs | status │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Executor (executor.py)                  │
│  Coordinates agents, manages workflow, creates checkpoints  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  LoopController│   │   TaskGraph     │   │ RollbackManager │
│  State machine │   │   DAG of tasks  │   │   Checkpoints   │
└───────────────┘   └─────────────────┘   └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         Agents Layer                         │
│  ┌─────────┐ ┌────────┐ ┌──────────┐ ┌────────┐            │
│  │ Planner │→│ Coder  │→│ Reviewer │→│ Fixer  │            │
│  └─────────┘ └────────┘ └──────────┘ └────────┘            │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   LLMClient   │   │   FileGuard     │   │   DiffEngine    │
│ Ollama client │   │ Safe file ops   │   │  Diff-based edits│
└───────────────┘   └─────────────────┘   └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         Tools Layer                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ Filesystem │  │  Testing   │  │   Shell    │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### Core Components

#### LLMClient (`core/llm_client.py`)
- HTTP client for Ollama API
- Streaming support for responses
- Token counting and estimation
- Model hot-swapping
- Health checks and retries

#### FileGuard (`core/file_guard.py`)
- Mediates all file system access
- Whitelist-based path validation
- Blocked pattern matching
- Rate limiting per operation type
- Automatic backup creation
- Operation logging

#### DiffEngine (`core/diff_engine.py`)
- Creates unified diffs from changes
- Validates diffs before applying
- Atomic application with rollback
- Preview mode for dry runs

#### ContextManager (`core/context_manager.py`)
- Token budget management
- Priority-based item selection
- Context window optimization
- Overflow handling

#### TelemetryCollector (`core/telemetry.py`)
- Structured event logging
- Run metrics aggregation
- JSONL export
- Local-only storage

#### Policy Profiles (`core/policy/profiles.py`)
- Runtime behavior modes: `strict`, `balanced`, `permissive`
- Tool allowlist enforcement per profile
- Tool-step and fallback policy controls
- LLM variability controls (temperature/top_p)

### Configuration (`config/`)

#### models.yaml
```yaml
default_model: "qwen2.5-coder:7b-instruct-q4_K_M"
ollama:
  base_url: "http://localhost:11434"
  timeout_seconds: 300
models:
  - name: "qwen2.5-coder:7b-instruct-q4_K_M"
    context_window: 32768
    max_output_tokens: 4096
```

#### limits.yaml
```yaml
tokens:
  max_per_run: 50000
  max_per_completion: 4096
iterations:
  max_planning_iterations: 10
  max_fix_iterations: 5
time:
  max_run_duration_seconds: 3600
```

#### policies.yaml
```yaml
file_access:
  blocked_patterns: ["*.pem", "*.key", ".env*"]
  allowed_extensions: [".py", ".ts", ".js", ...]
safety:
  no_self_modification: true
  no_credential_exposure: true
```

### Agents (`agents/`)

All agents follow the same pattern:
1. Receive typed input
2. Build system + user prompts
3. Call LLM for completion
4. Parse JSON response
5. Return typed output

#### PlannerAgent
- Input: Task description, workspace context
- Output: List of subtasks with dependencies

#### CoderAgent
- Input: Single subtask, relevant file contents
- Output: Code changes as diffs

#### ReviewerAgent
- Input: Code changes, original task
- Output: Issues list, verdict (APPROVE/REQUEST_CHANGES/REJECT)

#### FixerAgent
- Input: Code changes, review issues
- Output: Fixed code changes

### Orchestration (`orchestration/`)

#### TaskGraph
- DAG representation of subtasks
- Topological sort for execution order
- Cycle detection
- Failure propagation

#### LoopController
- State machine: PLANNING → EXECUTING → REVIEWING → FIXING
- Iteration counting and limits
- Timeout enforcement
- Termination handling

#### RollbackManager
- Checkpoint creation with file snapshots
- State serialization
- Rollback restoration
- Checkpoint pruning

#### Executor
- Main coordination loop
- Agent dispatch
- Error handling
- Result aggregation
- Policy-aware tool and planning enforcement
- Failure normalization and invariant-aware transitions

#### WorkspaceManager (`orchestration/workspace_manager.py`)
- Multi-workspace context orchestration (v1)
- Deterministic workspace ordering
- Context and memory isolation per workspace
- Sequential cross-workspace execution

### Tools (`tools/`)

#### Plugin Architecture
- `ToolPlugin` contract (`tools/base.py`)
- `ToolRegistry` deterministic registration and lookup (`tools/registry.py`)
- Explicit plugin modules in `tools/plugins/`
- No execution outside registry

#### FilesystemTools
- Read/write/create/delete files
- Directory listing
- Search within files
- Workspace search

#### TestRunner
- pytest execution
- unittest support
- Output parsing
- Timeout enforcement

#### ShellExecutor
- Command whitelist
- Working directory restriction
- Environment sanitization
- Output capture

#### Formal Guarantee Layer
- task-state transition invariants in `TaskGraph`
- explicit invalid-transition exceptions
- normalized failure classes for observability and memory learning
- postcondition checks on plugin tool outputs

### State Management (`state/`)

#### RunState
- Complete run state model
- Phase tracking
- Task state tracking
- Token usage tracking

#### CheckpointStore
- Checkpoint persistence
- File snapshot storage
- Index management
- Pruning

#### SummaryGenerator
- Multiple output formats
- Configurable verbosity
- Metrics aggregation

## Data Flow

### Execution Flow

```
1. User provides task description
                    │
                    ▼
2. Planner decomposes into subtasks
                    │
                    ▼
3. For each subtask (in dependency order):
   │
   ├─→ Coder generates changes
   │        │
   │        ▼
   ├─→ Reviewer evaluates changes
   │        │
   │        ├─→ APPROVE: Apply changes
   │        │
   │        ├─→ REQUEST_CHANGES: Fixer attempts repair
   │        │        │
   │        │        └─→ Loop back to Reviewer
   │        │
   │        └─→ REJECT: Skip subtask
   │
   └─→ Checkpoint after each subtask
                    │
                    ▼
4. Generate summary and exit
```

### File Modification Flow

```
1. Coder outputs diff
         │
         ▼
2. DiffEngine validates diff
         │
         ▼
3. FileGuard checks permissions
         │
         ▼
4. Backup created if file exists
         │
         ▼
5. Diff applied atomically
         │
         ▼
6. Telemetry records operation
```

## Error Handling

### Retry Strategy
- LLM calls: 3 retries with exponential backoff
- File operations: Single retry
- Network errors: Logged and reported

### Failure Modes
- Token budget exceeded → Terminate with partial results
- Iteration limit hit → Terminate with partial results
- Timeout → Terminate with partial results
- LLM unavailable → Fail fast
- Invalid diff → Skip and continue

### Recovery
- Checkpoints stored after each agent
- Resume from any checkpoint
- Rollback to previous state

## Security Model

### File Access
- Workspace-restricted (no escapes)
- Pattern-based blocking
- Extension whitelist
- Size limits

### Shell Execution
- Command whitelist only
- No network commands (by default)
- Sanitized environment
- Timeout enforcement

### LLM Interaction
- No prompt injection surface
- Output validation
- Structured JSON only

## Performance Considerations

### Memory Management
- Single model at a time
- Stream LLM responses
- Context window limits
- Checkpoint pruning

### Token Efficiency
- Priority-based context
- Progressive summarization
- Targeted file reading

### Disk Usage
- JSONL logs (append-only)
- Checkpoint rotation
- Backup cleanup

## Extension Points

### Adding New Agents
1. Create new agent class extending BaseAgent
2. Define input/output schemas
3. Implement execute() method
4. Register in agents/__init__.py

### Adding New Tools
1. Create tool class in tools/
2. Implement operations with logging
3. Register in tools/__init__.py
4. Add policy configuration

### Custom Models
1. Add model config to models.yaml
2. Ensure model pulled in Ollama
3. Update default_model if desired
