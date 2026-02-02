# Local Coding Agents

**Production-grade, extensible, auditable local AI coding agent system**

Designed for Apple Silicon (M-series, 16GB RAM) • Fully offline • Deterministic behavior

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Ensure Ollama is running with a model
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
ollama serve

# 3. Run a task
lca run "Create a Python function that calculates fibonacci numbers"

# 4. Check status
lca status
```

---

## Overview

Local Coding Agents (LCA) is a commercial-grade coding assistant that runs entirely on your local machine using Ollama for LLM inference. It implements a strict separation of concerns between agents, tools, orchestration, and state management.

### Key Features

- **Fully Offline**: No cloud APIs, no telemetry exfiltration
- **Apple Silicon Optimized**: Designed for 7B-8B quantized models (Q4/Q5)
- **Deterministic Execution**: Reproducible runs with checkpointing
- **Safe by Design**: All file operations mediated, diff-based edits, rollback support
- **Observable**: Structured logging, decision records, token tracking

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Task Graph  │─▶│  Executor   │─▶│   Loop      │              │
│  │             │  │             │  │ Controller  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          AGENTS                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Planner  │  │  Coder   │  │ Reviewer │  │  Fixer   │        │
│  │          │  │          │  │          │  │          │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                           CORE                                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                 │
│  │ LLM Client │  │ File Guard │  │Diff Engine │                 │
│  └────────────┘  └────────────┘  └────────────┘                 │
│  ┌────────────┐  ┌────────────┐                                 │
│  │  Context   │  │ Telemetry  │                                 │
│  │  Manager   │  │            │                                 │
│  └────────────┘  └────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          TOOLS                                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                 │
│  │ Filesystem │  │  Testing   │  │   Shell    │                 │
│  │  (guarded) │  │  (guarded) │  │  (guarded) │                 │
│  └────────────┘  └────────────┘  └────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          STATE                                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                 │
│  │ Run State  │  │Checkpoints │  │ Summaries  │                 │
│  └────────────┘  └────────────┘  └────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB RAM minimum
- Python 3.11+
- Ollama installed and running

### Supported Models

- `codellama:7b-instruct-q4_K_M`
- `deepseek-coder:6.7b-instruct-q4_K_M`
- `qwen2.5-coder:7b-instruct-q4_K_M`
- `mistral:7b-instruct-q4_K_M`

---

## Installation

```bash
# Clone the repository
git clone https://github.com/local-coding-agents/local-coding-agents.git
cd local-coding-agents

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Ensure Ollama is running
ollama serve

# Pull a supported model
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

---

## Usage

### Basic Usage

```bash
# Run the agent with a task
lca run --task "Implement a binary search function in Python"

# Run with specific workspace
lca run --task "Add unit tests for utils.py" --workspace ./my-project

# Resume from checkpoint
lca resume --run-id abc123

# Rollback a run
lca rollback --run-id abc123 --checkpoint 3
```

### Configuration

```bash
# Show current configuration
lca config show

# Validate configuration
lca config validate
```

### Observability

```bash
# View run logs
lca logs --run-id abc123

# Export run summary
lca export --run-id abc123 --format json
```

---

## Agent Contracts

### Planner
- **Input**: Task description, workspace context
- **Output**: Ordered list of subtasks with acceptance criteria
- **Constraints**: No code generation, max 10 subtasks

### Coder
- **Input**: Subtask, file context, constraints
- **Output**: Code changes as structured diffs
- **Constraints**: Max 3 files per subtask, max 500 lines

### Reviewer
- **Input**: Proposed changes, original code, requirements
- **Output**: Approval/rejection with specific feedback
- **Constraints**: Must cite line numbers, no code generation

### Fixer
- **Input**: Review feedback, original code, proposed changes
- **Output**: Minimal corrective diffs
- **Constraints**: Only address cited issues, max 50 lines changed

---

## Safety Guarantees

1. **File Access Mediation**: All file operations go through FileGuard
2. **Diff Validation**: Changes validated before application
3. **Rollback Support**: Full state recovery at any checkpoint
4. **Token Limits**: Hard caps per agent and per run
5. **Iteration Limits**: Maximum cycles to prevent infinite loops
6. **No Self-Modification**: Core logic is immutable at runtime

---

## Project Structure

```
local-coding-agents/
├── README.md
├── pyproject.toml
├── docs/
│   ├── architecture.md
│   ├── agent-contracts.md
│   └── failure-modes.md
├── src/
│   ├── main.py
│   ├── config/
│   │   ├── models.yaml
│   │   ├── limits.yaml
│   │   └── policies.yaml
│   ├── core/
│   │   ├── llm_client.py
│   │   ├── context_manager.py
│   │   ├── file_guard.py
│   │   ├── diff_engine.py
│   │   └── telemetry.py
│   ├── agents/
│   │   ├── base.py
│   │   ├── planner.py
│   │   ├── coder.py
│   │   ├── reviewer.py
│   │   └── fixer.py
│   ├── orchestration/
│   │   ├── task_graph.py
│   │   ├── executor.py
│   │   ├── loop_controller.py
│   │   └── rollback.py
│   ├── tools/
│   │   ├── filesystem.py
│   │   ├── testing.py
│   │   └── shell.py
│   └── state/
│       ├── run_state.py
│       ├── checkpoints.py
│       └── summaries.py
├── workspace/
│   ├── src/
│   ├── tests/
│   └── artifacts/
└── logs/
```

---

## Documentation

- [Architecture Guide](docs/architecture.md)
- [Agent Contracts](docs/agent-contracts.md)
- [Failure Modes](docs/failure-modes.md)

---

## License

MIT License - See LICENSE file for details.
