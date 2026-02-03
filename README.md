# Local Coding Agent

**Your AI pair programmer that runs entirely on your machine**

[![CI](https://github.com/local-coding-agents/local-coding-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/local-coding-agents/local-coding-agents/actions/workflows/ci.yml)

Designed for Apple Silicon (M-series, 16GB RAM) • Fully offline • Human-in-the-loop

---

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Ensure Ollama is running with a model
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
ollama serve

# 3. Start interactive session
agent

# 4. Or run a one-shot task
agent -t "Add error handling to the parse function in utils.py"
```

---

## Overview

Local Coding Agent is a **Claude Code-like CLI** that runs entirely on your machine using Ollama. It's designed for developers who want AI assistance without sending code to the cloud.

### Why This?

- **🔒 Private**: Your code never leaves your machine
- **⚡ Fast**: No network latency, runs on Apple Silicon GPU
- **🎯 Safe**: Human approval before any file changes
- **↩️ Reversible**: Full rollback support for every change
- **📊 Transparent**: See exactly what the AI is doing

### Key Features

| Feature | Description |
|---------|-------------|
| **Interactive CLI** | Chat with your codebase like Claude Code |
| **Human-in-the-Loop** | Review and approve all changes before they're applied |
| **Diff Preview** | See exactly what will change before accepting |
| **Rollback** | Undo any change with `/rollback` |
| **Session Memory** | Continue where you left off |
| **Large Project Support** | Handles codebases larger than context window |
| **Policy Guards** | Configurable safety limits and file protections |

---

## Installation

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB RAM minimum (8GB may work with smaller models)
- Python 3.10+
- [Ollama](https://ollama.ai) installed

### Install

```bash
# Clone the repository
git clone https://github.com/local-coding-agents/local-coding-agents.git
cd local-coding-agents

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .

# Verify installation
agent doctor
```

### Pull a Model

```bash
# Recommended: Qwen 2.5 Coder (best balance of quality/speed)
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

# Alternative: DeepSeek Coder
ollama pull deepseek-coder:6.7b-instruct-q4_K_M

# Alternative: CodeLlama
ollama pull codellama:7b-instruct-q4_K_M
```

---

## Usage

### Interactive Mode (Recommended)

```bash
# Start a session in current directory
agent

# Start in a specific project
agent --workspace ~/projects/my-app

# Resume previous session
agent --resume
```

Once in a session:

```
You: Add input validation to the signup form

◐ Planning...
◐ Writing code...

┌─ Pending Changes ──────────────────────────────────┐
│ src/forms/signup.py                                │
│ @@ -15,6 +15,12 @@                                 │
│  def validate(self):                               │
│ +    if not self.email or '@' not in self.email:  │
│ +        raise ValueError("Invalid email")         │
│ +    if len(self.password) < 8:                   │
│ +        raise ValueError("Password too short")   │
└────────────────────────────────────────────────────┘

Apply changes? [y/n/e(dit)]:
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/diff` | Show pending changes |
| `/apply` | Apply pending changes |
| `/reject` | Reject pending changes |
| `/undo` | Undo last applied change |
| `/rollback [id]` | Rollback to checkpoint |
| `/checkpoints` | List available checkpoints |
| `/project [on\|off]` | Enable/disable project mode for iterative development |
| `/plan` | Show current plan |
| `/tokens` | Show token usage |
| `/model [name]` | Switch model |
| `/clear` | Clear conversation |
| `/exit` | Exit session |

### Project Mode

For continuous iterative development on the same project:

```bash
agent

> /project on
[Project Mode] Enabled - executor will persist across tasks

🔄 [Project] > Create a web scraper for news articles
🔄 [Project] > Add caching to avoid duplicate requests
🔄 [Project] > Add unit tests for the scraper

> /project off
```

See [docs/project-mode.md](docs/project-mode.md) for details.

### One-Shot Mode

```bash
# Quick task (still requires approval)
agent -t "Fix the TypeError in api/routes.py"

# Auto-approve (use carefully!)
agent --auto-approve -t "Format all Python files with black"

# Specify model
agent --model codellama:7b -t "Explain the auth flow"
```

### Other Commands

```bash
# Check system status
agent doctor

# Show version
agent version

# List previous sessions
agent sessions

# Show/edit configuration
agent config
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           CLI                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Session    │  │  Commands   │  │  Display    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Task Graph  │  │  Executor   │  │   Loop      │              │
│  │             │  │             │  │ Controller  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│  ┌─────────────┐  ┌─────────────┐                               │
│  │  Rollback   │  │Large Project│                               │
│  │  Manager    │  │  Handler    │                               │
│  └─────────────┘  └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          AGENTS                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Planner  │  │  Coder   │  │ Reviewer │  │  Fixer   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                           CORE                                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                 │
│  │ LLM Client │  │ File Guard │  │Diff Engine │                 │
│  └────────────┘  └────────────┘  └────────────┘                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                 │
│  │  Context   │  │ Contracts  │  │ File Lock  │                 │
│  │  Manager   │  │ Enforcer   │  │  Manager   │                 │
│  └────────────┘  └────────────┘  └────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Safety Guarantees

| Guarantee | How It Works |
|-----------|--------------|
| **No surprise writes** | All changes require human approval (unless `--auto-approve`) |
| **Workspace containment** | Agent cannot access files outside your project |
| **Rollback always works** | Full file snapshots at every checkpoint |
| **Token limits** | Hard caps prevent runaway costs |
| **Iteration limits** | Maximum cycles prevent infinite loops |
| **Contract enforcement** | Agent outputs validated against schemas |

### Protected Files

By default, these files/patterns are protected:

- `.env`, `.env.*` - Secrets
- `.git/` - Git internals
- `*.pem`, `*.key` - Credentials
- `node_modules/`, `venv/` - Dependencies

Configure in `src/config/policies.yaml`.

---

## Agent Contracts

Each agent has a strict contract defining what it can and cannot do:

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

## Configuration

### Models Configuration (`src/config/models.yaml`)

```yaml
default_model: qwen2.5-coder:7b-instruct-q4_K_M

models:
  qwen2.5-coder:7b-instruct-q4_K_M:
    context_window: 32768
    max_output: 8192
    temperature: 0.1
```

### Limits Configuration (`src/config/limits.yaml`)

```yaml
max_iterations: 10
max_tokens_per_run: 100000
max_files_per_task: 5
max_lines_per_edit: 500
```

### Policies Configuration (`src/config/policies.yaml`)

```yaml
protected_paths:
  - ".env*"
  - ".git/"
  - "*.pem"
  
allowed_operations:
  - read
  - write
  - create
  # - delete  # Uncomment to allow deletions
```

---

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run safety tests only (critical!)
pytest tests/test_safety.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Project Structure

```
local-coding-agents/
├── src/
│   ├── cli/              # CLI interface (the product!)
│   │   ├── app.py        # Entry point
│   │   ├── session.py    # Session management
│   │   ├── commands.py   # Slash commands
│   │   └── display.py    # Rich terminal output
│   ├── agents/           # AI agents
│   ├── core/             # Core systems
│   ├── orchestration/    # Execution coordination
│   ├── state/            # State management
│   └── tools/            # Filesystem, shell, testing
├── tests/                # Test suite
├── docs/                 # Documentation
└── workspace/            # Default working directory
```

---

## Troubleshooting

### "Ollama not running"

```bash
ollama serve
```

### "Model not found"

```bash
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

### "Out of memory"

Try a smaller model:
```bash
ollama pull codellama:7b-instruct-q4_0
```

### "Command not found: agent"

Reinstall the package:
```bash
pip install -e .
```

### Check system status

```bash
agent doctor
```

---

## Documentation

- [Architecture Guide](docs/architecture.md)
- [Agent Contracts](docs/agent-contracts.md)
- [CLI Design](docs/cli-design.md)
- [Failure Modes](docs/failure-modes.md)

---

## License

MIT License - See LICENSE file for details.
