# CLI Design Document

## Product Vision

The `agent` CLI is a **fully offline, human-in-the-loop coding assistant** that runs on Apple Silicon with 16GB RAM. It provides a Claude Code–like conversational experience while maintaining strict safety guarantees and deterministic behavior.

**Design Philosophy:**
- Conversational, not transactional
- Human-in-the-loop, always
- Diff-first, no silent side effects
- Calm, professional tone
- Trust through transparency

---

## Interaction Model

### Entry Point

```bash
agent                          # Start interactive session
agent "your task here"         # One-shot mode with task
agent --resume <session-id>    # Resume previous session
agent doctor                   # Check system health
```

### Session Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                       SESSION STATES                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │  INIT    │───►│  ACTIVE  │───►│  PAUSED  │             │
│   └──────────┘    └──────────┘    └──────────┘             │
│        │               │  ▲            │                    │
│        │               │  │            │                    │
│        │               ▼  │            │                    │
│        │          ┌──────────┐         │                    │
│        │          │ PENDING  │─────────┤                    │
│        │          │ APPROVAL │         │                    │
│        │          └──────────┘         │                    │
│        │               │               │                    │
│        │               ▼               │                    │
│        │          ┌──────────┐         │                    │
│        └─────────►│  ENDED   │◄────────┘                    │
│                   └──────────┘                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**INIT**: Session starting, checking prerequisites
**ACTIVE**: Processing user input, executing agents
**PENDING APPROVAL**: Diff generated, awaiting user confirmation
**PAUSED**: Session saved to disk, can be resumed
**ENDED**: Session completed or terminated

### Conversation Flow

```
┌─────────────────────────────────────────────────────────────┐
│  User Input                                                  │
│  ───────────                                                │
│  Create a fibonacci function in Python                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Planning                                                    │
│  ────────                                                   │
│  I'll create a fibonacci function. Here's my plan:          │
│                                                              │
│  1. Create src/fibonacci.py with iterative implementation  │
│  2. Add comprehensive docstring and type hints              │
│  3. Handle edge cases (n < 0, n == 0, n == 1)              │
│                                                              │
│  Shall I proceed? [Y/n]                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Implementation                                              │
│  ──────────────                                             │
│  ◐ Implementing task 1/3: Create fibonacci function...     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Diff Preview                                                │
│  ────────────                                               │
│  I've made the following changes:                           │
│                                                              │
│  + src/fibonacci.py (new file, 35 lines)                   │
│                                                              │
│  --- /dev/null                                              │
│  +++ src/fibonacci.py                                       │
│  @@ -0,0 +1,35 @@                                           │
│  +"""Fibonacci sequence implementation."""                  │
│  +                                                          │
│  +def fibonacci(n: int) -> int:                             │
│  +    """Calculate the nth Fibonacci number.                │
│  +    ...                                                   │
│                                                              │
│  Apply these changes? [Y/n/e(dit)/r(eject)]                │
└─────────────────────────────────────────────────────────────┘
```

---

## Commands

### Entry Commands

| Command | Description |
|---------|-------------|
| `agent` | Start interactive session in current directory |
| `agent "task"` | One-shot: execute task, then exit |
| `agent --resume ID` | Resume a paused session |
| `agent --workspace PATH` | Use specific workspace |
| `agent doctor` | Check system prerequisites |
| `agent version` | Show version and model info |

### Slash Commands (In-Session)

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/diff` | Show pending diffs |
| `/apply` | Apply all pending diffs |
| `/reject` | Reject all pending diffs |
| `/undo` | Undo last applied change |
| `/rollback [ID]` | Rollback to checkpoint |
| `/summary` | Show current session summary |
| `/plan` | Show current execution plan |
| `/logs` | Show execution logs |
| `/policy` | Show active policies |
| `/models` | Show available models |
| `/model NAME` | Switch to a different model |
| `/tokens` | Show token usage |
| `/pause` | Save session and exit |
| `/exit` | Exit (prompts if unsaved changes) |
| `/clear` | Clear conversation context |

### Approval Commands

When a diff is pending:

| Input | Action |
|-------|--------|
| `y` / `yes` / Enter | Apply the changes |
| `n` / `no` | Reject and continue |
| `e` / `edit` | Open diff in editor |
| `r` / `reject` | Reject and stop |
| `?` | Explain what the change does |
| `/diff` | Show full diff again |

---

## Safety Guarantees

### 1. No Silent Side Effects

**Every file modification requires explicit user approval.**

```
Changes ready to apply:

  Modified: src/api.py (+15, -3 lines)
  Created:  src/utils.py (42 lines)

Apply these changes? [Y/n]
```

### 2. Automatic Checkpoints

Checkpoints are created:
- Before any file modification
- After each successful task completion
- On graceful session pause

```
✓ Checkpoint created: cp_20260203_143522
  Use /rollback cp_20260203_143522 to restore
```

### 3. Workspace Isolation

All file operations are restricted to:
- The workspace directory
- Explicitly allowed paths in config

**Blocked by default:**
- `.git/` directory
- `node_modules/`, `venv/`, `__pycache__/`
- Files > 10MB
- Binary files

### 4. Resource Limits

Hard limits enforced at runtime:

| Resource | Limit |
|----------|-------|
| Tokens per completion | 4,096 |
| Tokens per run | 50,000 |
| Modified files per run | 50 |
| Max iterations | 10 |
| Single file size | 1MB |

### 5. Graceful Degradation

When limits are approached:

```
⚠ Token usage: 45,000/50,000 (90%)

Options:
  1. Complete current task only
  2. Summarize progress and pause
  3. Continue (may hit limit)

Choice [1]:
```

---

## Failure Handling

### Ollama Connection Errors

```
✗ Cannot connect to Ollama at localhost:11434

Possible causes:
  • Ollama is not running
  • A different port is configured
  • Network/firewall issue

Solutions:
  • Run: ollama serve
  • Check: ollama list
  • Verify: curl http://localhost:11434/api/tags

Retry? [Y/n]
```

### Model Not Found

```
✗ Model 'qwen2.5-coder:7b' not found

Available models:
  • llama3.2:latest
  • codellama:7b

Options:
  1. Use llama3.2:latest instead
  2. Pull qwen2.5-coder:7b (requires network)
  3. Exit and configure manually

Choice [1]:
```

### Policy Violations

```
⚠ Policy violation: Cannot modify .gitignore

This file is protected by policy. To modify protected files:
  1. Edit ~/.config/agent/policies.yaml
  2. Or use --allow-protected flag (not recommended)

Skip this change? [Y/n]
```

### Parsing Failures

```
⚠ Agent output parsing failed

The model produced invalid output. This sometimes happens with complex tasks.

Options:
  1. Retry with same context
  2. Simplify the current task
  3. Show raw output for debugging

Choice [1]:
```

---

## Large Project Mode

For projects with >100 files or >50,000 lines:

### Automatic Sharding

```
Large project detected (347 files, 89,000 lines)

Activating large project mode:
  • Task will be split into shards
  • Each shard processes max 20 files
  • Context summaries carried between shards
  • Checkpoints after each shard

Continue? [Y/n]
```

### Shard Execution

```
Executing shard 1/4: Core API changes

Files in scope:
  • src/api/routes.py
  • src/api/handlers.py
  • src/models/user.py

Progress: ████████░░ 80% (3/4 tasks)
```

### Context Summaries

Between shards, the system carries forward:
- Completed task summaries (not full code)
- File structure changes
- Key decisions made
- Remaining work items

This keeps context under token limits while maintaining coherence.

---

## Performance Constraints

### Memory Budget

| Component | Allocation |
|-----------|------------|
| Ollama (model) | ~8GB |
| Agent process | ~500MB |
| Context window | ~100MB |
| File buffers | ~200MB |
| **Total** | **<10GB** |

Reserved: 6GB for system and other applications.

### Response Latency

Target latencies (7B model, M1/M2/M3):

| Operation | Target | Timeout |
|-----------|--------|---------|
| First token | <2s | 30s |
| Simple completion | <10s | 60s |
| Complex generation | <60s | 300s |
| File operation | <100ms | 5s |

### Token Efficiency

- System prompt: ~500 tokens (fixed)
- Task context: adaptive (500-3000 tokens)
- File content: truncated/summarized if >2000 tokens
- Response: max 2048 tokens per completion

---

## Session Persistence

### Automatic Saves

Sessions are saved to `~/.local/share/agent/sessions/`:

```
sessions/
  session_20260203_143012/
    state.json          # Session state
    context.json        # Conversation context
    checkpoints/        # Rollback points
    diffs/              # Pending diffs
```

### Resume Flow

```bash
$ agent --resume session_20260203_143012

Resuming session from Feb 3, 2026 2:30 PM

Last activity: Implementing user authentication
Progress: 3/5 tasks completed
Pending changes: 2 files modified

Continue where you left off? [Y/n]
```

### Session Cleanup

Sessions older than 7 days are automatically cleaned up.
Use `agent sessions --keep ID` to preserve important sessions.

---

## Configuration

### Config Files

```
~/.config/agent/
  config.yaml           # Main configuration
  models.yaml          # Model definitions
  policies.yaml        # Safety policies
  limits.yaml          # Resource limits
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_WORKSPACE` | Default workspace | `.` |
| `AGENT_MODEL` | Default model | `qwen2.5-coder:7b` |
| `AGENT_OLLAMA_URL` | Ollama endpoint | `http://localhost:11434` |
| `AGENT_LOG_LEVEL` | Logging verbosity | `info` |

### Per-Project Config

Create `.agent.yaml` in project root:

```yaml
workspace:
  include:
    - src/
    - tests/
  exclude:
    - "*.pyc"
    - __pycache__/

limits:
  max_files_per_task: 10
  
model: codellama:7b-instruct
```

---

## Output Formatting

### Status Indicators

| Symbol | Meaning |
|--------|---------|
| `◐` | In progress (spinning) |
| `✓` | Success |
| `✗` | Error |
| `⚠` | Warning |
| `●` | Pending |
| `○` | Not started |

### Color Scheme

| Color | Usage |
|-------|-------|
| Green | Success, additions |
| Red | Errors, deletions |
| Yellow | Warnings, pending |
| Blue | Info, prompts |
| Dim | Secondary info |

### Diff Display

```diff
--- src/old.py
+++ src/new.py
@@ -10,6 +10,8 @@
 def existing_function():
     pass
 
+def new_function():
+    """Added by agent."""
+    return True
+
 def another_function():
     pass
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Interrupt current operation |
| `Ctrl+D` | Exit session (same as /exit) |
| `Ctrl+L` | Clear screen |
| `↑` / `↓` | Navigate history |
| `Tab` | Autocomplete commands |

### Interrupt Handling

```
^C

Operation interrupted.

Options:
  1. Discard current work and continue
  2. Save checkpoint and pause
  3. Exit without saving

Choice [2]:
```

---

## Example Sessions

### Simple Task (One-Shot)

```bash
$ agent "Add a .gitignore for Python projects"

Planning...
I'll create a comprehensive .gitignore for Python.

Creating: .gitignore (34 lines)

--- /dev/null
+++ .gitignore
@@ -0,0 +1,34 @@
+# Byte-compiled files
+__pycache__/
+*.py[cod]
+...

Apply? [Y/n] y

✓ Changes applied
✓ Session complete
```

### Interactive Session

```bash
$ agent

Welcome to Agent v1.0.0
Model: qwen2.5-coder:7b-instruct-q4_K_M
Workspace: /Users/dev/myproject

Type your task or /help for commands.

> Create a REST API for user management

Planning...

I'll build a REST API with the following structure:

1. Create User model in src/models/user.py
2. Create API routes in src/api/users.py  
3. Add validation schemas
4. Write unit tests

Proceed? [Y/n] y

◐ Implementing task 1/4: User model...

Done. Ready to review:

+ src/models/user.py (45 lines)

Apply? [Y/n] y
✓ Applied

◐ Implementing task 2/4: API routes...

Done. Ready to review:

+ src/api/users.py (89 lines)
~ src/api/__init__.py (+3 lines)

Apply? [Y/n] n

Changes discarded. Want me to try a different approach? [Y/n]
```

---

## Error Recovery

### Crash Recovery

If the agent crashes:

```bash
$ agent

⚠ Previous session ended unexpectedly

Session: session_20260203_143012
Last checkpoint: cp_20260203_143520

Options:
  1. Resume from last checkpoint
  2. Review what was pending
  3. Start fresh

Choice [1]: 1

Restored to checkpoint cp_20260203_143520
3/5 tasks completed

Continue? [Y/n]
```

### Rollback

```bash
> /rollback

Available checkpoints:
  1. cp_20260203_143522 - After task 3 (current)
  2. cp_20260203_143412 - After task 2
  3. cp_20260203_143201 - After planning

Rollback to [1-3]: 2

Rolling back to cp_20260203_143412...
✓ Restored 3 files
✓ Session state restored

You are now at: After task 2
```

---

## Accessibility

- All output is screen-reader compatible
- No reliance on color alone (symbols always accompany colors)
- Keyboard-navigable
- Configurable output verbosity
- Compatible with terminal multiplexers (tmux, screen)

---

## Telemetry

**All telemetry stays local.**

Stored in `~/.local/share/agent/telemetry/`:
- Session duration
- Token usage
- Success/failure rates
- Model performance

Use `agent stats` to view local telemetry.
Use `agent stats --clear` to delete all telemetry.

---

## Version History

| Version | Changes |
|---------|---------|
| 1.0.0 | Initial release |

---

## Appendix: Full Command Reference

```
agent - Local AI Coding Agent

USAGE:
    agent [OPTIONS] [TASK]
    agent <COMMAND>

COMMANDS:
    doctor      Check system prerequisites
    version     Show version information
    config      Manage configuration
    sessions    List and manage sessions
    stats       View usage statistics

OPTIONS:
    -w, --workspace PATH    Set workspace directory
    -r, --resume ID         Resume a previous session
    -m, --model NAME        Use specific model
    -v, --verbose           Increase output verbosity
    -q, --quiet             Minimal output
    --dry-run               Show what would be done
    --no-checkpoint         Disable automatic checkpoints
    --allow-protected       Allow modifying protected files
    -h, --help              Show help

EXAMPLES:
    agent                              Start interactive session
    agent "fix the bug in auth.py"     One-shot task
    agent --resume abc123              Resume session
    agent doctor                       Check prerequisites
```
