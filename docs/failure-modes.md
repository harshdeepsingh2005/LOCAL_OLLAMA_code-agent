# Failure Modes and Recovery

This document catalogs all known failure modes, their detection, and recovery strategies.

## Overview

The system is designed to fail gracefully with these guarantees:
1. **No data loss**: Checkpoints preserve state
2. **No corruption**: Atomic operations, validated diffs
3. **Full audit trail**: Every operation logged
4. **Deterministic recovery**: Resume from any checkpoint

---

## Failure Categories

### 1. LLM Failures

#### 1.1 Ollama Unavailable

**Detection:**
- Health check fails at startup
- Connection refused on API call
- Timeout on completion request

**Recovery:**
```
1. Retry with exponential backoff (3 attempts)
2. If still failing, terminate with clear error
3. State saved to allow resume when Ollama starts
```

**User Action:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

#### 1.2 Model Not Available

**Detection:**
- Model not in `ollama list` output
- 404 on model pull

**Recovery:**
```
1. Log warning with model name
2. Attempt to use fallback model (if configured)
3. If no fallback, terminate with instructions
```

**User Action:**
```bash
# Pull required model
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

#### 1.3 Invalid LLM Response

**Detection:**
- JSON parsing fails
- Schema validation fails
- Required fields missing

**Recovery:**
```
1. Log the invalid response
2. Retry with clearer prompt (up to 3 times)
3. If still failing, mark task as failed
4. Continue with next task if possible
```

#### 1.4 Token Limit Exceeded

**Detection:**
- Response truncated mid-JSON
- Token counter exceeds budget

**Recovery:**
```
1. Reduce context size
2. Retry with minimal context
3. If still failing, skip current subtask
```

---

### 2. File System Failures

#### 2.1 Permission Denied

**Detection:**
- OSError with EACCES/EPERM
- FileGuard policy rejection

**Recovery:**
```
1. Log the permission error with path
2. Skip the file operation
3. Mark subtask as failed
4. Continue with independent subtasks
```

**User Action:**
```bash
# Check file permissions
ls -la /path/to/file

# Fix permissions if needed
chmod u+rw /path/to/file
```

#### 2.2 Disk Full

**Detection:**
- OSError with ENOSPC
- Write operation fails

**Recovery:**
```
1. Log disk space error
2. Clean up old checkpoints
3. Retry write
4. If still failing, abort run
```

**User Action:**
```bash
# Check disk space
df -h

# Clean up logs/checkpoints
lca logs cleanup --older-than 7d
```

#### 2.3 File Modified Externally

**Detection:**
- Hash mismatch on read
- Conflict with checkpoint

**Recovery:**
```
1. Log the conflict
2. Create backup of current state
3. Abort current task
4. User must resolve manually
```

#### 2.4 Path Escape Attempt

**Detection:**
- Path resolves outside workspace
- Symlink points outside workspace
- Path contains ".."

**Recovery:**
```
1. Block the operation immediately
2. Log security warning
3. Continue with other tasks
```

---

### 3. Execution Failures

#### 3.1 Planning Failure

**Detection:**
- Planner returns no subtasks
- Planner returns invalid structure
- All subtasks have cycles

**Recovery:**
```
1. Log planning output
2. Retry with simplified prompt
3. If still failing, abort run
```

**Symptoms:**
- Empty subtask list
- Subtasks reference non-existent dependencies
- Circular dependency detected

#### 3.2 Coding Failure

**Detection:**
- Coder returns invalid diff
- Diff doesn't apply cleanly
- Syntax errors in generated code

**Recovery:**
```
1. Log the invalid code
2. Retry with explicit constraints
3. If 3 retries fail, mark subtask failed
4. Continue with next subtask
```

#### 3.3 Review Infinite Loop

**Detection:**
- Fix iteration count exceeds limit
- Same issues reported repeatedly
- No progress between iterations

**Recovery:**
```
1. Detect repetition in issues
2. Force REJECT after max iterations
3. Move to next subtask
```

**Configuration:**
```yaml
limits:
  iterations:
    max_fix_iterations: 5
```

#### 3.4 Timeout

**Detection:**
- Run duration exceeds limit
- Single agent call exceeds limit

**Recovery:**
```
1. Graceful shutdown
2. Save current state to checkpoint
3. Log timeout location
4. User can resume
```

---

### 4. State Management Failures

#### 4.1 Checkpoint Corruption

**Detection:**
- JSON parse error on load
- Missing required fields
- File hash mismatch

**Recovery:**
```
1. Log corruption details
2. Fall back to previous checkpoint
3. If no valid checkpoints, start fresh
```

#### 4.2 Resume Failure

**Detection:**
- Run state not found
- Checkpoint not found
- Workspace state diverged

**Recovery:**
```
1. Log the mismatch
2. Offer to start fresh or abort
3. Keep corrupted state for analysis
```

---

## Error Codes

| Code | Category | Description |
|------|----------|-------------|
| E001 | LLM | Ollama connection failed |
| E002 | LLM | Model not available |
| E003 | LLM | Invalid response format |
| E004 | LLM | Token limit exceeded |
| E010 | FS | Permission denied |
| E011 | FS | Disk full |
| E012 | FS | External modification |
| E013 | FS | Path escape blocked |
| E020 | EXEC | Planning failed |
| E021 | EXEC | Coding failed |
| E022 | EXEC | Review loop exceeded |
| E023 | EXEC | Timeout |
| E030 | STATE | Checkpoint corrupted |
| E031 | STATE | Resume failed |

---

## Diagnostic Commands

### Check System Health
```bash
lca status
```

Shows:
- Ollama connection status
- Available models
- Disk space
- Recent errors

### View Run Logs
```bash
# List recent runs
lca logs list

# Show specific run
lca logs show run_abc123 --format detailed

# Export for analysis
lca logs export run_abc123 ./debug.json
```

### Inspect Checkpoints
```bash
# List checkpoints for a run
lca checkpoints list run_abc123

# Show checkpoint details
lca checkpoints show chk_456
```

### Validate Configuration
```bash
lca config validate
```

---

## Recovery Procedures

### Procedure 1: Resume Failed Run

```bash
# 1. Check the failure reason
lca logs show run_abc123

# 2. If recoverable, resume
lca resume run_abc123

# 3. If not, rollback and retry
lca rollback run_abc123 chk_previous
lca run "same task" --run-id run_abc123_retry
```

### Procedure 2: Recover from Corruption

```bash
# 1. List available checkpoints
lca checkpoints list run_abc123

# 2. Find last good checkpoint
lca checkpoints show chk_good

# 3. Rollback
lca rollback run_abc123 chk_good

# 4. Resume
lca resume run_abc123 --from-checkpoint chk_good
```

### Procedure 3: Debug Invalid Output

```bash
# 1. Enable verbose logging
export LCA_LOG_LEVEL=DEBUG

# 2. Run with dry-run first
lca run "task" --dry-run

# 3. Check telemetry
cat logs/telemetry/run_*.jsonl | jq 'select(.event_type == "llm_completion")'

# 4. Inspect raw LLM output
cat logs/debug/run_*/llm_responses/*.json
```

---

## Prevention Strategies

### 1. Pre-flight Checks

Before each run:
```python
def preflight_checks():
    # Check Ollama
    assert llm_client.health_check()
    
    # Check model availability
    models = llm_client.list_models()
    assert config.default_model in models
    
    # Check workspace
    assert workspace.exists()
    assert os.access(workspace, os.W_OK)
    
    # Check disk space (need 1GB free)
    usage = shutil.disk_usage(workspace)
    assert usage.free > 1_000_000_000
```

### 2. Defensive Coding

All agents:
```python
def execute(self, input, context):
    try:
        # Validate input
        validated = InputModel.model_validate(input)
        
        # Execute with timeout
        with timeout(context.max_tokens):
            result = self._execute_impl(validated, context)
        
        # Validate output
        return OutputModel.model_validate(result)
        
    except ValidationError as e:
        return OutputModel(status=FAILED, error=str(e))
    except TimeoutError:
        return OutputModel(status=TIMEOUT, error="Agent timed out")
```

### 3. Checkpoint Strategy

```python
# Checkpoint at safe points
CHECKPOINT_POINTS = [
    "after_planning",      # Task graph created
    "after_each_subtask",  # Subtask completed
    "before_applying",     # About to modify files
]

def should_checkpoint(point: str, state: RunState) -> bool:
    return (
        point in CHECKPOINT_POINTS and
        state.iteration_count > 0 and
        not state.has_recent_checkpoint(minutes=5)
    )
```

---

## Telemetry for Failures

Every failure is logged with:

```json
{
    "timestamp": "2024-01-15T10:30:00Z",
    "run_id": "run_abc123",
    "event_type": "error",
    "error_code": "E021",
    "error_message": "Coding failed: Invalid diff format",
    "context": {
        "agent": "coder",
        "task_id": "task_003",
        "iteration": 2,
        "tokens_used": 1250
    },
    "recovery_action": "retry_with_constraints",
    "stack_trace": "..."
}
```

---

## Known Limitations

1. **No automatic model switching**: If default model fails, manual intervention required
2. **No conflict resolution**: External file changes require manual handling
3. **Memory limits**: Large codebases may exceed context window
4. **Sequential only**: No parallel execution to reduce complexity

---

## Version History

| Version | Changes |
|---------|---------|
| 1.0.0 | Initial failure mode documentation |
