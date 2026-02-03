# Project Mode

Project mode enables continuous iterative development on the same project without recreating the executor for each task. This is ideal for building and refining a project incrementally.

## Overview

In **one-shot mode** (default), each task creates a new executor instance:
- Task 1: Create executor → Execute → Destroy
- Task 2: Create executor → Execute → Destroy
- Task 3: Create executor → Execute → Destroy

In **project mode**, the same executor persists across tasks:
- Task 1: Create executor → Execute
- Task 2: Same executor → Execute
- Task 3: Same executor → Execute → Destroy on exit

## Usage

### Enabling Project Mode

```bash
# Start interactive session
agent

# Enable project mode
/project on
```

### Using Project Mode

1. **Start with your initial project**:
   ```
   🔄 [Project] > Create a Python calculator with add, subtract, multiply, divide
   ```

2. **Continue iterating on the same project**:
   ```
   🔄 [Project] > Add support for square root and power operations
   ```

3. **Keep refining**:
   ```
   🔄 [Project] > Add input validation for division by zero
   ```

4. **Disable when done**:
   ```
   /project off
   ```

### Command Reference

| Command | Description |
|---------|-------------|
| `/project on` | Enable project mode |
| `/project off` | Disable project mode and cleanup executor |
| `/project` | Show current project mode status |

## Benefits

1. **Context Preservation**: The executor maintains context across tasks
2. **Performance**: Avoids overhead of creating new executors
3. **Continuity**: Seamless iterative development workflow
4. **Session Management**: Same run ID, telemetry, and checkpoints

## Workflow Example

```
$ agent

Welcome to Agent v0.1.0
Model: qwen2.5-coder:7b-instruct-q4_K_M
Workspace: /path/to/workspace

💡 Tip: Use /project on for continuous iterative development

> /project on
[Project Mode] Enabled - executor will persist across tasks

🔄 [Project] > Create a FastAPI web server with a hello endpoint

Planning...
Executing...
Done - 3/3 tasks • 12.4s • 4,521 tokens

Files Changed:
  + server.py
  + requirements.txt

🔄 [Project] > Add a POST endpoint to create users

[Project Mode] Continuing on same project...
Planning...
Executing...
Done - 2/2 tasks • 8.1s • 3,102 tokens

Files Changed:
  ~ server.py

🔄 [Project] > Add user authentication with JWT

[Project Mode] Continuing on same project...
Planning...
Executing...
Done - 4/4 tasks • 15.7s • 5,834 tokens

Files Changed:
  ~ server.py
  + auth.py
  ~ requirements.txt

🔄 [Project] > /project off
[Project Mode] Disabled

> /exit
```

## Implementation Details

### Executor Behavior

When in project mode:
- `execute()` is called for the first task (creates new executor)
- `execute_additional_task()` is called for subsequent tasks
- Same run ID, loop controller, telemetry, and checkpoints are reused
- LLM client is kept alive (no model unloading between tasks)

### State Management

The executor maintains:
- Task graph with all executed tasks
- Context from previous iterations
- Checkpoint history
- Telemetry data

### Cleanup

When project mode is disabled:
- Executor reference is cleared
- Resources are automatically cleaned up
- Next task will create a fresh executor

## Technical Notes

### Method Flow

**First Task (Project Start)**:
```python
executor = Executor(config, workspace, log_dir)
result = executor.execute(task, run_id)
```

**Subsequent Tasks**:
```python
# Same executor instance
result = executor.execute_additional_task(task)
```

### Key Differences from Continue

| Feature | Continue | Project Mode |
|---------|----------|--------------|
| Use Case | Resume paused execution | Add new tasks to project |
| Trigger | Max iterations reached | User provides new task |
| Method | `continue_execution()` | `execute_additional_task()` |
| Iterations | Extends existing limit | Full iteration budget for new task |
| Planning | Uses existing plan | Re-plans for new task |

## Best Practices

1. **Use for related tasks**: Project mode works best when tasks build on each other
2. **Monitor iterations**: Each task gets full iteration budget; watch for runaway loops
3. **Checkpoint frequently**: Project mode accumulates changes across tasks
4. **Clean exit**: Use `/project off` before switching to unrelated work
5. **RAM management**: Same executor means same model stays loaded

## Troubleshooting

### Executor State Issues

If you encounter unexpected behavior:
1. Disable project mode: `/project off`
2. Enable it again: `/project on`
3. Start fresh task

### Memory Issues

Long-running project sessions may accumulate state:
- Monitor RAM usage
- Use `/project off` periodically to reset
- Consider breaking into multiple sessions for very large projects

### Context Confusion

If the executor seems confused about project context:
- Provide explicit task descriptions
- Reference specific files that need changes
- Consider starting a new project mode session
