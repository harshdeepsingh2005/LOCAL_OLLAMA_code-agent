# Project Mode - Quick Reference

## What is Project Mode?

A workflow mode that keeps the same executor alive across multiple tasks, enabling continuous iterative development on a single project.

## Quick Commands

```bash
/project on      # Enable project mode
/project off     # Disable project mode
/project         # Show current status
```

## Visual Indicators

| Indicator | Meaning |
|-----------|---------|
| `🔄 [Project]` in prompt | Project mode is active |
| `[Project Mode] Starting new project...` | First task in project mode |
| `[Project Mode] Continuing on same project...` | Subsequent task in project mode |

## Typical Workflow

```
1. agent                          # Start interactive session
2. /project on                    # Enable project mode
3. Task 1: "Create web scraper"   # Executor created
4. Task 2: "Add caching"          # Same executor
5. Task 3: "Add tests"            # Same executor
6. /project off                   # Cleanup
```

## When to Use

✅ **Use Project Mode When**:
- Building a project incrementally
- Making related changes across multiple tasks
- Refining and extending existing code
- Iterating on a design

❌ **Don't Use Project Mode When**:
- Working on unrelated tasks
- Switching between projects
- One-off quick fixes
- Testing different approaches

## Comparison

| Feature | One-Shot Mode | Project Mode |
|---------|---------------|--------------|
| Executor | New per task | Persists across tasks |
| Context | Task-specific | Project-wide |
| Run ID | New per task | Same for all tasks |
| Use Case | Independent tasks | Related iterations |
| Performance | Setup overhead | Faster iterations |

## Key Benefits

1. **Context Preservation** - Executor remembers project state
2. **Performance** - No executor recreation overhead  
3. **Continuity** - Natural iterative workflow
4. **Session Unity** - Single run ID and checkpoints

## Examples

### Example 1: Web API Development

```
🔄 [Project] > Create FastAPI server with health check
🔄 [Project] > Add user CRUD endpoints
🔄 [Project] > Add JWT authentication
🔄 [Project] > Add rate limiting
🔄 [Project] > Add comprehensive tests
```

### Example 2: Data Pipeline

```
🔄 [Project] > Create data ingestion script
🔄 [Project] > Add data validation
🔄 [Project] > Add transformation pipeline
🔄 [Project] > Add error handling and logging
🔄 [Project] > Add unit tests
```

### Example 3: CLI Tool

```
🔄 [Project] > Create CLI with argument parsing
🔄 [Project] > Add config file support
🔄 [Project] > Add interactive mode
🔄 [Project] > Add help documentation
🔄 [Project] > Add packaging setup
```

## Tips

💡 **Enable early** - Turn on project mode before starting the first task

💡 **Descriptive tasks** - Be specific about what each iteration should add

💡 **Disable between projects** - Use `/project off` before switching contexts

💡 **Monitor progress** - Watch file changes accumulate across tasks

💡 **Use checkpoints** - Project mode accumulates checkpoints for rollback

## Troubleshooting

### Executor seems confused
- `/project off` then `/project on` to reset

### Memory issues
- Disable project mode periodically on long sessions

### Unexpected changes
- Check `/diff` to see accumulated changes
- Use `/rollback` if needed

## Learn More

- Full guide: [docs/project-mode.md](docs/project-mode.md)
- Implementation: [PROJECT_MODE_IMPLEMENTATION.md](PROJECT_MODE_IMPLEMENTATION.md)
- README: [README.md](README.md#project-mode)
