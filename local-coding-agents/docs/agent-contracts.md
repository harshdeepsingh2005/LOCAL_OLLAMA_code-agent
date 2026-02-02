# Agent Contracts

This document defines the strict input/output contracts for each agent in the system.

## Contract Philosophy

Every agent adheres to these principles:
1. **Typed I/O**: All inputs and outputs use Pydantic models
2. **JSON Communication**: LLM outputs structured JSON
3. **Deterministic Parsing**: Output parsing never fails silently
4. **Error Propagation**: Failures carry context

## Common Types

### AgentStatus
```python
class AgentStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"
```

### AgentContext
Provided to every agent execution:
```python
class AgentContext:
    run_id: str           # Current run identifier
    agent_type: AgentType # Type of agent executing
    config: Configuration # System configuration
    llm_client: LLMClient # LLM client instance
    context_manager: ContextManager
    telemetry: TelemetryCollector
    max_tokens: int       # Token limit for this agent
    max_retries: int      # Retry limit
```

---

## Planner Agent

### Purpose
Decomposes a high-level task into ordered, atomic subtasks.

### Input Schema
```python
class PlannerInput(BaseModel):
    task_id: str              # Unique task identifier
    run_id: str               # Parent run identifier
    task_description: str     # Natural language task
    workspace_context: dict   # File listing, structure info
    constraints: list[str]    # Optional constraints
```

### Output Schema
```python
class Subtask(BaseModel):
    id: str                   # e.g., "task_001"
    description: str          # What to accomplish
    target_files: list[str]   # Files to create/modify
    dependencies: list[str]   # IDs of prerequisite tasks
    estimated_complexity: str # "low", "medium", "high"
    acceptance_criteria: list[str]

class PlannerOutput(BaseModel):
    task_id: str
    status: AgentStatus
    subtasks: list[Subtask]   # Ordered list of subtasks
    reasoning: str            # Why this decomposition
    tokens_used: int
    error: str | None
```

### LLM Prompt Structure
```
System: You are a planning agent. Decompose tasks into subtasks.
Output JSON with this structure: { subtasks: [...], reasoning: "..." }

User: 
Task: {task_description}
Workspace:
{workspace_context}

Output subtasks in execution order with dependencies.
```

### Example

**Input:**
```json
{
  "task_id": "plan_001",
  "task_description": "Create a REST API for user management",
  "workspace_context": {
    "files": ["README.md", "requirements.txt"],
    "structure": "Empty Python project"
  }
}
```

**Output:**
```json
{
  "task_id": "plan_001",
  "status": "success",
  "subtasks": [
    {
      "id": "task_001",
      "description": "Create User model with SQLAlchemy",
      "target_files": ["src/models/user.py"],
      "dependencies": [],
      "estimated_complexity": "low",
      "acceptance_criteria": ["User model with id, email, name fields"]
    },
    {
      "id": "task_002", 
      "description": "Create user CRUD endpoints",
      "target_files": ["src/routes/users.py"],
      "dependencies": ["task_001"],
      "estimated_complexity": "medium",
      "acceptance_criteria": ["GET/POST/PUT/DELETE endpoints"]
    }
  ],
  "reasoning": "Started with data model, then built API layer",
  "tokens_used": 1250
}
```

---

## Coder Agent

### Purpose
Generates code changes for a single subtask.

### Input Schema
```python
class CoderInput(BaseModel):
    task_id: str
    run_id: str
    subtask: Subtask           # The subtask to implement
    file_contents: dict[str, str]  # Current file contents
    context_files: dict[str, str]  # Related files for context
```

### Output Schema
```python
class CodeChange(BaseModel):
    file_path: str            # Relative path
    change_type: str          # "create", "modify", "delete"
    description: str          # What changed
    original_content: str | None
    new_content: str          # Full new content
    diff: str                 # Unified diff

class CoderOutput(BaseModel):
    task_id: str
    status: AgentStatus
    changes: list[CodeChange]
    implementation_notes: str  # Explanation of approach
    tokens_used: int
    error: str | None
```

### LLM Prompt Structure
```
System: You are a coding agent. Generate code changes as diffs.
Output JSON: { changes: [...], implementation_notes: "..." }

User:
Subtask: {subtask.description}
Target Files: {subtask.target_files}

Current Contents:
{file_contents}

Generate complete file contents with unified diffs.
```

### Example

**Input:**
```json
{
  "task_id": "task_001",
  "subtask": {
    "id": "task_001",
    "description": "Create User model with SQLAlchemy",
    "target_files": ["src/models/user.py"]
  },
  "file_contents": {}
}
```

**Output:**
```json
{
  "task_id": "task_001",
  "status": "success",
  "changes": [
    {
      "file_path": "src/models/user.py",
      "change_type": "create",
      "description": "Created User SQLAlchemy model",
      "original_content": null,
      "new_content": "from sqlalchemy import Column, Integer, String...",
      "diff": "--- /dev/null\n+++ src/models/user.py\n@@ -0,0 +1,15 @@\n+from sqlalchemy..."
    }
  ],
  "implementation_notes": "Used SQLAlchemy ORM with standard fields",
  "tokens_used": 890
}
```

---

## Reviewer Agent

### Purpose
Evaluates code changes against acceptance criteria and best practices.

### Input Schema
```python
class ReviewerInput(BaseModel):
    task_id: str
    run_id: str
    subtask: Subtask
    code_changes: list[CodeChange]
    original_files: dict[str, str]
    implementation_notes: str
```

### Output Schema
```python
class IssueSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    SUGGESTION = "suggestion"

class ReviewIssue(BaseModel):
    severity: IssueSeverity
    file_path: str
    line_start: int | None
    line_end: int | None
    description: str
    suggestion: str

class ReviewVerdict(str, Enum):
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    REJECT = "reject"

class ReviewerOutput(BaseModel):
    task_id: str
    status: AgentStatus
    verdict: ReviewVerdict
    issues: list[ReviewIssue]
    summary: str
    tokens_used: int
    error: str | None
```

### LLM Prompt Structure
```
System: You are a code reviewer. Evaluate changes against criteria.
Output JSON: { verdict: "...", issues: [...], summary: "..." }

User:
Subtask: {subtask.description}
Acceptance Criteria: {subtask.acceptance_criteria}

Changes:
{code_changes}

Evaluate for correctness, style, and criteria compliance.
```

### Verdict Guidelines

| Verdict | When to Use |
|---------|-------------|
| APPROVE | All criteria met, no critical/major issues |
| REQUEST_CHANGES | Fixable issues found, worth retrying |
| REJECT | Fundamental problems, would need complete rewrite |

### Example

**Output:**
```json
{
  "task_id": "task_001",
  "status": "success",
  "verdict": "request_changes",
  "issues": [
    {
      "severity": "major",
      "file_path": "src/models/user.py",
      "line_start": 10,
      "line_end": 10,
      "description": "Missing email validation",
      "suggestion": "Add email validator using pydantic or regex"
    }
  ],
  "summary": "Good structure but missing input validation",
  "tokens_used": 650
}
```

---

## Fixer Agent

### Purpose
Addresses specific review issues with minimal changes.

### Input Schema
```python
class FixerInput(BaseModel):
    task_id: str
    run_id: str
    original_changes: list[CodeChange]
    review_issues: list[ReviewIssue]
    file_contents: dict[str, str]
```

### Output Schema
```python
class FixerOutput(BaseModel):
    task_id: str
    status: AgentStatus
    fixed_changes: list[CodeChange]
    fixes_applied: list[str]      # Issue descriptions fixed
    unfixable_issues: list[str]   # Issues that couldn't be fixed
    tokens_used: int
    error: str | None
```

### LLM Prompt Structure
```
System: You are a code fixer. Address specific issues minimally.
Output JSON: { fixed_changes: [...], fixes_applied: [...] }

User:
Issues to fix:
{review_issues}

Current Code:
{original_changes}

Make minimal changes to address each issue.
```

### Example

**Input:**
```json
{
  "review_issues": [
    {
      "severity": "major",
      "file_path": "src/models/user.py",
      "description": "Missing email validation",
      "suggestion": "Add email validator"
    }
  ]
}
```

**Output:**
```json
{
  "task_id": "task_001",
  "status": "success",
  "fixed_changes": [
    {
      "file_path": "src/models/user.py",
      "change_type": "modify",
      "description": "Added email validation",
      "diff": "@@ -10,1 +10,5 @@\n-    email: str\n+    email: str\n+\n+    @validator('email')\n+    def validate_email(cls, v):\n+        if '@' not in v: raise ValueError('Invalid email')\n+        return v"
    }
  ],
  "fixes_applied": ["Added email validation"],
  "unfixable_issues": [],
  "tokens_used": 450
}
```

---

## Contract Validation

### Pre-execution Checks
1. All required fields present
2. Task IDs match
3. File paths are relative
4. Dependencies reference existing tasks

### Post-execution Checks
1. Status is set
2. Token count is positive
3. Error set only if status != SUCCESS
4. Output matches schema

### Error Handling
```python
# Agent execution pattern
try:
    output = agent.execute(input, context)
    if output.status == AgentStatus.SUCCESS:
        # Process output
    else:
        # Handle failure with output.error
except ValidationError as e:
    # Schema validation failed
    output = OutputType(status=AgentStatus.ERROR, error=str(e))
except Exception as e:
    # Unexpected error
    output = OutputType(status=AgentStatus.ERROR, error=str(e))
```

---

## Token Budgets

Default allocation per agent (configurable):

| Agent | Default Tokens | Use Case |
|-------|---------------|----------|
| Planner | 4096 | Task decomposition |
| Coder | 8192 | Code generation |
| Reviewer | 4096 | Code review |
| Fixer | 4096 | Targeted fixes |

Total run budget: 50,000 tokens (configurable)

---

## Version History

| Version | Changes |
|---------|---------|
| 1.0.0 | Initial contract definitions |
