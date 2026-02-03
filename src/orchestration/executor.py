"""
Executor Module

Coordinates the execution of agents and manages the workflow.
This is the main orchestration engine that ties everything together.

Design Decisions:
- Single entry point for execution
- Checkpointing before each agent
- Deterministic execution order
- Full audit trail
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agents import (
    AgentContext,
    AgentStatus,
    AgentType,
    CodeChange,
    CoderAgent,
    CoderInput,
    CoderOutput,
    FixerAgent,
    FixerInput,
    FixerOutput,
    PlannerAgent,
    PlannerInput,
    PlannerOutput,
    ReviewerAgent,
    ReviewerInput,
    ReviewerOutput,
    ReviewVerdict,
    Subtask,
)
from src.config import Configuration
from src.core import (
    ContextBudget,
    ContextManager,
    ContextPriority,
    ContextType,
    DiffEngine,
    DiffType,
    FileDiff,
    FileGuard,
    FileGuardPolicy,
    LLMClient,
    TelemetryCollector,
)
from src.orchestration.loop_controller import (
    LoopController,
    LoopState,
    TerminationReason,
)
from src.orchestration.rollback import Checkpoint, RollbackManager
from src.orchestration.task_graph import TaskGraph, TaskNode, TaskStatus


@dataclass
class ExecutionResult:
    """Result of a complete execution run."""
    run_id: str
    success: bool
    
    # Task information
    task_description: str
    subtasks_total: int = 0
    subtasks_completed: int = 0
    subtasks_failed: int = 0
    
    # Execution metrics
    total_tokens: int = 0
    total_duration_ms: float = 0
    iterations: int = 0
    
    # Files
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    
    # Status
    termination_reason: str = ""
    error: str | None = None
    
    # Continuation flag: True if max iterations reached and user can continue
    needs_continuation: bool = True
    
    # Timestamps
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


class ExecutionError(Exception):
    """Raised when execution fails."""
    pass


class Executor:
    """
    Main execution engine for the agent system.
    
    Orchestrates:
    - Agent execution in sequence
    - Task graph management
    - Checkpointing and rollback
    - Telemetry collection
    
    Thread Safety: NOT thread-safe. Designed for sequential execution.
    """
    
    def __init__(
        self,
        config: Configuration,
        workspace_root: Path,
        log_dir: Path,
    ) -> None:
        """
        Initialize the executor.
        
        Args:
            config: Configuration instance
            workspace_root: Root directory for file operations
            log_dir: Directory for logs and checkpoints
        """
        self._config = config
        self._workspace_root = workspace_root.resolve()
        self._log_dir = log_dir
        
        # Ensure directories exist
        self._workspace_root.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize agents (stateless)
        self._planner = PlannerAgent()
        self._coder = CoderAgent()
        self._reviewer = ReviewerAgent()
        self._fixer = FixerAgent()
        
        # These will be initialized per run
        self._run_id: str | None = None
        self._llm_client: LLMClient | None = None
        self._file_guard: FileGuard | None = None
        self._diff_engine: DiffEngine | None = None
        self._telemetry: TelemetryCollector | None = None
        self._rollback: RollbackManager | None = None
        self._loop: LoopController | None = None
        self._task_graph: TaskGraph | None = None
    
    def _initialize_run(self, run_id: str | None = None) -> str:
        """Initialize components for a new run."""
        self._run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        
        # Initialize LLM client
        self._llm_client = LLMClient(
            base_url=self._config.models.ollama.base_url,
            timeout=float(self._config.models.ollama.timeout_seconds),
        )
        
        # Initialize file guard
        file_policy = FileGuardPolicy(
            allowed_roots=[self._workspace_root],
            blocked_patterns=self._config.policies.file_access.blocked_patterns,
            allowed_extensions=self._config.policies.file_access.allowed_extensions,
            max_file_size_bytes=self._config.limits.files.max_read_size_bytes,
            max_files_per_run=self._config.limits.files.max_modified_per_run,
        )
        self._telemetry = TelemetryCollector(
            run_id=self._run_id,
            log_dir=self._log_dir,
        )
        self._file_guard = FileGuard(
            workspace_root=self._workspace_root,
            policy=file_policy,
            telemetry=self._telemetry,
        )
        
        # Initialize diff engine
        self._diff_engine = DiffEngine(
            file_guard=self._file_guard,
            telemetry=self._telemetry,
        )
        
        # Initialize rollback manager
        checkpoint_dir = self._log_dir / "checkpoints" / self._run_id
        self._rollback = RollbackManager(
            workspace_root=self._workspace_root,
            checkpoint_dir=checkpoint_dir,
            file_guard=self._file_guard,
            max_checkpoints=self._config.limits.memory.max_checkpoints_per_run,
        )
        
        # Initialize loop controller
        self._loop = LoopController(
            config=self._config,
            run_id=self._run_id,
        )
        
        return self._run_id
    
    def _create_agent_context(self, agent_type: AgentType) -> AgentContext:
        """Create execution context for an agent."""
        return AgentContext(
            run_id=self._run_id or "",
            agent_type=agent_type,
            config=self._config,
            llm_client=self._llm_client,  # type: ignore
            context_manager=ContextManager(
                budget=ContextBudget(
                    total_tokens=self._config.limits.tokens.max_per_completion,
                    system_reserved=self._config.limits.context.system_reserved,
                    response_reserved=self._config.limits.context.response_reserved,
                )
            ),
            telemetry=self._telemetry,
            max_tokens=self._config.get_token_limit_for_agent(agent_type.value),
            max_retries=self._config.limits.iterations.max_agent_retries,
        )
    
    def _checkpoint(self, description: str, task_id: str | None = None) -> Checkpoint:
        """Create a checkpoint."""
        assert self._rollback is not None
        assert self._run_id is not None
        assert self._file_guard is not None
        assert self._telemetry is not None
        
        return self._rollback.create_checkpoint(
            run_id=self._run_id,
            description=description,
            task_graph_state=self._task_graph.to_dict() if self._task_graph else {},
            current_task_id=task_id,
            modified_files=list(self._file_guard.get_modified_files()),
            tokens_used=self._telemetry.run_metrics.tokens.total,
        )
    
    def execute(
        self,
        task_description: str,
        run_id: str | None = None,
    ) -> ExecutionResult:
        """
        Execute a complete task from description to implementation.
        
        This is the main entry point for running the agent system.
        
        Args:
            task_description: Natural language description of the task
            run_id: Optional run ID (generated if not provided)
            
        Returns:
            ExecutionResult with status and metrics
        """
        # Initialize run
        run_id = self._initialize_run(run_id)
        
        result = ExecutionResult(
            run_id=run_id,
            success=False,
            task_description=task_description,
        )
        
        assert self._telemetry is not None
        assert self._loop is not None
        assert self._llm_client is not None
        
        # Record start
        self._telemetry.record_run_start(
            task=task_description,
            config={"workspace": str(self._workspace_root)},
        )
        
        try:
            # Check LLM health
            if not self._llm_client.health_check():
                raise ExecutionError("Ollama is not running or not accessible")
            
            # Start execution loop
            self._loop.start()
            
            # Phase 1: Planning
            plan = self._execute_planning(task_description)
            if plan is None:
                # Iteration limit reached during planning
                result.needs_continuation = True
                result.error = "Max iterations reached during planning"
                return result
            if plan.status != AgentStatus.SUCCESS or not plan.subtasks:
                self._loop.complete_failure(
                    TerminationReason.FATAL_ERROR,
                    f"Planning failed: {plan.error or 'No subtasks generated'}"
                )
                result.error = plan.error
                return result
            
            # Create task graph
            self._task_graph = TaskGraph.from_subtasks(plan.subtasks)
            result.subtasks_total = len(plan.subtasks)
            
            # Checkpoint after planning
            self._checkpoint("After planning", None)
            
            # Phase 2: Execute tasks
            for task_node in self._task_graph.iter_execution_order():
                if not self._loop.is_running:
                    # Check if we're paused for user continuation
                    if self._loop.is_paused:
                        result.needs_continuation = True
                        result.error = f"Max iterations ({self._loop.iteration_count}) reached"
                        return result
                    break
                
                # Execute single task through code-review-fix cycle
                task_success = self._execute_task(task_node)
                
                if task_success is None:
                    # Iteration limit reached - needs user continuation
                    result.needs_continuation = True
                    result.error = f"Max iterations ({self._loop.iteration_count}) reached"
                    return result
                elif task_success:
                    result.subtasks_completed += 1
                else:
                    result.subtasks_failed += 1
                    # Propagate failure to dependents
                    self._task_graph.propagate_failure(task_node.id)
                
                # Reset fix counter for next task
                self._loop.reset_fix_counter()
            
            # Determine final status
            stats = self._task_graph.get_stats()
            if stats.failed + stats.blocked == 0:
                self._loop.complete_success()
                result.success = True
            else:
                self._loop.complete_failure(
                    TerminationReason.UNRECOVERABLE_FAILURE,
                    f"{stats.failed} tasks failed, {stats.blocked} blocked"
                )
            
        except Exception as e:
            self._loop.complete_failure(
                TerminationReason.FATAL_ERROR,
                str(e)
            )
            result.error = str(e)
            self._telemetry.record_error(str(e))
        
        finally:
            # Finalize result
            result.completed_at = datetime.now(timezone.utc)
            result.total_duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
            result.iterations = self._loop.iteration_count
            result.total_tokens = self._telemetry.run_metrics.tokens.total
            result.termination_reason = (
                self._loop.termination_reason.value 
                if self._loop.termination_reason else "unknown"
            )
            
            # Get modified files
            if self._file_guard:
                for f in self._file_guard.state.files_created:
                    result.files_created.append(str(f))
                for f in self._file_guard.state.files_modified:
                    result.files_modified.append(str(f))
            
            # Record run end
            self._telemetry.record_run_end(
                success=result.success,
                summary=f"Completed {result.subtasks_completed}/{result.subtasks_total} tasks"
            )
            
            # Cleanup - DON'T close LLM client if we might continue
            if self._llm_client and not result.needs_continuation:
                self._llm_client.close()
        
        return result
    
    def continue_execution(self, additional_iterations: int = 10) -> ExecutionResult:
        """
        Continue execution after hitting max iterations.
        
        This method resumes from where the executor paused, WITHOUT
        restarting planning or re-running completed tasks.
        
        Args:
            additional_iterations: Number of additional iterations to allow
            
        Returns:
            ExecutionResult with updated status
        """
        assert self._loop is not None, "Cannot continue - executor not initialized"
        assert self._task_graph is not None, "Cannot continue - no task graph"
        assert self._telemetry is not None
        assert self._run_id is not None
        
        # Extend iteration limit
        self._loop.extend_iterations(additional_iterations)
        
        # Re-initialize LLM client if it was closed
        if self._llm_client is None:
            self._llm_client = LLMClient(
                base_url=self._config.models.ollama.base_url,
                timeout=float(self._config.models.ollama.timeout_seconds),
            )
        
        # Create result tracking continuation progress
        result = ExecutionResult(
            run_id=self._run_id,
            success=False,
            task_description="(continued)",
            subtasks_total=len(self._task_graph),
        )
        
        # Count already completed tasks
        stats = self._task_graph.get_stats()
        result.subtasks_completed = stats.completed
        result.subtasks_failed = stats.failed
        
        try:
            # Resume execution from pending/running tasks
            for task_node in self._task_graph.iter_execution_order():
                # Skip already completed or failed tasks
                if task_node.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, 
                                        TaskStatus.SKIPPED, TaskStatus.BLOCKED):
                    continue
                
                if not self._loop.is_running:
                    if self._loop.is_paused:
                        result.needs_continuation = True
                        result.error = f"Max iterations ({self._loop.iteration_count}) reached"
                        return result
                    break
                
                # Execute the task (resume if it was running)
                task_success = self._execute_task(task_node)
                
                if task_success is None:
                    result.needs_continuation = True
                    result.error = f"Max iterations ({self._loop.iteration_count}) reached"
                    return result
                elif task_success:
                    result.subtasks_completed += 1
                else:
                    result.subtasks_failed += 1
                    self._task_graph.propagate_failure(task_node.id)
                
                self._loop.reset_fix_counter()
            
            # Check final status
            stats = self._task_graph.get_stats()
            if stats.failed + stats.blocked == 0 and stats.completed == stats.total_tasks:
                self._loop.complete_success()
                result.success = True
            elif stats.pending == 0 and stats.running == 0:
                # All tasks processed but some failed
                self._loop.complete_failure(
                    TerminationReason.UNRECOVERABLE_FAILURE,
                    f"{stats.failed} tasks failed, {stats.blocked} blocked"
                )
            # else: still have pending tasks, might need more iterations
            
        except Exception as e:
            self._loop.complete_failure(TerminationReason.FATAL_ERROR, str(e))
            result.error = str(e)
            self._telemetry.record_error(str(e))
        
        finally:
            result.completed_at = datetime.now(timezone.utc)
            result.total_duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
            result.iterations = self._loop.iteration_count
            result.total_tokens = self._telemetry.run_metrics.tokens.total
            result.termination_reason = (
                self._loop.termination_reason.value
                if self._loop.termination_reason else "in_progress"
            )
            
            if self._file_guard:
                result.files_created = [str(f) for f in self._file_guard.state.files_created]
                result.files_modified = [str(f) for f in self._file_guard.state.files_modified]
            
            # Only close LLM if we're done
            if self._llm_client and not result.needs_continuation:
                self._llm_client.close()
        
        return result
    
    def execute_additional_task(self, task_description: str) -> ExecutionResult:
        """
        Execute an additional task in the same project session.
        
        This method allows adding new tasks to an ongoing project WITHOUT
        reinitializing the executor. All previous context, checkpoints, and
        state are preserved.
        
        Use this for iterative project development where multiple tasks
        are executed sequentially in the same session.
        
        Args:
            task_description: New task to execute
            
        Returns:
            ExecutionResult for this specific task
            
        Raises:
            ExecutionError: If executor not initialized
        """
        if not self._run_id or not self._loop or not self._telemetry:
            raise ExecutionError("Cannot execute additional task - executor not initialized")
        
        # Reset the loop for new task if it was in terminal state
        if self._loop.is_terminal:
            self._loop._state = LoopState.PLANNING
            self._loop._termination_reason = None
            self._loop._termination_message = None
            self._loop._needs_user_continue = False
        
        # Re-initialize LLM client if needed
        if self._llm_client is None:
            self._llm_client = LLMClient(
                base_url=self._config.models.ollama.base_url,
                timeout=float(self._config.models.ollama.timeout_seconds),
            )
        
        result = ExecutionResult(
            run_id=self._run_id,
            success=False,
            task_description=task_description,
        )
        
        try:
            # Phase 1: Planning for the new task
            plan = self._execute_planning(task_description)
            if plan is None:
                result.needs_continuation = True
                result.error = "Max iterations reached during planning"
                return result
            if plan.status != AgentStatus.SUCCESS or not plan.subtasks:
                result.error = plan.error or "No subtasks generated"
                return result
            
            # Create or extend task graph
            new_task_graph = TaskGraph.from_subtasks(plan.subtasks)
            result.subtasks_total = len(plan.subtasks)
            
            # Checkpoint before new task execution
            self._checkpoint(f"Starting additional task: {task_description}", None)
            
            # Phase 2: Execute new tasks
            for task_node in new_task_graph.iter_execution_order():
                if not self._loop.is_running:
                    if self._loop.is_paused:
                        result.needs_continuation = True
                        result.error = f"Max iterations ({self._loop.iteration_count}) reached"
                        return result
                    break
                
                task_success = self._execute_task(task_node)
                
                if task_success is None:
                    result.needs_continuation = True
                    result.error = f"Max iterations ({self._loop.iteration_count}) reached"
                    return result
                elif task_success:
                    result.subtasks_completed += 1
                else:
                    result.subtasks_failed += 1
                    new_task_graph.propagate_failure(task_node.id)
                
                self._loop.reset_fix_counter()
            
            # Check completion
            stats = new_task_graph.get_stats()
            if stats.failed + stats.blocked == 0:
                result.success = True
            
        except Exception as e:
            result.error = str(e)
            self._telemetry.record_error(str(e))
        
        finally:
            result.completed_at = datetime.now(timezone.utc)
            result.total_duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
            result.iterations = self._loop.iteration_count
            result.total_tokens = self._telemetry.run_metrics.tokens.total
            
            if self._file_guard:
                result.files_created = [str(f) for f in self._file_guard.state.files_created]
                result.files_modified = [str(f) for f in self._file_guard.state.files_modified]
            
            # Don't close LLM - keep session alive for next task
        
        return result
    
    def _execute_planning(self, task_description: str) -> PlannerOutput | None:
        """Execute the planning phase. Returns None if iteration limit reached."""
        assert self._loop is not None
        
        # Get workspace context
        workspace_context = self._get_workspace_context()
        
        # Create planner input
        planner_input = PlannerInput(
            task_id="planning",
            run_id=self._run_id or "",
            task_description=task_description,
            workspace_context=workspace_context,
        )
        
        # Execute planner
        iteration = self._loop.begin_iteration("planner")
        if iteration is None:
            return None
        
        context = self._create_agent_context(AgentType.PLANNER)
        
        output = self._planner.execute(planner_input, context)
        
        self._loop.end_iteration(
            iteration,
            success=output.status == AgentStatus.SUCCESS,
            error=output.error,
            tokens_used=output.tokens_used,
        )
        
        return output
    
    def _execute_task(self, task_node: TaskNode) -> bool | None:
        """
        Execute a single task through the code-review-fix cycle.
        
        Returns:
            True if task completed successfully
            False if task failed
            None if iteration limit reached (needs user continuation)
        """
        assert self._loop is not None
        
        task_node.mark_running()
        self._checkpoint(f"Starting task {task_node.id}", task_node.id)
        
        # Get relevant file contents
        file_contents = self._get_file_contents(task_node.subtask.target_files)
        
        # Code phase
        coder_output = self._execute_coder(task_node.subtask, file_contents)
        if coder_output is None:
            # Iteration limit reached
            return None
        if coder_output.status != AgentStatus.SUCCESS:
            task_node.mark_failed(coder_output.error or "Coding failed")
            return False
        
        # Review-fix loop
        current_changes = coder_output.changes
        
        while self._loop.is_running:
            # Review phase
            reviewer_output = self._execute_reviewer(
                task_node.subtask,
                current_changes,
                coder_output.implementation_notes,
            )
            
            if reviewer_output is None:
                # Iteration limit reached
                return None
            
            if reviewer_output.status != AgentStatus.SUCCESS:
                task_node.mark_failed(reviewer_output.error or "Review failed")
                return False
            
            # ============================================================
            # TERMINAL STATE CHECK: This is the authoritative stop condition
            # ============================================================
            # 
            # The reviewer's task_complete field is the EXPLICIT terminal signal.
            # When task_complete=True, the task is DONE - no more iterations.
            #
            # This check MUST come first, before any other verdict handling,
            # to ensure we never accidentally continue after approval.
            # ============================================================
            
            if reviewer_output.task_complete:
                # TERMINAL: Task is complete. Apply changes and EXIT the loop.
                # This is the normal successful completion path.
                success = self._apply_changes(current_changes)
                if success:
                    task_node.mark_completed({
                        "changes": len(current_changes),
                        "verdict": reviewer_output.verdict.value,
                    })
                    # Return True to signal successful completion
                    # The outer loop will move to the next task
                    return True
                else:
                    task_node.mark_failed("Failed to apply approved changes")
                    return False
            
            # ============================================================
            # NON-TERMINAL STATES: Only reached if task_complete=False
            # ============================================================
            
            if reviewer_output.verdict == ReviewVerdict.REJECT:
                # ABORT: Fundamental problems, cannot continue
                task_node.mark_failed(
                    f"Changes rejected by reviewer: {reviewer_output.summary}"
                )
                return False
            
            # REQUEST_CHANGES: Invoke fixer to address issues
            # This is the ONLY path that continues the loop
            if not self._loop.can_fix_again():
                task_node.mark_failed("Max fix iterations exceeded")
                return False
            
            # Fix phase - address reviewer's issues
            fixer_output = self._execute_fixer(
                current_changes,
                reviewer_output.issues,
                file_contents,
            )
            
            if fixer_output is None:
                # Iteration limit reached
                return None
            
            if fixer_output.status != AgentStatus.SUCCESS:
                task_node.mark_failed(fixer_output.error or "Fix failed")
                return False
            
            # Use fixed changes for next review iteration
            current_changes = fixer_output.fixed_changes
            # Loop continues to review the fixed changes
        
        # Check if we're paused for user continuation
        if self._loop.is_paused:
            return None
        
        task_node.mark_failed("Execution interrupted")
        return False
    
    def _execute_coder(
        self,
        subtask: Subtask,
        file_contents: dict[str, str],
    ) -> CoderOutput | None:
        """Execute the coder agent. Returns None if iteration limit reached."""
        assert self._loop is not None
        
        coder_input = CoderInput(
            task_id=subtask.id,
            run_id=self._run_id or "",
            subtask=subtask,
            file_contents=file_contents,
        )
        
        iteration = self._loop.begin_iteration("coder", subtask.id)
        if iteration is None:
            # Max iterations reached - needs user continuation
            return None
        
        context = self._create_agent_context(AgentType.CODER)
        
        output = self._coder.execute(coder_input, context)
        
        self._loop.end_iteration(
            iteration,
            success=output.status == AgentStatus.SUCCESS,
            error=output.error,
            tokens_used=output.tokens_used,
        )
        
        return output
    
    def _execute_reviewer(
        self,
        subtask: Subtask,
        changes: list[CodeChange],
        implementation_notes: str,
    ) -> ReviewerOutput | None:
        """Execute the reviewer agent. Returns None if iteration limit reached."""
        assert self._loop is not None
        
        # Get original file contents
        original_files = {}
        for change in changes:
            if change.original_content:
                original_files[change.file_path] = change.original_content
        
        reviewer_input = ReviewerInput(
            task_id=subtask.id,
            run_id=self._run_id or "",
            subtask=subtask,
            code_changes=changes,
            original_files=original_files,
            implementation_notes=implementation_notes,
        )
        
        iteration = self._loop.begin_iteration("reviewer", subtask.id)
        if iteration is None:
            return None
        
        context = self._create_agent_context(AgentType.REVIEWER)
        
        output = self._reviewer.execute(reviewer_input, context)
        
        self._loop.end_iteration(
            iteration,
            success=output.status == AgentStatus.SUCCESS,
            error=output.error,
            tokens_used=output.tokens_used,
        )
        
        return output
    
    def _execute_fixer(
        self,
        changes: list[CodeChange],
        issues: list,
        file_contents: dict[str, str],
    ) -> FixerOutput | None:
        """Execute the fixer agent. Returns None if iteration limit reached."""
        assert self._loop is not None
        
        fixer_input = FixerInput(
            task_id="fix",
            run_id=self._run_id or "",
            original_changes=changes,
            review_issues=issues,
            file_contents=file_contents,
        )
        
        iteration = self._loop.begin_iteration("fixer")
        if iteration is None:
            return None
        
        context = self._create_agent_context(AgentType.FIXER)
        
        output = self._fixer.execute(fixer_input, context)
        
        self._loop.end_iteration(
            iteration,
            success=output.status == AgentStatus.SUCCESS,
            error=output.error,
            tokens_used=output.tokens_used,
        )
        
        return output
    
    def _apply_changes(self, changes: list[CodeChange]) -> bool:
        """Apply code changes through the diff engine."""
        assert self._diff_engine is not None
        assert self._file_guard is not None
        
        for change in changes:
            try:
                file_path = self._workspace_root / change.file_path
                
                if change.change_type == "delete":
                    diff = self._diff_engine.create_deletion_diff(
                        file_path,
                        description=change.description,
                    )
                else:
                    diff = self._diff_engine.create_diff(
                        file_path,
                        change.new_content,
                        description=change.description,
                    )
                
                result = self._diff_engine.apply(diff)
                if not result.success:
                    return False
                    
            except Exception as e:
                if self._telemetry:
                    self._telemetry.record_error(f"Failed to apply change: {e}")
                return False
        
        return True
    
    def _get_workspace_context(self) -> dict[str, Any]:
        """Gather workspace context for planning."""
        context: dict[str, Any] = {
            "files": [],
            "structure": "",
        }
        
        if self._file_guard:
            try:
                files = self._file_guard.list_dir(self._workspace_root)
                context["files"] = [str(f.relative_to(self._workspace_root)) for f in files]
            except Exception:
                pass
        
        return context
    
    def _get_file_contents(self, file_paths: list[str]) -> dict[str, str]:
        """Read contents of specified files."""
        contents: dict[str, str] = {}
        
        if not self._file_guard:
            return contents
        
        for path in file_paths:
            try:
                full_path = self._workspace_root / path
                if self._file_guard.exists(full_path):
                    contents[path] = self._file_guard.read(full_path)
            except Exception:
                pass
        
        return contents
    
    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Rollback to a specific checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to rollback to
            
        Returns:
            True if rollback succeeded
        """
        if not self._rollback:
            return False
        
        checkpoint = self._rollback.get_checkpoint(checkpoint_id)
        if not checkpoint:
            return False
        
        # Rollback file system
        self._rollback.rollback_to(checkpoint_id)
        
        # Restore task graph
        if self._task_graph and checkpoint.task_graph_state:
            self._task_graph = TaskGraph.from_dict(checkpoint.task_graph_state)
        
        return True
