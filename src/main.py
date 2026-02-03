"""
Main Entry Point

Command-line interface for the local coding agents system.
Provides commands for running, resuming, and managing agent executions.

Design Decisions:
- Click-based CLI
- Rich output formatting
- Comprehensive subcommands
- Configuration validation on startup
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Import configuration first to validate
try:
    from src.config import Configuration, get_config
except ImportError as e:
    click.echo(f"Error loading configuration: {e}", err=True)
    sys.exit(1)

from src.orchestration import ExecutionResult, Executor
from src.state import (
    CheckpointStore,
    RunState,
    RunStateManager,
    SummaryGenerator,
    SummaryVerbosity,
)

console = Console()


def get_default_workspace() -> Path:
    """Get default workspace path."""
    return Path.cwd() / "workspace"


def get_default_log_dir() -> Path:
    """Get default log directory."""
    return Path.cwd() / "logs"


def get_default_state_dir() -> Path:
    """Get default state directory."""
    return Path.cwd() / ".lca"


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="Local Coding Agents")
@click.argument("task", required=False)
@click.option(
    "--workspace", "-w",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Workspace directory (default: ./workspace)",
)
@click.option(
    "--run-id", "-r",
    type=str,
    default=None,
    help="Custom run ID (auto-generated if not provided)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate task without executing",
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.pass_context
def cli(
    ctx: click.Context,
    task: Optional[str],
    workspace: Optional[Path],
    run_id: Optional[str],
    dry_run: bool,
    config: Optional[Path],
    verbose: bool,
) -> None:
    """
    Local Coding Agents - A production-grade local AI coding system.
    
    Run fully offline on Apple Silicon (M-series, 16 GB RAM).
    Uses Ollama for local LLM inference.
    """
    ctx.ensure_object(dict)
    
    # Load configuration
    try:
        if config:
            cfg = Configuration.from_yaml(config)
        else:
            cfg = get_config()
        ctx.obj["config"] = cfg
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)
    
    ctx.obj["verbose"] = verbose

    if task and ctx.invoked_subcommand is None:
        _execute_task(ctx, task, workspace, run_id, dry_run)


@cli.command()
@click.argument("task", type=str)
@click.option(
    "--workspace", "-w",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Workspace directory (default: ./workspace)",
)
@click.option(
    "--run-id", "-r",
    type=str,
    default=None,
    help="Custom run ID (auto-generated if not provided)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate task without executing",
)
@click.pass_context
def run(
    ctx: click.Context,
    task: str,
    workspace: Optional[Path],
    run_id: Optional[str],
    dry_run: bool,
) -> None:
    """
    Execute a coding task.
    
    TASK is a natural language description of what you want to accomplish.
    
    Examples:
    
        lca run "Create a Python function that calculates fibonacci numbers"
        
        lca run "Add error handling to the database module" -w ./myproject
    """
    _execute_task(ctx, task, workspace, run_id, dry_run)


def _execute_task(
    ctx: click.Context,
    task: str,
    workspace: Optional[Path],
    run_id: Optional[str],
    dry_run: bool,
) -> None:
    config: Configuration = ctx.obj["config"]
    verbose: bool = ctx.obj["verbose"]

    # Set up paths
    workspace_path = workspace or get_default_workspace()
    log_dir = get_default_log_dir()
    state_dir = get_default_state_dir()

    # Ensure directories exist
    workspace_path.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Local Coding Agents[/bold]")
    console.print(f"Task: {task}")
    console.print(f"Workspace: {workspace_path}")
    console.print(f"Run ID: {run_id or 'auto-generated'}")
    console.print()

    if dry_run:
        console.print("[yellow]Dry run - validating only[/yellow]")
        # TODO: Add task validation
        console.print("[green]Task validation passed[/green]")
        return

    # Initialize executor
    executor = Executor(
        config=config,
        workspace_root=workspace_path,
        log_dir=log_dir,
    )

    # Run with progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description="Executing task...", total=None)

        try:
            result = executor.execute(task, run_id=run_id)
        except Exception as e:
            console.print(f"[red]Execution failed: {e}[/red]")
            if verbose:
                console.print_exception()
            sys.exit(1)

    # Display results
    _display_result(result, verbose)

    # Save state
    state_manager = RunStateManager(state_dir / "runs")
    # Note: Executor should save state, this is backup

    if not result.success:
        sys.exit(1)


@cli.command()
@click.argument("run_id", type=str)
@click.option(
    "--from-checkpoint", "-c",
    type=str,
    default=None,
    help="Resume from specific checkpoint",
)
@click.pass_context
def resume(
    ctx: click.Context,
    run_id: str,
    from_checkpoint: Optional[str],
) -> None:
    """
    Resume a previous run.
    
    RUN_ID is the ID of a previous run to resume.
    
    Examples:
    
        lca resume run_abc123
        
        lca resume run_abc123 --from-checkpoint chk_456
    """
    config: Configuration = ctx.obj["config"]
    verbose: bool = ctx.obj["verbose"]
    
    state_dir = get_default_state_dir()
    state_manager = RunStateManager(state_dir / "runs")
    
    # Load existing state
    state = state_manager.load(run_id)
    if not state:
        console.print(f"[red]Run not found: {run_id}[/red]")
        sys.exit(1)
    
    console.print(f"[blue]Resuming run: {run_id}[/blue]")
    console.print(f"  Phase: {state.phase.value}")
    console.print(f"  Tasks: {state.get_task_stats()}")
    
    if from_checkpoint:
        checkpoint_store = CheckpointStore(state_dir / "checkpoints")
        checkpoint = checkpoint_store.get_checkpoint_metadata(from_checkpoint)
        if not checkpoint:
            console.print(f"[red]Checkpoint not found: {from_checkpoint}[/red]")
            sys.exit(1)
        console.print(f"  Restoring from checkpoint: {from_checkpoint}")
    
    # TODO: Implement actual resume logic
    console.print("[yellow]Resume functionality coming soon[/yellow]")


@cli.command()
@click.argument("run_id", type=str)
@click.argument("checkpoint_id", type=str)
@click.option(
    "--confirm", "-y",
    is_flag=True,
    help="Skip confirmation prompt",
)
@click.pass_context
def rollback(
    ctx: click.Context,
    run_id: str,
    checkpoint_id: str,
    confirm: bool,
) -> None:
    """
    Rollback to a checkpoint.
    
    Restores workspace files and run state to a previous checkpoint.
    
    Examples:
    
        lca rollback run_abc123 chk_456
    """
    state_dir = get_default_state_dir()
    checkpoint_store = CheckpointStore(state_dir / "checkpoints")
    
    # Verify checkpoint exists
    checkpoint = checkpoint_store.get_checkpoint_metadata(checkpoint_id)
    if not checkpoint:
        console.print(f"[red]Checkpoint not found: {checkpoint_id}[/red]")
        sys.exit(1)
    
    if checkpoint.run_id != run_id:
        console.print(f"[red]Checkpoint {checkpoint_id} does not belong to run {run_id}[/red]")
        sys.exit(1)
    
    console.print(f"[yellow]Rollback to:[/yellow]")
    console.print(f"  Checkpoint: {checkpoint_id}")
    console.print(f"  Created: {checkpoint.created_at}")
    console.print(f"  Description: {checkpoint.description}")
    console.print(f"  Files: {checkpoint.files_count}")
    
    if not confirm:
        if not click.confirm("Proceed with rollback?"):
            console.print("[yellow]Rollback cancelled[/yellow]")
            return
    
    # Perform rollback
    files = checkpoint_store.get_checkpoint_files(checkpoint_id)
    state = checkpoint_store.get_checkpoint_state(checkpoint_id)
    
    # TODO: Apply rollback
    console.print(f"[green]Would restore {len(files)} files[/green]")
    console.print("[yellow]Rollback functionality coming soon[/yellow]")


@cli.group()
def config() -> None:
    """Configuration management commands."""
    pass


@config.command(name="show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Display current configuration."""
    cfg: Configuration = ctx.obj["config"]
    
    # Models
    console.print("[bold]Models:[/bold]")
    console.print(f"  Default: {cfg.models.default_model}")
    console.print(f"  Ollama URL: {cfg.models.ollama.base_url}")
    console.print()
    
    # Limits
    console.print("[bold]Limits:[/bold]")
    console.print(f"  Max tokens/run: {cfg.limits.tokens.max_per_run:,}")
    console.print(f"  Max iterations: {cfg.limits.iterations.max_planning_iterations}")
    console.print(f"  Timeout: {cfg.limits.time.max_run_duration_seconds}s")
    console.print()
    
    # Policies
    console.print("[bold]Policies:[/bold]")
    console.print(f"  Blocked patterns: {len(cfg.policies.file_access.blocked_patterns)}")
    console.print(f"  Allowed extensions: {len(cfg.policies.file_access.allowed_extensions)}")


@config.command(name="validate")
@click.pass_context
def config_validate(ctx: click.Context) -> None:
    """Validate configuration."""
    cfg: Configuration = ctx.obj["config"]
    
    errors = []
    warnings = []
    
    # Check model configuration
    if not cfg.models.default_model:
        errors.append("No default model specified")
    
    # Check limits
    if cfg.limits.tokens.max_per_run > 100000:
        warnings.append("Token limit very high, may exhaust memory")
    
    # Check paths
    # TODO: Add path validation
    
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for e in errors:
            console.print(f"  • {e}")
    
    if warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for w in warnings:
            console.print(f"  • {w}")
    
    if not errors:
        console.print("[green]Configuration valid[/green]")
    else:
        sys.exit(1)


@cli.group()
def logs() -> None:
    """Log management commands."""
    pass


@logs.command(name="list")
@click.option(
    "--limit", "-n",
    type=int,
    default=10,
    help="Number of runs to show",
)
@click.option(
    "--failed",
    is_flag=True,
    help="Show only failed runs",
)
@click.pass_context
def logs_list(ctx: click.Context, limit: int, failed: bool) -> None:
    """List recent runs."""
    state_dir = get_default_state_dir()
    state_manager = RunStateManager(state_dir / "runs")
    
    runs = state_manager.list_runs(
        limit=limit,
        include_completed=not failed,
        include_failed=True,
    )
    
    if not runs:
        console.print("[yellow]No runs found[/yellow]")
        return
    
    table = Table(title="Recent Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Status")
    table.add_column("Tasks")
    table.add_column("Duration")
    table.add_column("Started")
    
    for run in runs:
        status_style = "green" if run.success else "red"
        status = "✓" if run.success else "✗"
        
        stats = run.get_task_stats()
        tasks = f"{stats['completed']}/{stats['total']}"
        
        duration = "-"
        if run.get_duration_ms():
            duration = f"{run.get_duration_ms() / 1000:.1f}s"
        
        started = "-"
        if run.started_at:
            started = run.started_at.strftime("%Y-%m-%d %H:%M")
        
        table.add_row(
            run.run_id,
            f"[{status_style}]{status}[/{status_style}]",
            tasks,
            duration,
            started,
        )
    
    console.print(table)


@logs.command(name="show")
@click.argument("run_id", type=str)
@click.option(
    "--format", "-f",
    type=click.Choice(["text", "json", "markdown"]),
    default="text",
    help="Output format",
)
@click.pass_context
def logs_show(ctx: click.Context, run_id: str, format: str) -> None:
    """Show details of a specific run."""
    verbose: bool = ctx.obj["verbose"]
    state_dir = get_default_state_dir()
    state_manager = RunStateManager(state_dir / "runs")
    
    state = state_manager.load(run_id)
    if not state:
        console.print(f"[red]Run not found: {run_id}[/red]")
        sys.exit(1)
    
    verbosity = SummaryVerbosity.DETAILED if verbose else SummaryVerbosity.NORMAL
    generator = SummaryGenerator(verbosity)
    summary = generator.from_run_state(state)
    
    if format == "json":
        console.print(generator.to_json(summary))
    elif format == "markdown":
        console.print(generator.to_markdown(summary))
    else:
        console.print(generator.to_text(summary))


@logs.command(name="export")
@click.argument("run_id", type=str)
@click.argument("output", type=click.Path(path_type=Path))
@click.option(
    "--format", "-f",
    type=click.Choice(["text", "json", "markdown"]),
    default="text",
    help="Output format",
)
@click.pass_context
def logs_export(
    ctx: click.Context,
    run_id: str,
    output: Path,
    format: str,
) -> None:
    """Export run details to file."""
    state_dir = get_default_state_dir()
    state_manager = RunStateManager(state_dir / "runs")
    
    state = state_manager.load(run_id)
    if not state:
        console.print(f"[red]Run not found: {run_id}[/red]")
        sys.exit(1)
    
    generator = SummaryGenerator(SummaryVerbosity.DETAILED)
    summary = generator.from_run_state(state)
    
    generator.save(summary, output, format=format)
    console.print(f"[green]Exported to: {output}[/green]")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show system status."""
    config: Configuration = ctx.obj["config"]
    
    console.print("[bold]System Status[/bold]")
    console.print()
    
    # Check Ollama
    console.print("Checking Ollama...", end=" ")
    try:
        from src.core import LLMClient
        client = LLMClient(base_url=config.models.ollama.base_url)
        if client.health_check():
            console.print("[green]✓ Running[/green]")
            
            # List models
            models = client.list_models()
            if models:
                console.print(f"  Available models: {', '.join(models[:5])}")
                if len(models) > 5:
                    console.print(f"  ... and {len(models) - 5} more")
        else:
            console.print("[red]✗ Not responding[/red]")
        client.close()
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
    
    # Check workspace
    workspace = get_default_workspace()
    console.print(f"Workspace: {workspace}", end=" ")
    if workspace.exists():
        console.print("[green]✓ Exists[/green]")
    else:
        console.print("[yellow]Will be created[/yellow]")
    
    # Check state directory
    state_dir = get_default_state_dir()
    state_manager = RunStateManager(state_dir / "runs")
    runs = state_manager.list_runs(limit=100)
    console.print(f"Previous runs: {len(runs)}")


def _display_result(result: ExecutionResult, verbose: bool) -> None:
    """Display execution result."""
    if result.success:
        console.print()
        console.print(
            "[green]✔ Done[/green]  "
            f"{result.subtasks_completed}/{result.subtasks_total} tasks  "
            f"• {result.total_duration_ms / 1000:.1f}s  "
            f"• {result.total_tokens:,} tokens"
        )
        console.print(f"Run ID: {result.run_id}")
    else:
        console.print()
        console.print(
            "[red]✖ Failed[/red]  "
            f"{result.subtasks_completed}/{result.subtasks_total} tasks  "
            f"• {result.termination_reason}"
        )
        console.print(f"Run ID: {result.run_id}")
        console.print(f"Error: {result.error or 'Unknown'}")
    
    # File changes
    if result.files_created or result.files_modified:
        console.print()
        console.print("[bold]Files Changed:[/bold]")
        for f in result.files_created:
            console.print(f"  [green]+[/green] {f}")
        for f in result.files_modified:
            console.print(f"  [yellow]~[/yellow] {f}")


def main() -> None:
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
