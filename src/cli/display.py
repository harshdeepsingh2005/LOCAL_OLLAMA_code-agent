"""
CLI Display Module

Rich terminal output formatting for the agent CLI.
Provides consistent, accessible output with proper symbols and colors.
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme


class StatusSymbol(str, Enum):
    """Status indicator symbols."""
    PROGRESS = "◐"
    SUCCESS = "✓"
    ERROR = "✗"
    WARNING = "⚠"
    PENDING = "●"
    NOT_STARTED = "○"
    INFO = "ℹ"
    PROMPT = ">"


# Custom theme for consistent styling
AGENT_THEME = Theme({
    "success": "green",
    "error": "red",
    "warning": "yellow",
    "info": "blue",
    "dim": "dim",
    "prompt": "bold cyan",
    "file.add": "green",
    "file.modify": "yellow",
    "file.delete": "red",
    "diff.add": "green",
    "diff.remove": "red",
    "diff.context": "dim",
})


class Display:
    """
    Rich terminal display manager.
    
    Provides consistent output formatting for all CLI operations.
    Thread-safe for use with async operations.
    """
    
    def __init__(
        self,
        quiet: bool = False,
        verbose: bool = False,
        no_color: bool = False,
    ) -> None:
        """
        Initialize display.
        
        Args:
            quiet: Minimal output mode
            verbose: Detailed output mode
            no_color: Disable color output
        """
        self.quiet = quiet
        self.verbose = verbose
        self.console = Console(
            theme=AGENT_THEME,
            force_terminal=None if not no_color else False,
            no_color=no_color,
        )
        self._live: Optional[Live] = None
    
    # =========================================================================
    # Status Messages
    # =========================================================================
    
    def success(self, message: str) -> None:
        """Display success message."""
        self.console.print(f"[success]{StatusSymbol.SUCCESS}[/success] {message}")
    
    def error(self, message: str, details: Optional[str] = None) -> None:
        """Display error message."""
        self.console.print(f"[error]{StatusSymbol.ERROR}[/error] {message}")
        if details and self.verbose:
            self.console.print(f"  [dim]{details}[/dim]")
    
    def warning(self, message: str) -> None:
        """Display warning message."""
        self.console.print(f"[warning]{StatusSymbol.WARNING}[/warning] {message}")
    
    def info(self, message: str) -> None:
        """Display info message."""
        if not self.quiet:
            self.console.print(f"[info]{StatusSymbol.INFO}[/info] {message}")
    
    def status(self, message: str) -> None:
        """Display status message (dim)."""
        if self.verbose:
            self.console.print(f"[dim]{message}[/dim]")
    
    # =========================================================================
    # Welcome & Headers
    # =========================================================================
    
    def welcome(self, version: str, model: str, workspace: Path) -> None:
        """Display welcome banner."""
        if self.quiet:
            return
            
        self.console.print()
        self.console.print("[bold]Agent[/bold] - Local AI Coding Assistant", style="bold blue")
        self.console.print(f"[dim]Version {version} • Model: {model}[/dim]")
        self.console.print(f"[dim]Workspace: {workspace}[/dim]")
        self.console.print()
        self.console.print("[dim]Type your task or /help for commands.[/dim]")
        self.console.print()
    
    def header(self, title: str) -> None:
        """Display section header."""
        self.console.print()
        self.console.print(Rule(title, style="blue"))
    
    def subheader(self, title: str) -> None:
        """Display subsection header."""
        self.console.print(f"\n[bold]{title}[/bold]")
    
    # =========================================================================
    # Input Prompts
    # =========================================================================
    
    def prompt(self, message: str = "") -> str:
        """Display input prompt and get user input."""
        try:
            if message:
                return self.console.input(f"[prompt]{message}[/prompt] ")
            return self.console.input(f"[prompt]{StatusSymbol.PROMPT}[/prompt] ")
        except EOFError:
            return "/exit"
        except KeyboardInterrupt:
            self.console.print()
            return ""
    
    def confirm(
        self,
        message: str,
        default: bool = True,
        allow_edit: bool = False,
        allow_reject: bool = False,
    ) -> str:
        """
        Display confirmation prompt.
        
        Returns:
            'y', 'n', 'e' (edit), 'r' (reject), or '?' (explain)
        """
        options = "[Y/n"
        if allow_edit:
            options += "/e(dit)"
        if allow_reject:
            options += "/r(eject)"
        options += "/?]"
        
        hint = "y" if default else "n"
        
        try:
            response = self.console.input(
                f"{message} {options} "
            ).strip().lower()
            
            if not response:
                return "y" if default else "n"
            if response in ("y", "yes"):
                return "y"
            if response in ("n", "no"):
                return "n"
            if response in ("e", "edit") and allow_edit:
                return "e"
            if response in ("r", "reject") and allow_reject:
                return "r"
            if response == "?":
                return "?"
            
            return "y" if default else "n"
            
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return "n"
    
    def choice(
        self,
        message: str,
        options: list[str],
        default: int = 1,
    ) -> int:
        """
        Display numbered choice prompt.
        
        Returns:
            1-indexed choice number
        """
        self.console.print(f"\n{message}\n")
        for i, option in enumerate(options, 1):
            self.console.print(f"  {i}. {option}")
        
        try:
            response = self.console.input(f"\nChoice [{default}]: ").strip()
            if not response:
                return default
            choice = int(response)
            if 1 <= choice <= len(options):
                return choice
            return default
        except (ValueError, EOFError, KeyboardInterrupt):
            return default
    
    # =========================================================================
    # Progress & Spinners
    # =========================================================================
    
    def spinner(self, message: str) -> Progress:
        """Create a spinner progress display."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        )
    
    def progress_bar(self, total: int, description: str = "") -> Progress:
        """Create a progress bar display."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        )
    
    def task_progress(self, current: int, total: int, description: str) -> None:
        """Display task progress inline."""
        self.console.print(
            f"[info]{StatusSymbol.PROGRESS}[/info] {description} ({current}/{total})"
        )
    
    # =========================================================================
    # Plan Display
    # =========================================================================
    
    def plan(self, summary: str, tasks: list[dict[str, Any]]) -> None:
        """Display execution plan."""
        self.console.print()
        self.console.print("[bold]Plan[/bold]")
        self.console.print()
        self.console.print(summary)
        self.console.print()
        
        for i, task in enumerate(tasks, 1):
            status_symbol = StatusSymbol.NOT_STARTED
            style = "dim"
            
            if task.get("status") == "completed":
                status_symbol = StatusSymbol.SUCCESS
                style = "success"
            elif task.get("status") == "in_progress":
                status_symbol = StatusSymbol.PROGRESS
                style = "info"
            elif task.get("status") == "failed":
                status_symbol = StatusSymbol.ERROR
                style = "error"
            
            self.console.print(
                f"  [{style}]{status_symbol}[/{style}] {i}. {task.get('title', 'Untitled')}"
            )
            if self.verbose and task.get("description"):
                self.console.print(f"     [dim]{task['description'][:80]}...[/dim]")
        
        self.console.print()
    
    # =========================================================================
    # Diff Display
    # =========================================================================
    
    def diff_summary(self, files: list[dict[str, Any]]) -> None:
        """Display summary of file changes."""
        self.console.print()
        self.console.print("[bold]Changes ready to apply:[/bold]")
        self.console.print()
        
        for f in files:
            change_type = f.get("change_type", "modify")
            path = f.get("path", "unknown")
            
            if change_type == "create":
                lines = f.get("lines", 0)
                self.console.print(f"  [file.add]+[/file.add] {path} [dim](new file, {lines} lines)[/dim]")
            elif change_type == "delete":
                self.console.print(f"  [file.delete]-[/file.delete] {path} [dim](deleted)[/dim]")
            else:
                added = f.get("lines_added", 0)
                removed = f.get("lines_removed", 0)
                self.console.print(
                    f"  [file.modify]~[/file.modify] {path} "
                    f"[dim](+{added}, -{removed} lines)[/dim]"
                )
        
        self.console.print()
    
    def diff(self, file_path: str, diff_content: str) -> None:
        """Display a unified diff."""
        self.console.print()
        self.console.print(f"[bold]{file_path}[/bold]")
        
        lines = diff_content.split("\n")
        for line in lines:
            if line.startswith("+++") or line.startswith("---"):
                self.console.print(f"[bold]{line}[/bold]")
            elif line.startswith("@@"):
                self.console.print(f"[info]{line}[/info]")
            elif line.startswith("+"):
                self.console.print(f"[diff.add]{line}[/diff.add]")
            elif line.startswith("-"):
                self.console.print(f"[diff.remove]{line}[/diff.remove]")
            else:
                self.console.print(f"[diff.context]{line}[/diff.context]")
        
        self.console.print()
    
    def diff_panel(self, title: str, diff_content: str) -> None:
        """Display diff in a panel."""
        syntax = Syntax(diff_content, "diff", theme="monokai", line_numbers=False)
        panel = Panel(syntax, title=title, border_style="blue")
        self.console.print(panel)
    
    # =========================================================================
    # Tables
    # =========================================================================
    
    def table(
        self,
        title: str,
        columns: list[str],
        rows: list[list[str]],
    ) -> None:
        """Display a table."""
        table = Table(title=title)
        
        for col in columns:
            table.add_column(col)
        
        for row in rows:
            table.add_row(*row)
        
        self.console.print(table)
    
    def key_value(self, data: dict[str, Any], title: Optional[str] = None) -> None:
        """Display key-value pairs."""
        if title:
            self.console.print(f"\n[bold]{title}[/bold]")
        
        for key, value in data.items():
            self.console.print(f"  [dim]{key}:[/dim] {value}")
    
    # =========================================================================
    # Help & Commands
    # =========================================================================
    
    def help(self, commands: dict[str, str]) -> None:
        """Display help for available commands."""
        self.console.print()
        self.console.print("[bold]Available Commands[/bold]")
        self.console.print()
        
        max_cmd_len = max(len(cmd) for cmd in commands.keys())
        
        for cmd, description in commands.items():
            padding = " " * (max_cmd_len - len(cmd) + 2)
            self.console.print(f"  [info]{cmd}[/info]{padding}[dim]{description}[/dim]")
        
        self.console.print()
    
    # =========================================================================
    # Errors & Recovery
    # =========================================================================
    
    def error_panel(
        self,
        title: str,
        message: str,
        causes: Optional[list[str]] = None,
        solutions: Optional[list[str]] = None,
    ) -> None:
        """Display detailed error information."""
        content = [message]
        
        if causes:
            content.append("\n[bold]Possible causes:[/bold]")
            for cause in causes:
                content.append(f"  • {cause}")
        
        if solutions:
            content.append("\n[bold]Solutions:[/bold]")
            for solution in solutions:
                content.append(f"  • {solution}")
        
        panel = Panel(
            "\n".join(content),
            title=f"[error]{StatusSymbol.ERROR} {title}[/error]",
            border_style="red",
        )
        self.console.print(panel)
    
    def recovery_options(
        self,
        options: list[str],
        default: int = 1,
    ) -> int:
        """Display recovery options after an error."""
        return self.choice("Options:", options, default)
    
    # =========================================================================
    # Session Info
    # =========================================================================
    
    def session_info(
        self,
        session_id: str,
        status: str,
        tasks_completed: int,
        tasks_total: int,
        tokens_used: int,
        tokens_limit: int,
    ) -> None:
        """Display session information."""
        self.console.print()
        self.console.print("[bold]Session Summary[/bold]")
        self.console.print()
        self.console.print(f"  Session ID: [info]{session_id}[/info]")
        self.console.print(f"  Status: {status}")
        self.console.print(f"  Progress: {tasks_completed}/{tasks_total} tasks")
        
        token_pct = (tokens_used / tokens_limit * 100) if tokens_limit > 0 else 0
        token_style = "success" if token_pct < 80 else "warning" if token_pct < 95 else "error"
        self.console.print(
            f"  Tokens: [{token_style}]{tokens_used:,}/{tokens_limit:,}[/{token_style}] "
            f"({token_pct:.0f}%)"
        )
        self.console.print()
    
    def checkpoint_info(self, checkpoint_id: str, description: str) -> None:
        """Display checkpoint creation info."""
        self.console.print(
            f"[success]{StatusSymbol.SUCCESS}[/success] "
            f"Checkpoint: [info]{checkpoint_id}[/info]"
        )
        if self.verbose:
            self.console.print(f"  [dim]{description}[/dim]")
    
    # =========================================================================
    # Token/Resource Warnings
    # =========================================================================
    
    def token_warning(self, used: int, limit: int) -> None:
        """Display token usage warning."""
        pct = (used / limit * 100) if limit > 0 else 0
        self.console.print()
        self.console.print(
            f"[warning]{StatusSymbol.WARNING}[/warning] "
            f"Token usage: {used:,}/{limit:,} ({pct:.0f}%)"
        )
    
    # =========================================================================
    # Markdown
    # =========================================================================
    
    def markdown(self, content: str) -> None:
        """Display markdown content."""
        md = Markdown(content)
        self.console.print(md)
    
    # =========================================================================
    # Raw Output
    # =========================================================================
    
    def raw(self, content: str) -> None:
        """Display raw content without formatting."""
        self.console.print(content)
    
    def newline(self) -> None:
        """Print a blank line."""
        self.console.print()
