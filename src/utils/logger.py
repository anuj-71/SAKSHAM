"""
Logging utilities using Rich for beautiful console output.
"""

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from typing import Optional

console = Console()


class Logger:
    """Rich-based logger for beautiful console output."""
    
    @staticmethod
    def info(message: str):
        console.print(f"[blue]ℹ[/blue] {message}")
    
    @staticmethod
    def success(message: str):
        console.print(f"[green]✓[/green] {message}")
    
    @staticmethod
    def warning(message: str):
        console.print(f"[yellow]⚠[/yellow] {message}")
    
    @staticmethod
    def error(message: str):
        console.print(f"[red]✗[/red] {message}")
    
    @staticmethod
    def header(message: str):
        console.print(f"\n[bold cyan]{message}[/bold cyan]")
    
    @staticmethod
    def panel(message: str, title: Optional[str] = None, style: str = "blue"):
        console.print(Panel(message, title=title, border_style=style))
    
    @staticmethod
    def table(title: str, columns: list, rows: list):
        """Create and print a rich table."""
        table = Table(title=title, show_header=True, header_style="bold magenta")
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*[str(x) for x in row])
        console.print(table)
    
    @staticmethod
    def progress_bar(description: str = "Processing..."):
        """Create a progress bar context manager."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        )


logger = Logger()
