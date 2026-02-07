"""
Output formatting module for pysearch.

This module handles the formatting and rendering of search results in different output
formats including plain text, JSON, and rich console output with syntax highlighting.
It provides a unified interface for result presentation across CLI and API usage.

Key Functions:
    format_result: Main entry point for formatting results in any supported format
    to_json_bytes: Fast JSON serialization using orjson
    format_text: Plain text formatting with optional highlighting
    render_highlight_console: Rich console output with syntax highlighting

Supported Output Formats:
    - TEXT: Plain text with line numbers and file paths
    - JSON: Structured JSON for programmatic processing
    - HIGHLIGHT: Rich console output with syntax highlighting and colors

Key Features:
    - Fast JSON serialization with orjson
    - Syntax highlighting based on file type detection
    - Match span highlighting within code lines
    - Configurable context display
    - Performance-optimized for large result sets
    - Memory-efficient streaming for large outputs

Example:
    Basic formatting:
        >>> from pysearch.formatter import format_result
        >>> from pysearch.types import OutputFormat
        >>>
        >>> # Format as plain text
        >>> text_output = format_result(search_results, OutputFormat.TEXT)
        >>> print(text_output)
        >>>
        >>> # Format as JSON
        >>> json_output = format_result(search_results, OutputFormat.JSON)
        >>> print(json_output.decode('utf-8'))

    Rich console output:
        >>> from pysearch.formatter import render_highlight_console
        >>> from rich.console import Console
        >>>
        >>> console = Console()
        >>> render_highlight_console(search_results, console)
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import orjson
from rich.console import Console
from rich.syntax import Syntax

from ..core.types import OutputFormat, SearchResult
from .helpers import highlight_spans

# File extension to Pygments lexer name mapping
_EXTENSION_LEXER_MAP: dict[str, str] = {
    ".py": "python", ".pyw": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".kt": "kotlin", ".scala": "scala",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cxx": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".go": "go", ".rs": "rust",
    ".php": "php", ".rb": "ruby", ".swift": "swift",
    ".sh": "bash", ".bash": "bash", ".zsh": "zsh",
    ".ps1": "powershell", ".sql": "sql",
    ".html": "html", ".htm": "html", ".css": "css",
    ".scss": "scss", ".sass": "sass", ".less": "less",
    ".xml": "xml", ".json": "json",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".md": "markdown", ".r": "r", ".R": "r",
    ".m": "matlab",
}


def _detect_lexer(file_path: Path) -> str:
    """Detect Pygments lexer name from file extension."""
    return _EXTENSION_LEXER_MAP.get(file_path.suffix.lower(), "text")


def to_json_bytes(result: SearchResult) -> bytes:
    """
    Convert search results to JSON bytes using fast orjson serialization.

    This function provides high-performance JSON serialization optimized for
    search results. It handles Path objects and nested dataclasses correctly
    while maintaining compatibility with standard JSON parsers.

    Args:
        result: SearchResult object containing search results and statistics

    Returns:
        JSON-encoded bytes with pretty formatting (indented)

    Example:
        >>> json_bytes = to_json_bytes(search_results)
        >>> json_str = json_bytes.decode('utf-8')
        >>> print(json_str)

    Note:
        Uses orjson for performance, which is significantly faster than
        the standard library json module for large result sets.
    """
    def default(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        return asdict(obj)

    # Convert manually to avoid asdict recursion issues for nested dataclasses within lists
    payload = {
        "items": [
            {
                "file": str(it.file),
                "start_line": it.start_line,
                "end_line": it.end_line,
                "lines": it.lines,
                "match_spans": [(li, [a, b]) for li, (a, b) in it.match_spans],
            }
            for it in result.items
        ],
        "stats": asdict(result.stats),
    }
    return orjson.dumps(payload, option=orjson.OPT_INDENT_2)


def format_text(result: SearchResult, highlight: bool = False) -> str:
    """
    Format search results as plain text with line numbers.

    This function creates a human-readable text representation of search results
    with file paths, line numbers, and optional match highlighting. It's the
    fastest formatting option and suitable for console output or logging.

    Args:
        result: SearchResult object containing search results
        highlight: Whether to highlight match spans with markers (default: False)

    Returns:
        Formatted text string with file paths, line numbers, and content

    Example:
        >>> formatted = format_text(search_results, highlight=True)
        >>> print(formatted)
        example.py:10-12
            10 | def main():
            11 |     print("Hello, world!")
            12 |     return 0

    Note:
        When highlight=True, match spans are highlighted with special markers.
        This is useful for terminal output but may not render well in all contexts.
    """
    out: list[str] = []
    for it in result.items:
        header = f"{it.file}:{it.start_line}-{it.end_line}"
        out.append(header)
        ln = it.start_line
        spans_by_line: dict[int, list[tuple[int, int]]] = {}
        for li, (a, b) in it.match_spans:
            spans_by_line.setdefault(li, []).append((a, b))
        for idx, line in enumerate(it.lines, start=0):
            prefix = f"{ln + idx:6d} | "
            content = line
            if highlight and idx in spans_by_line:
                content = highlight_spans(
                    content, spans_by_line[idx], marker_left="[[", marker_right="]]"
                )
            out.append(prefix + content)
        out.append("")
    # stats
    s = result.stats
    stats_line = (
        f"# files_scanned={s.files_scanned} files_matched={s.files_matched} items={s.items} "
        f"elapsed_ms={s.elapsed_ms:.2f} indexed={s.indexed_files}"
    )
    out.append(stats_line)
    return "\n".join(out)


def render_highlight_console(result: SearchResult, console: Console | None = None) -> None:
    """Render search results with rich syntax highlighting to the console."""
    if console is None:
        console = Console()
    for it in result.items:
        code = "\n".join(it.lines)
        lexer = _detect_lexer(it.file)
        syntax = Syntax(
            code, lexer, line_numbers=True, start_line=it.start_line,
        )
        console.print(f"[bold]{it.file}:{it.start_line}-{it.end_line}[/bold]")
        console.print(syntax)
        console.print()
    s = result.stats
    stats_text = (
        f"[dim]files_scanned={s.files_scanned} files_matched={s.files_matched} items={s.items} "
        f"elapsed_ms={s.elapsed_ms:.2f} indexed={s.indexed_files}[/dim]"
    )
    console.print(stats_text)


def format_result(result: SearchResult, fmt: OutputFormat) -> str:
    """Format search results according to the specified output format."""
    if fmt == OutputFormat.JSON:
        return to_json_bytes(result).decode("utf-8")
    if fmt == OutputFormat.HIGHLIGHT:
        # Use rich console rendering when stdout is a real terminal
        if sys.stdout.isatty():
            render_highlight_console(result)
            return ""
        # For non-interactive environments, fall back to text with simple markers
        return format_text(result, highlight=True)
    return format_text(result, highlight=False)
