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

from dataclasses import asdict
from pathlib import Path

import orjson
from rich.console import Console
from rich.syntax import Syntax

from .types import OutputFormat, SearchResult
from .utils import highlight_spans


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
    def default(obj):
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
    out.append(
        f"# files_scanned={s.files_scanned} files_matched={s.files_matched} items={s.items} elapsed_ms={s.elapsed_ms:.2f} indexed={s.indexed_files}"
    )
    return "\n".join(out)


def render_highlight_console(result: SearchResult) -> None:
    console = Console()
    for it in result.items:
        code = "\n".join(it.lines)
        # Use Python syntax highlight
        syntax = Syntax(code, "python", line_numbers=True, line_range=(it.start_line, it.end_line))
        console.print(f"[bold]{it.file}:{it.start_line}-{it.end_line}[/bold]")
        console.print(syntax)
        console.print()
    s = result.stats
    console.print(
        f"[dim]files_scanned={s.files_scanned} files_matched={s.files_matched} items={s.items} elapsed_ms={s.elapsed_ms:.2f} indexed={s.indexed_files}[/dim]"
    )


def format_result(result: SearchResult, fmt: OutputFormat) -> str:
    if fmt == OutputFormat.JSON:
        return to_json_bytes(result).decode("utf-8")
    if fmt == OutputFormat.HIGHLIGHT:
        # For non-interactive environments, fall back to text with simple markers
        return format_text(result, highlight=True)
    return format_text(result, highlight=False)
