from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import orjson
from rich.console import Console
from rich.syntax import Syntax

from .types import OutputFormat, SearchItem, SearchResult
from .utils import highlight_spans


def to_json_bytes(result: SearchResult) -> bytes:
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
    out: List[str] = []
    for it in result.items:
        header = f"{it.file}:{it.start_line}-{it.end_line}"
        out.append(header)
        ln = it.start_line
        spans_by_line: dict[int, List[Tuple[int, int]]] = {}
        for li, (a, b) in it.match_spans:
            spans_by_line.setdefault(li, []).append((a, b))
        for idx, line in enumerate(it.lines, start=0):
            prefix = f"{ln + idx:6d} | "
            content = line
            if highlight and idx in spans_by_line:
                content = highlight_spans(content, spans_by_line[idx], marker_left="[[", marker_right="]]")
            out.append(prefix + content)
        out.append("")
    # stats
    s = result.stats
    out.append(f"# files_scanned={s.files_scanned} files_matched={s.files_matched} items={s.items} elapsed_ms={s.elapsed_ms:.2f} indexed={s.indexed_files}")
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
    console.print(f"[dim]files_scanned={s.files_scanned} files_matched={s.files_matched} items={s.items} elapsed_ms={s.elapsed_ms:.2f} indexed={s.indexed_files}[/dim]")


def format_result(result: SearchResult, fmt: OutputFormat) -> str:
    if fmt == OutputFormat.JSON:
        return to_json_bytes(result).decode("utf-8")
    if fmt == OutputFormat.HIGHLIGHT:
        # For non-interactive environments, fall back to text with simple markers
        return format_text(result, highlight=True)
    return format_text(result, highlight=False)