from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    HIGHLIGHT = "highlight"


@dataclass(slots=True)
class ASTFilters:
    func_name: Optional[str] = None
    class_name: Optional[str] = None
    decorator: Optional[str] = None
    imported: Optional[str] = None


# match_spans: list of (line_index, (start_col, end_col))
MatchSpan = Tuple[int, Tuple[int, int]]


@dataclass(slots=True)
class SearchItem:
    file: Path
    start_line: int
    end_line: int
    lines: List[str]
    match_spans: List[MatchSpan] = field(default_factory=list)


@dataclass(slots=True)
class SearchStats:
    files_scanned: int = 0
    files_matched: int = 0
    items: int = 0
    elapsed_ms: float = 0.0
    indexed_files: int = 0


@dataclass(slots=True)
class SearchResult:
    items: List[SearchItem] = field(default_factory=list)
    stats: SearchStats = field(default_factory=SearchStats)


@dataclass(slots=True)
class Query:
    pattern: str
    use_regex: bool = False
    use_ast: bool = False
    context: int = 2
    output: OutputFormat = OutputFormat.TEXT
    filters: Optional[ASTFilters] = None
    search_docstrings: bool = True
    search_comments: bool = True
    search_strings: bool = True