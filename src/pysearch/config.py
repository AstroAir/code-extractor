from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence

from .types import OutputFormat


class RankStrategy(str, Enum):
    DEFAULT = "default"


@dataclass(slots=True)
class SearchConfig:
    # Scope
    paths: List[str] = field(default_factory=lambda: ["."],
                             metadata={"help": "Root search paths."})
    include: List[str] = field(default_factory=lambda: ["**/*.py"],
                               metadata={"help": "Glob patterns to include."})
    exclude: List[str] = field(default_factory=lambda: ["**/.venv/**", "**/.git/**", "**/build/**", "**/dist/**", "**/__pycache__/**"],
                               metadata={"help": "Glob patterns to exclude."})

    # Behavior
    context: int = 2
    output_format: OutputFormat = OutputFormat.TEXT
    follow_symlinks: bool = False
    max_file_bytes: int = 2_000_000  # 2MB safeguard

    # Content toggles
    enable_docstrings: bool = True
    enable_comments: bool = True
    enable_strings: bool = True

    # Performance
    parallel: bool = True
    workers: int = 0  # 0 = auto(cpu_count)
    cache_dir: Optional[Path] = None  # default: .pysearch-cache under first path
    # New toggles
    strict_hash_check: bool = False  # if True, compute sha1 on scan for exact change detection
    dir_prune_exclude: bool = True   # if True, prune excluded directories during traversal

    # Ranking
    rank_strategy: RankStrategy = RankStrategy.DEFAULT
    ast_weight: float = 2.0
    text_weight: float = 1.0

    def resolve_cache_dir(self) -> Path:
        base = Path(self.paths[0]).resolve() if self.paths else Path(".").resolve()
        return (self.cache_dir or (base / ".pysearch-cache"))