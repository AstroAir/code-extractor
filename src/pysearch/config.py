"""
Configuration module for pysearch.

This module defines the SearchConfig class which serves as the central configuration
object for all search operations. It provides comprehensive settings for search scope,
behavior, performance, and output formatting.

Classes:
    RankStrategy: Enumeration of available ranking strategies
    SearchConfig: Main configuration class with all search parameters

Key Configuration Areas:
    - Search scope: paths, include/exclude patterns, language filtering
    - Search behavior: context lines, content toggles, symlink handling
    - Performance: parallel execution, caching, file size limits
    - Output: format selection, ranking strategies
    - Advanced: strict hash checking, directory pruning

Example:
    Basic configuration:
        >>> from pysearch.config import SearchConfig
        >>> from pysearch.types import OutputFormat
        >>>
        >>> config = SearchConfig(
        ...     paths=["."],
        ...     include=["**/*.py"],
        ...     exclude=["**/.venv/**", "**/__pycache__/**"],
        ...     context=3,
        ...     output_format=OutputFormat.JSON
        ... )

    Performance-optimized configuration:
        >>> config = SearchConfig(
        ...     paths=["./src", "./tests"],
        ...     parallel=True,
        ...     workers=4,
        ...     strict_hash_check=False,
        ...     dir_prune_exclude=True
        ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .types import Language, OutputFormat


class RankStrategy(str, Enum):
    DEFAULT = "default"


@dataclass(slots=True)
class SearchConfig:
    # Scope
    paths: list[str] = field(default_factory=lambda: ["."], metadata={"help": "Root search paths."})
    include: list[str] | None = None  # None = auto-detect based on languages
    exclude: list[str] | None = None  # None = use defaults

    # Language filtering
    languages: set[Language] | None = None  # None = auto-detect all supported languages
    file_size_limit: int = 2_000_000  # 2MB default limit

    # Behavior
    context: int = 2
    output_format: OutputFormat = OutputFormat.TEXT
    follow_symlinks: bool = False
    max_file_bytes: int = 2_000_000  # 2MB safeguard (kept for backward compatibility)

    # Content toggles
    enable_docstrings: bool = True
    enable_comments: bool = True
    enable_strings: bool = True

    # Performance
    parallel: bool = True
    workers: int = 0  # 0 = auto(cpu_count)
    cache_dir: Path | None = None  # default: .pysearch-cache under first path
    # New toggles
    strict_hash_check: bool = False  # if True, compute sha1 on scan for exact change detection
    dir_prune_exclude: bool = True  # if True, prune excluded directories during traversal

    # Ranking
    rank_strategy: RankStrategy = RankStrategy.DEFAULT
    ast_weight: float = 2.0
    text_weight: float = 1.0

    def resolve_cache_dir(self) -> Path:
        base = Path(self.paths[0]).resolve() if self.paths else Path(".").resolve()
        return self.cache_dir or (base / ".pysearch-cache")

    def get_include_patterns(self) -> list[str]:
        """Get include patterns, using defaults if not specified."""
        if self.include is not None:
            return self.include

        # Generate default patterns based on enabled languages
        if self.languages is None:
            # Include all common text file types
            return [
                "**/*.py",
                "**/*.js",
                "**/*.ts",
                "**/*.java",
                "**/*.c",
                "**/*.cpp",
                "**/*.h",
                "**/*.hpp",
                "**/*.cs",
                "**/*.go",
                "**/*.rs",
                "**/*.php",
                "**/*.rb",
                "**/*.swift",
                "**/*.kt",
                "**/*.scala",
                "**/*.r",
                "**/*.m",
                "**/*.sh",
                "**/*.ps1",
                "**/*.sql",
                "**/*.html",
                "**/*.css",
                "**/*.xml",
                "**/*.json",
                "**/*.yaml",
                "**/*.yml",
                "**/*.toml",
                "**/*.md",
                "**/*.txt",
                "**/Dockerfile",
                "**/Makefile",
            ]

        # Generate patterns for specific languages
        from .language_detection import get_language_extensions

        patterns = []
        for lang in self.languages:
            extensions = get_language_extensions(lang)
            for ext in extensions:
                patterns.append(f"**/*{ext}")

        # Add special filename patterns
        special_files = ["**/Dockerfile", "**/Makefile", "**/Rakefile", "**/Gemfile"]
        patterns.extend(special_files)

        return patterns

    def get_exclude_patterns(self) -> list[str]:
        """Get exclude patterns, using defaults if not specified."""
        if self.exclude is not None:
            return self.exclude

        return [
            "**/.venv/**",
            "**/.git/**",
            "**/build/**",
            "**/dist/**",
            "**/__pycache__/**",
            "**/.svn/**",
            "**/.hg/**",
            "**/node_modules/**",
            "**/target/**",
            "**/bin/**",
            "**/obj/**",
            "**/.pysearch-cache/**",
        ]

    def should_include_language(self, language: Language) -> bool:
        """Check if a language should be included in search."""
        if self.languages is None:
            return language != Language.UNKNOWN
        return language in self.languages
