"""
Command-line interface implementation.

This module provides the command-line interface for pysearch:
- CLI command definitions and parsing
- Interactive command-line features
- Terminal output formatting
- Command-line argument validation

The CLI module makes pysearch accessible from the command line
with a rich set of options and user-friendly interface.
"""

from .main import (
    bookmarks_cmd,
    cache_cmd,
    cli,
    config_cmd,
    deps_cmd,
    errors_cmd,
    find_cmd,
    history_cmd,
    index_cmd,
    main,
    repo_cmd,
    semantic_cmd,
    watch_cmd,
)

__all__ = [
    "main",
    "cli",
    "find_cmd",
    "history_cmd",
    "bookmarks_cmd",
    "semantic_cmd",
    "index_cmd",
    "deps_cmd",
    "watch_cmd",
    "cache_cmd",
    "config_cmd",
    "repo_cmd",
    "errors_cmd",
]
