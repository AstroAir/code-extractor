"""
Utility functions and helper modules.

This module contains various utility functions and helper classes:
- Output formatting and highlighting
- Metadata filtering and processing
- File watching and monitoring
- Error handling and logging
- Performance monitoring
- General utility functions

The utils module provides common functionality used throughout
the application, promoting code reuse and maintainability.
"""

from .error_handling import (
    EncodingError,
    FileAccessError,
    ParsingError,
    PermissionError,
    SearchError,
    ErrorCollector,
    create_error_report,
    handle_file_error,
)
from .formatter import format_result, render_highlight_console
from .logging_config import configure_logging, disable_logging, enable_debug_logging, get_logger
from .metadata_filters import create_metadata_filters, apply_metadata_filters, get_file_author
from .utils import create_file_metadata, read_text_safely

__all__ = [
    # Error handling
    "EncodingError",
    "FileAccessError",
    "ParsingError",
    "PermissionError",
    "SearchError",
    "ErrorCollector",
    "create_error_report",
    "handle_file_error",
    # Formatting
    "format_result",
    "render_highlight_console",
    # Logging
    "configure_logging",
    "disable_logging",
    "enable_debug_logging",
    "get_logger",
    # Metadata filtering
    "create_metadata_filters",
    "apply_metadata_filters",
    "get_file_author",
    # Utilities
    "create_file_metadata",
    "read_text_safely",
]
