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
    AdvancedErrorCollector,
    CircuitBreaker,
    ConfigurationError,
    EncodingError,
    EnhancedErrorHandler,
    ErrorCollector,
    FileAccessError,
    IndexingError,
    ParsingError,
    PermissionError,
    RecoveryManager,
    SearchError,
    create_error_report,
    handle_file_error,
)
from .file_watcher import ChangeProcessor, FileEvent, FileWatcher, WatchManager
from .formatter import format_result, render_highlight_console
from .helpers import (
    FileMeta,
    build_pathspec,
    create_file_metadata,
    extract_context,
    file_meta,
    file_sha1,
    get_ast_node_info,
    highlight_spans,
    iter_files,
    iter_python_ast_nodes,
    matches_patterns,
    read_text_safely,
    sha1_bytes,
    split_lines_keepends,
)
from .logging_config import configure_logging, disable_logging, enable_debug_logging, get_logger
from .metadata_filters import apply_metadata_filters, create_metadata_filters, get_file_author
from .performance_monitoring import (
    MetricsCollector,
    OptimizationEngine,
    PerformanceMonitor,
    PerformanceProfiler,
)

__all__ = [
    # Advanced error handling
    "EnhancedErrorHandler",
    "RecoveryManager",
    "CircuitBreaker",
    "IndexingError",
    "AdvancedErrorCollector",
    # Error handling
    "ConfigurationError",
    "EncodingError",
    "FileAccessError",
    "ParsingError",
    "PermissionError",
    "SearchError",
    "ErrorCollector",
    "create_error_report",
    "handle_file_error",
    # File watching
    "FileWatcher",
    "WatchManager",
    "ChangeProcessor",
    "FileEvent",
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
    # Performance monitoring
    "PerformanceMonitor",
    "PerformanceProfiler",
    "MetricsCollector",
    "OptimizationEngine",
    # Utilities - File operations
    "create_file_metadata",
    "read_text_safely",
    "file_sha1",
    "sha1_bytes",
    "file_meta",
    "FileMeta",
    # Utilities - Path operations
    "matches_patterns",
    "build_pathspec",
    "iter_files",
    # Utilities - Text processing
    "extract_context",
    "split_lines_keepends",
    "highlight_spans",
    # Utilities - AST
    "iter_python_ast_nodes",
    "get_ast_node_info",
]
