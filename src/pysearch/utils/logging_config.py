from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any


class LogLevel(str, Enum):
    """Available log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Available log formats."""

    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    STRUCTURED = "structured"


class SearchLogger:
    """
    Centralized logging system for pysearch with multiple output formats
    and configurable levels.
    """

    def __init__(
        self,
        name: str = "pysearch",
        level: LogLevel = LogLevel.INFO,
        format_type: LogFormat = LogFormat.SIMPLE,
        log_file: Path | None = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True,
        enable_file: bool = False,
    ):
        self.name = name
        self.level = level
        self.format_type = format_type
        self.log_file = log_file
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file

        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Setup handlers
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup logging handlers based on configuration."""
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(getattr(logging, self.level.value))
            console_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(console_handler)

        # File handler
        if self.enable_file and self.log_file:
            # Ensure log directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            # Use rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(getattr(logging, self.level.value))
            file_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(file_handler)

    def _get_formatter(self) -> logging.Formatter:
        """Get formatter based on format type."""
        if self.format_type == LogFormat.SIMPLE:
            return logging.Formatter("%(levelname)s: %(message)s")
        elif self.format_type == LogFormat.DETAILED:
            return logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
            )
        elif self.format_type == LogFormat.JSON:
            return JsonFormatter()
        elif self.format_type == LogFormat.STRUCTURED:
            return StructuredFormatter()
        else:
            return logging.Formatter("%(levelname)s: %(message)s")

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self.logger.critical(message, extra=kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, extra=kwargs)

    def log_search_start(self, pattern: str, paths: list, **kwargs: Any) -> None:
        """Log search operation start."""
        self.info(
            f"Starting search for pattern: '{pattern}' in paths: {paths}",
            operation="search_start",
            pattern=pattern,
            paths=paths,
            **kwargs,
        )

    def log_search_complete(
        self, pattern: str, results_count: int, elapsed_ms: float, **kwargs: Any
    ) -> None:
        """Log search operation completion."""
        self.info(
            f"Search completed: pattern='{pattern}', results={results_count}, time={elapsed_ms:.2f}ms",
            operation="search_complete",
            pattern=pattern,
            results_count=results_count,
            elapsed_ms=elapsed_ms,
            **kwargs,
        )

    def log_file_error(self, file_path: str, error: str, **kwargs: Any) -> None:
        """Log file processing error."""
        self.error(
            f"File error: {file_path} - {error}",
            operation="file_error",
            file_path=file_path,
            error=error,
            **kwargs,
        )

    def log_indexing_stats(
        self, files_scanned: int, files_indexed: int, elapsed_ms: float, **kwargs: Any
    ) -> None:
        """Log indexing statistics."""
        self.info(
            f"Indexing stats: scanned={files_scanned}, indexed={files_indexed}, time={elapsed_ms:.2f}ms",
            operation="indexing_stats",
            files_scanned=files_scanned,
            files_indexed=files_indexed,
            elapsed_ms=elapsed_ms,
            **kwargs,
        )


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "getMessage",
                ]:
                    log_entry[key] = value

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class StructuredFormatter(logging.Formatter):
    """Structured formatter for human-readable structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        # Format timestamp
        record.asctime = self.formatTime(record, self.datefmt)

        # Base format
        base = f"{record.asctime} [{record.levelname}] {record.name}: {record.getMessage()}"

        # Add structured fields
        extra_fields = []
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "getMessage",
                    "asctime",
                ]:
                    extra_fields.append(f"{key}={value}")

        if extra_fields:
            base += f" | {' '.join(extra_fields)}"

        # Add exception info if present
        if record.exc_info:
            base += f"\n{self.formatException(record.exc_info)}"

        return base


# Global logger instance
_global_logger: SearchLogger | None = None


def get_logger() -> SearchLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = SearchLogger()
    return _global_logger


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    format_type: LogFormat = LogFormat.SIMPLE,
    log_file: Path | None = None,
    enable_console: bool = True,
    enable_file: bool = False,
    **kwargs: Any,
) -> SearchLogger:
    """Configure global logging settings."""
    global _global_logger
    _global_logger = SearchLogger(
        level=level,
        format_type=format_type,
        log_file=log_file,
        enable_console=enable_console,
        enable_file=enable_file,
        **kwargs,
    )
    return _global_logger


def disable_logging() -> None:
    """Disable all logging."""
    global _global_logger
    if _global_logger:
        _global_logger.logger.setLevel(logging.CRITICAL + 1)


def enable_debug_logging() -> None:
    """Enable debug logging for troubleshooting."""
    global _global_logger
    if _global_logger:
        _global_logger.level = LogLevel.DEBUG
        _global_logger.logger.setLevel(logging.DEBUG)
        for handler in _global_logger.logger.handlers:
            handler.setLevel(logging.DEBUG)
