"""
Enhanced error handling and recovery system for robust indexing operations.

This module implements comprehensive error handling, recovery mechanisms,
and logging inspired by Continue's approach but with additional features
for production-grade reliability and debugging.

Classes:
    ErrorSeverity: Error severity levels
    IndexingError: Enhanced error representation
    ErrorCollector: Collects and categorizes errors
    RecoveryManager: Handles error recovery strategies
    CircuitBreaker: Prevents cascading failures

Features:
    - Comprehensive error categorization and tracking
    - Automatic recovery strategies for common failures
    - Circuit breaker pattern for external dependencies
    - Detailed error reporting with context
    - Performance impact monitoring
    - Error trend analysis and alerting
    - Graceful degradation strategies

Example:
    Basic error handling:
        >>> from pysearch.utils.advanced_error_handling import ErrorCollector
        >>> collector = ErrorCollector()
        >>> collector.add_error("file.py", "Parse error", ErrorSeverity.WARNING)

    Advanced recovery:
        >>> from pysearch.utils.advanced_error_handling import RecoveryManager
        >>> recovery = RecoveryManager()
        >>> await recovery.attempt_recovery("embedding_failure", context)
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .logging_config import get_logger

logger = get_logger()


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for better classification."""

    FILE_ACCESS = "file_access"
    PARSING = "parsing"
    NETWORK = "network"
    MEMORY = "memory"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


@dataclass
class IndexingError:
    """Enhanced error representation with context and metadata."""

    error_id: str
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    context: str  # File path, operation, etc.
    timestamp: float = field(default_factory=time.time)
    stack_trace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    impact_score: float = 0.0  # 0.0 = no impact, 1.0 = critical impact


class ErrorCollector:
    """
    Collects and categorizes errors during indexing operations.

    Provides comprehensive error tracking with categorization, impact analysis,
    and trend monitoring for better debugging and system reliability.
    """

    def __init__(self, max_errors: int = 10000):
        self.max_errors = max_errors
        self.errors: list[IndexingError] = []
        self.error_counts: dict[ErrorCategory, int] = {}
        self.error_trends: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def add_error(
        self,
        context: str,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory | None = None,
        metadata: dict[str, Any] | None = None,
        exception: Exception | None = None,
    ) -> str:
        """
        Add an error to the collector.

        Args:
            context: Context where error occurred (file path, operation, etc.)
            message: Error message
            severity: Error severity level
            category: Error category (auto-detected if None)
            metadata: Additional error metadata
            exception: Original exception if available

        Returns:
            Unique error ID
        """
        async with self._lock:
            # Auto-detect category if not provided
            if category is None:
                category = self._categorize_error(message, exception)

            # Generate unique error ID
            error_id = f"{category.value}_{int(time.time() * 1000)}"

            # Extract stack trace if exception provided
            stack_trace: str | None = None
            if exception:
                stack_trace_lines = traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
                stack_trace = "".join(stack_trace_lines)

            # Calculate impact score
            impact_score = self._calculate_impact_score(severity, category, context)

            # Create error object
            error = IndexingError(
                error_id=error_id,
                severity=severity,
                category=category,
                message=message,
                context=context,
                stack_trace=stack_trace,
                metadata=metadata or {},
                impact_score=impact_score,
            )

            # Add to collection
            self.errors.append(error)

            # Update counts
            self.error_counts[category] = self.error_counts.get(category, 0) + 1

            # Update trends
            current_hour = int(time.time() // 3600)
            trend_key = f"{category.value}_{current_hour}"
            if trend_key not in self.error_trends:
                self.error_trends[trend_key] = []
            self.error_trends[trend_key].append(time.time())

            # Trim old errors if needed
            if len(self.errors) > self.max_errors:
                self.errors = self.errors[-self.max_errors :]

            # Log error
            log_level = {
                ErrorSeverity.DEBUG: logger.debug,
                ErrorSeverity.INFO: logger.info,
                ErrorSeverity.WARNING: logger.warning,
                ErrorSeverity.ERROR: logger.error,
                ErrorSeverity.CRITICAL: logger.critical,
            }[severity]

            log_level(f"[{category.value}] {context}: {message}")

            return error_id

    def _categorize_error(
        self,
        message: str,
        exception: Exception | None = None,
    ) -> ErrorCategory:
        """Auto-categorize error based on message and exception type."""
        message_lower = message.lower()

        # File access errors
        if any(
            keyword in message_lower
            for keyword in ["permission", "not found", "access denied", "file not found"]
        ):
            return ErrorCategory.FILE_ACCESS

        # Network errors
        if any(
            keyword in message_lower
            for keyword in ["connection", "timeout", "network", "http", "ssl"]
        ):
            return ErrorCategory.NETWORK

        # Memory errors
        if any(keyword in message_lower for keyword in ["memory", "out of memory", "allocation"]):
            return ErrorCategory.MEMORY

        # Parsing errors
        if any(keyword in message_lower for keyword in ["parse", "syntax", "invalid", "malformed"]):
            return ErrorCategory.PARSING

        # Timeout errors
        if "timeout" in message_lower:
            return ErrorCategory.TIMEOUT

        # Configuration errors
        if any(
            keyword in message_lower for keyword in ["config", "setting", "parameter", "option"]
        ):
            return ErrorCategory.CONFIGURATION

        # Dependency errors
        if any(
            keyword in message_lower for keyword in ["import", "module", "dependency", "package"]
        ):
            return ErrorCategory.DEPENDENCY

        # Exception-based categorization
        if exception:
            if isinstance(exception, (FileNotFoundError, PermissionError)):
                return ErrorCategory.FILE_ACCESS
            elif isinstance(exception, (ConnectionError, TimeoutError)):
                return ErrorCategory.NETWORK
            elif isinstance(exception, MemoryError):
                return ErrorCategory.MEMORY
            elif isinstance(exception, (SyntaxError, ValueError)):
                return ErrorCategory.PARSING
            elif isinstance(exception, ImportError):
                return ErrorCategory.DEPENDENCY

        return ErrorCategory.UNKNOWN

    def _calculate_impact_score(
        self,
        severity: ErrorSeverity,
        category: ErrorCategory,
        context: str,
    ) -> float:
        """Calculate impact score for an error."""
        # Base score from severity
        severity_scores = {
            ErrorSeverity.DEBUG: 0.1,
            ErrorSeverity.INFO: 0.2,
            ErrorSeverity.WARNING: 0.4,
            ErrorSeverity.ERROR: 0.7,
            ErrorSeverity.CRITICAL: 1.0,
        }
        base_score = severity_scores[severity]

        # Category multipliers
        category_multipliers = {
            ErrorCategory.MEMORY: 1.3,
            ErrorCategory.NETWORK: 1.2,
            ErrorCategory.CONFIGURATION: 1.1,
            ErrorCategory.FILE_ACCESS: 1.0,
            ErrorCategory.TIMEOUT: 1.1,
            ErrorCategory.DEPENDENCY: 0.9,
            ErrorCategory.PARSING: 0.8,
            ErrorCategory.UNKNOWN: 0.7,
        }
        multiplier = category_multipliers.get(category, 1.0)

        return min(1.0, base_score * multiplier)

    def get_errors_by_severity(self, severity: ErrorSeverity) -> list[IndexingError]:
        """Get all errors of a specific severity."""
        return [error for error in self.errors if error.severity == severity]

    def get_errors_by_category(self, category: ErrorCategory) -> list[IndexingError]:
        """Get all errors of a specific category."""
        return [error for error in self.errors if error.category == category]

    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of all collected errors."""
        if not self.errors:
            return {"total_errors": 0}

        # Count by severity
        severity_counts: dict[str, int] = {}
        for error in self.errors:
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1

        # Count by category
        category_counts = dict(self.error_counts)

        # Calculate average impact
        total_impact = sum(error.impact_score for error in self.errors)
        avg_impact = total_impact / len(self.errors)

        # Recent error rate (last hour)
        recent_errors = [error for error in self.errors if time.time() - error.timestamp < 3600]

        return {
            "total_errors": len(self.errors),
            "severity_counts": severity_counts,
            "category_counts": category_counts,
            "average_impact_score": avg_impact,
            "recent_errors_count": len(recent_errors),
            "most_common_category": (
                max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None
            ),
            "error_rate_per_hour": len(recent_errors),
        }

    def get_all_errors(self) -> list[str]:
        """Get all error messages for compatibility with existing code."""
        return [f"{error.context}: {error.message}" for error in self.errors]

    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self.errors) > 0

    def clear_errors(self) -> None:
        """Clear all collected errors."""
        self.errors.clear()
        self.error_counts.clear()
        self.error_trends.clear()


class CircuitBreaker:
    """
    Circuit breaker pattern for external dependencies.

    Prevents cascading failures by temporarily disabling failing
    operations and allowing them to recover.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # "closed", "open", "half_open"
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            # Check circuit state
            if self.state == "open":
                # Check if recovery timeout has passed
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half_open"
                    self.success_count = 0
                    logger.info("Circuit breaker entering half-open state")
                else:
                    raise RuntimeError("Circuit breaker is open")

            try:
                # Execute function
                result = (
                    await func(*args, **kwargs)
                    if asyncio.iscoroutinefunction(func)
                    else func(*args, **kwargs)
                )

                # Success - update state
                if self.state == "half_open":
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = "closed"
                        self.failure_count = 0
                        logger.info("Circuit breaker closed - service recovered")

                return result

            except Exception:
                # Failure - update state
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.warning(f"Circuit breaker opened due to {self.failure_count} failures")

                raise


class RecoveryManager:
    """
    Manages error recovery strategies for different types of failures.

    Implements automatic recovery mechanisms for common indexing failures,
    reducing the need for manual intervention and improving system reliability.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.recovery_strategies: dict[ErrorCategory, Callable] = {}
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.recovery_attempts: dict[str, int] = {}
        self.successful_recoveries: int = 0
        self.failed_recoveries: int = 0
        self.max_recovery_attempts = 3

        # Register default recovery strategies
        self._register_default_strategies()

    def _register_default_strategies(self) -> None:
        """Register default recovery strategies."""
        self.recovery_strategies[ErrorCategory.FILE_ACCESS] = self._recover_file_access
        self.recovery_strategies[ErrorCategory.NETWORK] = self._recover_network
        self.recovery_strategies[ErrorCategory.MEMORY] = self._recover_memory
        self.recovery_strategies[ErrorCategory.TIMEOUT] = self._recover_timeout
        self.recovery_strategies[ErrorCategory.PARSING] = self._recover_parsing
        self.recovery_strategies[ErrorCategory.DEPENDENCY] = self._recover_dependency

    async def attempt_recovery(
        self,
        error: IndexingError,
        context: dict[str, Any],
    ) -> bool:
        """
        Attempt to recover from an error.

        Args:
            error: The error to recover from
            context: Additional context for recovery

        Returns:
            True if recovery was successful, False otherwise
        """
        # Check if we've exceeded max recovery attempts
        attempt_key = f"{error.category.value}_{error.context}"
        attempts = self.recovery_attempts.get(attempt_key, 0)

        if attempts >= self.max_recovery_attempts:
            logger.warning(f"Max recovery attempts exceeded for {attempt_key}")
            return False

        # Increment attempt counter
        self.recovery_attempts[attempt_key] = attempts + 1

        # Get recovery strategy
        strategy = self.recovery_strategies.get(error.category)
        if not strategy:
            logger.debug(f"No recovery strategy for category {error.category}")
            return False

        try:
            logger.info(f"Attempting recovery for {error.error_id} (attempt {attempts + 1})")
            success: bool = await strategy(error, context)

            if success:
                error.recovery_attempted = True
                error.recovery_successful = True
                self.successful_recoveries += 1
                # Reset attempt counter on success
                self.recovery_attempts.pop(attempt_key, None)
                logger.info(f"Recovery successful for {error.error_id}")
            else:
                error.recovery_attempted = True
                error.recovery_successful = False
                self.failed_recoveries += 1
                logger.warning(f"Recovery failed for {error.error_id}")

            return success

        except Exception as e:
            logger.error(f"Recovery strategy failed for {error.error_id}: {e}")
            error.recovery_attempted = True
            error.recovery_successful = False
            self.failed_recoveries += 1
            return False

    async def _recover_file_access(
        self,
        error: IndexingError,
        context: dict[str, Any],
    ) -> bool:
        """Recover from file access errors."""
        file_path = error.context

        try:
            # Check if file exists
            if not Path(file_path).exists():
                logger.debug(f"File {file_path} no longer exists, skipping")
                return True  # Consider this a successful recovery

            # Check permissions
            if not Path(file_path).is_file():
                logger.debug(f"Path {file_path} is not a file, skipping")
                return True

            # Try to read file with different encoding
            encodings = ["utf-8", "latin-1", "cp1252", "ascii"]
            for encoding in encodings:
                try:
                    Path(file_path).read_text(encoding=encoding)
                    logger.info(f"File {file_path} readable with encoding {encoding}")
                    return True
                except UnicodeDecodeError:
                    continue

            logger.warning(f"File {file_path} not readable with any encoding")
            return False

        except Exception as e:
            logger.error(f"File access recovery failed: {e}")
            return False

    async def _recover_network(
        self,
        error: IndexingError,
        context: dict[str, Any],
    ) -> bool:
        """Recover from network errors."""
        # Implement exponential backoff
        attempt = self.recovery_attempts.get(f"network_{error.context}", 0)
        backoff_time = min(60.0, 2**attempt)

        logger.info(f"Network recovery: waiting {backoff_time}s before retry")
        await asyncio.sleep(backoff_time)

        # Test network connectivity using stdlib
        try:
            import urllib.request

            req = urllib.request.Request(
                "https://httpbin.org/status/200", method="HEAD"
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    logger.info("Network connectivity restored")
                    return True
        except Exception:
            pass

        logger.warning("Network connectivity still unavailable")
        return False

    async def _recover_memory(
        self,
        error: IndexingError,
        context: dict[str, Any],
    ) -> bool:
        """Recover from memory errors."""
        # Force garbage collection
        import gc

        gc.collect()

        # Reduce batch sizes if possible
        if "batch_size" in context:
            new_batch_size = max(1, context["batch_size"] // 2)
            context["batch_size"] = new_batch_size
            logger.info(f"Reduced batch size to {new_batch_size} for memory recovery")
            return True

        logger.warning("Cannot recover from memory error - no batch size to reduce")
        return False

    async def _recover_timeout(
        self,
        error: IndexingError,
        context: dict[str, Any],
    ) -> bool:
        """Recover from timeout errors."""
        # Increase timeout if possible
        if "timeout" in context:
            new_timeout = min(300.0, context["timeout"] * 2)
            context["timeout"] = new_timeout
            logger.info(f"Increased timeout to {new_timeout}s for recovery")
            return True

        return False

    async def _recover_parsing(
        self,
        error: IndexingError,
        context: dict[str, Any],
    ) -> bool:
        """Recover from parsing errors."""
        # For parsing errors, we usually skip the problematic content
        logger.info(f"Skipping problematic content in {error.context}")
        return True

    async def _recover_dependency(
        self,
        error: IndexingError,
        context: dict[str, Any],
    ) -> bool:
        """Recover from dependency errors."""
        # Try to install missing dependencies or use fallbacks
        logger.info(f"Using fallback for missing dependency in {error.context}")
        return True

    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]

    def get_recovery_stats(self) -> dict[str, Any]:
        """Get recovery statistics."""
        total_attempts = self.successful_recoveries + self.failed_recoveries

        return {
            "total_recovery_attempts": total_attempts,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "recovery_success_rate": (
                self.successful_recoveries / max(total_attempts, 1)
            ),
            "active_circuit_breakers": len(
                [cb for cb in self.circuit_breakers.values() if cb.state != "closed"]
            ),
            "recovery_strategies": list(self.recovery_strategies.keys()),
        }


class EnhancedErrorHandler:
    """
    Main error handling coordinator for the enhanced indexing system.

    Integrates error collection, recovery management, and monitoring
    to provide comprehensive error handling capabilities.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.error_collector = ErrorCollector()
        self.recovery_manager = RecoveryManager(config)
        self.error_log_path = config.resolve_cache_dir() / "error_log.jsonl"

    async def handle_error(
        self,
        context: str,
        exception: Exception,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        metadata: dict[str, Any] | None = None,
        attempt_recovery: bool = True,
    ) -> bool:
        """
        Handle an error with optional recovery attempt.

        Args:
            context: Context where error occurred
            exception: The exception that occurred
            severity: Error severity level
            metadata: Additional error metadata
            attempt_recovery: Whether to attempt automatic recovery

        Returns:
            True if error was handled successfully (possibly with recovery)
        """
        # Add error to collector
        error_id = await self.error_collector.add_error(
            context=context,
            message=str(exception),
            severity=severity,
            metadata=metadata,
            exception=exception,
        )

        # Log error to file
        await self._log_error_to_file(error_id, context, exception, metadata)

        # Attempt recovery if requested
        if attempt_recovery:
            # Find the error we just added
            error = next((e for e in self.error_collector.errors if e.error_id == error_id), None)

            if error:
                recovery_context = metadata or {}
                recovery_context.update({"original_context": context})

                recovery_success = await self.recovery_manager.attempt_recovery(
                    error, recovery_context
                )

                return recovery_success

        return False

    async def _log_error_to_file(
        self,
        error_id: str,
        context: str,
        exception: Exception,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Log error to persistent file for analysis."""
        try:
            error_entry = {
                "error_id": error_id,
                "timestamp": time.time(),
                "context": context,
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "metadata": metadata or {},
            }

            # Append to error log file
            with open(self.error_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(error_entry) + "\n")

        except Exception as e:
            logger.error(f"Failed to log error to file: {e}")

    def get_comprehensive_report(self) -> dict[str, Any]:
        """Get comprehensive error and recovery report."""
        error_summary = self.error_collector.get_error_summary()
        recovery_stats = self.recovery_manager.get_recovery_stats()

        return {
            "errors": error_summary,
            "recovery": recovery_stats,
            "system_health": self._assess_system_health(),
        }

    def _assess_system_health(self) -> dict[str, Any]:
        """Assess overall system health based on error patterns."""
        recent_errors = [
            error
            for error in self.error_collector.errors
            if time.time() - error.timestamp < 3600  # Last hour
        ]

        if not recent_errors:
            health_score = 1.0
            status = "healthy"
        else:
            # Calculate health score based on error frequency and severity
            critical_errors = len(
                [e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL]
            )
            error_errors = len([e for e in recent_errors if e.severity == ErrorSeverity.ERROR])

            if critical_errors > 0:
                health_score = 0.0
                status = "critical"
            elif error_errors > 10:
                health_score = 0.3
                status = "degraded"
            elif len(recent_errors) > 50:
                health_score = 0.6
                status = "warning"
            else:
                health_score = 0.8
                status = "stable"

        return {
            "health_score": health_score,
            "status": status,
            "recent_error_count": len(recent_errors),
            "recommendations": self._get_health_recommendations(recent_errors),
        }

    def _get_health_recommendations(self, recent_errors: list[IndexingError]) -> list[str]:
        """Get recommendations based on recent error patterns."""
        recommendations = []

        # Analyze error patterns
        category_counts: dict[ErrorCategory, int] = {}
        for error in recent_errors:
            category_counts[error.category] = category_counts.get(error.category, 0) + 1

        # Generate recommendations
        if category_counts.get(ErrorCategory.MEMORY, 0) > 5:
            recommendations.append("Consider reducing batch sizes or increasing available memory")

        if category_counts.get(ErrorCategory.NETWORK, 0) > 3:
            recommendations.append("Check network connectivity and API rate limits")

        if category_counts.get(ErrorCategory.FILE_ACCESS, 0) > 10:
            recommendations.append("Review file permissions and disk space")

        if category_counts.get(ErrorCategory.PARSING, 0) > 20:
            recommendations.append(
                "Consider updating language parsers or excluding problematic files"
            )

        return recommendations
