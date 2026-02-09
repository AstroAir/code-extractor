#!/usr/bin/env python3
"""
MCP Input Validation and Error Handling for PySearch

This module provides comprehensive input validation, error handling,
and graceful degradation for all MCP tools and resources.

Features:
- Parameter validation with type checking
- Sanitization of user inputs
- Graceful error recovery
- Rate limiting and abuse prevention
- Performance monitoring and alerting
- Comprehensive logging and debugging
"""

from __future__ import annotations

import re
import time
from enum import Enum
from pathlib import Path
from typing import Any


# Validation error classes
class ValidationError(Exception):
    """Base class for validation errors."""

    def __init__(self, message: str, field: str = "", value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format validation error message."""
        if self.field:
            return f"Validation error in '{self.field}': {self.message}"
        return f"Validation error: {self.message}"


class SecurityValidationError(ValidationError):
    """Error raised for security-related validation failures."""

    pass


class PerformanceValidationError(ValidationError):
    """Error raised for performance-related validation failures."""

    pass


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationResult:
    """Result of input validation."""

    def __init__(
        self,
        is_valid: bool = True,
        errors: list[ValidationError] | None = None,
        warnings: list[str] | None = None,
        sanitized_value: Any = None,
    ) -> None:
        self.is_valid = is_valid
        self.errors: list[ValidationError] = errors or []
        self.warnings: list[str] = warnings or []
        self.sanitized_value = sanitized_value

    def add_error(self, error: ValidationError) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)


class RateLimiter:
    """Simple rate limiter for preventing abuse."""

    def __init__(self, max_requests: int = 100, time_window: int = 60) -> None:
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: dict[str, list[float]] = {}

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for the given identifier."""
        current_time = time.time()

        if identifier not in self.requests:
            self.requests[identifier] = []

        # Remove old requests outside the time window
        self.requests[identifier] = [
            req_time
            for req_time in self.requests[identifier]
            if current_time - req_time < self.time_window
        ]

        # Check if under the limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(current_time)
            return True

        return False

    def get_remaining_requests(self, identifier: str) -> int:
        """Get number of remaining requests for identifier."""
        current_time = time.time()

        if identifier not in self.requests:
            return self.max_requests

        # Count recent requests
        recent_requests = [
            req_time
            for req_time in self.requests[identifier]
            if current_time - req_time < self.time_window
        ]

        return max(0, self.max_requests - len(recent_requests))


class InputValidator:
    """Comprehensive input validator for MCP tools."""

    def __init__(self) -> None:
        self.rate_limiter = RateLimiter()
        self.validation_history: list[dict[str, Any]] = []

    def validate_pattern(self, pattern: str, field_name: str = "pattern") -> ValidationResult:
        """Validate search pattern input."""
        result = ValidationResult()

        # Check for None or empty
        if not pattern:
            result.add_error(ValidationError("Pattern cannot be empty", field_name, pattern))
            return result

        # Type check
        if not isinstance(pattern, str):
            result.add_error(ValidationError("Pattern must be a string", field_name, type(pattern)))
            return result

        # Length check
        if len(pattern) > 10000:  # 10KB limit
            result.add_error(
                PerformanceValidationError(
                    "Pattern too long (max 10,000 characters)", field_name, len(pattern)
                )
            )
            return result

        # Check for potential security issues
        if self._contains_dangerous_patterns(pattern):
            result.add_error(
                SecurityValidationError(
                    "Pattern contains potentially dangerous content", field_name, pattern[:100]
                )
            )
            return result

        # Sanitize and store result
        sanitized_pattern = self._sanitize_pattern(pattern)
        result.sanitized_value = sanitized_pattern

        if sanitized_pattern != pattern:
            result.add_warning("Pattern was sanitized for security")

        return result

    def validate_regex_pattern(self, pattern: str, field_name: str = "pattern") -> ValidationResult:
        """Validate regex pattern with additional checks."""
        result = self.validate_pattern(pattern, field_name)

        if not result.is_valid:
            return result

        # Test regex compilation
        try:
            import re

            re.compile(pattern, re.MULTILINE)

            # Check for potentially catastrophic backtracking
            if self._is_complex_regex(pattern):
                result.add_warning("Complex regex pattern may cause performance issues")

        except re.error as e:
            result.add_error(
                ValidationError(f"Invalid regex pattern: {str(e)}", field_name, pattern)
            )
            return result
        except Exception as e:
            result.add_error(
                ValidationError(f"Regex validation error: {str(e)}", field_name, pattern)
            )
            return result

        return result

    def validate_paths(
        self, paths: list[str] | None, field_name: str = "paths"
    ) -> ValidationResult:
        """Validate file paths input."""
        result = ValidationResult()

        # Allow None (will use default paths)
        if paths is None:
            result.sanitized_value = None
            return result

        # Type check
        if not isinstance(paths, list):
            result.add_error(ValidationError("Paths must be a list", field_name, type(paths)))
            return result

        # Length check
        if len(paths) > 100:
            result.add_error(
                PerformanceValidationError(
                    "Too many paths specified (max 100)", field_name, len(paths)
                )
            )
            return result

        sanitized_paths = []
        for i, path in enumerate(paths):
            if not isinstance(path, str):
                result.add_error(
                    ValidationError(
                        f"Path at index {i} must be a string", f"{field_name}[{i}]", type(path)
                    )
                )
                continue

            # Security check for path traversal
            if self._is_path_traversal_attempt(path):
                result.add_error(
                    SecurityValidationError(
                        "Path traversal attempt detected", f"{field_name}[{i}]", path
                    )
                )
                continue

            # Sanitize path
            sanitized_path = self._sanitize_path(path)
            sanitized_paths.append(sanitized_path)

            if sanitized_path != path:
                result.add_warning(f"Path at index {i} was sanitized")

        result.sanitized_value = sanitized_paths
        return result

    def validate_context_lines(self, context: int, field_name: str = "context") -> ValidationResult:
        """Validate context lines parameter."""
        result = ValidationResult()

        # Type check
        if not isinstance(context, int):
            result.add_error(
                ValidationError("Context must be an integer", field_name, type(context))
            )
            return result

        # Range check
        if context < 0:
            result.add_error(ValidationError("Context cannot be negative", field_name, context))
            return result

        if context > 1000:
            result.add_error(
                PerformanceValidationError(
                    "Context too large (max 1000 lines)", field_name, context
                )
            )
            return result

        # Performance warning
        if context > 50:
            result.add_warning("Large context values may impact performance")

        result.sanitized_value = context
        return result

    def validate_file_path(self, file_path: str, field_name: str = "file_path") -> ValidationResult:
        """Validate a single file path."""
        result = ValidationResult()

        # Type check
        if not isinstance(file_path, str):
            result.add_error(
                ValidationError("File path must be a string", field_name, type(file_path))
            )
            return result

        # Empty check
        if not file_path.strip():
            result.add_error(ValidationError("File path cannot be empty", field_name, file_path))
            return result

        # Security check
        if self._is_path_traversal_attempt(file_path):
            result.add_error(
                SecurityValidationError("Path traversal attempt detected", field_name, file_path)
            )
            return result

        # Path validation
        try:
            path_obj = Path(file_path)

            # Check if path is absolute and potentially unsafe
            if path_obj.is_absolute() and self._is_system_path(path_obj):
                result.add_error(
                    SecurityValidationError(
                        "Access to system paths not allowed", field_name, file_path
                    )
                )
                return result

        except Exception as e:
            result.add_error(ValidationError(f"Invalid file path: {str(e)}", field_name, file_path))
            return result

        result.sanitized_value = self._sanitize_path(file_path)
        return result

    def validate_session_id(
        self, session_id: str | None, field_name: str = "session_id"
    ) -> ValidationResult:
        """Validate session ID parameter."""
        result = ValidationResult()

        # Allow None
        if session_id is None:
            result.sanitized_value = None
            return result

        # Type check
        if not isinstance(session_id, str):
            result.add_error(
                ValidationError("Session ID must be a string", field_name, type(session_id))
            )
            return result

        # Format validation
        if not re.match(r"^[a-zA-Z0-9_-]+$", session_id):
            result.add_error(
                ValidationError("Session ID contains invalid characters", field_name, session_id)
            )
            return result

        # Length validation
        if len(session_id) > 128:
            result.add_error(
                ValidationError(
                    "Session ID too long (max 128 characters)", field_name, len(session_id)
                )
            )
            return result

        result.sanitized_value = session_id
        return result

    def validate_similarity_threshold(
        self, threshold: float, field_name: str = "similarity_threshold"
    ) -> ValidationResult:
        """Validate fuzzy search similarity threshold."""
        result = ValidationResult()

        # Type check
        if not isinstance(threshold, int | float):
            result.add_error(
                ValidationError(
                    "Similarity threshold must be a number", field_name, type(threshold)
                )
            )
            return result

        # Range check
        if threshold < 0.0 or threshold > 1.0:
            result.add_error(
                ValidationError(
                    "Similarity threshold must be between 0.0 and 1.0", field_name, threshold
                )
            )
            return result

        result.sanitized_value = float(threshold)
        return result

    def validate_max_results(
        self, max_results: int, field_name: str = "max_results"
    ) -> ValidationResult:
        """Validate maximum results parameter."""
        result = ValidationResult()

        # Type check
        if not isinstance(max_results, int):
            result.add_error(
                ValidationError("Max results must be an integer", field_name, type(max_results))
            )
            return result

        # Range check
        if max_results <= 0:
            result.add_error(
                ValidationError("Max results must be positive", field_name, max_results)
            )
            return result

        if max_results > 10000:
            result.add_error(
                PerformanceValidationError(
                    "Max results too large (max 10,000)", field_name, max_results
                )
            )
            return result

        # Performance warning
        if max_results > 1000:
            result.add_warning("Large max results values may impact performance")

        result.sanitized_value = max_results
        return result

    def validate_operation_id(
        self, operation_id: str, field_name: str = "operation_id"
    ) -> ValidationResult:
        """Validate operation ID for progress tracking."""
        result = ValidationResult()

        # Type check
        if not isinstance(operation_id, str):
            result.add_error(
                ValidationError("Operation ID must be a string", field_name, type(operation_id))
            )
            return result

        # Format validation
        if not re.match(r"^[a-zA-Z0-9_-]+$", operation_id):
            result.add_error(
                ValidationError(
                    "Operation ID contains invalid characters", field_name, operation_id
                )
            )
            return result

        # Length validation
        if len(operation_id) > 64:
            result.add_error(
                ValidationError(
                    "Operation ID too long (max 64 characters)", field_name, len(operation_id)
                )
            )
            return result

        result.sanitized_value = operation_id
        return result

    def check_rate_limit(self, identifier: str) -> ValidationResult:
        """Check rate limiting for requests."""
        result = ValidationResult()

        if not self.rate_limiter.is_allowed(identifier):
            remaining = self.rate_limiter.get_remaining_requests(identifier)
            result.add_error(
                ValidationError(
                    f"Rate limit exceeded. {remaining} requests remaining in window.",
                    "rate_limit",
                    identifier,
                )
            )

        return result

    def _contains_dangerous_patterns(self, pattern: str) -> bool:
        """Check for potentially dangerous patterns in input."""
        dangerous_patterns = [
            r"<script",
            r"javascript:",
            r"eval\s*\(",
            r"exec\s*\(",
            r"system\s*\(",
            r"os\.system",
            r"subprocess",
            r"\$\{.*\}",  # Template injection
            r"{{.*}}",  # Template injection
            r"\\x[0-9a-fA-F]{2}",  # Hex escapes
        ]

        pattern_lower = pattern.lower()
        for dangerous in dangerous_patterns:
            if re.search(dangerous, pattern_lower, re.IGNORECASE):
                return True

        return False

    def _is_complex_regex(self, pattern: str) -> bool:
        """Check if regex pattern is potentially complex/expensive."""
        # Look for patterns that could cause catastrophic backtracking
        complex_patterns = [
            r"\(\.\*\)\+",  # (.*)+
            r"\(\.\+\)\+",  # (.+)+
            r"\(\.\*\)\*",  # (.*)*
            r"\(\.\+\)\*",  # (.+)*
            r"\(\.\*\?\)\+",  # (.*?)+
            r"\(\w\*\)\+",  # (\w*)+
        ]

        for complex_pattern in complex_patterns:
            if re.search(complex_pattern, pattern):
                return True

        # Check for excessive quantifiers
        if len(re.findall(r"[+*]{2,}", pattern)) > 0:
            return True

        # Check for deeply nested groups
        if pattern.count("(") > 10:
            return True

        return False

    def _is_path_traversal_attempt(self, path: str) -> bool:
        """Check for path traversal attempts."""
        # Normalize path separators
        normalized = path.replace("\\", "/")

        # Check for obvious traversal patterns
        traversal_patterns = [
            "../",
            "..\\",
            "..%2f",
            "..%5c",
            "%2e%2e%2f",
            "%2e%2e%5c",
        ]

        path_lower = normalized.lower()
        for pattern in traversal_patterns:
            if pattern in path_lower:
                return True

        # Check for absolute paths to sensitive locations
        try:
            path_obj = Path(path)
            if path_obj.is_absolute():
                path_str = str(path_obj).lower()
                sensitive_paths = ["/etc", "/sys", "/proc", "/root", "c:\\windows", "c:\\system"]
                for sensitive in sensitive_paths:
                    if path_str.startswith(sensitive):
                        return True
        except Exception:
            pass

        return False

    def _is_system_path(self, path: Path) -> bool:
        """Check if path is a protected system path."""
        path_str = str(path).lower()

        system_paths = [
            "/etc",
            "/sys",
            "/proc",
            "/root",
            "/boot",
            "c:\\windows",
            "c:\\system32",
            "c:\\program files",
        ]

        for sys_path in system_paths:
            if path_str.startswith(sys_path):
                return True

        return False

    def _sanitize_pattern(self, pattern: str) -> str:
        """Sanitize search pattern for security."""
        # Remove null bytes
        sanitized = pattern.replace("\x00", "")

        # Limit line lengths to prevent DoS
        lines = sanitized.split("\n")
        sanitized_lines = [line[:1000] for line in lines[:100]]  # Max 100 lines, 1000 chars each

        return "\n".join(sanitized_lines)

    def _sanitize_path(self, path: str) -> str:
        """Sanitize file path for security."""
        # Remove null bytes and control characters
        sanitized = "".join(char for char in path if ord(char) >= 32 or char in "\t\n")

        # Normalize path separators
        sanitized = sanitized.replace("\\", "/")

        # Remove multiple consecutive slashes
        sanitized = re.sub(r"/+", "/", sanitized)

        return sanitized

    def record_validation(self, field: str, result: ValidationResult) -> None:
        """Record validation attempt for monitoring."""
        self.validation_history.append(
            {
                "timestamp": time.time(),
                "field": field,
                "is_valid": result.is_valid,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
                "errors": [{"type": type(e).__name__, "message": e.message} for e in result.errors],
            }
        )

        # Keep history manageable
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-500:]

    def get_validation_stats(self) -> dict[str, Any]:
        """Get validation statistics."""
        if not self.validation_history:
            return {"message": "No validation history available"}

        total_validations = len(self.validation_history)
        failed_validations = len([v for v in self.validation_history if not v["is_valid"]])

        # Recent validation stats (last hour)
        current_time = time.time()
        recent_validations = [
            v for v in self.validation_history if current_time - v["timestamp"] < 3600
        ]

        # Error type distribution
        error_types: dict[str, int] = {}
        for validation in self.validation_history:
            for error in validation.get("errors", []):
                error_type: str = error["type"]
                error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "total_validations": total_validations,
            "success_rate": (
                (total_validations - failed_validations) / total_validations
                if total_validations > 0
                else 0.0
            ),
            "failed_validations": failed_validations,
            "recent_validations_hour": len(recent_validations),
            "error_type_distribution": error_types,
            "most_common_errors": sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5],
        }


# Global validator instance
_validator = None


def get_validator() -> InputValidator:
    """Get global validator instance."""
    global _validator
    if _validator is None:
        _validator = InputValidator()
    return _validator


def validate_tool_input(**kwargs: Any) -> dict[str, ValidationResult]:
    """
    Validate all inputs for an MCP tool.

    Args:
        **kwargs: Tool parameters to validate

    Returns:
        Dictionary of validation results by parameter name
    """
    validator = get_validator()
    results: dict[str, ValidationResult] = {}

    # Common parameter validations
    if "pattern" in kwargs:
        results["pattern"] = validator.validate_pattern(kwargs["pattern"])

    if "paths" in kwargs:
        results["paths"] = validator.validate_paths(kwargs.get("paths"))

    if "context" in kwargs:
        results["context"] = validator.validate_context_lines(kwargs["context"])

    if "session_id" in kwargs:
        results["session_id"] = validator.validate_session_id(kwargs.get("session_id"))

    if "file_path" in kwargs:
        results["file_path"] = validator.validate_file_path(kwargs["file_path"])

    if "similarity_threshold" in kwargs:
        results["similarity_threshold"] = validator.validate_similarity_threshold(
            kwargs["similarity_threshold"]
        )

    if "max_results" in kwargs:
        results["max_results"] = validator.validate_max_results(kwargs["max_results"])

    if "operation_id" in kwargs:
        results["operation_id"] = validator.validate_operation_id(kwargs["operation_id"])

    # Record validation attempts
    for field, result in results.items():
        validator.record_validation(field, result)

    return results


def check_validation_results(results: dict[str, ValidationResult]) -> None:
    """
    Check validation results and raise errors if validation failed.

    Args:
        results: Dictionary of validation results

    Raises:
        ValidationError: If any validation failed
    """
    errors: list[ValidationError] = []
    for _field, result in results.items():
        if not result.is_valid:
            errors.extend(result.errors)

    if errors:
        # Raise the first error for now; could be enhanced to raise multiple
        raise errors[0]


def get_sanitized_values(results: dict[str, ValidationResult]) -> dict[str, Any]:
    """
    Extract sanitized values from validation results.

    Args:
        results: Dictionary of validation results

    Returns:
        Dictionary of sanitized parameter values
    """
    sanitized: dict[str, Any] = {}
    for field, result in results.items():
        if result.sanitized_value is not None:
            sanitized[field] = result.sanitized_value
    return sanitized
