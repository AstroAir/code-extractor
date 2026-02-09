"""
Tests for mcp/shared/validation.py — InputValidator and helper functions.

Covers: validate_pattern, validate_regex_pattern, validate_paths,
validate_context_lines, validate_file_path, validate_session_id,
validate_similarity_threshold, validate_max_results, validate_operation_id,
check_rate_limit, record_validation, get_validation_stats,
validate_tool_input, check_validation_results, get_sanitized_values.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.shared.validation import (
    InputValidator,
    PerformanceValidationError,
    SecurityValidationError,
    ValidationError,
    check_validation_results,
    get_sanitized_values,
    validate_tool_input,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def validator():
    """Create a fresh InputValidator instance."""
    return InputValidator()


# ---------------------------------------------------------------------------
# validate_pattern
# ---------------------------------------------------------------------------


class TestValidatePattern:
    """Tests for InputValidator.validate_pattern."""

    def test_success(self, validator):
        """Valid pattern returns is_valid=True with sanitized value."""
        result = validator.validate_pattern("test pattern")
        assert result.is_valid
        assert result.sanitized_value == "test pattern"

    def test_empty_string(self, validator):
        """Empty string is rejected."""
        result = validator.validate_pattern("")
        assert not result.is_valid
        assert any("cannot be empty" in str(e) for e in result.errors)

    def test_too_long(self, validator):
        """Pattern exceeding 10k chars is rejected as PerformanceValidationError."""
        result = validator.validate_pattern("x" * 20000)
        assert not result.is_valid
        assert any("Pattern too long" in str(e) for e in result.errors)
        assert any(isinstance(e, PerformanceValidationError) for e in result.errors)

    def test_dangerous_content(self, validator):
        """Pattern containing dangerous content is rejected."""
        dangerous = ".*" * 1000
        result = validator.validate_pattern(dangerous)
        assert not result.is_valid or len(result.warnings) > 0

    def test_sanitization_warning(self, validator):
        """Pattern with null bytes gets sanitized and a warning is added."""
        result = validator.validate_pattern("hello\x00world")
        assert result.is_valid
        assert "\x00" not in result.sanitized_value
        assert any("sanitized" in w for w in result.warnings)

    def test_non_string_rejected(self, validator):
        """Non-string input is rejected."""
        result = validator.validate_pattern(12345)  # type: ignore[arg-type]
        assert not result.is_valid


# ---------------------------------------------------------------------------
# validate_regex_pattern
# ---------------------------------------------------------------------------


class TestValidateRegexPattern:
    """Tests for InputValidator.validate_regex_pattern."""

    def test_valid_regex(self, validator):
        """Valid regex passes validation."""
        result = validator.validate_regex_pattern(r"def\s+\w+")
        assert result.is_valid

    def test_invalid_regex(self, validator):
        """Invalid regex is rejected."""
        result = validator.validate_regex_pattern("[unclosed")
        assert not result.is_valid
        assert any("Invalid regex" in str(e) for e in result.errors)

    def test_complex_regex_warning(self, validator):
        """Complex regex with deeply nested groups gets a performance warning."""
        # Pattern with > 10 groups triggers _is_complex_regex
        pattern = "(" * 11 + "a" + ")" * 11
        result = validator.validate_regex_pattern(pattern)
        # Should either fail or produce a warning about complexity
        if result.is_valid:
            assert len(result.warnings) > 0


# ---------------------------------------------------------------------------
# validate_paths
# ---------------------------------------------------------------------------


class TestValidatePaths:
    """Tests for InputValidator.validate_paths."""

    def test_success(self, validator):
        """Valid paths return sanitized list."""
        paths = ["./src", "./tests", "/home/user/project"]
        result = validator.validate_paths(paths)
        assert result.is_valid
        assert len(result.sanitized_value) == 3

    def test_none_allowed(self, validator):
        """None is accepted (will use default paths)."""
        result = validator.validate_paths(None)
        assert result.is_valid
        assert result.sanitized_value is None

    def test_traversal_filtered(self, validator):
        """Path traversal attempts are filtered out."""
        malicious_paths = ["../../../etc/passwd", "./safe/path"]
        result = validator.validate_paths(malicious_paths)
        assert len(result.sanitized_value) == 1
        assert result.sanitized_value[0] == "./safe/path"

    def test_too_many_paths(self, validator):
        """More than 100 paths is rejected."""
        paths = [f"./path_{i}" for i in range(101)]
        result = validator.validate_paths(paths)
        assert not result.is_valid

    def test_non_string_element(self, validator):
        """Non-string path elements are rejected."""
        result = validator.validate_paths([123, "./ok"])  # type: ignore[list-item]
        assert any("must be a string" in str(e) for e in result.errors)


# ---------------------------------------------------------------------------
# validate_context_lines
# ---------------------------------------------------------------------------


class TestValidateContextLines:
    """Tests for InputValidator.validate_context_lines."""

    def test_valid(self, validator):
        """Valid context value passes."""
        result = validator.validate_context_lines(5)
        assert result.is_valid
        assert result.sanitized_value == 5

    def test_too_large(self, validator):
        """Context > 1000 is rejected."""
        result = validator.validate_context_lines(2000)
        assert not result.is_valid
        assert any("too large" in str(e) for e in result.errors)

    def test_negative(self, validator):
        """Negative context is rejected."""
        result = validator.validate_context_lines(-1)
        assert not result.is_valid

    def test_large_warning(self, validator):
        """Context > 50 produces a performance warning."""
        result = validator.validate_context_lines(60)
        assert result.is_valid
        assert any("performance" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# validate_file_path
# ---------------------------------------------------------------------------


class TestValidateFilePath:
    """Tests for InputValidator.validate_file_path."""

    def test_valid(self, validator):
        """Valid relative file path passes."""
        result = validator.validate_file_path("./src/main.py")
        assert result.is_valid
        assert result.sanitized_value is not None

    def test_empty(self, validator):
        """Empty file path is rejected."""
        result = validator.validate_file_path("   ")
        assert not result.is_valid

    def test_traversal(self, validator):
        """Path traversal attempt is rejected."""
        result = validator.validate_file_path("../../../etc/passwd")
        assert not result.is_valid
        assert any(isinstance(e, SecurityValidationError) for e in result.errors)

    def test_system_path(self, validator):
        """Absolute system path is rejected."""
        # Use a Windows system path since tests run on Windows
        result = validator.validate_file_path("C:\\Windows\\System32\\cmd.exe")
        assert not result.is_valid


# ---------------------------------------------------------------------------
# validate_session_id
# ---------------------------------------------------------------------------


class TestValidateSessionId:
    """Tests for InputValidator.validate_session_id."""

    def test_valid(self, validator):
        """Valid session ID passes."""
        result = validator.validate_session_id("abc-123_XYZ")
        assert result.is_valid
        assert result.sanitized_value == "abc-123_XYZ"

    def test_none_allowed(self, validator):
        """None session ID is accepted."""
        result = validator.validate_session_id(None)
        assert result.is_valid
        assert result.sanitized_value is None

    def test_invalid_chars(self, validator):
        """Session ID with invalid characters is rejected."""
        result = validator.validate_session_id("abc!@#$%")
        assert not result.is_valid

    def test_too_long(self, validator):
        """Session ID exceeding 128 chars is rejected."""
        result = validator.validate_session_id("a" * 200)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# validate_similarity_threshold
# ---------------------------------------------------------------------------


class TestValidateSimilarityThreshold:
    """Tests for InputValidator.validate_similarity_threshold."""

    def test_valid(self, validator):
        """Valid threshold passes."""
        result = validator.validate_similarity_threshold(0.8)
        assert result.is_valid
        assert result.sanitized_value == 0.8

    def test_out_of_range(self, validator):
        """Threshold > 1.0 is rejected."""
        result = validator.validate_similarity_threshold(1.5)
        assert not result.is_valid
        assert any("must be between 0.0 and 1.0" in str(e) for e in result.errors)

    def test_negative(self, validator):
        """Negative threshold is rejected."""
        result = validator.validate_similarity_threshold(-0.1)
        assert not result.is_valid

    def test_boundary_zero(self, validator):
        """Threshold of 0.0 is valid."""
        result = validator.validate_similarity_threshold(0.0)
        assert result.is_valid

    def test_boundary_one(self, validator):
        """Threshold of 1.0 is valid."""
        result = validator.validate_similarity_threshold(1.0)
        assert result.is_valid


# ---------------------------------------------------------------------------
# validate_max_results
# ---------------------------------------------------------------------------


class TestValidateMaxResults:
    """Tests for InputValidator.validate_max_results."""

    def test_valid(self, validator):
        """Valid max_results passes."""
        result = validator.validate_max_results(100)
        assert result.is_valid
        assert result.sanitized_value == 100

    def test_too_large(self, validator):
        """max_results > 10000 is rejected."""
        result = validator.validate_max_results(50000)
        assert not result.is_valid
        assert any("too large" in str(e) for e in result.errors)

    def test_zero(self, validator):
        """max_results of 0 is rejected."""
        result = validator.validate_max_results(0)
        assert not result.is_valid

    def test_negative(self, validator):
        """Negative max_results is rejected."""
        result = validator.validate_max_results(-5)
        assert not result.is_valid

    def test_large_warning(self, validator):
        """max_results > 1000 produces a performance warning."""
        result = validator.validate_max_results(5000)
        assert result.is_valid
        assert any("performance" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# validate_operation_id
# ---------------------------------------------------------------------------


class TestValidateOperationId:
    """Tests for InputValidator.validate_operation_id."""

    def test_valid(self, validator):
        """Valid operation ID passes."""
        result = validator.validate_operation_id("op_123-abc")
        assert result.is_valid
        assert result.sanitized_value == "op_123-abc"

    def test_invalid_chars(self, validator):
        """Operation ID with invalid characters is rejected."""
        result = validator.validate_operation_id("op!@#")
        assert not result.is_valid

    def test_too_long(self, validator):
        """Operation ID exceeding 64 chars is rejected."""
        result = validator.validate_operation_id("x" * 100)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# check_rate_limit
# ---------------------------------------------------------------------------


class TestRateLimit:
    """Tests for InputValidator.check_rate_limit."""

    def test_first_request_allowed(self):
        """First request should be allowed."""
        validator = InputValidator()
        result = validator.check_rate_limit("test_user")
        assert result.is_valid

    def test_exceed_limit(self):
        """Exceeding 100 requests/minute triggers rate limiting."""
        validator = InputValidator()
        for _ in range(150):
            validator.check_rate_limit("test_user")

        result = validator.check_rate_limit("test_user")
        assert not result.is_valid
        assert any("rate limit" in str(e).lower() for e in result.errors)


# ---------------------------------------------------------------------------
# record_validation / get_validation_stats
# ---------------------------------------------------------------------------


class TestValidationStats:
    """Tests for record_validation and get_validation_stats."""

    def test_record_and_stats(self, validator):
        """Recording validations updates stats correctly."""
        r1 = validator.validate_pattern("good")
        validator.record_validation("pattern", r1)

        r2 = validator.validate_pattern("")
        validator.record_validation("pattern", r2)

        stats = validator.get_validation_stats()
        assert stats["total_validations"] >= 2
        assert stats["failed_validations"] >= 1
        assert 0.0 <= stats["success_rate"] <= 1.0

    def test_empty_stats(self):
        """Empty validation history returns message."""
        v = InputValidator()
        stats = v.get_validation_stats()
        assert "message" in stats or stats.get("total_validations", 0) == 0


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestModuleHelpers:
    """Tests for validate_tool_input, check_validation_results, get_sanitized_values."""

    def test_validate_tool_input_pattern(self):
        """validate_tool_input validates pattern parameter."""
        results = validate_tool_input(pattern="hello")
        assert "pattern" in results
        assert results["pattern"].is_valid

    def test_validate_tool_input_multiple(self):
        """validate_tool_input handles multiple parameters."""
        results = validate_tool_input(pattern="hello", context=3, paths=None)
        assert "pattern" in results
        assert "context" in results
        assert "paths" in results

    def test_check_validation_results_pass(self):
        """check_validation_results does not raise on valid results."""
        results = validate_tool_input(pattern="hello")
        check_validation_results(results)  # Should not raise

    def test_check_validation_results_fail(self):
        """check_validation_results raises ValidationError on invalid results."""
        results = validate_tool_input(pattern="")
        with pytest.raises(ValidationError):
            check_validation_results(results)

    def test_get_sanitized_values(self):
        """get_sanitized_values extracts sanitized values."""
        results = validate_tool_input(pattern="hello", context=5)
        sanitized = get_sanitized_values(results)
        assert sanitized["pattern"] == "hello"
        assert sanitized["context"] == 5
