"""Tests for pysearch.utils.error_handling module (advanced error handling classes)."""

from __future__ import annotations

import pytest

from pysearch.utils.error_handling import (
    AdvancedErrorCollector,
    CircuitBreaker,
    ErrorCategory,
    ErrorSeverity,
    IndexingError,
    RecoveryManager,
)


class TestErrorSeverity:
    """Tests for ErrorSeverity enum."""

    def test_values(self):
        assert ErrorSeverity.DEBUG == "debug"
        assert ErrorSeverity.INFO == "info"
        assert ErrorSeverity.WARNING == "warning"
        assert ErrorSeverity.ERROR == "error"
        assert ErrorSeverity.CRITICAL == "critical"


class TestIndexingError:
    """Tests for IndexingError dataclass."""

    def test_creation(self):
        err = IndexingError(
            error_id="e1",
            message="Test error",
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.UNKNOWN,
            context="test_module",
        )
        assert err.message == "Test error"
        assert err.severity == ErrorSeverity.WARNING

    def test_defaults(self):
        err = IndexingError(
            error_id="e2",
            message="err",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.UNKNOWN,
            context="src",
        )
        assert err.stack_trace is None
        assert err.timestamp > 0


class TestAdvancedErrorCollector:
    """Tests for advanced ErrorCollector class."""

    def test_init(self):
        collector = AdvancedErrorCollector()
        assert collector is not None

    @pytest.mark.asyncio
    async def test_add_error(self):
        collector = AdvancedErrorCollector()
        await collector.add_error("test.py", "Error occurred", ErrorSeverity.WARNING)
        assert len(collector.get_all_errors()) >= 1

    @pytest.mark.asyncio
    async def test_get_errors_by_severity(self):
        collector = AdvancedErrorCollector()
        await collector.add_error("a.py", "warn", ErrorSeverity.WARNING)
        await collector.add_error("b.py", "err", ErrorSeverity.ERROR)
        warnings = collector.get_errors_by_severity(ErrorSeverity.WARNING)
        assert len(warnings) >= 1

    @pytest.mark.asyncio
    async def test_clear(self):
        collector = AdvancedErrorCollector()
        await collector.add_error("a.py", "err", ErrorSeverity.ERROR)
        collector.clear_errors()
        assert len(collector.get_all_errors()) == 0


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_init(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        assert cb is not None

    @pytest.mark.asyncio
    async def test_call_success(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        result = await cb.call(lambda: 42)
        assert result == 42

    @pytest.mark.asyncio
    async def test_call_failure(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=30)

        def failing():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await cb.call(failing)
        with pytest.raises(ValueError):
            await cb.call(failing)


class TestRecoveryManager:
    """Tests for RecoveryManager class."""

    def test_init(self):
        from unittest.mock import MagicMock

        mgr = RecoveryManager(config=MagicMock())
        assert mgr is not None

    @pytest.mark.asyncio
    async def test_attempt_recovery(self):
        from unittest.mock import MagicMock

        mgr = RecoveryManager(config=MagicMock())
        err = IndexingError(
            error_id="e1",
            message="test",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.UNKNOWN,
            context="test",
        )
        result = await mgr.attempt_recovery(err, {})
        assert isinstance(result, bool)
