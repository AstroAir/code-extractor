"""Tests for pysearch.utils.logging_config module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.utils.logging_config import (
    LogFormat,
    LogLevel,
    SearchLogger,
    get_logger,
)


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_values(self):
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"


class TestLogFormat:
    """Tests for LogFormat enum."""

    def test_values(self):
        assert LogFormat.SIMPLE == "simple"
        assert LogFormat.DETAILED == "detailed"
        assert LogFormat.JSON == "json"
        assert LogFormat.STRUCTURED == "structured"


class TestSearchLogger:
    """Tests for SearchLogger class."""

    def test_init_default(self):
        logger = SearchLogger()
        assert logger.name == "pysearch"
        assert logger.level == LogLevel.INFO

    def test_init_custom(self):
        logger = SearchLogger(
            name="test", level=LogLevel.DEBUG,
            format_type=LogFormat.JSON,
        )
        assert logger.name == "test"
        assert logger.level == LogLevel.DEBUG
        assert logger.format_type == LogFormat.JSON

    def test_log_info(self):
        logger = SearchLogger()
        logger.info("test message")  # should not raise

    def test_with_file_logging(self, tmp_path: Path):
        log_file = tmp_path / "test.log"
        logger = SearchLogger(
            log_file=log_file, enable_file=True,
        )
        assert logger is not None


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger(self):
        logger = get_logger()
        assert logger is not None

    def test_returns_same_instance(self):
        l1 = get_logger()
        l2 = get_logger()
        assert l1 is l2
