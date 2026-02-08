"""Tests for pysearch.indexing.cache.cleanup module."""

from __future__ import annotations

import time

import pytest

from pysearch.indexing.cache.cleanup import CacheCleanup


class TestCacheCleanup:
    """Tests for CacheCleanup class."""

    @staticmethod
    def _noop_callback() -> int:
        return 0

    def test_init_auto_cleanup_enabled(self):
        cleanup = CacheCleanup(
            cleanup_callback=self._noop_callback,
            cleanup_interval=300,
            auto_cleanup=True,
        )
        assert cleanup.auto_cleanup is True
        assert cleanup.cleanup_interval == 300
        cleanup.shutdown()

    def test_init_auto_cleanup_disabled(self):
        cleanup = CacheCleanup(
            cleanup_callback=self._noop_callback,
            cleanup_interval=300,
            auto_cleanup=False,
        )
        assert cleanup.auto_cleanup is False
        assert cleanup.is_running() is False

    def test_start_and_stop_cleanup_thread(self):
        cleanup = CacheCleanup(
            cleanup_callback=self._noop_callback,
            cleanup_interval=300,
            auto_cleanup=False,
        )
        assert cleanup.is_running() is False

        cleanup.start_cleanup_thread()
        assert cleanup.is_running() is True

        cleanup.stop_cleanup_thread(timeout=2.0)
        time.sleep(0.1)
        assert cleanup.is_running() is False

    def test_start_cleanup_thread_idempotent(self):
        cleanup = CacheCleanup(
            cleanup_callback=self._noop_callback,
            cleanup_interval=300,
            auto_cleanup=False,
        )
        cleanup.start_cleanup_thread()
        cleanup.start_cleanup_thread()  # should not raise
        assert cleanup.is_running() is True
        cleanup.shutdown()

    def test_stop_cleanup_thread_when_not_running(self):
        cleanup = CacheCleanup(
            cleanup_callback=self._noop_callback,
            cleanup_interval=300,
            auto_cleanup=False,
        )
        cleanup.stop_cleanup_thread()  # should not raise

    def test_manual_cleanup(self):
        call_count = 0

        def counting_callback() -> int:
            nonlocal call_count
            call_count += 1
            return 5

        cleanup = CacheCleanup(
            cleanup_callback=counting_callback,
            auto_cleanup=False,
        )
        result = cleanup.manual_cleanup()
        assert result == 5
        assert call_count == 1

    def test_manual_cleanup_exception_returns_zero(self):
        def failing_callback() -> int:
            raise RuntimeError("test error")

        cleanup = CacheCleanup(
            cleanup_callback=failing_callback,
            auto_cleanup=False,
        )
        result = cleanup.manual_cleanup()
        assert result == 0

    def test_is_running(self):
        cleanup = CacheCleanup(
            cleanup_callback=self._noop_callback,
            auto_cleanup=False,
        )
        assert cleanup.is_running() is False
        cleanup.start_cleanup_thread()
        assert cleanup.is_running() is True
        cleanup.shutdown()

    def test_get_status(self):
        cleanup = CacheCleanup(
            cleanup_callback=self._noop_callback,
            cleanup_interval=120,
            auto_cleanup=False,
        )
        status = cleanup.get_status()
        assert isinstance(status, dict)
        assert status["auto_cleanup_enabled"] is False
        assert status["cleanup_interval"] == 120
        assert status["thread_running"] is False
        assert status["thread_alive"] is False

    def test_get_status_when_running(self):
        cleanup = CacheCleanup(
            cleanup_callback=self._noop_callback,
            cleanup_interval=300,
            auto_cleanup=True,
        )
        status = cleanup.get_status()
        assert status["auto_cleanup_enabled"] is True
        assert status["thread_running"] is True
        assert status["thread_alive"] is True
        cleanup.shutdown()

    def test_shutdown(self):
        cleanup = CacheCleanup(
            cleanup_callback=self._noop_callback,
            cleanup_interval=300,
            auto_cleanup=True,
        )
        assert cleanup.is_running() is True
        cleanup.shutdown(timeout=2.0)
        time.sleep(0.1)
        assert cleanup.is_running() is False
