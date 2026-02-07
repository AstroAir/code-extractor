"""Tests for pysearch.utils.file_watcher module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pysearch.core.config import SearchConfig
from pysearch.utils.file_watcher import (
    EventType,
    FileEvent,
    FileWatcher,
    WatchManager,
    matches_patterns,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_values(self):
        assert EventType.CREATED is not None
        assert EventType.MODIFIED is not None
        assert EventType.DELETED is not None


class TestFileEvent:
    """Tests for FileEvent dataclass."""

    def test_creation(self):
        evt = FileEvent(
            event_type=EventType.CREATED,
            path=Path("test.py"),
            timestamp=1.0,
        )
        assert evt.event_type == EventType.CREATED
        assert evt.path == Path("test.py")
        assert evt.timestamp == 1.0


class TestMatchesPatterns:
    """Tests for matches_patterns function."""

    def test_matches_include(self):
        result = matches_patterns(Path("test.py"), ["*.py"])
        assert result is True

    def test_no_match_include(self):
        result = matches_patterns(Path("test.txt"), ["*.py"])
        assert result is False

    def test_wildcard(self):
        result = matches_patterns(Path("test.py"), ["*"])
        assert result is True

    def test_empty_patterns(self):
        result = matches_patterns(Path("test.py"), [])
        assert result is False


class TestWatchManager:
    """Tests for WatchManager class."""

    def test_init(self):
        mgr = WatchManager()
        assert mgr is not None

    def test_list_watchers_empty(self):
        mgr = WatchManager()
        watchers = mgr.list_watchers()
        assert isinstance(watchers, list)

    def test_add_watcher(self, tmp_path: Path):
        mgr = WatchManager()
        result = mgr.add_watcher("test", tmp_path)
        assert result is True

    def test_remove_watcher(self, tmp_path: Path):
        mgr = WatchManager()
        mgr.add_watcher("w1", tmp_path)
        result = mgr.remove_watcher("w1")
        assert isinstance(result, bool)

    def test_remove_nonexistent(self):
        mgr = WatchManager()
        assert mgr.remove_watcher("nonexistent") is False

    def test_stop_all(self, tmp_path: Path):
        mgr = WatchManager()
        mgr.add_watcher("w1", tmp_path)
        mgr.stop_all()
