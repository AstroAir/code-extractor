"""Tests for pysearch.utils.file_watcher module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from pysearch.core.config import SearchConfig
from pysearch.utils.file_watcher import (
    EventType,
    FileEvent,
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

    def test_get_watcher_status_exists(self, tmp_path: Path):
        mgr = WatchManager()
        mgr.add_watcher("w1", tmp_path)
        status = mgr.get_watcher_status("w1")
        assert "is_watching" in status
        assert "paused" in status
        assert status["paused"] is False

    def test_get_watcher_status_nonexistent(self):
        mgr = WatchManager()
        assert mgr.get_watcher_status("missing") == {}

    def test_set_filters(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        mgr = WatchManager()
        mgr.add_watcher("w1", tmp_path, config=cfg)
        mgr.set_filters(include_patterns=["*.py"], exclude_patterns=["test_*"])
        watcher = mgr.get_watcher("w1")
        assert watcher is not None
        assert watcher.config.include == ["*.py"]
        assert watcher.config.exclude == ["test_*"]

    def test_set_filters_partial(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        mgr = WatchManager()
        mgr.add_watcher("w1", tmp_path, config=cfg)
        mgr.set_filters(exclude_patterns=["*.txt"])
        watcher = mgr.get_watcher("w1")
        assert watcher is not None
        assert watcher.config.include == ["**/*.py"]  # unchanged
        assert watcher.config.exclude == ["*.txt"]

    def test_pause_all_resume_all(self, tmp_path: Path):
        mgr = WatchManager()
        mgr.add_watcher("w1", tmp_path)
        # Before starting, pausing should be a no-op (not watching)
        mgr.pause_all()
        watcher = mgr.get_watcher("w1")
        assert watcher is not None
        # Not watching, so _paused should not be set
        assert not getattr(watcher, "_paused", False)

    def test_get_all_stats(self, tmp_path: Path):
        mgr = WatchManager()
        mgr.add_watcher("w1", tmp_path)
        stats = mgr.get_all_stats()
        assert "w1" in stats
        assert "is_watching" in stats["w1"]


class TestChangeProcessor:
    """Tests for ChangeProcessor class."""

    def test_cache_invalidation_callback_called(self, tmp_path: Path):
        """Verify cache invalidation callback is triggered on file events."""
        from pysearch.utils.file_watcher import ChangeProcessor

        mock_indexer = MagicMock()
        invalidated_paths: list[Path] = []

        def on_invalidate(paths: list[Path]) -> None:
            invalidated_paths.extend(paths)

        processor = ChangeProcessor(
            mock_indexer,
            debounce_delay=0.0,
            cache_invalidation_callback=on_invalidate,
        )

        # Create a test file so update_file won't fail
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        event = FileEvent(
            path=test_file,
            event_type=EventType.MODIFIED,
            timestamp=1.0,
        )
        processor.process_event(event)
        # Force flush
        processor.flush_pending()
        assert test_file in invalidated_paths

    def test_no_callback_does_not_error(self, tmp_path: Path):
        """Verify processing works fine without a callback."""
        from pysearch.utils.file_watcher import ChangeProcessor

        mock_indexer = MagicMock()
        processor = ChangeProcessor(mock_indexer, debounce_delay=0.0)

        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        event = FileEvent(
            path=test_file,
            event_type=EventType.CREATED,
            timestamp=1.0,
        )
        processor.process_event(event)
        processor.flush_pending()
        # Should not raise

    def test_stats_updated(self, tmp_path: Path):
        """Verify processing statistics are updated correctly."""
        from pysearch.utils.file_watcher import ChangeProcessor

        mock_indexer = MagicMock()
        processor = ChangeProcessor(mock_indexer, debounce_delay=0.0)

        test_file = tmp_path / "a.py"
        test_file.write_text("pass")

        event = FileEvent(
            path=test_file,
            event_type=EventType.MODIFIED,
            timestamp=1.0,
        )
        processor.process_event(event)
        processor.flush_pending()

        stats = processor.get_stats()
        assert stats["events_processed"] >= 1
        assert stats["batches_processed"] >= 1

    def test_deleted_event_calls_remove_file(self, tmp_path: Path):
        """Verify deleted events trigger indexer.remove_file."""
        from pysearch.utils.file_watcher import ChangeProcessor

        mock_indexer = MagicMock()
        processor = ChangeProcessor(mock_indexer, debounce_delay=0.0)

        deleted_file = tmp_path / "gone.py"

        event = FileEvent(
            path=deleted_file,
            event_type=EventType.DELETED,
            timestamp=1.0,
        )
        processor.process_event(event)
        processor.flush_pending()
        mock_indexer.remove_file.assert_called_with(deleted_file)

    def test_moved_event_handling(self, tmp_path: Path):
        """Verify moved events trigger remove on old path and update on new."""
        from pysearch.utils.file_watcher import ChangeProcessor

        mock_indexer = MagicMock()
        processor = ChangeProcessor(mock_indexer, debounce_delay=0.0)

        old_path = tmp_path / "old.py"
        new_path = tmp_path / "new.py"
        new_path.write_text("pass")

        event = FileEvent(
            path=new_path,
            event_type=EventType.MOVED,
            timestamp=1.0,
            old_path=old_path,
        )
        processor.process_event(event)
        processor.flush_pending()
        mock_indexer.remove_file.assert_called_with(old_path)
        mock_indexer.update_file.assert_called_with(new_path)
