"""Tests for pysearch.core.managers.file_watching module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from pysearch.core.config import SearchConfig
from pysearch.core.managers.file_watching import FileWatchingManager


class TestFileWatchingManager:
    """Tests for FileWatchingManager class."""

    def test_init(self):
        cfg = SearchConfig()
        mgr = FileWatchingManager(cfg)
        assert mgr.config is cfg
        assert mgr._auto_watch_enabled is False
        assert mgr.watch_manager is None
        assert mgr._indexer is None

    def test_set_indexer(self):
        mgr = FileWatchingManager(SearchConfig())
        mock_indexer = MagicMock()
        mgr.set_indexer(mock_indexer)
        assert mgr._indexer is mock_indexer

    def test_is_auto_watch_enabled_default(self):
        mgr = FileWatchingManager(SearchConfig())
        assert mgr.is_auto_watch_enabled() is False

    def test_list_watchers_no_manager(self):
        mgr = FileWatchingManager(SearchConfig())
        assert mgr.list_watchers() == []

    def test_get_watch_stats_no_manager(self):
        mgr = FileWatchingManager(SearchConfig())
        stats = mgr.get_watch_stats()
        assert stats == {}

    def test_disable_auto_watch_when_not_enabled(self):
        mgr = FileWatchingManager(SearchConfig())
        mgr.disable_auto_watch()
        assert mgr.is_auto_watch_enabled() is False

    def test_remove_watcher_no_manager(self):
        mgr = FileWatchingManager(SearchConfig())
        assert mgr.remove_watcher("nonexistent") is False

    def test_list_watchers_with_mock_manager(self):
        mgr = FileWatchingManager(SearchConfig())
        mock_wm = MagicMock()
        mock_wm.list_watchers.return_value = ["w1", "w2"]
        mgr.watch_manager = mock_wm
        assert mgr.list_watchers() == ["w1", "w2"]

    def test_get_watch_stats_with_mock_manager(self):
        mgr = FileWatchingManager(SearchConfig())
        mock_wm = MagicMock()
        mock_wm.get_all_stats.return_value = {"w1": {"events_processed": 10}}
        mgr.watch_manager = mock_wm
        stats = mgr.get_watch_stats()
        assert stats["w1"]["events_processed"] == 10

    def test_remove_watcher_with_mock_manager(self):
        mgr = FileWatchingManager(SearchConfig())
        mock_wm = MagicMock()
        mock_wm.remove_watcher.return_value = True
        mgr.watch_manager = mock_wm
        assert mgr.remove_watcher("w1") is True

    def test_add_custom_watcher_with_mock_manager(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        mgr = FileWatchingManager(cfg)
        mock_wm = MagicMock()
        mock_wm.add_watcher.return_value = True
        mgr.watch_manager = mock_wm
        handler = MagicMock()
        result = mgr.add_custom_watcher("test_watcher", tmp_path, handler)
        assert result is True

    def test_pause_watching_with_mock_manager(self):
        mgr = FileWatchingManager(SearchConfig())
        mock_wm = MagicMock()
        mgr.watch_manager = mock_wm
        mgr.pause_watching()
        mock_wm.pause_all.assert_called_once()

    def test_pause_watching_no_manager(self):
        mgr = FileWatchingManager(SearchConfig())
        mgr.pause_watching()  # should not raise

    def test_resume_watching_with_mock_manager(self):
        mgr = FileWatchingManager(SearchConfig())
        mock_wm = MagicMock()
        mgr.watch_manager = mock_wm
        mgr.resume_watching()
        mock_wm.resume_all.assert_called_once()

    def test_resume_watching_no_manager(self):
        mgr = FileWatchingManager(SearchConfig())
        mgr.resume_watching()  # should not raise

    def test_force_rescan_with_indexer(self):
        mgr = FileWatchingManager(SearchConfig())
        mock_indexer = MagicMock()
        mgr.set_indexer(mock_indexer)
        mock_wm = MagicMock()
        mgr.watch_manager = mock_wm
        assert mgr.force_rescan() is True
        mock_indexer.scan.assert_called_once()

    def test_force_rescan_no_indexer(self):
        mgr = FileWatchingManager(SearchConfig())
        assert mgr.force_rescan() is False

    def test_force_rescan_no_watch_manager(self):
        mgr = FileWatchingManager(SearchConfig())
        mgr._indexer = MagicMock()
        assert mgr.force_rescan() is False

    def test_get_watcher_status_no_manager(self):
        mgr = FileWatchingManager(SearchConfig())
        assert mgr.get_watcher_status("w1") == {}

    def test_get_watcher_status_with_mock(self):
        mgr = FileWatchingManager(SearchConfig())
        mock_wm = MagicMock()
        mock_wm.get_watcher_status.return_value = {"running": True}
        mgr.watch_manager = mock_wm
        assert mgr.get_watcher_status("w1")["running"] is True

    def test_set_watch_filters_no_manager(self):
        mgr = FileWatchingManager(SearchConfig())
        mgr.set_watch_filters(include_patterns=["*.py"])  # should not raise

    def test_set_watch_filters_with_mock(self):
        mgr = FileWatchingManager(SearchConfig())
        mock_wm = MagicMock()
        mgr.watch_manager = mock_wm
        mgr.set_watch_filters(include_patterns=["*.py"], exclude_patterns=["test_*"])
        mock_wm.set_filters.assert_called_once_with(["*.py"], ["test_*"])

    def test_get_watch_performance_metrics_no_manager(self):
        mgr = FileWatchingManager(SearchConfig())
        assert mgr.get_watch_performance_metrics() == {}

    def test_get_watch_performance_metrics_with_mock(self):
        mgr = FileWatchingManager(SearchConfig())
        mock_wm = MagicMock()
        mock_wm.get_all_stats.return_value = {
            "w1": {"events_processed": 100, "errors": 2, "avg_processing_time": 0.5}
        }
        mgr.watch_manager = mock_wm
        metrics = mgr.get_watch_performance_metrics()
        assert metrics["total_watchers"] == 1
        assert metrics["total_events_processed"] == 100
        assert metrics["total_errors"] == 2

    def test_enable_auto_watch_already_enabled(self):
        mgr = FileWatchingManager(SearchConfig())
        mgr._auto_watch_enabled = True
        assert mgr.enable_auto_watch() is True

    def test_disable_auto_watch_stops_watchers(self):
        mgr = FileWatchingManager(SearchConfig())
        mock_wm = MagicMock()
        mgr.watch_manager = mock_wm
        mgr._auto_watch_enabled = True
        mgr.disable_auto_watch()
        assert mgr._auto_watch_enabled is False
        mock_wm.stop_all.assert_called_once()

    def test_set_cache_invalidation_callback(self):
        mgr = FileWatchingManager(SearchConfig())
        callback = MagicMock()
        mgr.set_cache_invalidation_callback(callback)
        assert mgr._cache_invalidation_callback is callback

    def test_enable_auto_watch_passes_callback(self, tmp_path: Path):
        """Verify that cache_invalidation_callback is passed through to add_watcher."""
        cfg = SearchConfig(paths=[str(tmp_path)])
        mgr = FileWatchingManager(cfg)
        callback = MagicMock()
        mgr.set_cache_invalidation_callback(callback)
        mgr.set_indexer(MagicMock())

        mock_wm = MagicMock()
        mock_wm.add_watcher.return_value = True
        mock_wm.start_all.return_value = 1
        mgr.watch_manager = mock_wm

        result = mgr.enable_auto_watch()
        assert result is True
        # Verify callback was passed through
        call_kwargs = mock_wm.add_watcher.call_args
        assert call_kwargs[1].get("cache_invalidation_callback") is callback
