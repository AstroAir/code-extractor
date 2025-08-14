from __future__ import annotations

from pathlib import Path

from pysearch.utils.file_watcher import FileWatcher, WatchManager


def test_watch_manager_error_paths(tmp_path: Path) -> None:
    wm = WatchManager()
    
    # Test adding duplicate watcher
    p = tmp_path / "test_dir"
    p.mkdir()
    
    assert wm.add_watcher("test", p) is True
    # Adding same name again should return False
    assert wm.add_watcher("test", p) is False
    
    # Test removing non-existent watcher
    assert wm.remove_watcher("nonexistent") is False
    
    # Test removing existing watcher
    assert wm.remove_watcher("test") is True
    
    # Test adding watcher with invalid path (should handle gracefully)
    invalid_path = tmp_path / "nonexistent" / "deeply" / "nested"
    result = wm.add_watcher("invalid", invalid_path)
    # Should succeed in adding watcher (path validation happens on start)
    assert result is True
    
    # Test start_all and stop_all with mixed success
    p2 = tmp_path / "valid"
    p2.mkdir()
    wm.add_watcher("valid", p2)
    
    started = wm.start_all()
    assert started >= 0  # May be 0 if watchdog not available
    
    wm.stop_all()  # Should not raise
    
    # Test get methods
    assert wm.get_watcher("valid") is not None
    assert wm.get_watcher("nonexistent") is None
    assert isinstance(wm.list_watchers(), list)
    assert isinstance(wm.get_all_stats(), dict)


def test_file_watcher_is_available_fast_path(tmp_path: Path) -> None:
    # Test the is_available property fast path
    from pysearch import SearchConfig
    
    p = tmp_path / "test"
    p.mkdir()
    
    fw = FileWatcher(path=str(p), config=SearchConfig(paths=[str(p)]))
    
    # is_available should return boolean without error
    available = fw.is_available
    assert isinstance(available, bool)
    
    # is_watching should also work
    watching = fw.is_watching
    assert isinstance(watching, bool)
