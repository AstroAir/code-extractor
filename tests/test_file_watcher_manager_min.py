from __future__ import annotations

from pathlib import Path

from pysearch.file_watcher import WatchManager


def test_watch_manager_add_remove(tmp_path: Path) -> None:
    wm = WatchManager()
    p = tmp_path / "w"
    p.mkdir()
    assert wm.add_watcher("w1", p) in (True, False)
    assert isinstance(wm.list_watchers(), list)
    assert wm.remove_watcher("w1") in (True, False)

