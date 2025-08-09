from __future__ import annotations

import time
from pathlib import Path

from pysearch.config import SearchConfig
from pysearch.file_watcher import FileWatcher


def test_file_watcher_start_stop(tmp_path: Path) -> None:
    cfg = SearchConfig(paths=[str(tmp_path)], parallel=False)
    fw = FileWatcher(path=str(tmp_path), config=cfg)
    try:
        fw.start()
        # touch a file to exercise event path
        p = tmp_path / "t.py"
        p.write_text("print('x')\n", encoding="utf-8")
        time.sleep(0.05)
    finally:
        fw.stop()
        # multiple stops should be safe
        fw.stop()

