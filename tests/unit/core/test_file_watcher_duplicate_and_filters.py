from __future__ import annotations

from pathlib import Path

from pysearch import SearchConfig
from pysearch.utils.file_watcher import PySearchEventHandler


def test_duplicate_suppression_and_filters(tmp_path: Path) -> None:
    cfg = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], exclude=["**/ignored/**"])

    class DummyCP:
        def process_batch(self, events):
            # No-op
            return None

    handler = PySearchEventHandler(config=cfg, change_processor=DummyCP())

    class E:
        def __init__(self, p):
            self.src_path = str(p)
            self.is_directory = False

    f_ok = tmp_path / "x.py"
    f_ok.write_text("pass\n", encoding="utf-8")

    # duplicate suppression
    e1 = E(f_ok)
    handler.on_modified(e1)
    # Send immediately again; should be suppressed by duplicate window
    handler.on_modified(e1)

    # excluded path
    ignored_dir = tmp_path / "ignored"
    ignored_dir.mkdir()
    e_ex = E(ignored_dir / "y.py")
    handler.on_created(e_ex)

    # We just ensure methods run without raising under these branches

