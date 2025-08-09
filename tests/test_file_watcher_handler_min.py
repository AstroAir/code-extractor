from __future__ import annotations

from pysearch.config import SearchConfig
from pysearch.file_watcher import FileWatcher, PySearchEventHandler


def test_event_handler_methods(tmp_path):
    cfg = SearchConfig(paths=[str(tmp_path)])
    fw = FileWatcher(path=str(tmp_path), config=cfg)
    # Build a minimal handler with a dummy change processor
    class DummyCP:
        def process_batch(self, events):
            return None
    handler = PySearchEventHandler(config=cfg, change_processor=DummyCP())

    class E:
        def __init__(self, p):
            self.src_path = str(p)
            self.is_directory = False

    e = E(tmp_path / "x.py")
    # Ensure methods don't raise
    handler.on_created(e)
    handler.on_modified(e)
    handler.on_deleted(e)

