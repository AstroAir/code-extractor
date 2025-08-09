from __future__ import annotations

import time
from pathlib import Path

from pysearch.config import SearchConfig
from pysearch.file_watcher import ChangeProcessor, EventType, FileEvent
from pysearch.indexer import Indexer


def test_change_processor_batching_and_debounce(tmp_path: Path) -> None:
    # Create a mock indexer that tracks processed events
    config = SearchConfig(paths=[str(tmp_path)])
    indexer = Indexer(config)

    # Create change processor with small batch size for testing
    cp = ChangeProcessor(
        indexer=indexer,
        batch_size=3,
        debounce_delay=0.1
    )

    # Create file events
    current_time = time.time()
    event1 = FileEvent(path=tmp_path / "a.py", event_type=EventType.CREATED, timestamp=current_time, is_directory=False)
    event2 = FileEvent(path=tmp_path / "b.py", event_type=EventType.MODIFIED, timestamp=current_time + 0.01, is_directory=False)
    event3 = FileEvent(path=tmp_path / "c.py", event_type=EventType.DELETED, timestamp=current_time + 0.02, is_directory=False)

    # Add events one by one
    cp.process_event(event1)
    cp.process_event(event2)

    # Should not process yet (under batch size and within debounce)
    assert cp.events_processed == 0

    # Add third event to trigger batch processing
    cp.process_event(event3)

    # Give it a moment to process
    time.sleep(0.05)

    # Should have processed the batch
    assert cp.events_processed >= 3

    # Test debounce delay
    initial_count = cp.events_processed
    event4 = FileEvent(path=tmp_path / "d.py", event_type=EventType.CREATED, timestamp=time.time(), is_directory=False)
    cp.process_event(event4)

    # Should not process immediately
    assert cp.events_processed == initial_count

    # Wait for debounce
    time.sleep(0.15)

    # Should have processed the event
    assert cp.events_processed > initial_count
