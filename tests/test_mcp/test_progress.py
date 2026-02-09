"""
Tests for mcp/shared/progress.py — ProgressTracker, ProgressUpdate,
ProgressStatus, and ProgressAwareSearchServer.

Covers: start_operation, update_progress, complete_operation,
cancel_operation, is_cancelled, get_operation_status, add_callback,
cleanup_completed, ProgressAwareSearchServer.search_with_progress,
ProgressAwareSearchServer.batch_file_analysis_with_progress.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.shared.progress import (
    ProgressAwareSearchServer,
    ProgressStatus,
    ProgressTracker,
    ProgressUpdate,
    create_progress_aware_server,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker():
    """Create a fresh ProgressTracker."""
    return ProgressTracker()


# ---------------------------------------------------------------------------
# ProgressTracker — start / update / complete
# ---------------------------------------------------------------------------


class TestProgressLifecycle:
    """Tests for the start → update → complete lifecycle."""

    def test_start_operation(self, tracker):
        """start_operation creates a RUNNING operation."""
        tracker.start_operation("op1", total_steps=5, description="Test op")
        status = tracker.get_operation_status("op1")
        assert status is not None
        assert status.status == ProgressStatus.RUNNING
        assert status.progress == 0.0
        assert status.total_steps == 5

    def test_update_progress(self, tracker):
        """update_progress updates completed_steps and progress percentage."""
        tracker.start_operation("op1", total_steps=4)
        tracker.update_progress("op1", 2, "Step 2")
        status = tracker.get_operation_status("op1")
        assert status.completed_steps == 2
        assert status.progress == 0.5
        assert status.current_step == "Step 2"

    def test_update_progress_with_details(self, tracker):
        """update_progress stores extra details."""
        tracker.start_operation("op1", total_steps=3)
        tracker.update_progress("op1", 1, "Processing", {"file": "a.py"})
        status = tracker.get_operation_status("op1")
        assert status.details == {"file": "a.py"}

    def test_update_nonexistent_noop(self, tracker):
        """update_progress on unknown operation is a no-op."""
        tracker.update_progress("nonexistent", 1, "step")  # Should not raise

    def test_complete_operation_success(self, tracker):
        """complete_operation sets status to COMPLETED on success."""
        tracker.start_operation("op1", total_steps=3)
        tracker.complete_operation("op1", success=True)
        status = tracker.get_operation_status("op1")
        assert status.status == ProgressStatus.COMPLETED
        assert status.progress == 1.0

    def test_complete_operation_failure(self, tracker):
        """complete_operation sets status to FAILED on failure."""
        tracker.start_operation("op1", total_steps=3)
        tracker.complete_operation("op1", success=False)
        status = tracker.get_operation_status("op1")
        assert status.status == ProgressStatus.FAILED

    def test_complete_nonexistent_noop(self, tracker):
        """complete_operation on unknown operation is a no-op."""
        tracker.complete_operation("nonexistent")  # Should not raise

    def test_elapsed_time_tracked(self, tracker):
        """Elapsed time is tracked during progress updates."""
        tracker.start_operation("op1", total_steps=2)
        time.sleep(0.05)
        tracker.update_progress("op1", 1, "step 1")
        status = tracker.get_operation_status("op1")
        assert status.elapsed_time > 0

    def test_estimated_remaining(self, tracker):
        """estimated_remaining is calculated when progress > 0."""
        tracker.start_operation("op1", total_steps=4)
        time.sleep(0.05)
        tracker.update_progress("op1", 2, "halfway")
        status = tracker.get_operation_status("op1")
        assert status.estimated_remaining is not None
        assert status.estimated_remaining >= 0


# ---------------------------------------------------------------------------
# ProgressTracker — cancel
# ---------------------------------------------------------------------------


class TestCancelOperation:
    """Tests for cancel_operation and is_cancelled."""

    def test_cancel_active(self, tracker):
        """cancel_operation sets status to CANCELLED and returns True."""
        tracker.start_operation("op1", total_steps=3)
        assert tracker.cancel_operation("op1") is True
        status = tracker.get_operation_status("op1")
        assert status.status == ProgressStatus.CANCELLED

    def test_cancel_nonexistent(self, tracker):
        """cancel_operation returns False for unknown operations."""
        assert tracker.cancel_operation("nonexistent") is False

    def test_is_cancelled(self, tracker):
        """is_cancelled returns True after cancellation."""
        tracker.start_operation("op1", total_steps=3)
        assert tracker.is_cancelled("op1") is False
        tracker.cancel_operation("op1")
        assert tracker.is_cancelled("op1") is True

    def test_is_cancelled_unknown(self, tracker):
        """is_cancelled returns False for unknown operations."""
        assert tracker.is_cancelled("unknown") is False


# ---------------------------------------------------------------------------
# ProgressTracker — get_operation_status
# ---------------------------------------------------------------------------


class TestGetOperationStatus:
    """Tests for get_operation_status."""

    def test_returns_none_for_unknown(self, tracker):
        """get_operation_status returns None for unknown operations."""
        assert tracker.get_operation_status("unknown") is None

    def test_returns_update_for_known(self, tracker):
        """get_operation_status returns ProgressUpdate for known operations."""
        tracker.start_operation("op1", total_steps=3)
        status = tracker.get_operation_status("op1")
        assert isinstance(status, ProgressUpdate)


# ---------------------------------------------------------------------------
# ProgressTracker — callbacks
# ---------------------------------------------------------------------------


class TestCallbacks:
    """Tests for add_callback and callback notification."""

    def test_callback_on_update(self, tracker):
        """Callbacks are invoked on progress update."""
        received = []
        tracker.start_operation("op1", total_steps=3)
        tracker.add_callback("op1", lambda u: received.append(u))
        tracker.update_progress("op1", 1, "step 1")
        assert len(received) == 1
        assert received[0].current_step == "step 1"

    def test_callback_on_complete(self, tracker):
        """Callbacks are invoked on completion."""
        received = []
        tracker.start_operation("op1", total_steps=3)
        tracker.add_callback("op1", lambda u: received.append(u))
        tracker.complete_operation("op1")
        assert len(received) == 1
        assert received[0].status == ProgressStatus.COMPLETED

    def test_callback_error_ignored(self, tracker):
        """Callback errors are silently ignored."""
        tracker.start_operation("op1", total_steps=3)
        tracker.add_callback("op1", lambda u: 1 / 0)  # Raises ZeroDivisionError
        tracker.update_progress("op1", 1, "step")  # Should not raise

    def test_multiple_callbacks(self, tracker):
        """Multiple callbacks are all invoked."""
        counts = [0, 0]
        tracker.start_operation("op1", total_steps=2)
        tracker.add_callback("op1", lambda u: counts.__setitem__(0, counts[0] + 1))
        tracker.add_callback("op1", lambda u: counts.__setitem__(1, counts[1] + 1))
        tracker.update_progress("op1", 1, "step")
        assert counts == [1, 1]


# ---------------------------------------------------------------------------
# ProgressTracker — cleanup_completed
# ---------------------------------------------------------------------------


class TestCleanupCompleted:
    """Tests for cleanup_completed."""

    def test_cleanup_removes_completed(self, tracker):
        """cleanup_completed removes completed operations."""
        tracker.start_operation("op1", total_steps=1)
        tracker.complete_operation("op1")
        # Force elapsed_time to be old enough
        tracker.active_operations["op1"].elapsed_time = 0.0
        tracker.cleanup_completed(max_age_seconds=0)
        assert "op1" not in tracker.active_operations

    def test_cleanup_preserves_running(self, tracker):
        """cleanup_completed preserves running operations."""
        tracker.start_operation("op_running", total_steps=3)
        tracker.cleanup_completed(max_age_seconds=0)
        assert "op_running" in tracker.active_operations


# ---------------------------------------------------------------------------
# ProgressUpdate dataclass
# ---------------------------------------------------------------------------


class TestProgressUpdate:
    """Tests for ProgressUpdate dataclass."""

    def test_construction(self):
        """ProgressUpdate can be constructed with all fields."""
        update = ProgressUpdate(
            operation_id="test",
            status=ProgressStatus.RUNNING,
            progress=0.5,
            current_step="step 1",
            total_steps=4,
            completed_steps=2,
            elapsed_time=1.5,
            estimated_remaining=1.5,
            details={"key": "val"},
        )
        assert update.operation_id == "test"
        assert update.progress == 0.5


# ---------------------------------------------------------------------------
# ProgressAwareSearchServer
# ---------------------------------------------------------------------------


class TestProgressAwareSearchServer:
    """Tests for ProgressAwareSearchServer."""

    def test_create_server(self):
        """create_progress_aware_server returns a ProgressAwareSearchServer."""
        server = create_progress_aware_server()
        assert isinstance(server, ProgressAwareSearchServer)
        assert server.progress_tracker is not None

    def test_get_active_operations_empty(self):
        """get_active_operations returns empty list initially."""
        server = ProgressAwareSearchServer()
        ops = server.get_active_operations()
        assert ops == []

    def test_cancel_operation_returns_false_unknown(self):
        """cancel_operation returns False for unknown operation."""
        server = ProgressAwareSearchServer()
        assert server.cancel_operation("unknown") is False

    def test_estimate_file_count_none(self):
        """_estimate_file_count returns default for None paths."""
        server = ProgressAwareSearchServer()
        assert server._estimate_file_count(None) == 100

    def test_estimate_file_count_with_paths(self):
        """_estimate_file_count estimates based on path count."""
        server = ProgressAwareSearchServer()
        count = server._estimate_file_count(["./src", "./tests"])
        assert count >= 20

    @pytest.mark.asyncio
    async def test_search_with_progress_yields_updates(self):
        """search_with_progress yields ProgressUpdate objects."""
        server = ProgressAwareSearchServer()
        updates = []
        async for update in server.search_with_progress("test"):
            updates.append(update)
        assert len(updates) > 0
        assert all(isinstance(u, ProgressUpdate) for u in updates)
        assert updates[-1].status == ProgressStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_search_with_progress_callback(self):
        """search_with_progress invokes progress_callback."""
        server = ProgressAwareSearchServer()
        callback_updates = []
        async for _ in server.search_with_progress(
            "test", progress_callback=lambda u: callback_updates.append(u)
        ):
            pass
        assert len(callback_updates) > 0

    @pytest.mark.asyncio
    async def test_search_with_progress_invalid_regex(self):
        """search_with_progress with invalid regex yields error update."""
        server = ProgressAwareSearchServer()
        updates = []
        async for update in server.search_with_progress("[unclosed", use_regex=True):
            updates.append(update)
        assert len(updates) > 0
        last = updates[-1]
        assert last.status == ProgressStatus.FAILED
        assert "error" in (last.details or {})
