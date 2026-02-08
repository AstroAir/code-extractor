"""Tests for pysearch.indexing.advanced.locking module."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from pysearch.indexing.advanced.locking import IndexLock


class TestIndexLock:
    """Tests for IndexLock class."""

    # -- init --
    def test_init(self, tmp_path: Path):
        lock = IndexLock(tmp_path)
        assert lock.lock_file == tmp_path / "indexing.lock"
        assert lock.cache_dir == tmp_path

    # -- acquire / release --
    async def test_acquire_and_release(self, tmp_path: Path):
        lock = IndexLock(tmp_path)
        acquired = await lock.acquire(["/test"])
        assert acquired is True
        assert lock.lock_file.exists()
        await lock.release()
        assert not lock.lock_file.exists()

    async def test_release_without_acquire(self, tmp_path: Path):
        lock = IndexLock(tmp_path)
        await lock.release()  # should not raise

    async def test_acquire_creates_lock_file(self, tmp_path: Path):
        lock = IndexLock(tmp_path)
        assert not lock.lock_file.exists()
        await lock.acquire(["/test"])
        assert lock.lock_file.exists()
        await lock.release()

    # -- lock file content validation --
    async def test_lock_file_content(self, tmp_path: Path):
        lock = IndexLock(tmp_path)
        await lock.acquire(["/project", "/lib"])
        data = json.loads(lock.lock_file.read_text(encoding="utf-8"))
        assert "directories" in data
        assert data["directories"] == ["/project", "/lib"]
        assert "timestamp" in data
        assert isinstance(data["timestamp"], float)
        assert "pid" in data
        assert isinstance(data["pid"], int)
        await lock.release()

    # -- stale lock detection --
    async def test_stale_lock_removed(self, tmp_path: Path):
        lock = IndexLock(tmp_path)
        # Create a stale lock (timestamp > 600 seconds ago)
        stale_data = {
            "directories": ["/old"],
            "timestamp": time.time() - 700,  # 700s ago > 600s threshold
            "pid": 99999,
        }
        lock.lock_file.write_text(json.dumps(stale_data), encoding="utf-8")
        assert lock.lock_file.exists()

        # Acquiring should detect the stale lock, remove it, and succeed
        acquired = await lock.acquire(["/new"], timeout=5.0)
        assert acquired is True
        # New lock should have updated content
        new_data = json.loads(lock.lock_file.read_text(encoding="utf-8"))
        assert new_data["directories"] == ["/new"]
        assert new_data["timestamp"] > stale_data["timestamp"]
        await lock.release()

    async def test_corrupted_lock_removed(self, tmp_path: Path):
        lock = IndexLock(tmp_path)
        # Create corrupted lock file
        lock.lock_file.write_text("not valid json!", encoding="utf-8")

        acquired = await lock.acquire(["/test"], timeout=5.0)
        assert acquired is True
        await lock.release()

    # -- update_timestamp --
    async def test_update_timestamp(self, tmp_path: Path):
        lock = IndexLock(tmp_path)
        await lock.acquire(["/test"])
        original_data = json.loads(lock.lock_file.read_text(encoding="utf-8"))
        original_ts = original_data["timestamp"]

        # Small delay to ensure timestamp difference
        import asyncio
        await asyncio.sleep(0.05)

        await lock.update_timestamp()
        updated_data = json.loads(lock.lock_file.read_text(encoding="utf-8"))
        assert updated_data["timestamp"] >= original_ts
        await lock.release()

    async def test_update_timestamp_no_lock_file(self, tmp_path: Path):
        lock = IndexLock(tmp_path)
        # Should not raise when lock file doesn't exist
        await lock.update_timestamp()

    # -- double release --
    async def test_double_release(self, tmp_path: Path):
        lock = IndexLock(tmp_path)
        await lock.acquire(["/test"])
        await lock.release()
        # Second release should not raise
        await lock.release()
        assert not lock.lock_file.exists()
