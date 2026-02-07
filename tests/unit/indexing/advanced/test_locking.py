"""Tests for pysearch.indexing.advanced.locking module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.indexing.advanced.locking import IndexLock


class TestIndexLock:
    """Tests for IndexLock class."""

    def test_init(self, tmp_path: Path):
        lock = IndexLock(tmp_path)
        assert lock.lock_file == tmp_path / "indexing.lock"
        assert lock.cache_dir == tmp_path

    @pytest.mark.asyncio
    async def test_acquire_and_release(self, tmp_path: Path):
        lock = IndexLock(tmp_path)
        acquired = await lock.acquire(["/test"])
        assert acquired is True
        assert lock.lock_file.exists()
        await lock.release()
        assert not lock.lock_file.exists()

    @pytest.mark.asyncio
    async def test_release_without_acquire(self, tmp_path: Path):
        lock = IndexLock(tmp_path)
        await lock.release()  # should not raise

    @pytest.mark.asyncio
    async def test_acquire_creates_lock_file(self, tmp_path: Path):
        lock = IndexLock(tmp_path)
        assert not lock.lock_file.exists()
        await lock.acquire(["/test"])
        assert lock.lock_file.exists()
        await lock.release()
