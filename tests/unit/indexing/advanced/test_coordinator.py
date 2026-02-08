"""Tests for pysearch.indexing.advanced.coordinator module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pysearch.core.config import SearchConfig
from pysearch.indexing.advanced.coordinator import IndexCoordinator


def _make_coordinator(tmp_path: Path) -> IndexCoordinator:
    cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
    return IndexCoordinator(cfg)


def _make_mock_index(artifact_id: str = "test_idx") -> MagicMock:
    mock = MagicMock()
    mock.artifact_id = artifact_id
    mock.relative_expected_time = 1.0
    return mock


class TestIndexCoordinator:
    """Tests for IndexCoordinator class."""

    # -- init --
    def test_init(self, tmp_path: Path):
        coord = _make_coordinator(tmp_path)
        assert coord.indexes == []
        assert coord.lock is not None
        assert coord.error_collector is not None

    # -- add_index --
    def test_add_index(self, tmp_path: Path):
        coord = _make_coordinator(tmp_path)
        coord.add_index(_make_mock_index("idx_a"))
        assert len(coord.indexes) == 1

    def test_add_multiple_indexes(self, tmp_path: Path):
        coord = _make_coordinator(tmp_path)
        for i in range(3):
            coord.add_index(_make_mock_index(f"idx_{i}"))
        assert len(coord.indexes) == 3

    # -- remove_index --
    def test_remove_index_existing(self, tmp_path: Path):
        coord = _make_coordinator(tmp_path)
        coord.add_index(_make_mock_index("a"))
        coord.add_index(_make_mock_index("b"))
        assert coord.remove_index("a") is True
        assert len(coord.indexes) == 1
        assert coord.indexes[0].artifact_id == "b"

    def test_remove_index_nonexistent(self, tmp_path: Path):
        coord = _make_coordinator(tmp_path)
        coord.add_index(_make_mock_index("a"))
        assert coord.remove_index("nonexistent") is False
        assert len(coord.indexes) == 1

    def test_remove_from_empty(self, tmp_path: Path):
        coord = _make_coordinator(tmp_path)
        assert coord.remove_index("anything") is False

    # -- get_index --
    def test_get_index_found(self, tmp_path: Path):
        coord = _make_coordinator(tmp_path)
        idx = _make_mock_index("target")
        coord.add_index(_make_mock_index("other"))
        coord.add_index(idx)
        assert coord.get_index("target") is idx

    def test_get_index_not_found(self, tmp_path: Path):
        coord = _make_coordinator(tmp_path)
        coord.add_index(_make_mock_index("a"))
        assert coord.get_index("missing") is None

    def test_get_index_empty(self, tmp_path: Path):
        coord = _make_coordinator(tmp_path)
        assert coord.get_index("x") is None

    # -- build_content_addresses --
    async def test_build_content_addresses(self, tmp_path: Path):
        coord = _make_coordinator(tmp_path)
        f = tmp_path / "test.py"
        f.write_text("x = 1\n", encoding="utf-8")
        addresses = await coord.build_content_addresses([str(f)])
        assert isinstance(addresses, dict)
        assert str(f) in addresses

    async def test_build_content_addresses_nonexistent_file(self, tmp_path: Path):
        coord = _make_coordinator(tmp_path)
        addresses = await coord.build_content_addresses(["/no/such/file.py"])
        assert isinstance(addresses, dict)
        # Should skip files that fail
        assert "/no/such/file.py" not in addresses

    async def test_build_content_addresses_empty_list(self, tmp_path: Path):
        coord = _make_coordinator(tmp_path)
        addresses = await coord.build_content_addresses([])
        assert addresses == {}

    # -- cleanup_cache --
    async def test_cleanup_cache(self, tmp_path: Path):
        coord = _make_coordinator(tmp_path)
        coord.global_cache = MagicMock()
        coord.global_cache.cleanup_orphaned_content = AsyncMock(return_value=5)
        removed = await coord.cleanup_cache()
        assert removed == 5
        coord.global_cache.cleanup_orphaned_content.assert_awaited_once()

    # -- refresh_all_indexes with no indexes --
    async def test_refresh_all_indexes_no_indexes(self, tmp_path: Path):
        coord = _make_coordinator(tmp_path)
        from pysearch.analysis.content_addressing import IndexTag

        tag = IndexTag(directory=str(tmp_path), branch="main", artifact_id="*")
        updates = []
        async for update in coord.refresh_all_indexes(tag, {}, lambda p: "", None):
            updates.append(update)
        assert len(updates) == 1
        assert updates[0].status == "done"
        assert updates[0].progress == 1.0
