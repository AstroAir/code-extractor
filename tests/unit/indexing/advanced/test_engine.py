"""Tests for pysearch.indexing.advanced.engine module."""

from __future__ import annotations

from pathlib import Path

from pysearch.core.config import SearchConfig
from pysearch.indexing.advanced.engine import IndexingEngine


def _make_engine(tmp_path: Path) -> IndexingEngine:
    cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
    return IndexingEngine(cfg)


class TestIndexingEngine:
    """Tests for IndexingEngine class."""

    # -- init --
    def test_init(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        assert engine._paused is False
        assert engine.is_cancelled is False
        assert engine.coordinator is not None
        assert engine.error_collector is not None

    # -- pause / resume --
    def test_pause(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        engine.pause()
        assert engine.is_paused is True

    def test_resume(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        engine.pause()
        engine.resume()
        assert engine.is_paused is False

    def test_pause_resume_cycle(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        assert engine.is_paused is False
        engine.pause()
        assert engine.is_paused is True
        engine.resume()
        assert engine.is_paused is False

    # -- cancel --
    def test_cancel(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        assert engine.is_cancelled is False
        engine.cancel()
        assert engine.is_cancelled is True

    # -- is_paused / is_cancelled properties --
    def test_is_paused_property(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        assert engine.is_paused is False
        engine._paused = True
        assert engine.is_paused is True

    def test_is_cancelled_property(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        assert engine.is_cancelled is False
        engine._cancel_event.set()
        assert engine.is_cancelled is True

    # -- initialize --
    async def test_initialize(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        # initialize loads default indexes; they may not be available but should not error
        await engine.initialize()
        assert engine.coordinator is not None

    # -- refresh_index --
    async def test_refresh_index_empty_dir(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        updates = []
        async for update in engine.refresh_index():
            updates.append(update)
        assert isinstance(updates, list)
        assert len(updates) > 0

    async def test_refresh_index_with_files(self, tmp_path: Path):
        (tmp_path / "test.py").write_text("x = 1\n", encoding="utf-8")
        engine = _make_engine(tmp_path)
        updates = []
        async for update in engine.refresh_index():
            updates.append(update)
        assert isinstance(updates, list)

    async def test_refresh_index_explicit_params(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        updates = []
        async for update in engine.refresh_index(
            directories=[str(tmp_path)], branch="test-branch", repo_name="test-repo"
        ):
            updates.append(update)
        assert isinstance(updates, list)

    # -- refresh_file --
    async def test_refresh_file(self, tmp_path: Path):
        f = tmp_path / "single.py"
        f.write_text("y = 2\n", encoding="utf-8")
        engine = _make_engine(tmp_path)
        # Should complete without error
        await engine.refresh_file(str(f), str(tmp_path), branch="main")

    async def test_refresh_file_nonexistent(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        # Should handle gracefully without raising
        await engine.refresh_file("/no/such/file.py", str(tmp_path))

    # -- get_index_stats --
    async def test_get_index_stats(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        stats = await engine.get_index_stats()
        assert isinstance(stats, dict)
        assert "total_indexes" in stats
        assert "index_types" in stats
        assert "cache_dir" in stats
        assert "errors" in stats
