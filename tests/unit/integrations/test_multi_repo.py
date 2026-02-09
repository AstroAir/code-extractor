"""Tests for pysearch.integrations.multi_repo module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pysearch.core.config import SearchConfig
from pysearch.core.types import Query, SearchResult, SearchStats
from pysearch.integrations.multi_repo import (
    MultiRepoSearchEngine,
    MultiRepoSearchResult,
    RepositoryInfo,
    RepositoryManager,
    SearchCoordinator,
)

# ---------------------------------------------------------------------------
# RepositoryInfo
# ---------------------------------------------------------------------------


class TestRepositoryInfo:
    """Tests for RepositoryInfo dataclass."""

    def test_creation(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        info = RepositoryInfo(name="repo1", path=tmp_path, config=cfg)
        assert info.name == "repo1"
        assert info.path == tmp_path

    def test_defaults(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        info = RepositoryInfo(name="r", path=tmp_path, config=cfg)
        assert info.priority == "normal"
        assert info.enabled is True
        assert info.last_updated > 0
        assert info.metadata == {}
        assert info.git_remote == ""
        assert info.git_branch == ""
        assert info.git_commit == ""

    def test_missing_path_sets_error(self, tmp_path: Path):
        missing = tmp_path / "nonexistent"
        cfg = SearchConfig(paths=[str(missing)])
        info = RepositoryInfo(name="bad", path=missing, config=cfg)
        assert info.health_status == "error"

    def test_no_git_dir_sets_warning(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        info = RepositoryInfo(name="nogit", path=tmp_path, config=cfg)
        assert info.health_status == "warning"

    def test_refresh_status(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        info = RepositoryInfo(name="r", path=tmp_path, config=cfg)
        old_ts = info.last_updated
        info.refresh_status()
        assert info.last_updated >= old_ts


# ---------------------------------------------------------------------------
# MultiRepoSearchResult
# ---------------------------------------------------------------------------


class TestMultiRepoSearchResult:
    """Tests for MultiRepoSearchResult dataclass."""

    def test_total_matches(self):
        r1 = SearchResult(
            stats=SearchStats(files_scanned=1, files_matched=1, items=3, elapsed_ms=1.0),
        )
        r2 = SearchResult(
            stats=SearchStats(files_scanned=2, files_matched=1, items=5, elapsed_ms=2.0),
        )
        result = MultiRepoSearchResult(repository_results={"a": r1, "b": r2})
        assert result.total_matches == 8

    def test_total_matches_empty(self):
        result = MultiRepoSearchResult(repository_results={})
        assert result.total_matches == 0

    def test_success_rate(self):
        result = MultiRepoSearchResult(
            repository_results={"a": MagicMock()},
            total_repositories=4,
            successful_repositories=3,
        )
        assert result.success_rate == 0.75

    def test_success_rate_zero_repos(self):
        result = MultiRepoSearchResult(
            repository_results={},
            total_repositories=0,
            successful_repositories=0,
        )
        assert result.success_rate == 0.0

    def test_failed_repositories_default(self):
        result = MultiRepoSearchResult(repository_results={})
        assert result.failed_repositories == []


# ---------------------------------------------------------------------------
# RepositoryManager
# ---------------------------------------------------------------------------


class TestRepositoryManager:
    """Tests for RepositoryManager class."""

    def test_init(self):
        mgr = RepositoryManager()
        assert mgr.repositories == {}

    def test_add_repository(self, tmp_path: Path):
        mgr = RepositoryManager()
        assert mgr.add_repository("repo1", tmp_path) is True
        assert "repo1" in mgr.repositories

    def test_add_duplicate_repository(self, tmp_path: Path):
        mgr = RepositoryManager()
        mgr.add_repository("repo1", tmp_path)
        assert mgr.add_repository("repo1", tmp_path) is False

    def test_add_nonexistent_path(self, tmp_path: Path):
        mgr = RepositoryManager()
        assert mgr.add_repository("bad", tmp_path / "nonexistent") is False

    def test_add_with_config(self, tmp_path: Path):
        mgr = RepositoryManager()
        cfg = SearchConfig(paths=[str(tmp_path)])
        assert mgr.add_repository("r", tmp_path, config=cfg) is True
        assert mgr.get_repository("r").config is cfg

    def test_add_with_priority_and_metadata(self, tmp_path: Path):
        mgr = RepositoryManager()
        mgr.add_repository("r", tmp_path, priority="high", team="backend")
        repo = mgr.get_repository("r")
        assert repo.priority == "high"
        assert repo.metadata["team"] == "backend"

    def test_remove_repository(self, tmp_path: Path):
        mgr = RepositoryManager()
        mgr.add_repository("repo1", tmp_path)
        assert mgr.remove_repository("repo1") is True
        assert mgr.get_repository("repo1") is None

    def test_remove_nonexistent(self):
        mgr = RepositoryManager()
        assert mgr.remove_repository("nonexistent") is False

    def test_get_repository(self, tmp_path: Path):
        mgr = RepositoryManager()
        mgr.add_repository("r", tmp_path)
        assert mgr.get_repository("r") is not None
        assert mgr.get_repository("missing") is None

    def test_list_repositories(self, tmp_path: Path):
        mgr = RepositoryManager()
        mgr.add_repository("repo1", tmp_path)
        repos = mgr.list_repositories()
        assert "repo1" in repos

    def test_get_enabled_repositories(self, tmp_path: Path):
        mgr = RepositoryManager()
        mgr.add_repository("r1", tmp_path)
        enabled = mgr.get_enabled_repositories()
        assert "r1" in enabled

        mgr.repositories["r1"].enabled = False
        enabled = mgr.get_enabled_repositories()
        assert "r1" not in enabled

    def test_configure_repository(self, tmp_path: Path):
        mgr = RepositoryManager()
        mgr.add_repository("r", tmp_path)
        assert mgr.configure_repository("r", priority="high") is True
        assert mgr.get_repository("r").priority == "high"

    def test_configure_repository_custom_metadata(self, tmp_path: Path):
        mgr = RepositoryManager()
        mgr.add_repository("r", tmp_path)
        mgr.configure_repository("r", custom_key="value")
        assert mgr.get_repository("r").metadata["custom_key"] == "value"

    def test_configure_nonexistent(self):
        mgr = RepositoryManager()
        assert mgr.configure_repository("missing") is False

    def test_get_health_summary(self, tmp_path: Path):
        mgr = RepositoryManager()
        p1 = tmp_path / "a"
        p1.mkdir()
        p2 = tmp_path / "b"
        p2.mkdir()
        mgr.add_repository("a", p1)
        mgr.add_repository("b", p2)
        summary = mgr.get_health_summary()
        assert summary["total"] == 2
        assert "healthy" in summary
        assert "warning" in summary
        assert "error" in summary
        assert summary["enabled"] == 2
        assert summary["disabled"] == 0

    def test_refresh_all_status(self, tmp_path: Path):
        mgr = RepositoryManager()
        mgr.add_repository("r", tmp_path)
        mgr.refresh_all_status()
        # Should not raise


# ---------------------------------------------------------------------------
# SearchCoordinator
# ---------------------------------------------------------------------------


class TestSearchCoordinator:
    """Tests for SearchCoordinator class."""

    def test_init(self):
        coord = SearchCoordinator(max_workers=2)
        assert coord.max_workers == 2

    def test_search_empty_repositories(self):
        coord = SearchCoordinator()
        query = Query(pattern="test")
        result = coord.search_repositories({}, query)
        assert result.total_repositories == 0
        assert result.repository_results == {}

    def test_aggregate_results_empty(self):
        coord = SearchCoordinator()
        agg = coord.aggregate_results({})
        assert agg.items == []
        assert agg.stats.items == 0

    def test_aggregate_results_limits(self):
        items = [MagicMock(file=Path(f"f{i}.py"), start_line=i) for i in range(10)]
        r = SearchResult(
            items=items,
            stats=SearchStats(files_scanned=10, files_matched=10, items=10, elapsed_ms=5.0),
        )
        coord = SearchCoordinator()
        agg = coord.aggregate_results({"repo": r}, max_results=3)
        assert len(agg.items) == 3


# ---------------------------------------------------------------------------
# MultiRepoSearchEngine
# ---------------------------------------------------------------------------


class TestMultiRepoSearchEngine:
    """Tests for MultiRepoSearchEngine class."""

    def test_init(self):
        engine = MultiRepoSearchEngine()
        assert engine.total_searches == 0
        assert engine.search_history == []

    def test_init_custom_workers(self):
        engine = MultiRepoSearchEngine(max_workers=2)
        assert engine.search_coordinator.max_workers == 2

    def test_add_repository(self, tmp_path: Path):
        engine = MultiRepoSearchEngine()
        assert engine.add_repository("r", tmp_path) is True
        assert "r" in engine.list_repositories()

    def test_remove_repository(self, tmp_path: Path):
        engine = MultiRepoSearchEngine()
        engine.add_repository("r", tmp_path)
        assert engine.remove_repository("r") is True
        assert "r" not in engine.list_repositories()

    def test_configure_repository(self, tmp_path: Path):
        engine = MultiRepoSearchEngine()
        engine.add_repository("r", tmp_path)
        assert engine.configure_repository("r", priority="high") is True

    def test_get_repository_info(self, tmp_path: Path):
        engine = MultiRepoSearchEngine()
        engine.add_repository("r", tmp_path)
        info = engine.get_repository_info("r")
        assert info is not None
        assert info.name == "r"
        assert engine.get_repository_info("missing") is None

    def test_search_all_empty(self):
        engine = MultiRepoSearchEngine()
        results = engine.search_all("test")
        assert results.total_repositories == 0

    def test_search_repositories_no_pattern_no_query(self):
        engine = MultiRepoSearchEngine()
        with pytest.raises(ValueError, match="Either query or pattern"):
            engine.search_repositories()

    def test_search_repositories_with_pattern(self, tmp_path: Path):
        engine = MultiRepoSearchEngine()
        p = tmp_path / "r"
        p.mkdir()
        (p / "a.py").write_text("x = 1\n", encoding="utf-8")
        engine.add_repository("r", p, config=SearchConfig(paths=[str(p)], include=["**/*.py"]))
        result = engine.search_repositories(pattern="x")
        assert result.total_repositories >= 1

    def test_search_repositories_disabled_repo(self, tmp_path: Path):
        engine = MultiRepoSearchEngine()
        engine.add_repository("r", tmp_path)
        engine.repository_manager.get_repository("r").enabled = False
        result = engine.search_repositories(repositories=["r"], pattern="test")
        assert result.total_repositories == 0

    def test_get_health_status(self, tmp_path: Path):
        engine = MultiRepoSearchEngine()
        engine.add_repository("r", tmp_path)
        health = engine.get_health_status()
        assert "total" in health
        assert "total_searches" in health
        assert "average_search_time" in health

    def test_get_search_statistics_empty(self):
        engine = MultiRepoSearchEngine()
        stats = engine.get_search_statistics()
        assert stats["total_searches"] == 0
        assert stats["average_search_time"] == 0.0

    def test_search_history_tracking(self, tmp_path: Path):
        engine = MultiRepoSearchEngine()
        p = tmp_path / "r"
        p.mkdir()
        (p / "a.py").write_text("hello\n", encoding="utf-8")
        engine.add_repository("r", p, config=SearchConfig(paths=[str(p)], include=["**/*.py"]))
        engine.search_all("hello")
        engine.search_all("world")
        assert engine.total_searches == 2
        assert len(engine.search_history) == 2
        stats = engine.get_search_statistics()
        assert stats["total_searches"] == 2
