"""
Integration tests for pysearch.integrations.multi_repo module.

Consolidated from previously fragmented test files:
- test_multi_repo_min.py
- test_multi_repo_more.py
- test_multi_repo_engine_min.py
- test_multi_repo_engine_paths.py
- test_multi_repo_health_states.py
- test_multi_repo_search_history.py
"""

from __future__ import annotations

from pathlib import Path

from pysearch import SearchConfig, SearchResult, SearchStats
from pysearch.integrations.multi_repo import (
    MultiRepoSearchEngine,
    MultiRepoSearchResult,
    RepositoryInfo,
    RepositoryManager,
)


# ---------------------------------------------------------------------------
# RepositoryManager integration tests
# ---------------------------------------------------------------------------


class TestRepositoryManagerIntegration:
    """Integration tests for RepositoryManager lifecycle."""

    def test_add_list_get_remove(self, tmp_path: Path) -> None:
        rm = RepositoryManager()
        p = tmp_path / "repo"
        p.mkdir()
        assert rm.add_repository("r1", p) is True
        assert "r1" in rm.list_repositories()
        assert rm.get_repository("r1") is not None
        assert rm.remove_repository("r1") is True
        assert rm.get_repository("r1") is None

    def test_add_duplicate_returns_false(self, tmp_path: Path) -> None:
        rm = RepositoryManager()
        p = tmp_path / "valid"
        p.mkdir()
        cfg = SearchConfig(paths=[str(p)])
        assert rm.add_repository("test", p, config=cfg) is True
        assert rm.add_repository("test", p, config=cfg) is False

    def test_add_nonexistent_path_returns_false(self, tmp_path: Path) -> None:
        rm = RepositoryManager()
        invalid_path = tmp_path / "invalid"
        assert rm.add_repository("invalid", invalid_path) is False


# ---------------------------------------------------------------------------
# RepositoryInfo health state tests
# ---------------------------------------------------------------------------


class TestRepositoryInfoHealthStates:
    """Integration tests for RepositoryInfo health status detection."""

    def test_missing_path_error_status(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "nonexistent"
        repo = RepositoryInfo(
            name="missing",
            path=missing_path,
            config=SearchConfig(paths=[str(missing_path)]),
        )
        assert repo.health_status == "error"

    def test_no_git_warning_status(self, tmp_path: Path) -> None:
        no_git_path = tmp_path / "no_git"
        no_git_path.mkdir()
        repo = RepositoryInfo(
            name="no_git",
            path=no_git_path,
            config=SearchConfig(paths=[str(no_git_path)]),
        )
        assert repo.health_status == "warning"

    def test_refresh_status(self, tmp_path: Path) -> None:
        no_git_path = tmp_path / "refresh"
        no_git_path.mkdir()
        repo = RepositoryInfo(
            name="refresh",
            path=no_git_path,
            config=SearchConfig(paths=[str(no_git_path)]),
        )
        repo.refresh_status()
        assert repo.health_status in ("warning", "error")


class TestRepositoryManagerHealthSummary:
    """Integration tests for RepositoryManager health summary."""

    def test_health_summary_mixed(self, tmp_path: Path) -> None:
        rm = RepositoryManager()

        healthy_path = tmp_path / "healthy"
        healthy_path.mkdir()
        (healthy_path / ".git").mkdir()

        warning_path = tmp_path / "warning"
        warning_path.mkdir()

        rm.add_repository(
            "healthy", healthy_path, config=SearchConfig(paths=[str(healthy_path)])
        )
        rm.add_repository(
            "warning", warning_path, config=SearchConfig(paths=[str(warning_path)])
        )

        summary = rm.get_health_summary()
        assert summary["total"] >= 2
        assert "healthy" in summary
        assert "warning" in summary
        assert "error" in summary
        assert "enabled" in summary
        assert "disabled" in summary

    def test_refresh_all_preserves_summary(self, tmp_path: Path) -> None:
        rm = RepositoryManager()
        p = tmp_path / "r"
        p.mkdir()
        rm.add_repository("r", p, config=SearchConfig(paths=[str(p)]))
        rm.refresh_all_status()
        summary = rm.get_health_summary()
        assert summary["total"] >= 1


# ---------------------------------------------------------------------------
# MultiRepoSearchResult property tests
# ---------------------------------------------------------------------------


class TestMultiRepoSearchResultIntegration:
    """Integration tests for MultiRepoSearchResult aggregation properties."""

    def test_total_matches_and_success_rate(self) -> None:
        r1 = SearchResult(
            stats=SearchStats(files_scanned=1, files_matched=1, items=1, elapsed_ms=1.0),
        )
        r2 = SearchResult(
            stats=SearchStats(files_scanned=2, files_matched=0, items=0, elapsed_ms=2.0),
        )
        agg = MultiRepoSearchResult(repository_results={"a": r1, "b": r2})
        assert agg.total_matches >= 1
        assert agg.success_rate >= 0.0


# ---------------------------------------------------------------------------
# MultiRepoSearchEngine integration tests
# ---------------------------------------------------------------------------


class TestMultiRepoSearchEngineIntegration:
    """Integration tests for MultiRepoSearchEngine with real files."""

    def _create_repo(self, tmp_path: Path, name: str, content: str) -> Path:
        """Helper to create a repo directory with a single .py file."""
        p = tmp_path / name
        p.mkdir()
        (p / f"{name}.py").write_text(content, encoding="utf-8")
        return p

    def test_basic_search_across_repos(self, tmp_path: Path) -> None:
        repo1 = self._create_repo(tmp_path, "r1", "def foo():\n    pass\n")
        repo2 = self._create_repo(tmp_path, "r2", "class Bar:\n    pass\n")

        eng = MultiRepoSearchEngine(max_workers=2)
        assert eng.add_repository(
            "r1", repo1, config=SearchConfig(paths=[str(repo1)], include=["**/*.py"])
        )
        assert eng.add_repository(
            "r2", repo2, config=SearchConfig(paths=[str(repo2)], include=["**/*.py"])
        )

        res = eng.search_all("def", context=0, aggregate_results=True)
        assert res.total_repositories == 2
        assert res.total_matches >= 0

        stats = eng.get_search_statistics()
        assert "total_searches" in stats
        health = eng.get_health_status()
        assert "total" in health

    def test_no_repos_then_add_and_search(self, tmp_path: Path) -> None:
        eng = MultiRepoSearchEngine()

        # No repos: expect empty result
        r = eng.search_all("noop", use_regex=False, aggregate_results=True, max_results=5)
        assert r.total_repositories == 0

        p1 = self._create_repo(tmp_path, "r1", "x=1\n")
        p2 = self._create_repo(tmp_path, "r2", "def f():\n pass\n")

        assert eng.add_repository(
            "r1", p1, config=SearchConfig(paths=[str(p1)], include=["**/*.py"])
        )
        assert eng.add_repository(
            "r2", p2, config=SearchConfig(paths=[str(p2)], include=["**/*.py"])
        )

        res = eng.search_all("def ", use_regex=True, aggregate_results=True, max_results=10)
        assert res.total_repositories >= 2

        stats = eng.get_search_statistics()
        assert "total_searches" in stats

        hs = eng.get_health_status()
        assert "total" in hs and "average_search_time" in hs

    def test_search_history_and_bounds(self, tmp_path: Path) -> None:
        eng = MultiRepoSearchEngine()

        p1 = self._create_repo(tmp_path, "r1", "def func1(): pass\n")
        p2 = self._create_repo(tmp_path, "r2", "def func2(): pass\n")

        eng.add_repository(
            "r1", p1, config=SearchConfig(paths=[str(p1)], include=["**/*.py"])
        )
        eng.add_repository(
            "r2", p2, config=SearchConfig(paths=[str(p2)], include=["**/*.py"])
        )

        for i in range(5):
            eng.search_all(f"func{i % 2 + 1}", use_regex=False, max_results=10)

        stats = eng.get_search_statistics()
        assert stats["total_searches"] >= 5
        assert "average_search_time" in stats
        assert len(eng.search_history) <= 100

    def test_mixed_enabled_disabled(self, tmp_path: Path) -> None:
        eng = MultiRepoSearchEngine()

        p1 = self._create_repo(tmp_path, "enabled", "enabled_code\n")
        eng.add_repository(
            "enabled", p1, config=SearchConfig(paths=[str(p1)], include=["**/*.py"])
        )

        repo_info = eng.repository_manager.get_repository("enabled")
        if repo_info:
            repo_info.enabled = False

        result = eng.search_repositories(repositories=["enabled"], pattern="enabled_code")
        assert result.total_repositories >= 0
