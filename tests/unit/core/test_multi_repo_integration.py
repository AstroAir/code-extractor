"""Tests for pysearch.core.integrations.multi_repo_integration module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pysearch.core.config import SearchConfig
from pysearch.core.integrations.multi_repo_integration import MultiRepoIntegrationManager


class TestMultiRepoIntegrationManager:
    """Tests for MultiRepoIntegrationManager class."""

    def test_init(self):
        cfg = SearchConfig()
        mgr = MultiRepoIntegrationManager(cfg)
        assert mgr.config is cfg
        assert mgr.multi_repo_engine is None
        assert mgr._multi_repo_enabled is False

    def test_is_multi_repo_enabled_default(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        assert mgr.is_multi_repo_enabled() is False

    def test_disable_multi_repo_when_not_enabled(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        mgr.disable_multi_repo()  # should not raise
        assert mgr.is_multi_repo_enabled() is False

    def test_disable_multi_repo_with_engine(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mgr.multi_repo_engine = mock_engine
        mgr._multi_repo_enabled = True
        mgr.disable_multi_repo()
        assert mgr.is_multi_repo_enabled() is False
        assert mgr.multi_repo_engine is None

    def test_add_repository_no_engine(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        result = mgr.add_repository("repo1", "/path/to/repo")
        assert result is False

    def test_add_repository_with_engine(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mock_engine.add_repository.return_value = True
        mgr.multi_repo_engine = mock_engine
        result = mgr.add_repository("repo1", "/path/to/repo")
        assert result is True

    def test_add_repository_exception(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mock_engine.add_repository.side_effect = RuntimeError("fail")
        mgr.multi_repo_engine = mock_engine
        result = mgr.add_repository("repo1", "/path")
        assert result is False

    def test_remove_repository_no_engine(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        assert mgr.remove_repository("repo1") is False

    def test_remove_repository_with_engine(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mock_engine.remove_repository.return_value = True
        mgr.multi_repo_engine = mock_engine
        assert mgr.remove_repository("repo1") is True

    def test_remove_repository_exception(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mock_engine.remove_repository.side_effect = RuntimeError("fail")
        mgr.multi_repo_engine = mock_engine
        assert mgr.remove_repository("repo1") is False

    def test_list_repositories_no_engine(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        assert mgr.list_repositories() == []

    def test_list_repositories_with_engine(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mock_engine.list_repositories.return_value = [{"name": "repo1"}, {"name": "repo2"}]
        mgr.multi_repo_engine = mock_engine
        repos = mgr.list_repositories()
        assert len(repos) == 2

    def test_list_repositories_exception(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mock_engine.list_repositories.side_effect = RuntimeError("fail")
        mgr.multi_repo_engine = mock_engine
        assert mgr.list_repositories() == []

    def test_search_repositories_no_engine(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        assert mgr.search_repositories("query") is None

    def test_search_repositories_with_engine(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mock_engine.search_repositories.return_value = {"results": []}
        mgr.multi_repo_engine = mock_engine
        result = mgr.search_repositories("test")
        assert result is not None

    def test_search_repositories_exception(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mock_engine.search_repositories.side_effect = RuntimeError("fail")
        mgr.multi_repo_engine = mock_engine
        assert mgr.search_repositories("test") is None

    def test_search_repositories_specific_repos(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mock_engine.search_repositories.return_value = {"results": []}
        mgr.multi_repo_engine = mock_engine
        result = mgr.search_repositories("test", repositories=["repo1"])
        mock_engine.search_repositories.assert_called_once_with(
            repositories=["repo1"], query="test",
        )

    def test_get_repository_stats_no_engine(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        assert mgr.get_repository_stats() == {}

    def test_get_repository_stats_with_engine(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mock_engine.get_search_statistics.return_value = {"total_repos": 3}
        mgr.multi_repo_engine = mock_engine
        stats = mgr.get_repository_stats()
        assert stats["total_repos"] == 3

    def test_get_repository_stats_exception(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mock_engine.get_search_statistics.side_effect = RuntimeError("fail")
        mgr.multi_repo_engine = mock_engine
        assert mgr.get_repository_stats() == {}

    def test_sync_repositories_no_engine(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        assert mgr.sync_repositories() == {}

    def test_sync_repositories_with_engine(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mock_repo = MagicMock()
        mock_repo.health_status = "healthy"
        mock_engine.list_repositories.return_value = ["repo1"]
        mock_engine.get_repository_info.return_value = mock_repo
        mgr.multi_repo_engine = mock_engine
        result = mgr.sync_repositories()
        assert result["repo1"] is True

    def test_get_repository_health_no_engine(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        assert mgr.get_repository_health() == {}

    def test_get_repository_health_with_engine(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mock_engine.get_health_status.return_value = {
            "repo1": {"status": "healthy"},
        }
        mgr.multi_repo_engine = mock_engine
        health = mgr.get_repository_health()
        assert health["repo1"]["status"] == "healthy"

    def test_get_repository_health_exception(self):
        mgr = MultiRepoIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mock_engine.get_health_status.side_effect = RuntimeError("fail")
        mgr.multi_repo_engine = mock_engine
        assert mgr.get_repository_health() == {}
