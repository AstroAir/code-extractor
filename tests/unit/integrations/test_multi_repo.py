"""Tests for pysearch.integrations.multi_repo module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pysearch.core.config import SearchConfig
from pysearch.integrations.multi_repo import (
    MultiRepoSearchEngine,
    RepositoryInfo,
    RepositoryManager,
)


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
        assert info.last_updated > 0  # __post_init__ sets to time.time()
        assert info.metadata == {}


class TestRepositoryManager:
    """Tests for RepositoryManager class."""

    def test_init(self):
        mgr = RepositoryManager()
        assert mgr is not None

    def test_add_repository(self, tmp_path: Path):
        mgr = RepositoryManager()
        result = mgr.add_repository("repo1", tmp_path)
        assert result is True

    def test_add_duplicate_repository(self, tmp_path: Path):
        mgr = RepositoryManager()
        mgr.add_repository("repo1", tmp_path)
        result = mgr.add_repository("repo1", tmp_path)
        assert result is False

    def test_remove_repository(self, tmp_path: Path):
        mgr = RepositoryManager()
        mgr.add_repository("repo1", tmp_path)
        result = mgr.remove_repository("repo1")
        assert result is True

    def test_remove_nonexistent(self):
        mgr = RepositoryManager()
        assert mgr.remove_repository("nonexistent") is False

    def test_list_repositories(self, tmp_path: Path):
        mgr = RepositoryManager()
        mgr.add_repository("repo1", tmp_path)
        repos = mgr.list_repositories()
        assert "repo1" in repos


class TestMultiRepoSearchEngine:
    """Tests for MultiRepoSearchEngine class."""

    def test_init(self):
        engine = MultiRepoSearchEngine()
        assert engine is not None

    def test_add_repository(self, tmp_path: Path):
        engine = MultiRepoSearchEngine()
        engine.add_repository("repo1", str(tmp_path))
        repos = engine.list_repositories()
        assert len(repos) >= 1

    def test_search_all_empty(self):
        engine = MultiRepoSearchEngine()
        results = engine.search_all("test")
        assert results is not None
        assert results.total_repositories == 0
