"""
Integration tests for workspace management functionality.

Tests the end-to-end workspace lifecycle including creation, discovery,
persistence, and interaction with the multi-repo search engine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.core.config import SearchConfig
from pysearch.core.workspace import (
    RepositoryConfig,
    WorkspaceConfig,
    WorkspaceManager,
)


class TestWorkspaceDiscoveryIntegration:
    """Integration tests for repository auto-discovery."""

    def _create_repo(
        self,
        base: Path,
        name: str,
        project_type: str = "",
        files: dict[str, str] | None = None,
    ) -> Path:
        """Helper: create a fake Git repo directory."""
        repo = base / name
        repo.mkdir(parents=True, exist_ok=True)
        (repo / ".git").mkdir(exist_ok=True)

        if project_type == "python":
            (repo / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        elif project_type == "node":
            (repo / "package.json").write_text("{}\n", encoding="utf-8")
        elif project_type == "go":
            (repo / "go.mod").write_text("module example\n", encoding="utf-8")
        elif project_type == "rust":
            (repo / "Cargo.toml").write_text("[package]\n", encoding="utf-8")

        if files:
            for fname, content in files.items():
                fpath = repo / fname
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(content, encoding="utf-8")

        return repo

    def test_discover_mixed_project_types(self, tmp_path: Path) -> None:
        self._create_repo(tmp_path, "py-app", "python")
        self._create_repo(tmp_path, "js-app", "node")
        self._create_repo(tmp_path, "go-svc", "go")

        mgr = WorkspaceManager()
        ws = mgr.create_workspace("mixed", str(tmp_path))
        discovered = mgr.discover_repositories(ws)

        assert len(discovered) == 3
        types = {r.project_type for r in discovered}
        assert "python" in types
        assert "node" in types
        assert "go" in types

    def test_discover_nested_repos(self, tmp_path: Path) -> None:
        # Create repos at different depths
        self._create_repo(tmp_path, "level0", "python")
        self._create_repo(tmp_path / "sub", "level1", "node")
        self._create_repo(tmp_path / "sub" / "deep", "level2", "go")

        mgr = WorkspaceManager()
        ws = mgr.create_workspace("nested", str(tmp_path))

        # depth=1 should find level0 only
        d1 = mgr.discover_repositories(ws, max_depth=1, auto_add=False)
        names1 = {r.name for r in d1}
        assert "level0" in names1

        # depth=3 should find all
        d3 = mgr.discover_repositories(ws, max_depth=3, auto_add=False)
        assert len(d3) >= 2  # at least level0 and sub-level1

    def test_discover_skips_hidden_dirs(self, tmp_path: Path) -> None:
        # Hidden dir should be skipped
        hidden_repo = tmp_path / ".hidden" / "repo"
        hidden_repo.mkdir(parents=True)
        (hidden_repo / ".git").mkdir()

        # Normal repo
        self._create_repo(tmp_path, "visible", "python")

        mgr = WorkspaceManager()
        ws = mgr.create_workspace("test", str(tmp_path))
        discovered = mgr.discover_repositories(ws)

        names = {r.name for r in discovered}
        assert "visible" in names
        # Hidden dir should not be traversed
        assert not any("hidden" in r.name for r in discovered)

    def test_discover_includes_correct_patterns(self, tmp_path: Path) -> None:
        self._create_repo(tmp_path, "py-proj", "python")
        self._create_repo(tmp_path, "node-proj", "node")

        mgr = WorkspaceManager()
        ws = mgr.create_workspace("test", str(tmp_path))
        discovered = mgr.discover_repositories(ws)

        py_repo = next(r for r in discovered if r.project_type == "python")
        assert py_repo.include is not None
        assert "**/*.py" in py_repo.include

        node_repo = next(r for r in discovered if r.project_type == "node")
        assert node_repo.include is not None
        assert "**/*.js" in node_repo.include or "**/*.ts" in node_repo.include


class TestWorkspacePersistenceIntegration:
    """Integration tests for workspace config persistence."""

    def test_full_lifecycle(self, tmp_path: Path) -> None:
        """Test create → add repos → save → load → verify."""
        mgr = WorkspaceManager()

        # Create
        ws = mgr.create_workspace("lifecycle", str(tmp_path), description="test")
        ws.include = ["**/*.py", "**/*.js"]
        ws.context = 5
        ws.max_workers = 8

        # Add repos
        r1 = RepositoryConfig(
            name="core", path=str(tmp_path / "core"),
            priority="high", project_type="python",
            include=["**/*.py"],
        )
        r2 = RepositoryConfig(
            name="web", path=str(tmp_path / "web"),
            priority="normal", project_type="node",
        )
        ws.add_repository(r1)
        ws.add_repository(r2)

        # Save
        config_path = tmp_path / ".pysearch-workspace.toml"
        mgr.save_workspace(ws, config_path)
        assert config_path.exists()
        content = config_path.read_text(encoding="utf-8")
        assert "lifecycle" in content
        assert "core" in content
        assert "web" in content

        # Load
        loaded = mgr.load_workspace(config_path)
        assert loaded.name == "lifecycle"
        assert loaded.description == "test"
        assert loaded.include == ["**/*.py", "**/*.js"]
        assert loaded.context == 5
        assert loaded.max_workers == 8
        assert len(loaded.repositories) == 2
        assert loaded.repositories[0].name == "core"
        assert loaded.repositories[0].priority == "high"
        assert loaded.repositories[1].name == "web"

    def test_save_and_load_empty_workspace(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager()
        ws = mgr.create_workspace("empty", str(tmp_path))

        config_path = tmp_path / "empty.toml"
        mgr.save_workspace(ws, config_path)
        loaded = mgr.load_workspace(config_path)

        assert loaded.name == "empty"
        assert len(loaded.repositories) == 0

    def test_discover_then_save_load(self, tmp_path: Path) -> None:
        # Create repos
        repo1 = tmp_path / "repo1"
        repo1.mkdir()
        (repo1 / ".git").mkdir()
        (repo1 / "pyproject.toml").write_text("", encoding="utf-8")

        repo2 = tmp_path / "repo2"
        repo2.mkdir()
        (repo2 / ".git").mkdir()
        (repo2 / "package.json").write_text("{}", encoding="utf-8")

        mgr = WorkspaceManager()
        ws = mgr.create_workspace("disco", str(tmp_path))
        mgr.discover_repositories(ws, auto_add=True)

        # Save
        config_path = tmp_path / "disco.toml"
        mgr.save_workspace(ws, config_path)

        # Load
        loaded = mgr.load_workspace(config_path)
        assert len(loaded.repositories) >= 2
        repo_names = {r.name for r in loaded.repositories}
        assert "repo1" in repo_names
        assert "repo2" in repo_names


class TestWorkspaceSummaryIntegration:
    """Integration tests for workspace summary generation."""

    def test_summary_with_repos(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager()

        ws = WorkspaceConfig(
            name="summary-test",
            root_path=str(tmp_path),
            include=["**/*.py"],
            context=3,
            max_workers=4,
            repositories=[
                RepositoryConfig(name="a", path="/a", priority="high", project_type="python"),
                RepositoryConfig(name="b", path="/b", priority="normal", project_type="node"),
                RepositoryConfig(name="c", path="/c", priority="low", enabled=False),
            ],
        )

        summary = mgr.get_workspace_summary(ws)
        assert summary["total_repositories"] == 3
        assert summary["enabled_repositories"] == 2
        assert summary["disabled_repositories"] == 1
        assert summary["repositories_by_priority"]["high"] == 1
        assert summary["repositories_by_priority"]["normal"] == 1
        assert summary["repositories_by_priority"]["low"] == 1
        assert summary["repositories_by_type"]["python"] == 1
        assert summary["repositories_by_type"]["node"] == 1
        assert summary["search_settings"]["include"] == ["**/*.py"]
        assert summary["search_settings"]["max_workers"] == 4

    def test_summary_empty_workspace(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager()
        ws = WorkspaceConfig(name="empty", root_path=str(tmp_path))
        summary = mgr.get_workspace_summary(ws)
        assert summary["total_repositories"] == 0
        assert summary["enabled_repositories"] == 0


class TestMultiRepoIntegrationWithWorkspace:
    """Integration tests for workspace features via MultiRepoIntegrationManager."""

    def test_load_workspace_initializes_repos(self, tmp_path: Path) -> None:
        # Create workspace config with real repo paths
        repo1 = tmp_path / "r1"
        repo1.mkdir()
        (repo1 / "main.py").write_text("def hello(): pass\n", encoding="utf-8")

        mgr = WorkspaceManager()
        ws = mgr.create_workspace("test", str(tmp_path))
        ws.add_repository(RepositoryConfig(
            name="r1", path=str(repo1), priority="high",
            include=["**/*.py"],
        ))

        config_path = tmp_path / ".pysearch-workspace.toml"
        mgr.save_workspace(ws, config_path)

        # Load via integration manager
        from pysearch.core.managers.multi_repo_integration import MultiRepoIntegrationManager

        int_mgr = MultiRepoIntegrationManager(SearchConfig())
        result = int_mgr.load_workspace(config_path)
        assert result is True
        assert int_mgr.is_multi_repo_enabled()
        assert "r1" in int_mgr.list_repositories()

    def test_create_workspace_via_integration(self, tmp_path: Path) -> None:
        from pysearch.core.managers.multi_repo_integration import MultiRepoIntegrationManager

        int_mgr = MultiRepoIntegrationManager(SearchConfig())
        ws = int_mgr.create_workspace("via-integration", str(tmp_path))
        assert ws is not None
        assert ws.name == "via-integration"
        assert int_mgr.get_workspace_config() is ws

    def test_discover_via_integration(self, tmp_path: Path) -> None:
        # Create a repo to discover
        repo = tmp_path / "discoverable"
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "pyproject.toml").write_text("", encoding="utf-8")

        from pysearch.core.managers.multi_repo_integration import MultiRepoIntegrationManager

        int_mgr = MultiRepoIntegrationManager(SearchConfig())
        discovered = int_mgr.discover_repositories(root_path=str(tmp_path), auto_add=False)
        assert len(discovered) >= 1
        assert any(d["name"] == "discoverable" for d in discovered)

    def test_workspace_summary_via_integration(self, tmp_path: Path) -> None:
        from pysearch.core.managers.multi_repo_integration import MultiRepoIntegrationManager

        int_mgr = MultiRepoIntegrationManager(SearchConfig())
        int_mgr.create_workspace("summary-test", str(tmp_path))
        summary = int_mgr.get_workspace_summary()
        assert summary["name"] == "summary-test"
        assert summary["total_repositories"] == 0

    def test_no_workspace_returns_empty_summary(self) -> None:
        from pysearch.core.managers.multi_repo_integration import MultiRepoIntegrationManager

        int_mgr = MultiRepoIntegrationManager(SearchConfig())
        assert int_mgr.get_workspace_summary() == {}
        assert int_mgr.get_workspace_config() is None
