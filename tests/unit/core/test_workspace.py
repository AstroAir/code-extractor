"""Tests for pysearch.core.workspace module."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pysearch.core.workspace import (
    RepositoryConfig,
    WorkspaceConfig,
    WorkspaceManager,
)

# ---------------------------------------------------------------------------
# RepositoryConfig
# ---------------------------------------------------------------------------


class TestRepositoryConfig:
    """Tests for RepositoryConfig dataclass."""

    def test_creation(self):
        cfg = RepositoryConfig(name="repo1", path="/tmp/repo1")
        assert cfg.name == "repo1"
        assert cfg.path == "/tmp/repo1"

    def test_defaults(self):
        cfg = RepositoryConfig(name="r", path="/tmp/r")
        assert cfg.priority == "normal"
        assert cfg.enabled is True
        assert cfg.project_type == ""
        assert cfg.include is None
        assert cfg.exclude is None
        assert cfg.metadata == {}

    def test_to_dict_minimal(self):
        cfg = RepositoryConfig(name="r", path="/tmp/r")
        d = cfg.to_dict()
        assert d["name"] == "r"
        assert d["path"] == "/tmp/r"
        assert d["priority"] == "normal"
        assert d["enabled"] is True
        assert "project_type" not in d
        assert "include" not in d
        assert "exclude" not in d
        assert "metadata" not in d

    def test_to_dict_full(self):
        cfg = RepositoryConfig(
            name="r",
            path="/tmp/r",
            priority="high",
            enabled=False,
            project_type="python",
            include=["**/*.py"],
            exclude=["**/build/**"],
            metadata={"team": "backend"},
        )
        d = cfg.to_dict()
        assert d["priority"] == "high"
        assert d["enabled"] is False
        assert d["project_type"] == "python"
        assert d["include"] == ["**/*.py"]
        assert d["exclude"] == ["**/build/**"]
        assert d["metadata"]["team"] == "backend"

    def test_from_dict_minimal(self):
        cfg = RepositoryConfig.from_dict({"name": "r", "path": "/p"})
        assert cfg.name == "r"
        assert cfg.path == "/p"
        assert cfg.priority == "normal"
        assert cfg.enabled is True

    def test_from_dict_full(self):
        cfg = RepositoryConfig.from_dict(
            {
                "name": "r",
                "path": "/p",
                "priority": "low",
                "enabled": False,
                "project_type": "node",
                "include": ["**/*.js"],
                "exclude": ["**/dist/**"],
                "metadata": {"version": "1.0"},
            }
        )
        assert cfg.priority == "low"
        assert cfg.enabled is False
        assert cfg.project_type == "node"
        assert cfg.include == ["**/*.js"]
        assert cfg.metadata["version"] == "1.0"

    def test_roundtrip(self):
        original = RepositoryConfig(
            name="r",
            path="/tmp/r",
            priority="high",
            project_type="python",
            include=["**/*.py"],
        )
        restored = RepositoryConfig.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.path == original.path
        assert restored.priority == original.priority
        assert restored.project_type == original.project_type
        assert restored.include == original.include


# ---------------------------------------------------------------------------
# WorkspaceConfig
# ---------------------------------------------------------------------------


class TestWorkspaceConfig:
    """Tests for WorkspaceConfig dataclass."""

    def test_creation(self):
        ws = WorkspaceConfig(name="test-ws")
        assert ws.name == "test-ws"
        assert ws.created_at > 0
        assert ws.updated_at > 0

    def test_defaults(self):
        ws = WorkspaceConfig()
        assert ws.name == "default"
        assert ws.description == ""
        assert ws.root_path == "."
        assert ws.repositories == []
        assert ws.include is None
        assert ws.exclude is None
        assert ws.context == 2
        assert ws.parallel is True
        assert ws.workers == 0
        assert ws.max_workers == 4
        assert ws.follow_symlinks is False

    def test_get_repository_found(self):
        repo = RepositoryConfig(name="r1", path="/p1")
        ws = WorkspaceConfig(repositories=[repo])
        assert ws.get_repository("r1") is repo

    def test_get_repository_not_found(self):
        ws = WorkspaceConfig()
        assert ws.get_repository("missing") is None

    def test_get_enabled_repositories(self):
        r1 = RepositoryConfig(name="r1", path="/p1", enabled=True)
        r2 = RepositoryConfig(name="r2", path="/p2", enabled=False)
        r3 = RepositoryConfig(name="r3", path="/p3", enabled=True)
        ws = WorkspaceConfig(repositories=[r1, r2, r3])
        enabled = ws.get_enabled_repositories()
        assert len(enabled) == 2
        assert r1 in enabled
        assert r3 in enabled
        assert r2 not in enabled

    def test_get_repository_names(self):
        r1 = RepositoryConfig(name="r1", path="/p1")
        r2 = RepositoryConfig(name="r2", path="/p2")
        ws = WorkspaceConfig(repositories=[r1, r2])
        assert ws.get_repository_names() == ["r1", "r2"]

    def test_add_repository(self):
        ws = WorkspaceConfig()
        repo = RepositoryConfig(name="r1", path="/p1")
        assert ws.add_repository(repo) is True
        assert len(ws.repositories) == 1
        assert ws.get_repository("r1") is repo

    def test_add_repository_duplicate(self):
        ws = WorkspaceConfig()
        repo = RepositoryConfig(name="r1", path="/p1")
        ws.add_repository(repo)
        assert ws.add_repository(RepositoryConfig(name="r1", path="/p2")) is False
        assert len(ws.repositories) == 1

    def test_remove_repository(self):
        repo = RepositoryConfig(name="r1", path="/p1")
        ws = WorkspaceConfig(repositories=[repo])
        assert ws.remove_repository("r1") is True
        assert len(ws.repositories) == 0

    def test_remove_repository_not_found(self):
        ws = WorkspaceConfig()
        assert ws.remove_repository("missing") is False

    def test_add_updates_timestamp(self):
        ws = WorkspaceConfig()
        old_ts = ws.updated_at
        time.sleep(0.01)
        ws.add_repository(RepositoryConfig(name="r", path="/p"))
        assert ws.updated_at >= old_ts

    def test_remove_updates_timestamp(self):
        repo = RepositoryConfig(name="r", path="/p")
        ws = WorkspaceConfig(repositories=[repo])
        old_ts = ws.updated_at
        time.sleep(0.01)
        ws.remove_repository("r")
        assert ws.updated_at >= old_ts

    def test_to_dict(self):
        ws = WorkspaceConfig(
            name="test",
            description="desc",
            root_path="/root",
            include=["**/*.py"],
            context=5,
        )
        d = ws.to_dict()
        assert d["workspace"]["name"] == "test"
        assert d["workspace"]["description"] == "desc"
        assert d["workspace"]["root_path"] == "/root"
        assert d["workspace"]["search"]["include"] == ["**/*.py"]
        assert d["workspace"]["search"]["context"] == 5

    def test_from_dict(self):
        data = {
            "workspace": {
                "name": "test",
                "description": "desc",
                "root_path": "/root",
                "search": {
                    "include": ["**/*.py"],
                    "context": 5,
                    "parallel": False,
                },
                "repositories": [
                    {"name": "r1", "path": "/p1", "priority": "high"},
                ],
            }
        }
        ws = WorkspaceConfig.from_dict(data)
        assert ws.name == "test"
        assert ws.description == "desc"
        assert ws.root_path == "/root"
        assert ws.include == ["**/*.py"]
        assert ws.context == 5
        assert ws.parallel is False
        assert len(ws.repositories) == 1
        assert ws.repositories[0].name == "r1"
        assert ws.repositories[0].priority == "high"

    def test_roundtrip(self):
        original = WorkspaceConfig(
            name="test",
            root_path="/root",
            include=["**/*.py"],
            exclude=["**/build/**"],
            context=3,
            max_workers=8,
            repositories=[
                RepositoryConfig(name="r1", path="/p1", priority="high"),
                RepositoryConfig(name="r2", path="/p2", project_type="node"),
            ],
        )
        restored = WorkspaceConfig.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.root_path == original.root_path
        assert restored.include == original.include
        assert restored.exclude == original.exclude
        assert restored.context == original.context
        assert restored.max_workers == original.max_workers
        assert len(restored.repositories) == 2
        assert restored.repositories[0].name == "r1"
        assert restored.repositories[1].project_type == "node"


# ---------------------------------------------------------------------------
# WorkspaceManager
# ---------------------------------------------------------------------------


class TestWorkspaceManager:
    """Tests for WorkspaceManager class."""

    def test_create_workspace(self, tmp_path: Path):
        mgr = WorkspaceManager()
        ws = mgr.create_workspace("test", tmp_path, description="my ws")
        assert ws.name == "test"
        assert ws.root_path == str(tmp_path.resolve())
        assert ws.description == "my ws"
        assert ws.created_at > 0

    def test_create_workspace_with_metadata(self, tmp_path: Path):
        mgr = WorkspaceManager()
        ws = mgr.create_workspace("test", tmp_path, team="backend")
        assert ws.metadata["team"] == "backend"

    def test_save_and_load_workspace(self, tmp_path: Path):
        mgr = WorkspaceManager()
        ws = mgr.create_workspace("test", str(tmp_path))
        ws.add_repository(RepositoryConfig(name="r1", path=str(tmp_path), priority="high"))
        ws.include = ["**/*.py"]
        ws.context = 5

        config_path = tmp_path / ".pysearch-workspace.toml"
        mgr.save_workspace(ws, config_path)

        assert config_path.exists()

        loaded = mgr.load_workspace(config_path)
        assert loaded.name == "test"
        assert loaded.include == ["**/*.py"]
        assert loaded.context == 5
        assert len(loaded.repositories) == 1
        assert loaded.repositories[0].name == "r1"
        assert loaded.repositories[0].priority == "high"

    def test_save_default_path(self, tmp_path: Path):
        mgr = WorkspaceManager()
        ws = mgr.create_workspace("test", str(tmp_path))

        saved_path = mgr.save_workspace(ws)
        assert saved_path == tmp_path / ".pysearch-workspace.toml"
        assert saved_path.exists()

    def test_load_nonexistent_raises(self, tmp_path: Path):
        mgr = WorkspaceManager()
        with pytest.raises(FileNotFoundError):
            mgr.load_workspace(tmp_path / "nonexistent.toml")

    def test_discover_repositories(self, tmp_path: Path):
        # Create some fake repos with .git directories
        repo1 = tmp_path / "repo1"
        repo1.mkdir()
        (repo1 / ".git").mkdir()
        (repo1 / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

        repo2 = tmp_path / "repo2"
        repo2.mkdir()
        (repo2 / ".git").mkdir()
        (repo2 / "package.json").write_text("{}\n", encoding="utf-8")

        # Non-repo dir (no .git)
        non_repo = tmp_path / "not_a_repo"
        non_repo.mkdir()

        mgr = WorkspaceManager()
        ws = mgr.create_workspace("test", str(tmp_path))

        discovered = mgr.discover_repositories(ws, max_depth=2)
        assert len(discovered) >= 2
        names = [r.name for r in discovered]
        assert "repo1" in names
        assert "repo2" in names

        # Check project type detection
        repo1_cfg = next(r for r in discovered if r.name == "repo1")
        assert repo1_cfg.project_type == "python"
        repo2_cfg = next(r for r in discovered if r.name == "repo2")
        assert repo2_cfg.project_type == "node"

    def test_discover_no_auto_add(self, tmp_path: Path):
        repo1 = tmp_path / "repo1"
        repo1.mkdir()
        (repo1 / ".git").mkdir()

        mgr = WorkspaceManager()
        ws = mgr.create_workspace("test", str(tmp_path))

        discovered = mgr.discover_repositories(ws, auto_add=False)
        assert len(discovered) >= 1
        # Should NOT have been added to workspace
        assert len(ws.repositories) == 0

    def test_discover_auto_add(self, tmp_path: Path):
        repo1 = tmp_path / "repo1"
        repo1.mkdir()
        (repo1 / ".git").mkdir()

        mgr = WorkspaceManager()
        ws = mgr.create_workspace("test", str(tmp_path))

        mgr.discover_repositories(ws, auto_add=True)
        assert len(ws.repositories) >= 1

    def test_discover_skips_existing(self, tmp_path: Path):
        repo1 = tmp_path / "repo1"
        repo1.mkdir()
        (repo1 / ".git").mkdir()

        mgr = WorkspaceManager()
        ws = mgr.create_workspace("test", str(tmp_path))

        # Add repo1 manually first
        ws.add_repository(RepositoryConfig(name="repo1", path=str(repo1)))

        # Discover should skip already-existing paths
        discovered = mgr.discover_repositories(ws, auto_add=True)
        assert len(discovered) == 0

    def test_discover_max_depth(self, tmp_path: Path):
        # Create nested repo structure
        deep = tmp_path / "a" / "b" / "c" / "d" / "deep_repo"
        deep.mkdir(parents=True)
        (deep / ".git").mkdir()

        mgr = WorkspaceManager()
        ws = mgr.create_workspace("test", str(tmp_path))

        # max_depth=2 should NOT find the deep repo
        discovered = mgr.discover_repositories(ws, max_depth=2)
        deep_names = [r.name for r in discovered]
        assert "deep_repo" not in deep_names

    def test_discover_nonexistent_root(self, tmp_path: Path):
        mgr = WorkspaceManager()
        ws = mgr.create_workspace("test", str(tmp_path / "nonexistent"))
        discovered = mgr.discover_repositories(ws)
        assert discovered == []

    def test_get_workspace_summary(self, tmp_path: Path):
        mgr = WorkspaceManager()
        ws = WorkspaceConfig(
            name="test",
            root_path=str(tmp_path),
            repositories=[
                RepositoryConfig(name="r1", path="/p1", priority="high", project_type="python"),
                RepositoryConfig(name="r2", path="/p2", priority="normal", project_type="node"),
                RepositoryConfig(name="r3", path="/p3", priority="low", enabled=False),
            ],
        )
        summary = mgr.get_workspace_summary(ws)
        assert summary["name"] == "test"
        assert summary["total_repositories"] == 3
        assert summary["enabled_repositories"] == 2
        assert summary["disabled_repositories"] == 1
        assert summary["repositories_by_priority"]["high"] == 1
        assert summary["repositories_by_priority"]["normal"] == 1
        assert summary["repositories_by_priority"]["low"] == 1
        assert summary["repositories_by_type"]["python"] == 1
        assert summary["repositories_by_type"]["node"] == 1

    def test_detect_project_type_python(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        assert WorkspaceManager._detect_project_type(tmp_path) == "python"

    def test_detect_project_type_node(self, tmp_path: Path):
        (tmp_path / "package.json").write_text("", encoding="utf-8")
        assert WorkspaceManager._detect_project_type(tmp_path) == "node"

    def test_detect_project_type_go(self, tmp_path: Path):
        (tmp_path / "go.mod").write_text("", encoding="utf-8")
        assert WorkspaceManager._detect_project_type(tmp_path) == "go"

    def test_detect_project_type_rust(self, tmp_path: Path):
        (tmp_path / "Cargo.toml").write_text("", encoding="utf-8")
        assert WorkspaceManager._detect_project_type(tmp_path) == "rust"

    def test_detect_project_type_java(self, tmp_path: Path):
        (tmp_path / "pom.xml").write_text("", encoding="utf-8")
        assert WorkspaceManager._detect_project_type(tmp_path) == "java"

    def test_detect_project_type_unknown(self, tmp_path: Path):
        assert WorkspaceManager._detect_project_type(tmp_path) == ""


# ---------------------------------------------------------------------------
# TOML Parser
# ---------------------------------------------------------------------------


class TestTomlParsing:
    """Tests for the built-in TOML parsing/writing."""

    def test_parse_basic_values(self):
        text = """
[workspace]
name = "test"
context = 5
parallel = true
workers = 0
"""
        data = WorkspaceManager._parse_toml(text)
        assert data["workspace"]["name"] == "test"
        assert data["workspace"]["context"] == 5
        assert data["workspace"]["parallel"] is True
        assert data["workspace"]["workers"] == 0

    def test_parse_arrays(self):
        text = """
[workspace.search]
include = ["**/*.py", "**/*.js"]
"""
        data = WorkspaceManager._parse_toml(text)
        assert data["workspace"]["search"]["include"] == ["**/*.py", "**/*.js"]

    def test_parse_array_of_tables(self):
        text = """
[[workspace.repositories]]
name = "repo1"
path = "/path/to/repo1"
priority = "high"
enabled = true

[[workspace.repositories]]
name = "repo2"
path = "/path/to/repo2"
"""
        data = WorkspaceManager._parse_toml(text)
        repos = data["workspace"]["repositories"]
        assert len(repos) == 2
        assert repos[0]["name"] == "repo1"
        assert repos[0]["priority"] == "high"
        assert repos[0]["enabled"] is True
        assert repos[1]["name"] == "repo2"

    def test_roundtrip_toml(self, tmp_path: Path):
        """Test that saving and loading produces equivalent data."""
        mgr = WorkspaceManager()
        ws = WorkspaceConfig(
            name="roundtrip-test",
            description="test description",
            root_path=str(tmp_path),
            include=["**/*.py", "**/*.js"],
            exclude=["**/build/**"],
            context=3,
            parallel=True,
            max_workers=8,
            repositories=[
                RepositoryConfig(
                    name="r1",
                    path="/path/r1",
                    priority="high",
                    project_type="python",
                    include=["**/*.py"],
                ),
                RepositoryConfig(
                    name="r2",
                    path="/path/r2",
                    priority="low",
                    enabled=False,
                ),
            ],
        )

        config_path = tmp_path / "test.toml"
        mgr.save_workspace(ws, config_path)

        loaded = mgr.load_workspace(config_path)
        assert loaded.name == ws.name
        assert loaded.description == ws.description
        assert loaded.include == ws.include
        assert loaded.exclude == ws.exclude
        assert loaded.context == ws.context
        assert loaded.max_workers == ws.max_workers
        assert len(loaded.repositories) == 2
        assert loaded.repositories[0].name == "r1"
        assert loaded.repositories[0].priority == "high"
        assert loaded.repositories[0].project_type == "python"
        assert loaded.repositories[1].name == "r2"
        assert loaded.repositories[1].enabled is False
