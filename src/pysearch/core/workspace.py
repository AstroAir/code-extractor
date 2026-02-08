"""
Workspace management module for pysearch.

This module provides a high-level Workspace concept that groups multiple
repositories together with shared configuration, persistence, and
auto-discovery capabilities.

Classes:
    RepositoryConfig: Configuration for a single repository in a workspace
    WorkspaceConfig: Complete workspace configuration with global settings
    WorkspaceManager: Manages workspace lifecycle (create, load, save, discover)

Key Features:
    - Workspace creation and configuration
    - TOML-based persistence (.pysearch-workspace.toml)
    - Repository auto-discovery (scan directory trees for Git repos)
    - Global search settings with per-repository overrides
    - Project type detection (Python, Node, Java, Go, Rust, etc.)

Example:
    Creating and using a workspace:
        >>> from pysearch.core.workspace import WorkspaceManager
        >>>
        >>> manager = WorkspaceManager()
        >>> ws = manager.create_workspace("my-project", "/path/to/root")
        >>> ws = manager.discover_repositories(ws, max_depth=3)
        >>> manager.save_workspace(ws, "/path/to/root/.pysearch-workspace.toml")
        >>>
        >>> # Later, load it back
        >>> ws = manager.load_workspace("/path/to/root/.pysearch-workspace.toml")

    Loading a workspace in PySearch:
        >>> from pysearch import PySearch, SearchConfig
        >>> engine = PySearch(SearchConfig())
        >>> engine.open_workspace("/path/to/.pysearch-workspace.toml")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.logging_config import get_logger

# Project type detection markers
_PROJECT_MARKERS: dict[str, list[str]] = {
    "python": ["pyproject.toml", "setup.py", "setup.cfg", "Pipfile", "requirements.txt"],
    "node": ["package.json", "tsconfig.json"],
    "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
    "go": ["go.mod", "go.sum"],
    "rust": ["Cargo.toml"],
    "dotnet": ["*.csproj", "*.sln", "*.fsproj"],
    "ruby": ["Gemfile", "Rakefile", "*.gemspec"],
    "php": ["composer.json"],
    "swift": ["Package.swift", "*.xcodeproj"],
    "kotlin": ["build.gradle.kts"],
    "scala": ["build.sbt"],
}

# Default include patterns per project type
_PROJECT_INCLUDE_PATTERNS: dict[str, list[str]] = {
    "python": ["**/*.py", "**/*.pyi", "**/*.pyx"],
    "node": ["**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx", "**/*.mjs", "**/*.cjs"],
    "java": ["**/*.java"],
    "go": ["**/*.go"],
    "rust": ["**/*.rs"],
    "dotnet": ["**/*.cs", "**/*.fs", "**/*.vb"],
    "ruby": ["**/*.rb", "**/*.erb"],
    "php": ["**/*.php"],
    "swift": ["**/*.swift"],
    "kotlin": ["**/*.kt", "**/*.kts"],
    "scala": ["**/*.scala", "**/*.sc"],
}

# Default exclude patterns per project type
_PROJECT_EXCLUDE_PATTERNS: dict[str, list[str]] = {
    "python": [
        "**/.venv/**", "**/venv/**", "**/__pycache__/**",
        "**/build/**", "**/dist/**", "**/*.egg-info/**",
    ],
    "node": [
        "**/node_modules/**", "**/dist/**", "**/build/**",
        "**/.next/**", "**/.nuxt/**",
    ],
    "java": [
        "**/target/**", "**/build/**", "**/.gradle/**",
    ],
    "go": [
        "**/vendor/**",
    ],
    "rust": [
        "**/target/**",
    ],
    "dotnet": [
        "**/bin/**", "**/obj/**",
    ],
}


@dataclass
class RepositoryConfig:
    """Configuration for a single repository within a workspace.

    Attributes:
        name: Unique name for the repository within the workspace
        path: Absolute or relative path to the repository root
        priority: Search priority ("high", "normal", "low")
        enabled: Whether this repository is active for searches
        project_type: Detected project type (e.g., "python", "node")
        include: File include patterns (overrides workspace defaults if set)
        exclude: File exclude patterns (overrides workspace defaults if set)
        metadata: Arbitrary key-value metadata
    """

    name: str
    path: str
    priority: str = "normal"
    enabled: bool = True
    project_type: str = ""
    include: list[str] | None = None
    exclude: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary suitable for TOML output."""
        d: dict[str, Any] = {
            "name": self.name,
            "path": self.path,
            "priority": self.priority,
            "enabled": self.enabled,
        }
        if self.project_type:
            d["project_type"] = self.project_type
        if self.include is not None:
            d["include"] = self.include
        if self.exclude is not None:
            d["exclude"] = self.exclude
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepositoryConfig:
        """Deserialize from a dictionary (e.g., parsed from TOML)."""
        return cls(
            name=data["name"],
            path=data["path"],
            priority=data.get("priority", "normal"),
            enabled=data.get("enabled", True),
            project_type=data.get("project_type", ""),
            include=data.get("include"),
            exclude=data.get("exclude"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class WorkspaceConfig:
    """Complete workspace configuration.

    A workspace groups multiple repositories together with shared search
    settings and per-repository overrides.

    Attributes:
        name: Human-readable workspace name
        description: Optional description
        root_path: Root directory of the workspace
        created_at: Unix timestamp of creation
        updated_at: Unix timestamp of last modification
        repositories: List of repository configurations
        include: Global file include patterns (applied to repos without overrides)
        exclude: Global file exclude patterns (applied to repos without overrides)
        context: Default number of context lines for search results
        parallel: Whether to use parallel search
        workers: Number of parallel workers (0 = auto)
        max_workers: Max workers for cross-repo parallel search
        follow_symlinks: Whether to follow symlinks
        metadata: Arbitrary key-value metadata for the workspace
    """

    name: str = "default"
    description: str = ""
    root_path: str = "."

    created_at: float = 0.0
    updated_at: float = 0.0

    repositories: list[RepositoryConfig] = field(default_factory=list)

    # Global search settings
    include: list[str] | None = None
    exclude: list[str] | None = None
    context: int = 2
    parallel: bool = True
    workers: int = 0
    max_workers: int = 4
    follow_symlinks: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        now = time.time()
        if self.created_at == 0.0:
            self.created_at = now
        if self.updated_at == 0.0:
            self.updated_at = now

    def get_repository(self, name: str) -> RepositoryConfig | None:
        """Get a repository config by name."""
        for repo in self.repositories:
            if repo.name == name:
                return repo
        return None

    def get_enabled_repositories(self) -> list[RepositoryConfig]:
        """Get all enabled repository configs."""
        return [r for r in self.repositories if r.enabled]

    def get_repository_names(self) -> list[str]:
        """Get list of all repository names."""
        return [r.name for r in self.repositories]

    def add_repository(self, repo: RepositoryConfig) -> bool:
        """Add a repository config. Returns False if name already exists."""
        if any(r.name == repo.name for r in self.repositories):
            return False
        self.repositories.append(repo)
        self.updated_at = time.time()
        return True

    def remove_repository(self, name: str) -> bool:
        """Remove a repository by name. Returns False if not found."""
        for i, r in enumerate(self.repositories):
            if r.name == name:
                self.repositories.pop(i)
                self.updated_at = time.time()
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire workspace config to a nested dict."""
        return {
            "workspace": {
                "name": self.name,
                "description": self.description,
                "root_path": self.root_path,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "metadata": self.metadata,
                "search": {
                    "include": self.include,
                    "exclude": self.exclude,
                    "context": self.context,
                    "parallel": self.parallel,
                    "workers": self.workers,
                    "max_workers": self.max_workers,
                    "follow_symlinks": self.follow_symlinks,
                },
                "repositories": [r.to_dict() for r in self.repositories],
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkspaceConfig:
        """Deserialize from a nested dict (e.g., parsed TOML)."""
        ws_data = data.get("workspace", data)
        search_data = ws_data.get("search", {})
        repos_data = ws_data.get("repositories", [])

        repos = [RepositoryConfig.from_dict(r) for r in repos_data]

        return cls(
            name=ws_data.get("name", "default"),
            description=ws_data.get("description", ""),
            root_path=ws_data.get("root_path", "."),
            created_at=ws_data.get("created_at", 0.0),
            updated_at=ws_data.get("updated_at", 0.0),
            repositories=repos,
            include=search_data.get("include"),
            exclude=search_data.get("exclude"),
            context=search_data.get("context", 2),
            parallel=search_data.get("parallel", True),
            workers=search_data.get("workers", 0),
            max_workers=search_data.get("max_workers", 4),
            follow_symlinks=search_data.get("follow_symlinks", False),
            metadata=ws_data.get("metadata", {}),
        )


class WorkspaceManager:
    """
    Manages workspace lifecycle: creation, loading, saving, and repository discovery.

    This class provides high-level operations for working with workspaces,
    including TOML-based persistence and automatic repository detection.
    """

    def __init__(self) -> None:
        self.logger = get_logger()

    # -- Creation -----------------------------------------------------------

    def create_workspace(
        self,
        name: str,
        root_path: str | Path,
        description: str = "",
        **metadata: Any,
    ) -> WorkspaceConfig:
        """
        Create a new workspace configuration.

        Args:
            name: Human-readable workspace name
            root_path: Root directory for the workspace
            description: Optional description
            **metadata: Arbitrary metadata

        Returns:
            A new WorkspaceConfig instance
        """
        root = Path(root_path).resolve()
        ws = WorkspaceConfig(
            name=name,
            description=description,
            root_path=str(root),
            metadata=dict(metadata),
        )
        self.logger.info(f"Created workspace '{name}' at {root}")
        return ws

    # -- Persistence --------------------------------------------------------

    def save_workspace(
        self, config: WorkspaceConfig, config_path: str | Path | None = None
    ) -> Path:
        """
        Save workspace configuration to a TOML file.

        Args:
            config: The workspace configuration to save
            config_path: Path for the output file. If None, saves to
                         <root_path>/.pysearch-workspace.toml

        Returns:
            Path to the saved configuration file
        """
        if config_path is None:
            config_path = Path(config.root_path) / ".pysearch-workspace.toml"
        else:
            config_path = Path(config_path)

        config.updated_at = time.time()
        data = config.to_dict()

        # Write TOML manually to avoid extra dependencies
        lines = self._dict_to_toml(data)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        self.logger.info(f"Saved workspace config to {config_path}")
        return config_path

    def load_workspace(self, config_path: str | Path) -> WorkspaceConfig:
        """
        Load workspace configuration from a TOML file.

        Args:
            config_path: Path to the .pysearch-workspace.toml file

        Returns:
            Loaded WorkspaceConfig

        Raises:
            FileNotFoundError: If the config file does not exist
            ValueError: If the config file cannot be parsed
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Workspace config not found: {config_path}")

        text = config_path.read_text(encoding="utf-8")
        data = self._parse_toml(text)
        ws = WorkspaceConfig.from_dict(data)

        self.logger.info(f"Loaded workspace '{ws.name}' from {config_path}")
        return ws

    # -- Repository Discovery -----------------------------------------------

    def discover_repositories(
        self,
        workspace: WorkspaceConfig,
        root_path: str | Path | None = None,
        max_depth: int = 3,
        auto_add: bool = True,
    ) -> list[RepositoryConfig]:
        """
        Scan a directory tree for Git repositories and optionally add them.

        Args:
            workspace: The workspace to add discovered repos to
            root_path: Directory to scan (defaults to workspace.root_path)
            max_depth: Maximum directory depth to search
            auto_add: Whether to automatically add discovered repos to workspace

        Returns:
            List of discovered RepositoryConfig objects
        """
        root = Path(root_path or workspace.root_path).resolve()
        if not root.exists():
            self.logger.warning(f"Discovery root does not exist: {root}")
            return []

        discovered: list[RepositoryConfig] = []
        existing_paths = {Path(r.path).resolve() for r in workspace.repositories}

        self._scan_for_repos(root, root, max_depth, 0, discovered, existing_paths)

        if auto_add:
            for repo_cfg in discovered:
                workspace.add_repository(repo_cfg)
            if discovered:
                workspace.updated_at = time.time()
                self.logger.info(
                    f"Auto-added {len(discovered)} repositories to workspace '{workspace.name}'"
                )

        return discovered

    def _scan_for_repos(
        self,
        root: Path,
        current: Path,
        max_depth: int,
        depth: int,
        results: list[RepositoryConfig],
        existing: set[Path],
    ) -> None:
        """Recursively scan directories for Git repositories."""
        if depth > max_depth:
            return

        try:
            git_dir = current / ".git"
            if git_dir.exists() and current.resolve() not in existing:
                repo_cfg = self._create_repo_config(root, current)
                results.append(repo_cfg)
                existing.add(current.resolve())
                # Don't descend into sub-repos of this repo
                return

            # Scan subdirectories
            for child in sorted(current.iterdir()):
                if not child.is_dir():
                    continue
                # Skip hidden dirs and common non-project dirs
                name = child.name
                if name.startswith(".") or name in {
                    "node_modules", "__pycache__", ".venv", "venv",
                    "build", "dist", "target", ".git", ".hg", ".svn",
                    "bin", "obj", ".tox", ".mypy_cache", ".pytest_cache",
                }:
                    continue
                self._scan_for_repos(root, child, max_depth, depth + 1, results, existing)
        except PermissionError:
            pass

    def _create_repo_config(self, workspace_root: Path, repo_path: Path) -> RepositoryConfig:
        """Create a RepositoryConfig from a discovered repository directory."""
        # Derive name from relative path
        try:
            rel = repo_path.relative_to(workspace_root)
            name = str(rel).replace("\\", "/").replace("/", "-") or repo_path.name
        except ValueError:
            name = repo_path.name

        # Detect project type
        project_type = self._detect_project_type(repo_path)

        # Get default patterns for the project type
        include = _PROJECT_INCLUDE_PATTERNS.get(project_type)
        exclude = _PROJECT_EXCLUDE_PATTERNS.get(project_type)

        return RepositoryConfig(
            name=name,
            path=str(repo_path),
            priority="normal",
            enabled=True,
            project_type=project_type,
            include=list(include) if include else None,
            exclude=list(exclude) if exclude else None,
        )

    @staticmethod
    def _detect_project_type(repo_path: Path) -> str:
        """Detect the project type based on marker files."""
        for ptype, markers in _PROJECT_MARKERS.items():
            for marker in markers:
                if "*" in marker:
                    # Glob pattern
                    if list(repo_path.glob(marker)):
                        return ptype
                else:
                    if (repo_path / marker).exists():
                        return ptype
        return ""

    # -- Workspace Summary --------------------------------------------------

    def get_workspace_summary(self, workspace: WorkspaceConfig) -> dict[str, Any]:
        """
        Get a summary of the workspace configuration.

        Args:
            workspace: The workspace to summarize

        Returns:
            Dictionary with workspace summary information
        """
        repos_by_priority: dict[str, int] = {"high": 0, "normal": 0, "low": 0}
        repos_by_type: dict[str, int] = {}
        enabled_count = 0

        for repo in workspace.repositories:
            repos_by_priority[repo.priority] = repos_by_priority.get(repo.priority, 0) + 1
            if repo.project_type:
                repos_by_type[repo.project_type] = repos_by_type.get(repo.project_type, 0) + 1
            if repo.enabled:
                enabled_count += 1

        return {
            "name": workspace.name,
            "description": workspace.description,
            "root_path": workspace.root_path,
            "total_repositories": len(workspace.repositories),
            "enabled_repositories": enabled_count,
            "disabled_repositories": len(workspace.repositories) - enabled_count,
            "repositories_by_priority": repos_by_priority,
            "repositories_by_type": repos_by_type,
            "search_settings": {
                "include": workspace.include,
                "exclude": workspace.exclude,
                "context": workspace.context,
                "parallel": workspace.parallel,
                "workers": workspace.workers,
                "max_workers": workspace.max_workers,
            },
        }

    # -- TOML Helpers (minimal, no external dependency) ---------------------

    @staticmethod
    def _dict_to_toml(data: dict[str, Any], prefix: str = "") -> list[str]:
        """Convert a nested dict to TOML lines (minimal implementation)."""
        lines: list[str] = []
        ws = data.get("workspace", {})

        # Header comment
        lines.append("# pysearch workspace configuration")
        lines.append("# Generated by pysearch - do not edit manually unless you know what you're doing")
        lines.append("")

        # [workspace] section
        lines.append("[workspace]")
        for key in ("name", "description", "root_path"):
            val = ws.get(key, "")
            lines.append(f'{key} = {WorkspaceManager._toml_value(val)}')
        for key in ("created_at", "updated_at"):
            val = ws.get(key, 0.0)
            lines.append(f"{key} = {val}")
        lines.append("")

        # [workspace.metadata]
        meta = ws.get("metadata", {})
        if meta:
            lines.append("[workspace.metadata]")
            for k, v in meta.items():
                lines.append(f'{k} = {WorkspaceManager._toml_value(v)}')
            lines.append("")

        # [workspace.search]
        search = ws.get("search", {})
        lines.append("[workspace.search]")
        for key in ("context", "parallel", "workers", "max_workers", "follow_symlinks"):
            val = search.get(key)
            if val is not None:
                lines.append(f"{key} = {WorkspaceManager._toml_value(val)}")
        for key in ("include", "exclude"):
            val = search.get(key)
            if val is not None:
                lines.append(f"{key} = {WorkspaceManager._toml_value(val)}")
        lines.append("")

        # [[workspace.repositories]]
        repos = ws.get("repositories", [])
        for repo in repos:
            lines.append("[[workspace.repositories]]")
            for key in ("name", "path", "priority"):
                val = repo.get(key, "")
                lines.append(f'{key} = {WorkspaceManager._toml_value(val)}')
            lines.append(f'enabled = {WorkspaceManager._toml_value(repo.get("enabled", True))}')
            if repo.get("project_type"):
                lines.append(f'project_type = {WorkspaceManager._toml_value(repo["project_type"])}')
            for key in ("include", "exclude"):
                val = repo.get(key)
                if val is not None:
                    lines.append(f"{key} = {WorkspaceManager._toml_value(val)}")
            meta = repo.get("metadata", {})
            if meta:
                # Inline table for metadata
                parts = ", ".join(
                    f'{k} = {WorkspaceManager._toml_value(v)}' for k, v in meta.items()
                )
                lines.append(f"metadata = {{ {parts} }}")
            lines.append("")

        return lines

    @staticmethod
    def _toml_value(val: Any) -> str:
        """Convert a Python value to a TOML literal string."""
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, int):
            return str(val)
        if isinstance(val, float):
            return str(val)
        if isinstance(val, str):
            # Escape backslashes for valid TOML (important for Windows paths)
            escaped = val.replace("\\", "\\\\")
            return f'"{escaped}"'
        if isinstance(val, list):
            items = ", ".join(WorkspaceManager._toml_value(v) for v in val)
            return f"[{items}]"
        if isinstance(val, dict):
            parts = ", ".join(
                f"{k} = {WorkspaceManager._toml_value(v)}" for k, v in val.items()
            )
            return f"{{ {parts} }}"
        return str(val)

    @staticmethod
    def _parse_toml(text: str) -> dict[str, Any]:
        """
        Parse a TOML string into a nested dict.

        Uses Python 3.11+ tomllib if available, otherwise falls back
        to a basic parser sufficient for workspace config files.
        """
        try:
            import tomllib

            return tomllib.loads(text)
        except ImportError:
            pass

        try:
            import tomli

            return tomli.loads(text)
        except ImportError:
            pass

        # Fallback: very basic line-by-line parser for workspace TOML files
        return WorkspaceManager._basic_toml_parse(text)

    @staticmethod
    def _basic_toml_parse(text: str) -> dict[str, Any]:
        """
        Minimal TOML parser supporting the subset used by workspace configs.

        Supports: sections, key=value, strings, booleans, integers, floats,
        arrays, and array-of-tables ([[section]]).
        """
        import re as _re

        result: dict[str, Any] = {}
        current_section: list[str] = []
        is_array_table = False

        for raw_line in text.splitlines():
            line = raw_line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Array of tables: [[section.path]]
            m = _re.match(r"^\[\[(.+)\]\]$", line)
            if m:
                current_section = m.group(1).strip().split(".")
                is_array_table = True
                # Ensure parent path exists and current key is a list
                parent = result
                for key in current_section[:-1]:
                    parent = parent.setdefault(key, {})
                arr_key = current_section[-1]
                if arr_key not in parent:
                    parent[arr_key] = []
                parent[arr_key].append({})
                continue

            # Regular table: [section.path]
            m = _re.match(r"^\[(.+)\]$", line)
            if m:
                current_section = m.group(1).strip().split(".")
                is_array_table = False
                # Ensure path exists
                parent = result
                for key in current_section:
                    parent = parent.setdefault(key, {})
                continue

            # Key = value
            m = _re.match(r"^(\w[\w_-]*)\s*=\s*(.+)$", line)
            if m:
                key = m.group(1).strip()
                raw_val = m.group(2).strip()
                value = WorkspaceManager._parse_toml_value(raw_val)

                # Navigate to current section
                parent = result
                for skey in current_section[:-1] if is_array_table else current_section:
                    parent = parent.setdefault(skey, {})

                if is_array_table:
                    # Get the last item in the array
                    arr_key = current_section[-1]
                    grand = result
                    for skey in current_section[:-1]:
                        grand = grand.setdefault(skey, {})
                    arr = grand.get(arr_key, [])
                    if arr:
                        arr[-1][key] = value
                else:
                    parent[key] = value

        return result

    @staticmethod
    def _parse_toml_value(raw: str) -> Any:
        """Parse a single TOML value string."""
        # String
        if raw.startswith('"') and raw.endswith('"'):
            return raw[1:-1]
        # Boolean
        if raw == "true":
            return True
        if raw == "false":
            return False
        # Array
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1].strip()
            if not inner:
                return []
            # Split by comma, handling quoted strings
            items: list[Any] = []
            in_quote = False
            current = ""
            for ch in inner:
                if ch == '"':
                    in_quote = not in_quote
                    current += ch
                elif ch == "," and not in_quote:
                    items.append(WorkspaceManager._parse_toml_value(current.strip()))
                    current = ""
                else:
                    current += ch
            if current.strip():
                items.append(WorkspaceManager._parse_toml_value(current.strip()))
            return items
        # Inline table
        if raw.startswith("{") and raw.endswith("}"):
            inner = raw[1:-1].strip()
            if not inner:
                return {}
            result: dict[str, Any] = {}
            for part in inner.split(","):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    result[k.strip()] = WorkspaceManager._parse_toml_value(v.strip())
            return result
        # Integer
        try:
            return int(raw)
        except ValueError:
            pass
        # Float
        try:
            return float(raw)
        except ValueError:
            pass
        return raw
