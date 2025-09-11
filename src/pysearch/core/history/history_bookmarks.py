"""
Bookmark management functionality for search history.

This module provides comprehensive bookmark organization and management capabilities,
including folder organization, bookmark categorization, and persistent storage.

Classes:
    BookmarkFolder: Organize bookmarks into folders
    BookmarkManager: Main bookmark management class

Key Features:
    - Bookmark favorite searches for quick access
    - Organize bookmarks into folders
    - Persistent storage of bookmarks and folders
    - Search and filter bookmarks

Example:
    Bookmark management:
        >>> from pysearch.core.history.history_bookmarks import BookmarkManager
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig()
        >>> manager = BookmarkManager(config)
        >>>
        >>> # Add a bookmark
        >>> manager.add_bookmark("find_main", query, results)
        >>>
        >>> # Create a folder and organize bookmarks
        >>> manager.create_folder("Python Functions", "Python-related searches")
        >>> manager.add_bookmark_to_folder("find_main", "Python Functions")
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from ..config import SearchConfig
from ..types import Query, SearchResult
from .history_core import SearchHistoryEntry


@dataclass(slots=True)
class BookmarkFolder:
    """Organize bookmarks into folders."""

    name: str
    description: str | None = None
    created_time: float | None = None
    bookmarks: set[str] | None = None

    def __post_init__(self) -> None:
        if self.created_time is None:
            self.created_time = time.time()
        if self.bookmarks is None:
            self.bookmarks = set()


class BookmarkManager:
    """Bookmark organization and management."""

    def __init__(self, cfg: SearchConfig) -> None:
        self.cfg = cfg
        self.cache_dir = cfg.resolve_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bookmarks_file = self.cache_dir / "bookmarks.json"
        self.folders_file = self.cache_dir / "bookmark_folders.json"

        self._bookmarks: dict[str, SearchHistoryEntry] = {}
        self._folders: dict[str, BookmarkFolder] = {}
        self._loaded = False

    def load(self) -> None:
        """Load bookmarks and folders from disk."""
        if self._loaded:
            return

        # Load bookmarks
        if self.bookmarks_file.exists():
            try:
                data = json.loads(
                    self.bookmarks_file.read_text(encoding="utf-8"))
                bookmarks = data.get("bookmarks", {})
                for name, entry_data in bookmarks.items():
                    # Handle legacy entries
                    if "category" not in entry_data:
                        from .history_core import SearchCategory
                        entry_data["category"] = SearchCategory.GENERAL.value
                    if "languages" in entry_data and entry_data["languages"]:
                        entry_data["languages"] = set(entry_data["languages"])
                    if "tags" in entry_data and entry_data["tags"]:
                        entry_data["tags"] = set(entry_data["tags"])

                    entry = SearchHistoryEntry(**entry_data)
                    self._bookmarks[name] = entry
            except Exception:
                pass

        # Load bookmark folders
        if self.folders_file.exists():
            try:
                data = json.loads(
                    self.folders_file.read_text(encoding="utf-8"))
                folders = data.get("folders", {})
                for folder_name, folder_data in folders.items():
                    if "bookmarks" in folder_data and folder_data["bookmarks"]:
                        folder_data["bookmarks"] = set(
                            folder_data["bookmarks"])

                    folder = BookmarkFolder(**folder_data)
                    self._folders[folder_name] = folder
            except Exception:
                pass

        self._loaded = True

    def save_bookmarks(self) -> None:
        """Save bookmarks to disk."""
        try:
            data = {
                "version": 1,
                "last_updated": time.time(),
                "bookmarks": {name: asdict(entry) for name, entry in self._bookmarks.items()},
            }
            tmp_file = self.bookmarks_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(
                data, ensure_ascii=False), encoding="utf-8")
            tmp_file.replace(self.bookmarks_file)
        except Exception:
            pass

    def save_folders(self) -> None:
        """Save bookmark folders to disk."""
        try:
            # Convert sets to lists for JSON serialization
            folders_data = {}
            for folder_name, folder in self._folders.items():
                folder_dict = asdict(folder)
                if folder_dict.get("bookmarks"):
                    folder_dict["bookmarks"] = list(folder_dict["bookmarks"])
                folders_data[folder_name] = folder_dict

            data = {"version": 1, "last_updated": time.time(),
                    "folders": folders_data}
            tmp_file = self.folders_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(
                data, ensure_ascii=False), encoding="utf-8")
            tmp_file.replace(self.folders_file)
        except Exception:
            pass

    def add_bookmark(self, name: str, query: Query, result: SearchResult) -> None:
        """Add a search as a bookmark."""
        self.load()

        entry = SearchHistoryEntry(
            timestamp=time.time(),
            query_pattern=query.pattern,
            use_regex=query.use_regex,
            use_ast=query.use_ast,
            context=query.context,
            files_matched=result.stats.files_matched,
            items_count=result.stats.items,
            elapsed_ms=result.stats.elapsed_ms,
            filters=str(query.filters) if query.filters else None,
        )

        self._bookmarks[name] = entry
        self.save_bookmarks()

    def get_bookmarks(self) -> dict[str, SearchHistoryEntry]:
        """Get all bookmarks."""
        self.load()
        return dict(self._bookmarks)

    def remove_bookmark(self, name: str) -> bool:
        """Remove a bookmark by name."""
        self.load()
        if name in self._bookmarks:
            del self._bookmarks[name]
            
            # Remove from all folders
            for folder in self._folders.values():
                if folder.bookmarks and name in folder.bookmarks:
                    folder.bookmarks.remove(name)
            
            self.save_bookmarks()
            self.save_folders()
            return True
        return False

    def create_folder(self, name: str, description: str | None = None) -> bool:
        """Create a new bookmark folder."""
        self.load()
        if name in self._folders:
            return False

        self._folders[name] = BookmarkFolder(
            name=name, description=description)
        self.save_folders()
        return True

    def delete_folder(self, name: str) -> bool:
        """Delete a bookmark folder."""
        self.load()
        if name not in self._folders:
            return False

        del self._folders[name]
        self.save_folders()
        return True

    def add_bookmark_to_folder(self, bookmark_name: str, folder_name: str) -> bool:
        """Add a bookmark to a folder."""
        self.load()
        if bookmark_name not in self._bookmarks or folder_name not in self._folders:
            return False

        folder = self._folders[folder_name]
        if folder.bookmarks is None:
            folder.bookmarks = set()
        folder.bookmarks.add(bookmark_name)
        self.save_folders()
        return True

    def remove_bookmark_from_folder(self, bookmark_name: str, folder_name: str) -> bool:
        """Remove a bookmark from a folder."""
        self.load()
        if folder_name not in self._folders:
            return False

        folder = self._folders[folder_name]
        if folder.bookmarks and bookmark_name in folder.bookmarks:
            folder.bookmarks.remove(bookmark_name)
            self.save_folders()
            return True
        return False

    def get_folders(self) -> dict[str, BookmarkFolder]:
        """Get all bookmark folders."""
        self.load()
        return self._folders.copy()

    def get_bookmarks_in_folder(self, folder_name: str) -> list[SearchHistoryEntry]:
        """Get all bookmarks in a specific folder."""
        self.load()
        if folder_name not in self._folders:
            return []

        folder = self._folders[folder_name]
        if not folder.bookmarks:
            return []

        return [self._bookmarks[name] for name in folder.bookmarks if name in self._bookmarks]

    def search_bookmarks(self, pattern: str) -> list[tuple[str, SearchHistoryEntry]]:
        """Search bookmarks by name or query pattern."""
        self.load()
        pattern_lower = pattern.lower()
        results = []

        for name, entry in self._bookmarks.items():
            if (pattern_lower in name.lower() or 
                pattern_lower in entry.query_pattern.lower()):
                results.append((name, entry))

        return results

    def get_bookmark_stats(self) -> dict[str, Any]:
        """Get bookmark statistics."""
        self.load()
        return {
            "total_bookmarks": len(self._bookmarks),
            "total_folders": len(self._folders),
            "bookmarks_in_folders": sum(
                len(folder.bookmarks) if folder.bookmarks else 0 
                for folder in self._folders.values()
            ),
        }
