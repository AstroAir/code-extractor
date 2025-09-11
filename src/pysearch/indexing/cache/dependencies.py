"""
File dependency tracking for cache management.

This module provides functionality to track which cache entries depend on
specific files, enabling automatic invalidation when files change.

Classes:
    DependencyTracker: Manages file dependencies for cache entries

Features:
    - Thread-safe dependency tracking
    - Efficient dependency lookup
    - Automatic cleanup of orphaned dependencies
    - Bulk dependency operations
"""

from __future__ import annotations

import builtins
import threading
from typing import Set


class DependencyTracker:
    """
    Tracks file dependencies for cache entries.
    
    This class manages the mapping between files and cache entries that
    depend on them, enabling efficient invalidation when files change.
    """

    def __init__(self):
        # file_path -> set of cache_keys that depend on it
        self._file_dependencies: dict[str, set[str]] = {}
        self._dependency_lock = threading.RLock()

    def add_dependencies(self, cache_key: str, file_paths: builtins.set[str]) -> None:
        """
        Add file dependencies for a cache key.
        
        Args:
            cache_key: The cache key that depends on the files
            file_paths: Set of file paths the cache entry depends on
        """
        with self._dependency_lock:
            for file_path in file_paths:
                if file_path not in self._file_dependencies:
                    self._file_dependencies[file_path] = set()
                self._file_dependencies[file_path].add(cache_key)

    def remove_dependencies(self, cache_key: str) -> None:
        """
        Remove all file dependencies for a cache key.
        
        Args:
            cache_key: The cache key to remove dependencies for
        """
        with self._dependency_lock:
            files_to_remove = []
            for file_path, keys in self._file_dependencies.items():
                keys.discard(cache_key)
                if not keys:
                    files_to_remove.append(file_path)

            for file_path in files_to_remove:
                del self._file_dependencies[file_path]

    def get_dependent_keys(self, file_path: str) -> Set[str]:
        """
        Get all cache keys that depend on a specific file.
        
        Args:
            file_path: Path of the file to check
            
        Returns:
            Set of cache keys that depend on the file
        """
        with self._dependency_lock:
            return self._file_dependencies.get(file_path, set()).copy()

    def get_dependency_count(self) -> int:
        """
        Get the total number of file dependencies tracked.
        
        Returns:
            Number of files being tracked for dependencies
        """
        with self._dependency_lock:
            return len(self._file_dependencies)

    def clear_all_dependencies(self) -> None:
        """Clear all file dependencies."""
        with self._dependency_lock:
            self._file_dependencies.clear()

    def get_files_with_dependencies(self) -> Set[str]:
        """
        Get all files that have cache dependencies.
        
        Returns:
            Set of file paths that have cache entries depending on them
        """
        with self._dependency_lock:
            return set(self._file_dependencies.keys())

    def cleanup_empty_dependencies(self) -> int:
        """
        Remove file entries that have no dependent cache keys.
        
        Returns:
            Number of file entries removed
        """
        with self._dependency_lock:
            files_to_remove = []
            for file_path, keys in self._file_dependencies.items():
                if not keys:
                    files_to_remove.append(file_path)

            for file_path in files_to_remove:
                del self._file_dependencies[file_path]

            return len(files_to_remove)
