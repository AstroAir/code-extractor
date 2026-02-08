"""Tests for pysearch.indexing.cache.dependencies module."""

from __future__ import annotations

import pytest

from pysearch.indexing.cache.dependencies import DependencyTracker


class TestDependencyTracker:
    """Tests for DependencyTracker class."""

    def test_init(self):
        tracker = DependencyTracker()
        assert tracker.get_dependency_count() == 0

    def test_add_dependencies_single_file(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("cache_key_1", {"file_a.py"})
        keys = tracker.get_dependent_keys("file_a.py")
        assert "cache_key_1" in keys

    def test_add_dependencies_multiple_files(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("k1", {"a.py", "b.py"})
        assert "k1" in tracker.get_dependent_keys("a.py")
        assert "k1" in tracker.get_dependent_keys("b.py")

    def test_add_dependencies_multiple_keys_same_file(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("k1", {"a.py"})
        tracker.add_dependencies("k2", {"a.py"})
        tracker.add_dependencies("k3", {"b.py"})
        affected = tracker.get_dependent_keys("a.py")
        assert affected == {"k1", "k2"}
        assert "k3" not in affected

    def test_get_dependent_keys_nonexistent_file(self):
        tracker = DependencyTracker()
        keys = tracker.get_dependent_keys("nonexistent.py")
        assert keys == set()

    def test_get_dependent_keys_returns_copy(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("k1", {"a.py"})
        keys = tracker.get_dependent_keys("a.py")
        keys.add("k_injected")
        assert "k_injected" not in tracker.get_dependent_keys("a.py")

    def test_remove_dependencies(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("k1", {"a.py", "b.py"})
        tracker.remove_dependencies("k1")
        assert "k1" not in tracker.get_dependent_keys("a.py")
        assert "k1" not in tracker.get_dependent_keys("b.py")

    def test_remove_dependencies_cleans_empty_files(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("k1", {"a.py"})
        tracker.remove_dependencies("k1")
        # a.py had no other keys, so it should be removed from tracking
        assert tracker.get_dependency_count() == 0

    def test_remove_dependencies_preserves_other_keys(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("k1", {"a.py"})
        tracker.add_dependencies("k2", {"a.py"})
        tracker.remove_dependencies("k1")
        assert "k2" in tracker.get_dependent_keys("a.py")
        assert tracker.get_dependency_count() == 1

    def test_remove_dependencies_nonexistent_key(self):
        tracker = DependencyTracker()
        tracker.remove_dependencies("nonexistent")  # should not raise

    def test_get_dependency_count(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("k1", {"a.py", "b.py"})
        assert tracker.get_dependency_count() == 2

    def test_clear_all_dependencies(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("k1", {"a.py"})
        tracker.add_dependencies("k2", {"b.py"})
        tracker.clear_all_dependencies()
        assert tracker.get_dependency_count() == 0

    def test_get_files_with_dependencies(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("k1", {"a.py", "b.py"})
        tracker.add_dependencies("k2", {"c.py"})
        files = tracker.get_files_with_dependencies()
        assert files == {"a.py", "b.py", "c.py"}

    def test_get_files_with_dependencies_empty(self):
        tracker = DependencyTracker()
        assert tracker.get_files_with_dependencies() == set()

    def test_cleanup_empty_dependencies(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("k1", {"a.py", "b.py"})
        # Manually clear keys from a file to simulate orphan
        tracker._file_dependencies["a.py"].clear()
        removed = tracker.cleanup_empty_dependencies()
        assert removed == 1
        assert tracker.get_dependency_count() == 1

    def test_cleanup_empty_dependencies_nothing_to_clean(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("k1", {"a.py"})
        removed = tracker.cleanup_empty_dependencies()
        assert removed == 0
