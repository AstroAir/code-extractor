"""Tests for pysearch.indexing.cache.dependencies module."""

from __future__ import annotations

import pytest

from pysearch.indexing.cache.dependencies import DependencyTracker


class TestDependencyTracker:
    """Tests for DependencyTracker class."""

    def test_init(self):
        tracker = DependencyTracker()
        assert tracker is not None

    def test_add_dependencies(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("cache_key_1", {"file_a.py"})
        keys = tracker.get_dependent_keys("file_a.py")
        assert "cache_key_1" in keys

    def test_add_multiple_files(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("k1", {"a.py", "b.py"})
        keys_a = tracker.get_dependent_keys("a.py")
        keys_b = tracker.get_dependent_keys("b.py")
        assert "k1" in keys_a
        assert "k1" in keys_b

    def test_get_dependent_keys(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("k1", {"a.py"})
        tracker.add_dependencies("k2", {"a.py"})
        tracker.add_dependencies("k3", {"b.py"})
        affected = tracker.get_dependent_keys("a.py")
        assert "k1" in affected
        assert "k2" in affected
        assert "k3" not in affected

    def test_remove_dependencies(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("k1", {"a.py"})
        tracker.remove_dependencies("k1")
        keys = tracker.get_dependent_keys("a.py")
        assert "k1" not in keys

    def test_get_dependency_count(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("k1", {"a.py", "b.py"})
        count = tracker.get_dependency_count()
        assert count >= 1

    def test_clear_all(self):
        tracker = DependencyTracker()
        tracker.add_dependencies("k1", {"a.py"})
        tracker.clear_all_dependencies()
        assert tracker.get_dependency_count() == 0
