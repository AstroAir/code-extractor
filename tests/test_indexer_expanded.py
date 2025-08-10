"""
Expanded comprehensive tests for indexer module.

This module tests indexer functionality that is currently not covered,
including edge cases, error conditions, cache management, and advanced features.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pysearch.config import SearchConfig
from pysearch.indexer import Indexer, IndexRecord, CACHE_FILE
from pysearch.utils import file_meta, file_sha1


class TestIndexerInitialization:
    """Test Indexer initialization and configuration."""

    def test_default_initialization(self, tmp_path: Path) -> None:
        """Test Indexer initialization with default configuration."""
        config = SearchConfig(paths=[str(tmp_path)])
        indexer = Indexer(config)
        
        assert indexer.cfg == config
        assert indexer.cache_dir == config.resolve_cache_dir()
        assert indexer.cache_path == indexer.cache_dir / CACHE_FILE
        assert indexer._index == {}
        assert not indexer._loaded
        assert indexer._hot_cache == {}
        assert indexer._hot_cache_max_size == 100

    def test_cache_directory_creation(self, tmp_path: Path) -> None:
        """Test that cache directory is created if it doesn't exist."""
        cache_dir = tmp_path / "custom_cache"
        config = SearchConfig(paths=[str(tmp_path)], cache_dir=str(cache_dir))
        
        assert not cache_dir.exists()
        
        indexer = Indexer(config)
        
        assert cache_dir.exists()
        assert indexer.cache_dir == cache_dir

    def test_multiple_indexer_instances(self, tmp_path: Path) -> None:
        """Test multiple indexer instances with same configuration."""
        config = SearchConfig(paths=[str(tmp_path)])
        
        indexer1 = Indexer(config)
        indexer2 = Indexer(config)
        
        # Should have separate internal state
        assert indexer1._index is not indexer2._index
        assert indexer1._lock is not indexer2._lock
        assert indexer1._hot_cache is not indexer2._hot_cache


class TestIndexerCacheManagement:
    """Test Indexer cache loading and saving."""

    def test_load_empty_cache(self, tmp_path: Path) -> None:
        """Test loading when no cache file exists."""
        config = SearchConfig(paths=[str(tmp_path)])
        indexer = Indexer(config)
        
        indexer.load()
        
        assert indexer._loaded is True
        assert indexer._index == {}
        assert indexer._hot_cache == {}

    def test_load_valid_cache(self, tmp_path: Path) -> None:
        """Test loading valid cache file."""
        config = SearchConfig(paths=[str(tmp_path)])
        indexer = Indexer(config)
        
        # Create valid cache file
        cache_data = {
            "version": 1,
            "generated_at": time.time(),
            "files": {
                "test.py": {
                    "path": "test.py",
                    "size": 100,
                    "mtime": time.time(),
                    "sha1": "abc123",
                    "last_accessed": time.time(),
                    "access_count": 5
                }
            }
        }
        
        indexer.cache_path.parent.mkdir(parents=True, exist_ok=True)
        indexer.cache_path.write_text(json.dumps(cache_data), encoding="utf-8")
        
        indexer.load()
        
        assert indexer._loaded is True
        assert len(indexer._index) == 1
        assert "test.py" in indexer._index
        
        record = indexer._index["test.py"]
        assert record.path == "test.py"
        assert record.size == 100
        assert record.sha1 == "abc123"
        assert record.access_count == 5

    def test_load_corrupted_cache(self, tmp_path: Path) -> None:
        """Test loading corrupted cache file."""
        config = SearchConfig(paths=[str(tmp_path)])
        indexer = Indexer(config)
        
        # Create corrupted cache file
        indexer.cache_path.parent.mkdir(parents=True, exist_ok=True)
        indexer.cache_path.write_text("invalid json", encoding="utf-8")
        
        indexer.load()
        
        assert indexer._loaded is True
        assert indexer._index == {}

    def test_load_invalid_record_format(self, tmp_path: Path) -> None:
        """Test loading cache with invalid record format."""
        config = SearchConfig(paths=[str(tmp_path)])
        indexer = Indexer(config)
        
        # Create cache with invalid record (missing required fields)
        cache_data = {
            "version": 1,
            "files": {
                "invalid.py": {
                    "path": "invalid.py",
                    "size": 100,
                    # Missing mtime and sha1
                }
            }
        }
        
        indexer.cache_path.parent.mkdir(parents=True, exist_ok=True)
        indexer.cache_path.write_text(json.dumps(cache_data), encoding="utf-8")
        
        indexer.load()
        
        assert indexer._loaded is True
        assert indexer._index == {}  # Invalid records should be filtered out

    def test_save_cache(self, tmp_path: Path) -> None:
        """Test saving cache to file."""
        config = SearchConfig(paths=[str(tmp_path)])
        indexer = Indexer(config)
        
        # Add some records
        record = IndexRecord(
            path="test.py",
            size=100,
            mtime=time.time(),
            sha1="abc123",
            last_accessed=time.time(),
            access_count=1
        )
        indexer._index["test.py"] = record
        
        indexer.save()
        
        assert indexer.cache_path.exists()
        
        # Verify saved content
        saved_data = json.loads(indexer.cache_path.read_text(encoding="utf-8"))
        assert saved_data["version"] == 1
        assert "generated_at" in saved_data
        assert "files" in saved_data
        assert "test.py" in saved_data["files"]
        
        saved_record = saved_data["files"]["test.py"]
        assert saved_record["path"] == "test.py"
        assert saved_record["size"] == 100
        assert saved_record["sha1"] == "abc123"

    def test_save_atomic_write(self, tmp_path: Path) -> None:
        """Test that save uses atomic write (tmp file + rename)."""
        config = SearchConfig(paths=[str(tmp_path)])
        indexer = Indexer(config)
        
        record = IndexRecord(
            path="test.py",
            size=100,
            mtime=time.time(),
            sha1="abc123"
        )
        indexer._index["test.py"] = record
        
        # Mock os.replace to verify it's called
        with patch('os.replace') as mock_replace:
            indexer.save()
            
            # Should call os.replace with tmp file and final file
            mock_replace.assert_called_once()
            args = mock_replace.call_args[0]
            assert str(args[0]).endswith(".tmp")
            assert args[1] == indexer.cache_path

    def test_hot_cache_population(self, tmp_path: Path) -> None:
        """Test hot cache population from loaded data."""
        config = SearchConfig(paths=[str(tmp_path)])
        indexer = Indexer(config)
        
        # Create cache with multiple records with different access patterns
        current_time = time.time()
        cache_data = {
            "version": 1,
            "files": {
                "hot1.py": {
                    "path": "hot1.py",
                    "size": 100,
                    "mtime": current_time,
                    "sha1": "abc1",
                    "last_accessed": current_time - 100,  # Recent
                    "access_count": 10
                },
                "hot2.py": {
                    "path": "hot2.py",
                    "size": 200,
                    "mtime": current_time,
                    "sha1": "abc2",
                    "last_accessed": current_time - 50,   # More recent
                    "access_count": 5
                },
                "cold.py": {
                    "path": "cold.py",
                    "size": 300,
                    "mtime": current_time,
                    "sha1": "abc3",
                    "last_accessed": current_time - 1000,  # Old
                    "access_count": 1
                }
            }
        }
        
        indexer.cache_path.parent.mkdir(parents=True, exist_ok=True)
        indexer.cache_path.write_text(json.dumps(cache_data), encoding="utf-8")
        
        indexer.load()
        
        # Hot cache should contain most recently accessed files
        assert len(indexer._hot_cache) <= indexer._hot_cache_max_size
        # hot2.py should be in hot cache (most recent access)
        assert "hot2.py" in indexer._hot_cache


class TestIndexerFileScanning:
    """Test Indexer file scanning functionality."""

    def test_scan_new_files(self, tmp_path: Path) -> None:
        """Test scanning new files."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        indexer = Indexer(config)
        
        # Create test files
        test_file1 = tmp_path / "test1.py"
        test_file2 = tmp_path / "test2.py"
        test_file1.write_text("print('test1')")
        test_file2.write_text("print('test2')")
        
        changed, removed, total = indexer.scan()
        
        assert total == 2
        assert len(changed) == 2
        assert len(removed) == 0
        assert test_file1 in changed
        assert test_file2 in changed

    def test_scan_no_changes(self, tmp_path: Path) -> None:
        """Test scanning when no files have changed."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        indexer = Indexer(config)
        
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("print('test')")
        
        # First scan
        changed1, removed1, total1 = indexer.scan()
        assert len(changed1) == 1
        
        # Second scan without changes
        changed2, removed2, total2 = indexer.scan()
        assert len(changed2) == 0
        assert len(removed2) == 0
        assert total2 == 1

    def test_scan_modified_files(self, tmp_path: Path) -> None:
        """Test scanning modified files."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        indexer = Indexer(config)
        
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("print('test')")
        
        # First scan
        changed1, _, _ = indexer.scan()
        assert len(changed1) == 1
        
        # Modify file
        time.sleep(0.01)  # Ensure mtime changes
        test_file.write_text("print('modified')")
        
        # Second scan
        changed2, removed2, total2 = indexer.scan()
        assert len(changed2) == 1
        assert test_file in changed2
        assert len(removed2) == 0

    def test_scan_removed_files(self, tmp_path: Path) -> None:
        """Test scanning when files are removed."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        indexer = Indexer(config)
        
        # Create test files
        test_file1 = tmp_path / "test1.py"
        test_file2 = tmp_path / "test2.py"
        test_file1.write_text("print('test1')")
        test_file2.write_text("print('test2')")
        
        # First scan
        changed1, _, total1 = indexer.scan()
        assert total1 == 2
        assert len(changed1) == 2
        
        # Remove one file
        test_file1.unlink()
        
        # Second scan
        changed2, removed2, total2 = indexer.scan()
        assert total2 == 1
        assert len(changed2) == 0
        assert len(removed2) == 1
        assert test_file1 in removed2

    def test_scan_with_strict_hash_check(self, tmp_path: Path) -> None:
        """Test scanning with strict hash checking enabled."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], strict_hash_check=True)
        indexer = Indexer(config)
        
        # Create test file
        test_file = tmp_path / "test.py"
        original_content = "print('test')"
        test_file.write_text(original_content)
        
        # First scan
        changed1, _, _ = indexer.scan()
        assert len(changed1) == 1
        
        # Touch file (same content, different mtime)
        time.sleep(0.01)
        test_file.write_text(original_content)  # Same content
        
        # Second scan with strict checking
        changed2, _, _ = indexer.scan()
        # Should not be marked as changed since content is the same
        assert len(changed2) == 0

    def test_scan_error_handling(self, tmp_path: Path) -> None:
        """Test error handling during file scanning."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        indexer = Indexer(config)
        
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("print('test')")
        
        # Mock file_meta to raise exception
        with patch('pysearch.indexer.file_meta', side_effect=OSError("Permission denied")):
            changed, removed, total = indexer.scan()
            
            # Should handle error gracefully
            assert total >= 0
            assert isinstance(changed, list)
            assert isinstance(removed, list)


class TestIndexerUtilityMethods:
    """Test Indexer utility methods."""

    def test_rel_path_conversion(self, tmp_path: Path) -> None:
        """Test relative path conversion."""
        config = SearchConfig(paths=[str(tmp_path)])
        indexer = Indexer(config)

        # Test normal file
        test_file = tmp_path / "subdir" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("test")

        rel_path = indexer._rel(test_file)
        assert rel_path == "subdir/test.py" or rel_path == "subdir\\test.py"  # Handle Windows paths

    def test_rel_path_error_handling(self, tmp_path: Path) -> None:
        """Test relative path conversion error handling."""
        config = SearchConfig(paths=[str(tmp_path)])
        indexer = Indexer(config)

        # Test with path outside of base directory
        external_path = Path("/some/external/path.py")
        rel_path = indexer._rel(external_path)

        # Should return string representation when relative path fails
        assert isinstance(rel_path, str)

    def test_iter_all_paths(self, tmp_path: Path) -> None:
        """Test iterating all indexed paths."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        indexer = Indexer(config)

        # Create test files and scan
        test_file1 = tmp_path / "test1.py"
        test_file2 = tmp_path / "test2.py"
        test_file1.write_text("test1")
        test_file2.write_text("test2")

        indexer.scan()

        # Test iteration
        all_paths = list(indexer.iter_all_paths())
        assert len(all_paths) == 2
        assert test_file1 in all_paths
        assert test_file2 in all_paths

    def test_iter_all_paths_empty_index(self, tmp_path: Path) -> None:
        """Test iterating paths with empty index."""
        config = SearchConfig(paths=[str(tmp_path)])
        indexer = Indexer(config)

        all_paths = list(indexer.iter_all_paths())
        assert all_paths == []

    def test_count_indexed(self, tmp_path: Path) -> None:
        """Test counting indexed files."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        indexer = Indexer(config)

        # Initially should be 0
        assert indexer.count_indexed() == 0

        # Create test files and scan
        test_file1 = tmp_path / "test1.py"
        test_file2 = tmp_path / "test2.py"
        test_file1.write_text("test1")
        test_file2.write_text("test2")

        indexer.scan()

        # Should count indexed files
        assert indexer.count_indexed() == 2

    def test_get_cache_stats(self, tmp_path: Path) -> None:
        """Test getting cache statistics."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        indexer = Indexer(config)

        # Create test files and scan
        test_file = tmp_path / "test.py"
        test_file.write_text("test")

        indexer.scan()

        stats = indexer.get_cache_stats()

        assert "total_files" in stats
        assert "hot_cache_size" in stats
        assert "avg_access_count" in stats
        assert stats["total_files"] == 1
        assert isinstance(stats["hot_cache_size"], int)
        assert isinstance(stats["avg_access_count"], int)

    def test_update_access_stats(self, tmp_path: Path) -> None:
        """Test updating access statistics."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        indexer = Indexer(config)

        # Create test file and scan
        test_file = tmp_path / "test.py"
        test_file.write_text("test")

        indexer.scan()

        rel_path = indexer._rel(test_file)
        original_count = indexer._index[rel_path].access_count

        # Update access stats
        indexer._update_access_stats(rel_path)

        # Access count should increase
        assert indexer._index[rel_path].access_count == original_count + 1


class TestIndexerAdvancedFeatures:
    """Test Indexer advanced features."""

    def test_cleanup_old_entries(self, tmp_path: Path) -> None:
        """Test cleanup of old index entries."""
        config = SearchConfig(paths=[str(tmp_path)])
        indexer = Indexer(config)

        # Add old and new records
        old_time = time.time() - (40 * 24 * 60 * 60)  # 40 days ago
        new_time = time.time() - (10 * 24 * 60 * 60)  # 10 days ago

        old_record = IndexRecord(
            path="old.py",
            size=100,
            mtime=time.time(),
            sha1="old",
            last_accessed=old_time,
            access_count=2
        )

        new_record = IndexRecord(
            path="new.py",
            size=200,
            mtime=time.time(),
            sha1="new",
            last_accessed=new_time,
            access_count=3
        )

        indexer._index["old.py"] = old_record
        indexer._index["new.py"] = new_record

        # Cleanup entries older than 30 days
        removed_count = indexer.cleanup_old_entries(days_old=30)

        assert removed_count == 1
        assert "old.py" not in indexer._index
        assert "new.py" in indexer._index

    def test_cleanup_preserves_frequently_accessed(self, tmp_path: Path) -> None:
        """Test that cleanup preserves frequently accessed files."""
        config = SearchConfig(paths=[str(tmp_path)])
        indexer = Indexer(config)

        # Add old but frequently accessed record
        old_time = time.time() - (40 * 24 * 60 * 60)  # 40 days ago

        frequent_record = IndexRecord(
            path="frequent.py",
            size=100,
            mtime=time.time(),
            sha1="frequent",
            last_accessed=old_time,
            access_count=10  # High access count
        )

        indexer._index["frequent.py"] = frequent_record

        # Cleanup should preserve frequently accessed files
        removed_count = indexer.cleanup_old_entries(days_old=30)

        assert removed_count == 0
        assert "frequent.py" in indexer._index

    def test_hot_cache_management(self, tmp_path: Path) -> None:
        """Test hot cache management."""
        config = SearchConfig(paths=[str(tmp_path)])
        indexer = Indexer(config)
        indexer._hot_cache_max_size = 2  # Small size for testing

        # Add records to hot cache
        record1 = IndexRecord(path="test1.py", size=100, mtime=time.time(), sha1="1")
        record2 = IndexRecord(path="test2.py", size=200, mtime=time.time(), sha1="2")
        record3 = IndexRecord(path="test3.py", size=300, mtime=time.time(), sha1="3")

        indexer._hot_cache["test1.py"] = record1
        indexer._hot_cache["test2.py"] = record2

        # Hot cache should be at max size
        assert len(indexer._hot_cache) == 2

        # Cleanup should remove from hot cache too
        indexer._index["test1.py"] = record1
        indexer._index["test2.py"] = record2
        indexer._hot_cache["test1.py"] = record1
        indexer._hot_cache["test2.py"] = record2

        # Make test1.py old
        old_record = IndexRecord(
            path="test1.py",
            size=100,
            mtime=time.time(),
            sha1="1",
            last_accessed=time.time() - (40 * 24 * 60 * 60),
            access_count=1
        )
        indexer._index["test1.py"] = old_record
        indexer._hot_cache["test1.py"] = old_record

        removed_count = indexer.cleanup_old_entries(days_old=30)

        assert removed_count == 1
        assert "test1.py" not in indexer._hot_cache
        assert "test2.py" in indexer._hot_cache


class TestIndexerThreadSafety:
    """Test Indexer thread safety."""

    def test_concurrent_load_calls(self, tmp_path: Path) -> None:
        """Test concurrent load calls."""
        config = SearchConfig(paths=[str(tmp_path)])
        indexer = Indexer(config)

        # Create cache file
        cache_data = {
            "version": 1,
            "files": {
                "test.py": {
                    "path": "test.py",
                    "size": 100,
                    "mtime": time.time(),
                    "sha1": "abc123"
                }
            }
        }

        indexer.cache_path.parent.mkdir(parents=True, exist_ok=True)
        indexer.cache_path.write_text(json.dumps(cache_data), encoding="utf-8")

        def load_cache() -> None:
            indexer.load()

        # Run multiple load calls concurrently
        threads = []
        for _ in range(5):
            t = threading.Thread(target=load_cache)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should be loaded only once
        assert indexer._loaded is True
        assert len(indexer._index) == 1

    def test_concurrent_scan_calls(self, tmp_path: Path) -> None:
        """Test concurrent scan calls."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        indexer = Indexer(config)

        # Create test files
        for i in range(5):
            test_file = tmp_path / f"test{i}.py"
            test_file.write_text(f"print('test{i}')")

        results = []

        def scan_files() -> None:
            result = indexer.scan()
            results.append(result)

        # Run multiple scan calls concurrently
        threads = []
        for _ in range(3):
            t = threading.Thread(target=scan_files)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All scans should complete successfully
        assert len(results) == 3
        for changed, removed, total in results:
            assert isinstance(changed, list)
            assert isinstance(removed, list)
            assert isinstance(total, int)
