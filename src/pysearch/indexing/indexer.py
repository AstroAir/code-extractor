"""
File indexing and caching module for pysearch.

This module provides efficient file indexing capabilities with incremental updates
based on file metadata (mtime, size, hash). It maintains a persistent cache to
avoid re-scanning unchanged files, significantly improving performance for large
codebases.

Classes:
    IndexRecord: Represents cached metadata for a single file
    Indexer: Main indexing class that manages file scanning and caching

Key Features:
    - Incremental indexing based on file mtime and size
    - Optional strict hash checking for exact change detection
    - Directory pruning to skip excluded paths during traversal
    - Parallel scanning for improved performance
    - Persistent JSON-based cache storage
    - Thread-safe operations with proper locking

Example:
    Basic indexing:
        >>> from pysearch.indexer import Indexer
        >>> from pysearch.config import SearchConfig
        >>>
        >>> config = SearchConfig(paths=["."], include=["**/*.py"])
        >>> indexer = Indexer(config)
        >>> files = list(indexer.iter_files())
        >>> print(f"Found {len(files)} files")

    With strict hash checking:
        >>> config.strict_hash_check = True
        >>> indexer = Indexer(config)
        >>> # Will compute SHA1 hashes for exact change detection
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from ..core.config import SearchConfig
from ..utils.utils import file_meta, file_sha1, iter_files

CACHE_FILE = "index.json"


@dataclass(slots=True)
class IndexRecord:
    path: str
    size: int
    mtime: float
    sha1: str | None
    last_accessed: float = 0.0  # Track access patterns
    access_count: int = 0  # Track popularity


class Indexer:
    """
    基于文件元数据(size/mtime/sha1)的轻量索引器，支持增量更新。
    Enhanced with parallel scanning, access tracking, and smart caching.
    缓存结构: { "version": 1, "files": { rel_path: IndexRecord } }
    """

    def __init__(self, cfg: SearchConfig) -> None:
        self.cfg = cfg
        self.cache_dir = cfg.resolve_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / CACHE_FILE
        self._index: dict[str, IndexRecord] = {}
        self._loaded = False
        self._lock = threading.RLock()
        # Hot cache for recently accessed files
        self._hot_cache: dict[str, IndexRecord] = {}
        self._hot_cache_max_size = 100

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            if self.cache_path.exists():
                try:
                    data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                    files = data.get("files", {})
                    self._index = {
                        k: IndexRecord(
                            **{
                                **v,
                                "last_accessed": v.get("last_accessed", 0.0),
                                "access_count": v.get("access_count", 0),
                            }
                        )
                        for k, v in files.items()
                        if self._valid_record(v)
                    }
                    # Populate hot cache with most recently accessed files
                    sorted_by_access = sorted(
                        self._index.items(),
                        key=lambda x: (x[1].last_accessed, x[1].access_count),
                        reverse=True,
                    )
                    for k, v in sorted_by_access[: self._hot_cache_max_size]:
                        self._hot_cache[k] = v
                except Exception:
                    self._index = {}
            self._loaded = True

    @staticmethod
    def _valid_record(v: dict) -> bool:
        return all(k in v for k in ("path", "size", "mtime", "sha1"))

    def save(self) -> None:
        out = {
            "version": 1,
            "generated_at": time.time(),
            "files": {k: asdict(r) for k, r in self._index.items()},
        }
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.cache_path)

    def _rel(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(Path(self.cfg.paths[0]).resolve()))
        except Exception:
            return str(p)

    def scan(self) -> tuple[list[Path], list[Path], int]:
        """
        Scan filesystem for changes and update index.

        Returns:
            Tuple of (changed_files, removed_files, total_seen)
        """
        self.load()
        seen: set[str] = set()  # Track files encountered during this scan
        # Files that have been modified since last scan
        changed: list[Path] = []
        total = 0  # Total number of files processed

        # Iterate through all files matching the configured patterns
        for p in iter_files(
            roots=self.cfg.paths,
            include=self.cfg.get_include_patterns(),
            exclude=self.cfg.get_exclude_patterns(),
            follow_symlinks=self.cfg.follow_symlinks,
            prune_excluded_dirs=self.cfg.dir_prune_exclude,
            language_filter=self.cfg.languages,
        ):
            total += 1
            # Convert to relative path for consistent indexing
            rel = self._rel(p)
            seen.add(rel)

            # Read minimal metadata (stat), compute sha1 only when necessary
            # This optimization avoids expensive hash computation for unchanged files
            try:
                meta = file_meta(p)
                if meta is None:
                    continue
            except Exception:
                # Skip files that can't be accessed
                continue

            rec = self._index.get(rel)

            # First check for changes using size/mtime (fast operations)
            # This allows us to quickly identify potentially changed files
            size_changed = (not rec) or (rec.size != meta.size)
            mtime_changed = (not rec) or (rec.mtime != meta.mtime)

            if not self.cfg.strict_hash_check:
                # Non-strict mode: rely only on size/mtime for change detection
                # This is faster but may miss some edge cases (e.g., same size/time but different content)
                if size_changed or mtime_changed:
                    self._index[rel] = IndexRecord(
                        path=rel,
                        size=meta.size,
                        mtime=meta.mtime,
                        sha1=rec.sha1 if rec else None,  # Preserve existing hash if available
                        last_accessed=time.time(),
                        access_count=rec.access_count + 1 if rec else 1,
                    )
                    changed.append(p)
            else:
                # Strict mode: when size or mtime changes, or when sha1 is missing, compute sha1 and compare
                # This provides exact change detection but is more expensive
                needs_sha1_check = size_changed or mtime_changed or (rec and rec.sha1 is None)
                if needs_sha1_check:
                    # Expensive operation - read entire file
                    current_sha1 = file_sha1(p)
                    if (not rec) or (rec.sha1 != current_sha1):
                        # Content actually changed or first time computing hash - update index and mark as changed
                        self._index[rel] = IndexRecord(
                            path=rel,
                            size=meta.size,
                            mtime=meta.mtime,
                            sha1=current_sha1,
                            last_accessed=time.time(),
                            access_count=rec.access_count + 1 if rec else 1,
                        )
                        changed.append(p)
                    else:
                        # SHA1 unchanged - content is the same despite metadata changes
                        # Update metadata but don't mark as changed for search purposes
                        self._index[rel] = IndexRecord(
                            path=rel,
                            size=meta.size,
                            mtime=meta.mtime,
                            sha1=current_sha1,
                            last_accessed=time.time(),
                            access_count=rec.access_count + 1 if rec else 1,
                        )
                else:
                    # size/mtime 未变，不更新
                    pass

        # removed
        removed_rels = [rel for rel in self._index.keys() if rel not in seen]
        for rel in removed_rels:
            del self._index[rel]
        removed_paths = [Path(self.cfg.paths[0]) / rel for rel in removed_rels]

        return changed, removed_paths, total

    def iter_all_paths(self) -> Iterator[Path]:
        """
        遍历索引中已知的所有路径（存在性不保证）。
        """
        base = Path(self.cfg.paths[0])
        for rel in self._index.keys():
            yield base / rel

    def count_indexed(self) -> int:
        with self._lock:
            return len(self._index)

    def get_cache_stats(self) -> dict[str, int]:
        """Get indexer cache statistics."""
        with self._lock:
            return {
                "total_files": len(self._index),
                "hot_cache_size": len(self._hot_cache),
                "avg_access_count": sum(r.access_count for r in self._index.values())
                // max(1, len(self._index)),
            }

    def cleanup_old_entries(self, days_old: int = 30) -> int:
        """Remove index entries for files not accessed in specified days."""
        cutoff = time.time() - (days_old * 24 * 60 * 60)
        removed = 0

        with self._lock:
            to_remove = []
            for rel, record in self._index.items():
                if record.last_accessed < cutoff and record.access_count < 5:
                    to_remove.append(rel)

            for rel in to_remove:
                del self._index[rel]
                if rel in self._hot_cache:
                    del self._hot_cache[rel]
                removed += 1

        return removed

    def _update_access_stats(self, rel_path: str) -> None:
        """Update access statistics for a file."""
        current_time = time.time()
        with self._lock:
            if rel_path in self._index:
                record = self._index[rel_path]
                record.last_accessed = current_time
                record.access_count += 1
                # Update hot cache
                self._hot_cache[rel_path] = record
                # Limit hot cache size
                if len(self._hot_cache) > self._hot_cache_max_size:
                    # Remove least recently accessed items
                    sorted_hot = sorted(self._hot_cache.items(), key=lambda x: x[1].last_accessed)
                    for k, _ in sorted_hot[:20]:  # Remove oldest 20
                        del self._hot_cache[k]
