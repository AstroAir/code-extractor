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
from ..utils.helpers import file_meta, file_sha1, iter_files

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

    # Fields accepted by IndexRecord constructor
    _RECORD_FIELDS = frozenset(IndexRecord.__dataclass_fields__.keys())

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
                                fk: fv
                                for fk, fv in {
                                    **v,
                                    "last_accessed": v.get("last_accessed", 0.0),
                                    "access_count": v.get("access_count", 0),
                                }.items()
                                if fk in self._RECORD_FIELDS
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
        with self._lock:
            out = {
                "version": 1,
                "generated_at": time.time(),
                "files": {k: asdict(r) for k, r in self._index.items()},
            }
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.cache_path)

    def _rel(self, p: Path) -> str:
        """Convert an absolute path to a relative path based on the best-matching configured root."""
        resolved = p.resolve()
        # Try each configured root to find the best match
        for root in self.cfg.paths:
            try:
                return str(resolved.relative_to(Path(root).resolve()))
            except ValueError:
                continue
        # Fallback: use absolute path string as key
        return str(resolved)

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

            with self._lock:
                rec = self._index.get(rel)

                # First check for changes using size/mtime (fast operations)
                # This allows us to quickly identify potentially changed files
                size_changed = (not rec) or (rec.size != meta.size)
                mtime_changed = (not rec) or (rec.mtime != meta.mtime)

                if not self.cfg.strict_hash_check:
                    # Non-strict mode: rely only on size/mtime for change detection
                    if size_changed or mtime_changed:
                        new_rec = IndexRecord(
                            path=rel,
                            size=meta.size,
                            mtime=meta.mtime,
                            sha1=rec.sha1 if rec else None,
                            last_accessed=time.time(),
                            access_count=rec.access_count + 1 if rec else 1,
                        )
                        self._index[rel] = new_rec
                        self._hot_cache[rel] = new_rec
                        changed.append(p)
                else:
                    # Strict mode: compute sha1 when metadata changes or sha1 missing
                    needs_sha1_check = size_changed or mtime_changed or (rec and rec.sha1 is None)
                    if needs_sha1_check:
                        current_sha1 = file_sha1(p)
                        new_rec = IndexRecord(
                            path=rel,
                            size=meta.size,
                            mtime=meta.mtime,
                            sha1=current_sha1,
                            last_accessed=time.time(),
                            access_count=rec.access_count + 1 if rec else 1,
                        )
                        self._index[rel] = new_rec
                        self._hot_cache[rel] = new_rec
                        if (not rec) or (rec.sha1 != current_sha1):
                            changed.append(p)

        # Remove entries for files no longer present on disk
        with self._lock:
            removed_rels = [rel for rel in self._index if rel not in seen]
            for rel in removed_rels:
                del self._index[rel]
                self._hot_cache.pop(rel, None)

            # Trim hot cache if it exceeded max size
            if len(self._hot_cache) > self._hot_cache_max_size:
                sorted_hot = sorted(
                    self._hot_cache.items(), key=lambda x: x[1].last_accessed
                )
                for k, _ in sorted_hot[: len(self._hot_cache) - self._hot_cache_max_size]:
                    del self._hot_cache[k]

        removed_paths = [self._resolve_rel_to_abs(rel) for rel in removed_rels]

        return changed, removed_paths, total

    def _resolve_rel_to_abs(self, rel: str) -> Path:
        """Resolve a relative index key back to an absolute path.

        Tries each configured root and returns the first that exists on disk.
        Falls back to joining with the first root if nothing is found (the
        relative key may already be absolute if _rel() used the fallback).
        """
        rel_path = Path(rel)
        # If the key is already absolute (fallback from _rel), return directly
        if rel_path.is_absolute():
            return rel_path
        for root in self.cfg.paths:
            candidate = Path(root).resolve() / rel
            if candidate.exists():
                return candidate
        # Fallback: use the first root
        return Path(self.cfg.paths[0]).resolve() / rel

    def iter_all_paths(self) -> Iterator[Path]:
        """
        遍历索引中已知的所有路径（存在性不保证）。
        """
        self.load()
        with self._lock:
            rels = list(self._index.keys())
        for rel in rels:
            self._update_access_stats(rel)
            yield self._resolve_rel_to_abs(rel)

    def count_indexed(self) -> int:
        self.load()
        with self._lock:
            return len(self._index)

    def get_cache_stats(self) -> dict[str, int]:
        """Get indexer cache statistics."""
        self.load()
        with self._lock:
            return {
                "total_files": len(self._index),
                "hot_cache_size": len(self._hot_cache),
                "avg_access_count": sum(r.access_count for r in self._index.values())
                // max(1, len(self._index)),
            }

    def clear(self) -> None:
        """Clear all index data (in-memory and on-disk cache)."""
        with self._lock:
            self._index.clear()
            self._hot_cache.clear()
            self._loaded = False
            if self.cache_path.exists():
                try:
                    self.cache_path.unlink()
                except Exception:
                    pass

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

    def update_file(self, path: Path) -> bool:
        """Incrementally update the index for a single file.

        Reads the file's current metadata and updates (or creates) its index
        record without rescanning the entire file tree.

        Args:
            path: Absolute or relative path to the file.

        Returns:
            True if the index was updated, False if the file could not be read.
        """
        self.load()
        rel = self._rel(path)

        try:
            meta = file_meta(path)
            if meta is None:
                return False
        except Exception:
            return False

        rec = self._index.get(rel)
        current_time = time.time()

        if not self.cfg.strict_hash_check:
            self._index[rel] = IndexRecord(
                path=rel,
                size=meta.size,
                mtime=meta.mtime,
                sha1=rec.sha1 if rec else None,
                last_accessed=current_time,
                access_count=rec.access_count + 1 if rec else 1,
            )
        else:
            current_sha1 = file_sha1(path)
            self._index[rel] = IndexRecord(
                path=rel,
                size=meta.size,
                mtime=meta.mtime,
                sha1=current_sha1,
                last_accessed=current_time,
                access_count=rec.access_count + 1 if rec else 1,
            )

        # Update hot cache
        with self._lock:
            self._hot_cache[rel] = self._index[rel]
            if len(self._hot_cache) > self._hot_cache_max_size:
                sorted_hot = sorted(self._hot_cache.items(), key=lambda x: x[1].last_accessed)
                for k, _ in sorted_hot[:20]:
                    del self._hot_cache[k]

        return True

    def remove_file(self, path: Path) -> bool:
        """Remove a file from the index.

        Args:
            path: Absolute or relative path to the file.

        Returns:
            True if the file was removed from the index, False if it wasn't indexed.
        """
        self.load()
        rel = self._rel(path)

        with self._lock:
            if rel in self._index:
                del self._index[rel]
                self._hot_cache.pop(rel, None)
                return True

        return False

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
