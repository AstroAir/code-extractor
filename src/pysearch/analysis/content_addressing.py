"""
Content-addressed caching system for enhanced code indexing.

This module implements SHA256-based content addressing similar to Continue's approach,
providing efficient incremental updates and cross-branch caching capabilities.

Classes:
    ContentAddress: Represents content-addressed file metadata
    IndexTag: Tag system for managing index versions
    GlobalCacheManager: Cross-branch content cache
    ContentAddressedIndexer: Enhanced indexer with content addressing

Features:
    - SHA256-based content addressing for exact change detection
    - Tag-based index management (directory + branch + artifact)
    - Global cache for cross-branch content sharing
    - Incremental update optimization
    - Memory-efficient batch processing
    - Thread-safe operations

Example:
    Basic content addressing:
        >>> from pysearch.content_addressing import ContentAddressedIndexer
        >>> from pysearch.config import SearchConfig
        >>>
        >>> config = SearchConfig(paths=["./src"])
        >>> indexer = ContentAddressedIndexer(config)
        >>> await indexer.refresh_index()

    Advanced usage with tags:
        >>> from pysearch.content_addressing import IndexTag
        >>> tag = IndexTag(
        ...     directory="/path/to/repo",
        ...     branch="main",
        ...     artifact_id="code_snippets"
        ... )
        >>> results = await indexer.get_refresh_results(tag)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.config import SearchConfig
from ..core.types import Language
from ..utils.error_handling import ErrorCollector
from ..utils.logging_config import get_logger
from ..utils.utils import file_meta, read_text_safely

logger = get_logger()


@dataclass(frozen=True)
class ContentAddress:
    """Content-addressed file metadata using SHA256 hashing."""

    path: str
    content_hash: str  # SHA256 of file contents
    size: int
    mtime: float
    language: Language = Language.UNKNOWN

    @classmethod
    async def from_file(cls, path: str) -> ContentAddress:
        """Create ContentAddress from file path."""
        try:
            content = read_text_safely(Path(path))
            if content is None:
                raise ValueError(f"Cannot read file content: {path}")
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            meta = file_meta(Path(path))
            if meta is None:
                raise ValueError(f"Cannot read file metadata: {path}")

            from .language_detection import detect_language

            language = detect_language(Path(path), content)

            return cls(
                path=path,
                content_hash=content_hash,
                size=meta.size,
                mtime=meta.mtime,
                language=language,
            )
        except Exception as e:
            logger.error(f"Error creating ContentAddress for {path}: {e}")
            raise


@dataclass(frozen=True)
class IndexTag:
    """Tag system for managing index versions across branches and artifacts."""

    directory: str
    branch: str
    artifact_id: str

    def to_string(self) -> str:
        """Convert tag to string representation."""
        return f"{self.directory}::{self.branch}::{self.artifact_id}"

    @classmethod
    def from_string(cls, tag_string: str) -> IndexTag:
        """Create IndexTag from string representation."""
        parts = tag_string.split("::")
        if len(parts) != 3:
            raise ValueError(f"Invalid tag string format: {tag_string}")
        return cls(directory=parts[0], branch=parts[1], artifact_id=parts[2])


@dataclass
class PathAndCacheKey:
    """Path and cache key pair for index operations."""

    path: str
    cache_key: str  # SHA256 content hash


@dataclass
class RefreshIndexResults:
    """Results of index refresh operation."""

    compute: list[PathAndCacheKey]  # New files to index
    delete: list[PathAndCacheKey]  # Files to remove completely
    add_tag: list[PathAndCacheKey]  # Existing content, new tag
    remove_tag: list[PathAndCacheKey]  # Remove tag, keep content


@dataclass
class IndexingProgressUpdate:
    """Progress update for indexing operations."""

    progress: float  # 0.0 to 1.0
    description: str
    status: str  # "loading", "indexing", "done", "failed", "paused", "cancelled"
    warnings: list[str] | None = None
    debug_info: str | None = None


# Type aliases
MarkCompleteCallback = Callable[[list[PathAndCacheKey], str], None]


class GlobalCacheManager:
    """
    Cross-branch content cache to avoid duplicate indexing work.

    This cache stores indexed content by content hash, allowing the same
    content to be reused across different branches and repositories.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_dir / "global_cache.db"
        self._connection: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def _get_connection(self) -> sqlite3.Connection:
        """Get database connection, creating tables if needed."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA busy_timeout=3000")
            await self._create_tables()
        return self._connection

    async def _create_tables(self) -> None:
        """Create global cache tables."""
        conn = self._connection
        if conn is None:
            raise RuntimeError("Database connection not established before creating tables")

        # Global cache table - stores content by hash
        conn.execute("""
            CREATE TABLE IF NOT EXISTS global_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                content_data BLOB NOT NULL,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                UNIQUE(content_hash, artifact_id)
            )
        """)

        # Tag associations table - tracks which tags use which content
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(content_hash, artifact_id, tag),
                FOREIGN KEY (content_hash, artifact_id)
                    REFERENCES global_cache (content_hash, artifact_id)
            )
        """)

        # Create indexes for performance
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_global_cache_hash_artifact
            ON global_cache(content_hash, artifact_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_tags_hash_artifact
            ON cache_tags(content_hash, artifact_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_tags_tag
            ON cache_tags(tag)
        """)

        conn.commit()

    async def get_cached_content(
        self,
        content_hash: str,
        artifact_id: str,
    ) -> Any | None:
        """Get cached content by hash and artifact type."""
        async with self._lock:
            try:
                conn = await self._get_connection()
                cursor = conn.execute(
                    """
                    SELECT content_data FROM global_cache
                    WHERE content_hash = ? AND artifact_id = ?
                """,
                    (content_hash, artifact_id),
                )

                row = cursor.fetchone()
                if row:
                    # Update access statistics
                    conn.execute(
                        """
                        UPDATE global_cache
                        SET last_accessed = ?, access_count = access_count + 1
                        WHERE content_hash = ? AND artifact_id = ?
                    """,
                        (time.time(), content_hash, artifact_id),
                    )
                    conn.commit()

                    return json.loads(row[0])
                return None
            except Exception as e:
                logger.error(f"Error getting cached content: {e}")
                return None

    async def store_cached_content(
        self,
        content_hash: str,
        artifact_id: str,
        content: Any,
        tags: list[IndexTag],
    ) -> None:
        """Store content in global cache with associated tags."""
        async with self._lock:
            try:
                conn = await self._get_connection()
                current_time = time.time()

                # Store content
                conn.execute(
                    """
                    INSERT OR REPLACE INTO global_cache
                    (content_hash, artifact_id, content_data, created_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (content_hash, artifact_id, json.dumps(content), current_time, current_time),
                )

                # Store tag associations
                for tag in tags:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO cache_tags
                        (content_hash, artifact_id, tag, created_at)
                        VALUES (?, ?, ?, ?)
                    """,
                        (content_hash, artifact_id, tag.to_string(), current_time),
                    )

                conn.commit()
            except Exception as e:
                logger.error(f"Error storing cached content: {e}")
                raise

    async def get_tags_for_content(
        self,
        content_hash: str,
        artifact_id: str,
    ) -> list[IndexTag]:
        """Get all tags associated with content."""
        async with self._lock:
            try:
                conn = await self._get_connection()
                cursor = conn.execute(
                    """
                    SELECT tag FROM cache_tags
                    WHERE content_hash = ? AND artifact_id = ?
                """,
                    (content_hash, artifact_id),
                )

                rows = cursor.fetchall()
                return [IndexTag.from_string(row[0]) for row in rows]
            except Exception as e:
                logger.error(f"Error getting tags for content: {e}")
                return []

    async def remove_tag(
        self,
        content_hash: str,
        artifact_id: str,
        tag: IndexTag,
    ) -> bool:
        """Remove tag association, return True if content should be deleted."""
        async with self._lock:
            try:
                conn = await self._get_connection()

                # Remove tag association
                conn.execute(
                    """
                    DELETE FROM cache_tags
                    WHERE content_hash = ? AND artifact_id = ? AND tag = ?
                """,
                    (content_hash, artifact_id, tag.to_string()),
                )

                # Check if any tags remain
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM cache_tags
                    WHERE content_hash = ? AND artifact_id = ?
                """,
                    (content_hash, artifact_id),
                )

                count = cursor.fetchone()[0]

                # If no tags remain, delete content
                if count == 0:
                    conn.execute(
                        """
                        DELETE FROM global_cache
                        WHERE content_hash = ? AND artifact_id = ?
                    """,
                        (content_hash, artifact_id),
                    )
                    conn.commit()
                    return True

                conn.commit()
                return False
            except Exception as e:
                logger.error(f"Error removing tag: {e}")
                return False

    async def cleanup_orphaned_content(self) -> int:
        """Remove content with no associated tags."""
        async with self._lock:
            try:
                conn = await self._get_connection()

                # Find orphaned content
                cursor = conn.execute("""
                    SELECT gc.content_hash, gc.artifact_id
                    FROM global_cache gc
                    LEFT JOIN cache_tags ct ON gc.content_hash = ct.content_hash
                        AND gc.artifact_id = ct.artifact_id
                    WHERE ct.content_hash IS NULL
                """)

                orphaned = cursor.fetchall()

                # Delete orphaned content
                for content_hash, artifact_id in orphaned:
                    conn.execute(
                        """
                        DELETE FROM global_cache
                        WHERE content_hash = ? AND artifact_id = ?
                    """,
                        (content_hash, artifact_id),
                    )

                conn.commit()
                return len(orphaned)
            except Exception as e:
                logger.error(f"Error cleaning up orphaned content: {e}")
                return 0


class ContentAddressedIndexer:
    """
    Enhanced indexer with content addressing and tag-based management.

    This indexer builds upon the existing pysearch indexer with Continue's
    content addressing approach for better incremental updates and caching.
    """

    def __init__(self, config: SearchConfig):
        self.config = config
        self.cache_dir = config.resolve_cache_dir()
        self.global_cache = GlobalCacheManager(self.cache_dir)
        self.error_collector = ErrorCollector()

        # Database for tag catalog
        self.db_path = self.cache_dir / "tag_catalog.db"
        self._connection: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def _get_connection(self) -> sqlite3.Connection:
        """Get database connection, creating tables if needed."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA busy_timeout=3000")
            await self._create_tables()
        return self._connection

    async def _create_tables(self) -> None:
        """Create tag catalog tables."""
        conn = self._connection
        if conn is None:
            raise RuntimeError("Database connection not established before creating tables")

        # Tag catalog - tracks what's indexed for each tag
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tag_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                directory TEXT NOT NULL,
                branch TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                last_updated REAL NOT NULL,
                UNIQUE(directory, branch, artifact_id, path)
            )
        """)

        # Create indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tag_catalog_tag
            ON tag_catalog(directory, branch, artifact_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tag_catalog_hash
            ON tag_catalog(content_hash)
        """)

        conn.commit()

    async def get_saved_items_for_tag(self, tag: IndexTag) -> list[dict[str, Any]]:
        """Get all items currently indexed for a tag."""
        async with self._lock:
            try:
                conn = await self._get_connection()
                cursor = conn.execute(
                    """
                    SELECT path, content_hash, last_updated
                    FROM tag_catalog
                    WHERE directory = ? AND branch = ? AND artifact_id = ?
                """,
                    (tag.directory, tag.branch, tag.artifact_id),
                )

                return [
                    {"path": row[0], "content_hash": row[1], "last_updated": row[2]}
                    for row in cursor.fetchall()
                ]
            except Exception as e:
                logger.error(f"Error getting saved items for tag: {e}")
                return []

    async def calculate_refresh_results(
        self,
        tag: IndexTag,
        current_files: dict[str, Any],
        read_file: Callable[[str], str],
    ) -> RefreshIndexResults:
        """
        Calculate what needs to be updated in the index.

        This implements Continue's logic for determining compute/delete/addTag/removeTag
        operations based on current file state vs indexed state.
        """
        saved_items = await self.get_saved_items_for_tag(tag)

        # Group saved items by path for efficient lookup
        saved_by_path: dict[str, list[dict[str, Any]]] = {}
        for item in saved_items:
            path = item["path"]
            if path not in saved_by_path:
                saved_by_path[path] = []
            saved_by_path[path].append(item)

        # Find latest version for each path
        saved_latest = {}
        for path, versions in saved_by_path.items():
            latest = max(versions, key=lambda x: x["last_updated"])
            saved_latest[path] = latest

        compute = []
        delete = []
        add_tag = []
        remove_tag = []

        # Process current files
        for path, file_stats in current_files.items():
            if path in saved_latest:
                saved_item = saved_latest[path]

                # Check if file was modified
                if file_stats["mtime"] > saved_item["last_updated"]:
                    # File was modified, check if content actually changed
                    try:
                        content = read_file(path)
                        current_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                        if current_hash != saved_item["content_hash"]:
                            # Content changed - need to compute new index
                            # First remove old versions
                            for version in saved_by_path[path]:
                                remove_tag.append(
                                    PathAndCacheKey(path=path, cache_key=version["content_hash"])
                                )

                            # Check if new content exists in global cache
                            cached_content = await self.global_cache.get_cached_content(
                                current_hash, tag.artifact_id
                            )

                            if cached_content:
                                add_tag.append(PathAndCacheKey(path=path, cache_key=current_hash))
                            else:
                                compute.append(PathAndCacheKey(path=path, cache_key=current_hash))
                        # If content unchanged, no action needed
                    except Exception as e:
                        logger.error(f"Error processing file {path}: {e}")
                        self.error_collector.add_error(e, file_path=Path(path))
            else:
                # New file
                try:
                    content = read_file(path)
                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                    # Check if content exists in global cache
                    cached_content = await self.global_cache.get_cached_content(
                        content_hash, tag.artifact_id
                    )

                    if cached_content:
                        add_tag.append(PathAndCacheKey(path=path, cache_key=content_hash))
                    else:
                        compute.append(PathAndCacheKey(path=path, cache_key=content_hash))
                except Exception as e:
                    logger.error(f"Error processing new file {path}: {e}")
                    self.error_collector.add_error(e, file_path=Path(path))

        # Find deleted files
        current_paths = set(current_files.keys())
        for path in saved_latest:
            if path not in current_paths:
                # File was deleted
                for version in saved_by_path[path]:
                    # Check if other tags use this content
                    tags = await self.global_cache.get_tags_for_content(
                        version["content_hash"], tag.artifact_id
                    )

                    if len(tags) > 1:
                        remove_tag.append(
                            PathAndCacheKey(path=path, cache_key=version["content_hash"])
                        )
                    else:
                        delete.append(PathAndCacheKey(path=path, cache_key=version["content_hash"]))

        return RefreshIndexResults(
            compute=compute, delete=delete, add_tag=add_tag, remove_tag=remove_tag
        )

    async def mark_complete(
        self,
        items: list[PathAndCacheKey],
        result_type: str,
        tag: IndexTag,
    ) -> None:
        """Mark items as completed in the tag catalog."""
        async with self._lock:
            try:
                conn = await self._get_connection()
                current_time = time.time()

                for item in items:
                    if result_type == "compute" or result_type == "add_tag":
                        # Add or update in tag catalog
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO tag_catalog
                            (directory, branch, artifact_id, path, content_hash, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """,
                            (
                                tag.directory,
                                tag.branch,
                                tag.artifact_id,
                                item.path,
                                item.cache_key,
                                current_time,
                            ),
                        )
                    elif result_type == "delete" or result_type == "remove_tag":
                        # Remove from tag catalog
                        conn.execute(
                            """
                            DELETE FROM tag_catalog
                            WHERE directory = ? AND branch = ? AND artifact_id = ?
                                AND path = ? AND content_hash = ?
                        """,
                            (tag.directory, tag.branch, tag.artifact_id, item.path, item.cache_key),
                        )

                conn.commit()
            except Exception as e:
                logger.error(f"Error marking items complete: {e}")
                raise
