"""
Enhanced full-text search index implementation.

This module implements an enhanced version of Continue's FullTextSearchCodebaseIndex
with improved ranking, filtering, and multi-language support using SQLite FTS5.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from ...analysis.content_addressing import (
    IndexingProgressUpdate,
    IndexTag,
    MarkCompleteCallback,
    PathAndCacheKey,  # noqa: F401
    RefreshIndexResults,
)
from ...analysis.language_detection import detect_language
from ...utils.logging_config import get_logger
from ...utils.helpers import read_text_safely
from ..advanced.base import CodebaseIndex

logger = get_logger()


class FullTextIndex(CodebaseIndex):
    """
    Enhanced full-text search index using SQLite FTS5.

    This index provides fast full-text search across all code content
    with improved ranking, filtering, and language-aware tokenization.
    """

    @property
    def artifact_id(self) -> str:
        return "enhanced_full_text"

    @property
    def relative_expected_time(self) -> float:
        return 1.0

    def __init__(self, config: Any) -> None:
        self.config = config
        self.cache_dir = config.resolve_cache_dir()
        self.db_path = self.cache_dir / "full_text.db"
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
        """Create full-text search tables."""
        conn = await self._get_connection()

        # FTS5 virtual table for content search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_content USING fts5(
                path,
                content,
                language,
                file_type,
                tokenize='trigram'
            )
        """)

        # Metadata table for additional information
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fts_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                language TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                line_count INTEGER NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(path, content_hash)
            )
        """)

        # Tags table for multi-branch support
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fts_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metadata_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(metadata_id, tag),
                FOREIGN KEY (metadata_id) REFERENCES fts_metadata (id)
            )
        """)

        # Create indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fts_metadata_path_hash
            ON fts_metadata(path, content_hash)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fts_tags_tag
            ON fts_tags(tag)
        """)

        conn.commit()

    async def update(
        self,
        tag: IndexTag,
        results: RefreshIndexResults,
        mark_complete: MarkCompleteCallback,
        repo_name: str | None = None,
    ) -> AsyncGenerator[IndexingProgressUpdate, None]:
        """Update the full-text search index."""
        conn = await self._get_connection()
        tag_string = tag.to_string()

        total_operations = (
            len(results.compute)
            + len(results.delete)
            + len(results.add_tag)
            + len(results.remove_tag)
        )
        completed_operations = 0

        # Process compute operations (new files)
        for item in results.compute:
            yield IndexingProgressUpdate(
                progress=completed_operations / max(total_operations, 1),
                description=f"Indexing content of {Path(item.path).name}",
                status="indexing",
            )

            try:
                # Read file content
                content = read_text_safely(Path(item.path))
                if not content:
                    continue

                # Detect language and file type
                language = detect_language(Path(item.path), content)
                file_type = Path(item.path).suffix.lower()

                # Insert into FTS table
                conn.execute(
                    """
                    INSERT OR REPLACE INTO fts_content (path, content, language, file_type)
                    VALUES (?, ?, ?, ?)
                """,
                    (item.path, content, language.value, file_type),
                )

                # Insert metadata
                line_count = len(content.split("\n"))
                file_size = len(content.encode("utf-8"))
                current_time = time.time()

                cursor = conn.execute(
                    """
                    INSERT OR REPLACE INTO fts_metadata
                    (path, content_hash, language, file_size, line_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        item.path,
                        item.cache_key,
                        language.value,
                        file_size,
                        line_count,
                        current_time,
                    ),
                )

                metadata_id = cursor.lastrowid

                # Add tag association
                conn.execute(
                    """
                    INSERT OR REPLACE INTO fts_tags (metadata_id, tag, created_at)
                    VALUES (?, ?, ?)
                """,
                    (metadata_id, tag_string, current_time),
                )

                conn.commit()
                mark_complete([item], "compute")
                completed_operations += 1

            except Exception as e:
                logger.error(f"Error processing file {item.path}: {e}")
                completed_operations += 1

        # Process add_tag operations (existing content, new tag)
        for item in results.add_tag:
            try:
                cursor = conn.execute(
                    "SELECT id FROM fts_metadata WHERE path = ? AND content_hash = ?",
                    (item.path, item.cache_key),
                )
                for row in cursor.fetchall():
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO fts_tags (metadata_id, tag, created_at)
                        VALUES (?, ?, ?)
                    """,
                        (row[0], tag_string, time.time()),
                    )
                conn.commit()
                mark_complete([item], "add_tag")
                completed_operations += 1
            except Exception as e:
                logger.error(f"Error adding tag for {item.path}: {e}")
                completed_operations += 1

        # Process remove_tag operations
        for item in results.remove_tag:
            try:
                cursor = conn.execute(
                    """
                    SELECT ft.id FROM fts_tags ft
                    JOIN fts_metadata fm ON ft.metadata_id = fm.id
                    WHERE fm.path = ? AND fm.content_hash = ? AND ft.tag = ?
                """,
                    (item.path, item.cache_key, tag_string),
                )
                tag_ids = [row[0] for row in cursor.fetchall()]
                for tag_id in tag_ids:
                    conn.execute("DELETE FROM fts_tags WHERE id = ?", (tag_id,))
                conn.commit()
                mark_complete([item], "remove_tag")
                completed_operations += 1
            except Exception as e:
                logger.error(f"Error removing tag for {item.path}: {e}")
                completed_operations += 1

        # Process delete operations
        for item in results.delete:
            try:
                # Delete FTS content
                conn.execute("DELETE FROM fts_content WHERE path = ?", (item.path,))

                # Find metadata IDs and delete tags first
                cursor = conn.execute(
                    "SELECT id FROM fts_metadata WHERE path = ? AND content_hash = ?",
                    (item.path, item.cache_key),
                )
                metadata_ids = [row[0] for row in cursor.fetchall()]
                for metadata_id in metadata_ids:
                    conn.execute("DELETE FROM fts_tags WHERE metadata_id = ?", (metadata_id,))
                    conn.execute("DELETE FROM fts_metadata WHERE id = ?", (metadata_id,))

                conn.commit()
                mark_complete([item], "delete")
                completed_operations += 1
            except Exception as e:
                logger.error(f"Error deleting content for {item.path}: {e}")
                completed_operations += 1

        yield IndexingProgressUpdate(
            progress=1.0, description="Full-text indexing complete", status="done"
        )

    async def retrieve(
        self,
        query: str,
        tag: IndexTag,
        limit: int = 50,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve files matching the full-text query."""
        conn = await self._get_connection()
        tag_string = tag.to_string()

        # Build FTS5 query
        fts_query = self._build_fts_query(query)

        # Build filters
        where_conditions = ["ft.tag = ?"]
        params = [tag_string]

        if "language" in kwargs:
            where_conditions.append("fm.language = ?")
            params.append(kwargs["language"])

        if "file_type" in kwargs:
            where_conditions.append("fc.file_type = ?")
            params.append(kwargs["file_type"])

        if "min_size" in kwargs:
            where_conditions.append("fm.file_size >= ?")
            params.append(kwargs["min_size"])

        where_clause = " AND ".join(where_conditions)

        # Execute search
        cursor = conn.execute(
            f"""
            SELECT
                fc.path,
                fc.content,
                fc.language,
                fc.file_type,
                fm.file_size,
                fm.line_count,
                rank
            FROM fts_content fc
            JOIN fts_metadata fm ON fc.path = fm.path
            JOIN fts_tags ft ON fm.id = ft.metadata_id
            WHERE fts_content MATCH ? AND {where_clause}
            ORDER BY rank
            LIMIT ?
        """,
            [fts_query] + params + [limit],
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "path": row[0],
                    "content": row[1],
                    "language": row[2],
                    "file_type": row[3],
                    "file_size": row[4],
                    "line_count": row[5],
                    "rank": row[6],
                }
            )

        return results

    def _build_fts_query(self, query: str) -> str:
        """Build FTS5 query from user query."""
        # Clean and prepare query for FTS5
        terms = query.split()

        # Handle special characters and build FTS5 query
        fts_terms = []
        for term in terms:
            # Escape special FTS5 characters
            escaped_term = term.replace('"', '""')

            # Add as phrase if contains spaces, otherwise as term
            if " " in escaped_term:
                fts_terms.append(f'"{escaped_term}"')
            else:
                fts_terms.append(escaped_term)

        return " AND ".join(fts_terms)

    async def search_in_file(
        self,
        file_path: str,
        query: str,
        tag: IndexTag,
        context_lines: int = 3,
    ) -> list[dict[str, Any]]:
        """Search for query within a specific file with context."""
        conn = await self._get_connection()
        tag_string = tag.to_string()

        # Get file content
        cursor = conn.execute(
            """
            SELECT fc.content FROM fts_content fc
            JOIN fts_metadata fm ON fc.path = fm.path
            JOIN fts_tags ft ON fm.id = ft.metadata_id
            WHERE fc.path = ? AND ft.tag = ?
        """,
            (file_path, tag_string),
        )

        row = cursor.fetchone()
        if not row:
            return []

        content = row[0]
        lines = content.split("\n")

        # Find matching lines
        matches = []
        query_lower = query.lower()

        for i, line in enumerate(lines):
            if query_lower in line.lower():
                # Get context lines
                start_line = max(0, i - context_lines)
                end_line = min(len(lines), i + context_lines + 1)

                context_content = "\n".join(lines[start_line:end_line])

                matches.append(
                    {
                        "line_number": i + 1,
                        "line_content": line,
                        "context": context_content,
                        "start_line": start_line + 1,
                        "end_line": end_line,
                    }
                )

        return matches

    async def get_statistics(self, tag: IndexTag) -> dict[str, Any]:
        """Get statistics for this full-text index."""
        conn = await self._get_connection()
        tag_string = tag.to_string()

        # Total files
        cursor = conn.execute(
            """
            SELECT COUNT(*) FROM fts_metadata fm
            JOIN fts_tags ft ON fm.id = ft.metadata_id
            WHERE ft.tag = ?
        """,
            (tag_string,),
        )
        total_files = cursor.fetchone()[0]

        # Files by language
        cursor = conn.execute(
            """
            SELECT fm.language, COUNT(*) FROM fts_metadata fm
            JOIN fts_tags ft ON fm.id = ft.metadata_id
            WHERE ft.tag = ?
            GROUP BY fm.language
        """,
            (tag_string,),
        )
        files_by_language = dict(cursor.fetchall())

        # Total size and lines
        cursor = conn.execute(
            """
            SELECT SUM(fm.file_size), SUM(fm.line_count) FROM fts_metadata fm
            JOIN fts_tags ft ON fm.id = ft.metadata_id
            WHERE ft.tag = ?
        """,
            (tag_string,),
        )
        totals = cursor.fetchone()

        return {
            "total_files": total_files,
            "files_by_language": files_by_language,
            "total_size_bytes": totals[0] or 0,
            "total_lines": totals[1] or 0,
        }
