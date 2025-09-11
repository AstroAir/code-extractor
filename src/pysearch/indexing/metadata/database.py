"""
Database operations for the metadata indexing system.

This module provides the MetadataIndex class that handles all database
operations for storing and querying file and entity metadata using SQLite.

Classes:
    MetadataIndex: SQLite-based metadata storage and querying

Features:
    - Efficient SQLite storage with proper indexing
    - Complex query support with multiple filter criteria
    - Automatic database schema creation and migration
    - JSON serialization for complex data types
    - Performance optimized queries
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, List, Optional

from .models import EntityMetadata, FileMetadata, IndexQuery, IndexStats

logger = logging.getLogger(__name__)


class MetadataIndex:
    """
    Comprehensive metadata index using SQLite for efficient storage and querying.

    This class provides a persistent, queryable index of file and entity metadata
    with support for complex queries and efficient updates.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None

    async def initialize(self) -> None:
        """Initialize the metadata index database."""
        self._connection = sqlite3.connect(str(self.db_path))
        self._connection.row_factory = sqlite3.Row

        # Create tables
        await self._create_tables()

        logger.info(f"Metadata index initialized: {self.db_path}")

    async def _create_tables(self) -> None:
        """Create database tables for metadata storage."""
        if not self._connection:
            return

        cursor = self._connection.cursor()

        # Files table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                file_path TEXT PRIMARY KEY,
                size INTEGER,
                mtime REAL,
                sha1 TEXT,
                language TEXT,
                line_count INTEGER,
                entity_count INTEGER,
                complexity_score REAL,
                semantic_summary TEXT,
                imports TEXT,  -- JSON array
                exports TEXT,  -- JSON array
                dependencies TEXT,  -- JSON array
                last_indexed REAL,
                access_count INTEGER,
                last_accessed REAL
            )
        """)

        # Entities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                name TEXT,
                entity_type TEXT,
                file_path TEXT,
                start_line INTEGER,
                end_line INTEGER,
                signature TEXT,
                docstring TEXT,
                language TEXT,
                scope TEXT,
                complexity_score REAL,
                semantic_embedding TEXT,  -- JSON array
                properties TEXT,  -- JSON object
                last_updated REAL,
                FOREIGN KEY (file_path) REFERENCES files (file_path)
            )
        """)

        # Create indexes for efficient querying
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_files_language ON files (language)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_files_mtime ON files (mtime)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_files_size ON files (size)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (entity_type)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities (name)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_file ON entities (file_path)")

        self._connection.commit()

    async def add_file_metadata(self, metadata: FileMetadata) -> None:
        """Add or update file metadata."""
        if not self._connection:
            return

        cursor = self._connection.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO files (
                file_path, size, mtime, sha1, language, line_count, entity_count,
                complexity_score, semantic_summary, imports, exports, dependencies,
                last_indexed, access_count, last_accessed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.file_path, metadata.size, metadata.mtime, metadata.sha1,
            metadata.language, metadata.line_count, metadata.entity_count,
            metadata.complexity_score, metadata.semantic_summary,
            json.dumps(metadata.imports), json.dumps(metadata.exports),
            json.dumps(metadata.dependencies), metadata.last_indexed,
            metadata.access_count, metadata.last_accessed
        ))
        self._connection.commit()

    async def add_entity_metadata(self, metadata: EntityMetadata) -> None:
        """Add or update entity metadata."""
        if not self._connection:
            return

        cursor = self._connection.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO entities (
                entity_id, name, entity_type, file_path, start_line, end_line,
                signature, docstring, language, scope, complexity_score,
                semantic_embedding, properties, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.entity_id, metadata.name, metadata.entity_type,
            metadata.file_path, metadata.start_line, metadata.end_line,
            metadata.signature, metadata.docstring, metadata.language,
            metadata.scope, metadata.complexity_score,
            json.dumps(
                metadata.semantic_embedding) if metadata.semantic_embedding else None,
            json.dumps(metadata.properties), metadata.last_updated
        ))
        self._connection.commit()

    async def query_files(self, query: IndexQuery) -> List[FileMetadata]:
        """Query files based on criteria."""
        if not self._connection:
            return []

        cursor = self._connection.cursor()

        # Build WHERE clause
        conditions = []
        params: list[Any] = []

        if query.languages:
            conditions.append(
                f"language IN ({','.join(['?'] * len(query.languages))})")
            params.extend(query.languages)

        if query.min_size is not None:
            conditions.append("size >= ?")
            params.append(query.min_size)

        if query.max_size is not None:
            conditions.append("size <= ?")
            params.append(query.max_size)

        if query.min_lines is not None:
            conditions.append("line_count >= ?")
            params.append(query.min_lines)

        if query.max_lines is not None:
            conditions.append("line_count <= ?")
            params.append(query.max_lines)

        if query.modified_after is not None:
            conditions.append("mtime >= ?")
            params.append(query.modified_after)

        if query.modified_before is not None:
            conditions.append("mtime <= ?")
            params.append(query.modified_before)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Build full query
        sql = f"SELECT * FROM files WHERE {where_clause}"

        if query.limit is not None:
            sql += f" LIMIT {query.limit}"

        if query.offset > 0:
            sql += f" OFFSET {query.offset}"

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        # Convert to FileMetadata objects
        files = []
        for row in rows:
            metadata = FileMetadata(
                file_path=row['file_path'],
                size=row['size'],
                mtime=row['mtime'],
                sha1=row['sha1'],
                language=row['language'],
                line_count=row['line_count'],
                entity_count=row['entity_count'],
                complexity_score=row['complexity_score'],
                semantic_summary=row['semantic_summary'],
                imports=json.loads(row['imports']) if row['imports'] else [],
                exports=json.loads(row['exports']) if row['exports'] else [],
                dependencies=json.loads(
                    row['dependencies']) if row['dependencies'] else [],
                last_indexed=row['last_indexed'],
                access_count=row['access_count'],
                last_accessed=row['last_accessed']
            )
            files.append(metadata)

        return files

    async def query_entities(self, query: IndexQuery) -> List[EntityMetadata]:
        """Query entities based on criteria."""
        if not self._connection:
            return []

        cursor = self._connection.cursor()

        # Build WHERE clause
        conditions = []
        params: list[Any] = []

        if query.entity_types:
            conditions.append(
                f"entity_type IN ({','.join(['?'] * len(query.entity_types))})")
            params.extend(query.entity_types)

        if query.entity_names:
            name_conditions = []
            for name in query.entity_names:
                name_conditions.append("name LIKE ?")
                params.append(f"%{name}%")
            conditions.append(f"({' OR '.join(name_conditions)})")

        if query.languages:
            conditions.append(
                f"language IN ({','.join(['?'] * len(query.languages))})")
            params.extend(query.languages)

        if query.has_docstring is not None:
            if query.has_docstring:
                conditions.append("docstring IS NOT NULL AND docstring != ''")
            else:
                conditions.append("(docstring IS NULL OR docstring = '')")

        if query.min_complexity is not None:
            conditions.append("complexity_score >= ?")
            params.append(query.min_complexity)

        if query.max_complexity is not None:
            conditions.append("complexity_score <= ?")
            params.append(query.max_complexity)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Build full query
        sql = f"SELECT * FROM entities WHERE {where_clause}"

        if query.limit is not None:
            sql += f" LIMIT {query.limit}"

        if query.offset > 0:
            sql += f" OFFSET {query.offset}"

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        # Convert to EntityMetadata objects
        entities = []
        for row in rows:
            metadata = EntityMetadata(
                entity_id=row['entity_id'],
                name=row['name'],
                entity_type=row['entity_type'],
                file_path=row['file_path'],
                start_line=row['start_line'],
                end_line=row['end_line'],
                signature=row['signature'],
                docstring=row['docstring'],
                language=row['language'],
                scope=row['scope'],
                complexity_score=row['complexity_score'],
                semantic_embedding=json.loads(
                    row['semantic_embedding']) if row['semantic_embedding'] else None,
                properties=json.loads(
                    row['properties']) if row['properties'] else {},
                last_updated=row['last_updated']
            )
            entities.append(metadata)

        return entities

    async def get_stats(self) -> IndexStats:
        """Get index statistics."""
        if not self._connection:
            return IndexStats()

        cursor = self._connection.cursor()

        # File statistics
        cursor.execute(
            "SELECT COUNT(*), AVG(size), AVG(entity_count) FROM files")
        file_stats = cursor.fetchone()
        total_files = file_stats[0] if file_stats[0] else 0
        avg_file_size = file_stats[1] if file_stats[1] else 0.0
        avg_entities_per_file = file_stats[2] if file_stats[2] else 0.0

        # Entity statistics
        cursor.execute("SELECT COUNT(*) FROM entities")
        entity_result = cursor.fetchone()
        total_entities = entity_result[0] if entity_result else 0

        # Language distribution
        cursor.execute(
            "SELECT language, COUNT(*) FROM files GROUP BY language")
        languages = dict(cursor.fetchall())

        # Entity type distribution
        cursor.execute(
            "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type")
        entity_types = dict(cursor.fetchall())

        # Database size
        cursor.execute(
            "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        db_result = cursor.fetchone()
        db_size = db_result[0] if db_result else 0
        index_size_mb = db_size / (1024 * 1024)

        return IndexStats(
            total_files=total_files,
            total_entities=total_entities,
            languages=languages,
            entity_types=entity_types,
            avg_file_size=avg_file_size,
            avg_entities_per_file=avg_entities_per_file,
            index_size_mb=index_size_mb
        )

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
