"""
Enhanced code snippets index implementation.

This module implements an enhanced version of Continue's CodeSnippetsIndex
with broader language support, better entity extraction, and improved
metadata handling.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from ...analysis.content_addressing import IndexTag, IndexingProgressUpdate, MarkCompleteCallback, PathAndCacheKey, RefreshIndexResults
from ..advanced.engine import EnhancedCodebaseIndex
from ...analysis.language_support import language_registry
from ...utils.logging_config import get_logger
from ...core.types import CodeEntity, Language
from ...utils.utils import read_text_safely

logger = get_logger()


class EnhancedCodeSnippetsIndex(EnhancedCodebaseIndex):
    """
    Enhanced code snippets index with multi-language support.

    This index extracts code entities (functions, classes, variables, etc.)
    from source files using tree-sitter parsing and stores them in SQLite
    for fast retrieval and filtering.
    """

    artifact_id = "enhanced_code_snippets"
    relative_expected_time = 1.5

    def __init__(self, config: Any) -> None:
        self.config = config
        self.cache_dir = config.resolve_cache_dir()
        self.db_path = self.cache_dir / "code_snippets.db"
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    async def _get_connection(self) -> sqlite3.Connection:
        """Get database connection, creating tables if needed."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA busy_timeout=3000")
            await self._create_tables()
        return self._connection

    async def _create_tables(self) -> None:
        """Create code snippets tables."""
        conn = await self._get_connection()

        # Enhanced code snippets table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS code_snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                signature TEXT,
                docstring TEXT,
                content TEXT NOT NULL,
                language TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                complexity_score REAL DEFAULT 0.0,
                quality_score REAL DEFAULT 0.0,
                dependencies TEXT,  -- JSON array
                metadata TEXT,      -- JSON object
                created_at REAL NOT NULL,
                UNIQUE(path, content_hash, name, start_line, end_line)
            )
        """)

        # Tags table for multi-branch support
        conn.execute("""
            CREATE TABLE IF NOT EXISTS snippet_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snippet_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(snippet_id, tag),
                FOREIGN KEY (snippet_id) REFERENCES code_snippets (id)
            )
        """)

        # Create indexes for performance
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_snippets_path_hash
            ON code_snippets(path, content_hash)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_snippets_name_type
            ON code_snippets(name, entity_type)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_snippets_language
            ON code_snippets(language)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_snippet_tags_tag
            ON snippet_tags(tag)
        """)

        conn.commit()

    async def update(
        self,
        tag: IndexTag,
        results: RefreshIndexResults,
        mark_complete: MarkCompleteCallback,
        repo_name: Optional[str] = None,
    ) -> AsyncGenerator[IndexingProgressUpdate, None]:
        """Update the code snippets index."""
        conn = await self._get_connection()
        tag_string = tag.to_string()

        # Process compute operations (new files)
        for i, item in enumerate(results.compute):
            yield IndexingProgressUpdate(
                progress=i / max(len(results.compute), 1),
                description=f"Extracting snippets from {Path(item.path).name}",
                status="indexing"
            )

            try:
                # Read file content
                content = await read_text_safely(Path(item.path))

                # Detect language
                from ...analysis.language_detection import detect_language
                language = detect_language(Path(item.path), content)

                # Extract entities using language processor
                processor = language_registry.get_processor(language)
                if processor:
                    entities = processor.extract_entities(content)
                else:
                    entities = []

                # Store entities in database
                current_time = time.time()
                for entity in entities:
                    # Calculate quality score
                    quality_score = self._calculate_entity_quality(entity)

                    # Insert snippet
                    cursor = conn.execute("""
                        INSERT OR REPLACE INTO code_snippets (
                            path, content_hash, name, entity_type, signature, docstring,
                            content, language, start_line, end_line, complexity_score,
                            quality_score, dependencies, metadata, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        item.path,
                        item.cache_key,
                        entity.name,
                        entity.entity_type.value,
                        entity.signature,
                        entity.docstring,
                        entity.content,
                        language.value,
                        entity.start_line,
                        entity.end_line,
                        entity.complexity_score,
                        quality_score,
                        json.dumps(entity.dependencies),
                        json.dumps(entity.properties),
                        current_time
                    ))

                    snippet_id = cursor.lastrowid

                    # Add tag association
                    conn.execute("""
                        INSERT OR REPLACE INTO snippet_tags (snippet_id, tag, created_at)
                        VALUES (?, ?, ?)
                    """, (snippet_id, tag_string, current_time))

                conn.commit()
                await mark_complete([item], "compute")

            except Exception as e:
                logger.error(f"Error processing file {item.path}: {e}")
                # Continue with other files

        # Process add_tag operations (existing content, new tag)
        for item in results.add_tag:
            try:
                # Find existing snippets for this content
                cursor = conn.execute("""
                    SELECT id FROM code_snippets
                    WHERE path = ? AND content_hash = ?
                """, (item.path, item.cache_key))

                snippet_ids = [row[0] for row in cursor.fetchall()]

                # Add tag to existing snippets
                current_time = time.time()
                for snippet_id in snippet_ids:
                    conn.execute("""
                        INSERT OR REPLACE INTO snippet_tags (snippet_id, tag, created_at)
                        VALUES (?, ?, ?)
                    """, (snippet_id, tag_string, current_time))

                conn.commit()
                await mark_complete([item], "add_tag")

            except Exception as e:
                logger.error(f"Error adding tag for {item.path}: {e}")

        # Process remove_tag operations
        for item in results.remove_tag:
            try:
                # Find snippets for this content and tag
                cursor = conn.execute("""
                    SELECT st.snippet_id FROM snippet_tags st
                    JOIN code_snippets cs ON st.snippet_id = cs.id
                    WHERE cs.path = ? AND cs.content_hash = ? AND st.tag = ?
                """, (item.path, item.cache_key, tag_string))

                snippet_ids = [row[0] for row in cursor.fetchall()]

                # Remove tag associations
                for snippet_id in snippet_ids:
                    conn.execute("""
                        DELETE FROM snippet_tags
                        WHERE snippet_id = ? AND tag = ?
                    """, (snippet_id, tag_string))

                conn.commit()
                await mark_complete([item], "remove_tag")

            except Exception as e:
                logger.error(f"Error removing tag for {item.path}: {e}")

        # Process delete operations
        for item in results.delete:
            try:
                # Find snippets to delete
                cursor = conn.execute("""
                    SELECT id FROM code_snippets
                    WHERE path = ? AND content_hash = ?
                """, (item.path, item.cache_key))

                snippet_ids = [row[0] for row in cursor.fetchall()]

                # Delete snippets and their tags
                for snippet_id in snippet_ids:
                    conn.execute("DELETE FROM snippet_tags WHERE snippet_id = ?", (snippet_id,))
                    conn.execute("DELETE FROM code_snippets WHERE id = ?", (snippet_id,))

                conn.commit()
                await mark_complete([item], "delete")

            except Exception as e:
                logger.error(f"Error deleting snippets for {item.path}: {e}")

        yield IndexingProgressUpdate(
            progress=1.0,
            description="Code snippets indexing complete",
            status="done"
        )

    async def retrieve(
        self,
        query: str,
        tag: IndexTag,
        limit: int = 50,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Retrieve code snippets matching the query."""
        conn = await self._get_connection()
        tag_string = tag.to_string()

        # Build search query
        search_terms = query.split()
        where_conditions = []
        params = []

        # Search in name, signature, and content
        for term in search_terms:
            where_conditions.append("""
                (cs.name LIKE ? OR cs.signature LIKE ? OR cs.content LIKE ?)
            """)
            params.extend([f"%{term}%", f"%{term}%", f"%{term}%"])

        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

        # Additional filters from kwargs
        if "entity_type" in kwargs:
            where_clause += " AND cs.entity_type = ?"
            params.append(kwargs["entity_type"])

        if "language" in kwargs:
            where_clause += " AND cs.language = ?"
            params.append(kwargs["language"])

        if "min_quality" in kwargs:
            where_clause += " AND cs.quality_score >= ?"
            params.append(kwargs["min_quality"])

        # Execute query
        cursor = conn.execute(f"""
            SELECT cs.* FROM code_snippets cs
            JOIN snippet_tags st ON cs.id = st.snippet_id
            WHERE st.tag = ? AND ({where_clause})
            ORDER BY cs.quality_score DESC, cs.complexity_score DESC
            LIMIT ?
        """, [tag_string] + params + [limit])

        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "path": row[1],
                "content_hash": row[2],
                "name": row[3],
                "entity_type": row[4],
                "signature": row[5],
                "docstring": row[6],
                "content": row[7],
                "language": row[8],
                "start_line": row[9],
                "end_line": row[10],
                "complexity_score": row[11],
                "quality_score": row[12],
                "dependencies": json.loads(row[13]) if row[13] else [],
                "metadata": json.loads(row[14]) if row[14] else {},
            })

        return results

    def _calculate_entity_quality(self, entity: CodeEntity) -> float:
        """Calculate quality score for a code entity."""
        quality = 0.0

        # Name quality (prefer descriptive names)
        if len(entity.name) > 3:
            quality += 0.2
        if not entity.name.startswith('_'):  # Not private
            quality += 0.1

        # Documentation quality
        if entity.docstring:
            quality += 0.3
        if entity.signature:
            quality += 0.2

        # Content quality
        content_lines = len(entity.content.split('\n'))
        if 5 <= content_lines <= 50:  # Reasonable size
            quality += 0.2

        return min(1.0, quality)

    async def get_entity_by_id(self, entity_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific entity by ID."""
        conn = await self._get_connection()

        cursor = conn.execute("""
            SELECT * FROM code_snippets WHERE id = ?
        """, (entity_id,))

        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "path": row[1],
                "content_hash": row[2],
                "name": row[3],
                "entity_type": row[4],
                "signature": row[5],
                "docstring": row[6],
                "content": row[7],
                "language": row[8],
                "start_line": row[9],
                "end_line": row[10],
                "complexity_score": row[11],
                "quality_score": row[12],
                "dependencies": json.loads(row[13]) if row[13] else [],
                "metadata": json.loads(row[14]) if row[14] else {},
            }
        return None

    async def get_entities_by_file(
        self,
        file_path: str,
        tag: IndexTag,
    ) -> List[Dict[str, Any]]:
        """Get all entities for a specific file."""
        conn = await self._get_connection()
        tag_string = tag.to_string()

        cursor = conn.execute("""
            SELECT cs.* FROM code_snippets cs
            JOIN snippet_tags st ON cs.id = st.snippet_id
            WHERE cs.path = ? AND st.tag = ?
            ORDER BY cs.start_line
        """, (file_path, tag_string))

        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "name": row[3],
                "entity_type": row[4],
                "signature": row[5],
                "start_line": row[9],
                "end_line": row[10],
                "complexity_score": row[11],
                "quality_score": row[12],
            })

        return results

    async def get_statistics(self, tag: IndexTag) -> Dict[str, Any]:
        """Get statistics for this index."""
        conn = await self._get_connection()
        tag_string = tag.to_string()

        # Total entities
        cursor = conn.execute("""
            SELECT COUNT(*) FROM code_snippets cs
            JOIN snippet_tags st ON cs.id = st.snippet_id
            WHERE st.tag = ?
        """, (tag_string,))
        total_entities = cursor.fetchone()[0]

        # Entities by type
        cursor = conn.execute("""
            SELECT cs.entity_type, COUNT(*) FROM code_snippets cs
            JOIN snippet_tags st ON cs.id = st.snippet_id
            WHERE st.tag = ?
            GROUP BY cs.entity_type
        """, (tag_string,))
        entities_by_type = dict(cursor.fetchall())

        # Entities by language
        cursor = conn.execute("""
            SELECT cs.language, COUNT(*) FROM code_snippets cs
            JOIN snippet_tags st ON cs.id = st.snippet_id
            WHERE st.tag = ?
            GROUP BY cs.language
        """, (tag_string,))
        entities_by_language = dict(cursor.fetchall())

        # Average quality and complexity
        cursor = conn.execute("""
            SELECT AVG(cs.quality_score), AVG(cs.complexity_score)
            FROM code_snippets cs
            JOIN snippet_tags st ON cs.id = st.snippet_id
            WHERE st.tag = ?
        """, (tag_string,))
        avg_scores = cursor.fetchone()

        return {
            "total_entities": total_entities,
            "entities_by_type": entities_by_type,
            "entities_by_language": entities_by_language,
            "average_quality": avg_scores[0] or 0.0,
            "average_complexity": avg_scores[1] or 0.0,
        }

    async def search_entities(
        self,
        query: str,
        tag: IndexTag,
        entity_types: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        min_quality: float = 0.0,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for entities with advanced filtering.

        Args:
            query: Search query
            tag: Index tag to search within
            entity_types: Filter by entity types
            languages: Filter by programming languages
            min_quality: Minimum quality score
            limit: Maximum number of results

        Returns:
            List of matching entities
        """
        conn = await self._get_connection()
        tag_string = tag.to_string()

        # Build query
        where_conditions = ["st.tag = ?"]
        params = [tag_string]

        # Text search
        if query.strip():
            where_conditions.append("""
                (cs.name LIKE ? OR cs.signature LIKE ? OR cs.docstring LIKE ? OR cs.content LIKE ?)
            """)
            query_pattern = f"%{query}%"
            params.extend([query_pattern, query_pattern, query_pattern, query_pattern])

        # Entity type filter
        if entity_types:
            placeholders = ",".join("?" * len(entity_types))
            where_conditions.append(f"cs.entity_type IN ({placeholders})")
            params.extend(entity_types)

        # Language filter
        if languages:
            placeholders = ",".join("?" * len(languages))
            where_conditions.append(f"cs.language IN ({placeholders})")
            params.extend(languages)

        # Quality filter
        if min_quality > 0:
            where_conditions.append("cs.quality_score >= ?")
            params.append(min_quality)

        where_clause = " AND ".join(where_conditions)

        # Execute search
        cursor = conn.execute(f"""
            SELECT cs.* FROM code_snippets cs
            JOIN snippet_tags st ON cs.id = st.snippet_id
            WHERE {where_clause}
            ORDER BY cs.quality_score DESC, cs.complexity_score DESC
            LIMIT ?
        """, params + [limit])

        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "path": row[1],
                "name": row[3],
                "entity_type": row[4],
                "signature": row[5],
                "docstring": row[6],
                "content": row[7],
                "language": row[8],
                "start_line": row[9],
                "end_line": row[10],
                "complexity_score": row[11],
                "quality_score": row[12],
                "dependencies": json.loads(row[13]) if row[13] else [],
                "metadata": json.loads(row[14]) if row[14] else {},
            })

        return results
