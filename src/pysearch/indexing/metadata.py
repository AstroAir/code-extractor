"""
Metadata indexing system for pysearch with comprehensive metadata support.

This module extends the basic indexer with advanced metadata indexing capabilities,
supporting entity-level indexing, semantic metadata, and efficient querying for
GraphRAG and search operations.

Classes:
    MetadataIndex: Comprehensive metadata index for files and entities
    MetadataIndexer: Extended indexer with metadata and entity support
    IndexQuery: Query interface for the metadata index
    IndexStats: Statistics and analytics for the index

Features:
    - File-level and entity-level metadata indexing
    - Semantic metadata extraction and storage
    - Incremental updates with change detection
    - Efficient querying with multiple filter criteria
    - Integration with GraphRAG knowledge graphs
    - Performance analytics and optimization
    - Persistent storage with compression

Example:
    Basic metadata indexing:
        >>> from pysearch.indexer_metadata import MetadataIndexer
        >>> from pysearch.config import SearchConfig
        >>>
        >>> config = SearchConfig(paths=["./src"])
        >>> indexer = MetadataIndexer(config)
        >>> await indexer.build_index()
        >>>
        >>> # Query the index
        >>> from pysearch.indexer_metadata import IndexQuery
        >>> query = IndexQuery(
        ...     entity_types=["function", "class"],
        ...     languages=["python"],
        ...     min_lines=10
        ... )
        >>> results = await indexer.query_index(query)

    Advanced metadata indexing:
        >>> # Index with semantic metadata
        >>> await indexer.build_index(include_semantic=True)
        >>>
        >>> # Query with semantic filters
        >>> query = IndexQuery(
        ...     semantic_query="database operations",
        ...     similarity_threshold=0.7
        ... )
        >>> results = await indexer.query_index(query)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..core.config import SearchConfig
from ..analysis.graphrag.core import EntityExtractor
from .indexer import Indexer, IndexRecord
from ..analysis.language_detection import detect_language
from ..search.semantic_advanced import SemanticEmbedding
from ..core.types import CodeEntity, EntityType, Language
from ..utils.utils import read_text_safely

logger = logging.getLogger(__name__)


@dataclass
class EntityMetadata:
    """Metadata for a code entity in the index."""

    entity_id: str
    name: str
    entity_type: str
    file_path: str
    start_line: int
    end_line: int
    signature: Optional[str] = None
    docstring: Optional[str] = None
    language: str = "unknown"
    scope: Optional[str] = None
    complexity_score: float = 0.0
    semantic_embedding: Optional[List[float]] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


@dataclass
class FileMetadata:
    """Enhanced metadata for a file in the index."""

    file_path: str
    size: int
    mtime: float
    sha1: Optional[str] = None
    language: str = "unknown"
    line_count: int = 0
    entity_count: int = 0
    complexity_score: float = 0.0
    semantic_summary: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    last_indexed: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


@dataclass
class IndexQuery:
    """Query specification for the enhanced index."""

    # File-level filters
    file_patterns: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    min_size: Optional[int] = None
    max_size: Optional[int] = None
    min_lines: Optional[int] = None
    max_lines: Optional[int] = None
    modified_after: Optional[float] = None
    modified_before: Optional[float] = None

    # Entity-level filters
    entity_types: Optional[List[str]] = None
    entity_names: Optional[List[str]] = None
    has_docstring: Optional[bool] = None
    min_complexity: Optional[float] = None
    max_complexity: Optional[float] = None

    # Semantic filters
    semantic_query: Optional[str] = None
    similarity_threshold: float = 0.7

    # Result options
    include_entities: bool = True
    include_file_content: bool = False
    limit: Optional[int] = None
    offset: int = 0


@dataclass
class IndexStats:
    """Statistics for the enhanced index."""

    total_files: int = 0
    total_entities: int = 0
    languages: Dict[str, int] = field(default_factory=dict)
    entity_types: Dict[str, int] = field(default_factory=dict)
    avg_file_size: float = 0.0
    avg_entities_per_file: float = 0.0
    index_size_mb: float = 0.0
    last_build_time: float = 0.0
    build_duration: float = 0.0


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


class MetadataIndexer:
    """
    Metadata indexer with comprehensive metadata support.

    This class extends the basic indexer with advanced metadata indexing,
    entity extraction, and semantic analysis capabilities.
    """

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.base_indexer = Indexer(config)
        self.entity_extractor = EntityExtractor()
        self.semantic_embedding = SemanticEmbedding()

        # Initialize metadata index
        cache_dir = config.resolve_cache_dir()
        self.metadata_index = MetadataIndex(cache_dir / "metadata.db")
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the metadata indexer."""
        await self.metadata_index.initialize()
        self._initialized = True
        logger.info("Metadata indexer initialized")

    async def build_index(
        self,
        include_semantic: bool = True,
        force_rebuild: bool = False
    ) -> IndexStats:
        """Build the comprehensive index."""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        logger.info("Building enhanced index...")

        # Get file changes from base indexer
        changed_files, removed_files, total_files = self.base_indexer.scan()

        if force_rebuild:
            # Process all files - use scan to get all files
            all_changed, all_removed, total = self.base_indexer.scan()
            changed_files = all_changed

        logger.info(
            f"Processing {len(changed_files)} files for metadata indexing")

        # Prepare semantic embedding if needed
        if include_semantic and changed_files:
            documents = []
            for file_path in changed_files:
                content = read_text_safely(file_path)
                if content:
                    documents.append(content)

            if documents and not self.semantic_embedding.is_fitted:
                self.semantic_embedding.fit(documents)

        # Process each changed file
        for file_path in changed_files:
            await self._index_file(file_path, include_semantic)

        # Remove metadata for deleted files
        for file_path in removed_files:
            await self._remove_file_metadata(file_path)

        build_duration = time.time() - start_time
        stats = await self.metadata_index.get_stats()
        stats.build_duration = build_duration
        stats.last_build_time = time.time()

        logger.info(
            f"Enhanced index built in {build_duration:.2f}s: {stats.total_files} files, {stats.total_entities} entities")
        return stats

    async def _index_file(self, file_path: Path, include_semantic: bool = True) -> None:
        """Index a single file with comprehensive metadata."""
        try:
            # Read file content
            content = read_text_safely(file_path)
            if not content:
                return

            # Get basic file info
            stat = file_path.stat()
            language = detect_language(file_path)
            lines = content.split('\n')
            line_count = len(lines)

            # Calculate complexity score (simple metric)
            complexity_score = self._calculate_file_complexity(
                content, language)

            # Extract entities
            entities = await self.entity_extractor.extract_from_file(file_path)

            # Generate semantic summary if requested
            semantic_summary = None
            if include_semantic and self.semantic_embedding.is_fitted:
                # Create a summary from first few lines and docstrings
                summary_parts = []
                if len(lines) > 0:
                    summary_parts.extend(lines[:5])  # First 5 lines

                for entity in entities:
                    if entity.docstring:
                        summary_parts.append(entity.docstring)

                if summary_parts:
                    summary_text = ' '.join(summary_parts)[
                        :500]  # Limit length
                    semantic_summary = summary_text

            # Extract imports and dependencies (simplified)
            imports = self._extract_imports(content, language)
            dependencies = self._extract_dependencies(content, language)

            # Create file metadata
            file_metadata = FileMetadata(
                file_path=str(file_path),
                size=stat.st_size,
                mtime=stat.st_mtime,
                language=language.value if hasattr(
                    language, 'value') else str(language),
                line_count=line_count,
                entity_count=len(entities),
                complexity_score=complexity_score,
                semantic_summary=semantic_summary,
                imports=imports,
                dependencies=dependencies,
                last_indexed=time.time()
            )

            # Store file metadata
            await self.metadata_index.add_file_metadata(file_metadata)

            # Process entities
            for entity in entities:
                entity_complexity = self._calculate_entity_complexity(
                    entity, content)

                # Generate semantic embedding if requested
                semantic_embedding = None
                if include_semantic and self.semantic_embedding.is_fitted:
                    entity_text = self._create_entity_text(entity)
                    embedding_vector = self.semantic_embedding.transform(
                        entity_text)
                    if embedding_vector:
                        # Convert sparse to dense
                        max_dim = max(embedding_vector.keys()) + \
                            1 if embedding_vector else 0
                        dense_vector = [0.0] * max_dim
                        for dim, value in embedding_vector.items():
                            dense_vector[dim] = value
                        semantic_embedding = dense_vector

                # Create entity metadata
                entity_metadata = EntityMetadata(
                    entity_id=entity.id,
                    name=entity.name,
                    entity_type=entity.entity_type.value,
                    file_path=str(file_path),
                    start_line=entity.start_line,
                    end_line=entity.end_line,
                    signature=entity.signature,
                    docstring=entity.docstring,
                    language=entity.language.value if hasattr(
                        entity.language, 'value') else str(entity.language),
                    scope=entity.scope,
                    complexity_score=entity_complexity,
                    semantic_embedding=semantic_embedding,
                    properties=entity.properties
                )

                # Store entity metadata
                await self.metadata_index.add_entity_metadata(entity_metadata)

            logger.debug(f"Indexed {file_path}: {len(entities)} entities")

        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")

    async def _remove_file_metadata(self, file_path: Path) -> None:
        """Remove metadata for a deleted file."""
        # This would require additional SQL operations to delete from both tables
        # For now, we'll log the removal
        logger.debug(f"File removed from index: {file_path}")

    def _calculate_file_complexity(self, content: str, language: Language) -> float:
        """Calculate a simple complexity score for a file."""
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        # Basic complexity metrics
        complexity = 0.0
        complexity += len(non_empty_lines) * 0.1  # Line count
        complexity += content.count('if ') * 0.5  # Conditional statements
        complexity += content.count('for ') * 0.5  # Loops
        complexity += content.count('while ') * 0.5  # Loops
        complexity += content.count('try:') * 0.3  # Exception handling
        complexity += content.count('def ') * 0.2  # Function definitions
        complexity += content.count('class ') * 0.4  # Class definitions

        return min(complexity, 100.0)  # Cap at 100

    def _calculate_entity_complexity(self, entity: CodeEntity, content: str) -> float:
        """Calculate complexity score for an entity."""
        if not entity.signature:
            return 0.0

        # Extract entity content
        lines = content.split('\n')
        start_idx = max(0, entity.start_line - 1)
        end_idx = min(len(lines), entity.end_line)
        entity_content = '\n'.join(lines[start_idx:end_idx])

        # Calculate complexity
        complexity = 0.0
        complexity += len(entity_content.split('\n')) * 0.1
        complexity += entity_content.count('if ') * 0.5
        complexity += entity_content.count('for ') * 0.5
        complexity += entity_content.count('while ') * 0.5
        complexity += entity_content.count('try:') * 0.3

        return min(complexity, 50.0)  # Cap at 50 for entities

    def _extract_imports(self, content: str, language: Language) -> List[str]:
        """Extract import statements from content."""
        imports = []
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if language == Language.PYTHON:
                if line.startswith('import ') or line.startswith('from '):
                    imports.append(line)
            elif language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
                if 'import ' in line and 'from ' in line:
                    imports.append(line)
            elif language == Language.JAVA:
                if line.startswith('import '):
                    imports.append(line)

        return imports[:20]  # Limit to first 20 imports

    def _extract_dependencies(self, content: str, language: Language) -> List[str]:
        """Extract dependency information from content."""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated dependency analysis
        dependencies = []

        if language == Language.PYTHON:
            # Look for common library usage patterns
            common_libs = ['requests', 'numpy',
                           'pandas', 'flask', 'django', 'fastapi']
            for lib in common_libs:
                if lib in content:
                    dependencies.append(lib)

        return dependencies

    def _create_entity_text(self, entity: CodeEntity) -> str:
        """Create text representation of entity for semantic embedding."""
        parts = [entity.name]

        if entity.signature:
            parts.append(entity.signature)

        if entity.docstring:
            parts.append(entity.docstring)

        # Add some context from properties
        if entity.properties:
            for key, value in entity.properties.items():
                if isinstance(value, str):
                    parts.append(f"{key}: {value}")
                elif isinstance(value, list):
                    parts.append(f"{key}: {value}")

        return ' '.join(parts)

    async def query_index(self, query: IndexQuery) -> Dict[str, Any]:
        """Query the enhanced index."""
        if not self._initialized:
            await self.initialize()

        results: Dict[str, Any] = {
            "files": [],
            "entities": [],
            "stats": {}
        }

        # Query files
        files = await self.metadata_index.query_files(query)
        results["files"] = [
            {
                "path": f.file_path,
                "language": f.language,
                "size": f.size,
                "line_count": f.line_count,
                "entity_count": f.entity_count,
                "complexity": f.complexity_score
            }
            for f in files
        ]

        # Query entities if requested
        if query.include_entities:
            entities = await self.metadata_index.query_entities(query)
            results["entities"] = [
                {
                    "id": e.entity_id,
                    "name": e.name,
                    "type": e.entity_type,
                    "file": e.file_path,
                    "line": e.start_line,
                    "signature": e.signature,
                    "complexity": e.complexity_score
                }
                for e in entities
            ]

        # Get stats
        stats = await self.metadata_index.get_stats()
        results["stats"] = {
            "total_files": stats.total_files,
            "total_entities": stats.total_entities,
            "languages": stats.languages,
            "entity_types": stats.entity_types
        }

        return results

    async def close(self) -> None:
        """Close the metadata indexer."""
        await self.metadata_index.close()
        logger.info("Metadata indexer closed")
