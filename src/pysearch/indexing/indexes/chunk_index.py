"""
Enhanced chunk index implementation for code chunking and analysis.

This module implements intelligent code chunking with metadata storage,
providing the foundation for vector embeddings and semantic analysis.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..advanced_chunking import ChunkingEngine, ChunkingConfig, ChunkingStrategy
from ..content_addressing import IndexTag, IndexingProgressUpdate, MarkCompleteCallback, PathAndCacheKey, RefreshIndexResults
from ..enhanced_indexing_engine import EnhancedCodebaseIndex
from ..language_detection import detect_language
from ..logging_config import get_logger
from ..utils import read_text_safely

logger = get_logger(__name__)


class EnhancedChunkIndex(EnhancedCodebaseIndex):
    """
    Enhanced chunk index with intelligent code-aware chunking.
    
    This index creates meaningful code chunks that respect language
    boundaries and provide optimal segments for embedding and analysis.
    """
    
    artifact_id = "enhanced_chunks"
    relative_expected_time = 1.2
    
    def __init__(self, config):
        self.config = config
        self.cache_dir = config.resolve_cache_dir()
        self.db_path = self.cache_dir / "chunks.db"
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        
        # Initialize chunking engine
        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,
            max_chunk_size=getattr(config, 'chunk_size', 1000),
            min_chunk_size=50,
            overlap_size=100,
            respect_boundaries=True,
            quality_threshold=0.5,
        )
        self.chunking_engine = ChunkingEngine(chunking_config)
    
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
        """Create chunk tables."""
        conn = await self._get_connection()
        
        # Enhanced chunks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS code_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                content TEXT NOT NULL,
                language TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                chunk_type TEXT NOT NULL,
                entity_name TEXT,
                entity_type TEXT,
                complexity_score REAL DEFAULT 0.0,
                quality_score REAL DEFAULT 0.0,
                dependencies TEXT,  -- JSON array
                overlap_with TEXT,  -- JSON array of chunk IDs
                metadata TEXT,      -- JSON object
                created_at REAL NOT NULL,
                UNIQUE(chunk_id, content_hash)
            )
        """)
        
        # Tags table for multi-branch support
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_db_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(chunk_db_id, tag),
                FOREIGN KEY (chunk_db_id) REFERENCES code_chunks (id)
            )
        """)
        
        # Create indexes for performance
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_path_hash 
            ON code_chunks(path, content_hash)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id 
            ON code_chunks(chunk_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_entity 
            ON code_chunks(entity_name, entity_type)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunk_tags_tag 
            ON chunk_tags(tag)
        """)
        
        conn.commit()
    
    async def update(
        self,
        tag: IndexTag,
        results: RefreshIndexResults,
        mark_complete: MarkCompleteCallback,
        repo_name: Optional[str] = None,
    ) -> AsyncGenerator[IndexingProgressUpdate, None]:
        """Update the chunk index."""
        conn = await self._get_connection()
        tag_string = tag.to_string()
        
        total_operations = len(results.compute) + len(results.delete) + len(results.add_tag) + len(results.remove_tag)
        completed_operations = 0
        
        # Process compute operations (new files)
        for item in results.compute:
            yield IndexingProgressUpdate(
                progress=completed_operations / max(total_operations, 1),
                description=f"Chunking {Path(item.path).name}",
                status="indexing"
            )
            
            try:
                # Read and chunk file
                content = await read_text_safely(Path(item.path))
                chunks = await self.chunking_engine.chunk_file(item.path, content)
                
                # Store chunks in database
                current_time = time.time()
                for chunk in chunks:
                    # Insert chunk
                    cursor = conn.execute("""
                        INSERT OR REPLACE INTO code_chunks (
                            chunk_id, path, content_hash, content, language,
                            start_line, end_line, chunk_type, entity_name, entity_type,
                            complexity_score, quality_score, dependencies, overlap_with,
                            metadata, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chunk.chunk_id,
                        item.path,
                        item.cache_key,
                        chunk.content,
                        chunk.language.value,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.chunk_type,
                        chunk.entity_name,
                        chunk.entity_type.value if chunk.entity_type else None,
                        chunk.complexity_score,
                        chunk.quality_score,
                        json.dumps(chunk.dependencies),
                        json.dumps(chunk.overlap_with),
                        json.dumps(chunk.metadata.__dict__ if chunk.metadata else {}),
                        current_time
                    ))
                    
                    chunk_db_id = cursor.lastrowid
                    
                    # Add tag association
                    conn.execute("""
                        INSERT OR REPLACE INTO chunk_tags (chunk_db_id, tag, created_at)
                        VALUES (?, ?, ?)
                    """, (chunk_db_id, tag_string, current_time))
                
                conn.commit()
                await mark_complete([item], "compute")
                completed_operations += 1
                
            except Exception as e:
                logger.error(f"Error chunking file {item.path}: {e}")
                completed_operations += 1
        
        # Process other operations (simplified)
        for item in results.add_tag + results.remove_tag + results.delete:
            completed_operations += 1
            await mark_complete([item], "processed")
        
        yield IndexingProgressUpdate(
            progress=1.0,
            description="Chunk indexing complete",
            status="done"
        )
    
    async def retrieve(
        self,
        query: str,
        tag: IndexTag,
        limit: int = 50,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks matching the query."""
        conn = await self._get_connection()
        tag_string = tag.to_string()
        
        # Build search conditions
        where_conditions = ["ct.tag = ?"]
        params = [tag_string]
        
        # Text search in content
        if query.strip():
            where_conditions.append("cc.content LIKE ?")
            params.append(f"%{query}%")
        
        # Additional filters
        if "chunk_type" in kwargs:
            where_conditions.append("cc.chunk_type = ?")
            params.append(kwargs["chunk_type"])
        
        if "language" in kwargs:
            where_conditions.append("cc.language = ?")
            params.append(kwargs["language"])
        
        if "entity_type" in kwargs:
            where_conditions.append("cc.entity_type = ?")
            params.append(kwargs["entity_type"])
        
        if "min_quality" in kwargs:
            where_conditions.append("cc.quality_score >= ?")
            params.append(kwargs["min_quality"])
        
        where_clause = " AND ".join(where_conditions)
        
        # Execute query
        cursor = conn.execute(f"""
            SELECT cc.* FROM code_chunks cc
            JOIN chunk_tags ct ON cc.id = ct.chunk_db_id
            WHERE {where_clause}
            ORDER BY cc.quality_score DESC, cc.complexity_score DESC
            LIMIT ?
        """, params + [limit])
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "chunk_id": row[1],
                "path": row[2],
                "content": row[4],
                "language": row[5],
                "start_line": row[6],
                "end_line": row[7],
                "chunk_type": row[8],
                "entity_name": row[9],
                "entity_type": row[10],
                "complexity_score": row[11],
                "quality_score": row[12],
                "dependencies": json.loads(row[13]) if row[13] else [],
                "overlap_with": json.loads(row[14]) if row[14] else [],
                "metadata": json.loads(row[15]) if row[15] else {},
            })
        
        return results
    
    async def get_chunks_by_file(
        self,
        file_path: str,
        tag: IndexTag,
    ) -> List[Dict[str, Any]]:
        """Get all chunks for a specific file."""
        conn = await self._get_connection()
        tag_string = tag.to_string()
        
        cursor = conn.execute("""
            SELECT cc.* FROM code_chunks cc
            JOIN chunk_tags ct ON cc.id = ct.chunk_db_id
            WHERE cc.path = ? AND ct.tag = ?
            ORDER BY cc.start_line
        """, (file_path, tag_string))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "chunk_id": row[1],
                "start_line": row[6],
                "end_line": row[7],
                "chunk_type": row[8],
                "entity_name": row[9],
                "quality_score": row[12],
                "content_preview": row[4][:100] + "..." if len(row[4]) > 100 else row[4],
            })
        
        return results
    
    async def get_statistics(self, tag: IndexTag) -> Dict[str, Any]:
        """Get statistics for this chunk index."""
        conn = await self._get_connection()
        tag_string = tag.to_string()
        
        # Total chunks
        cursor = conn.execute("""
            SELECT COUNT(*) FROM code_chunks cc
            JOIN chunk_tags ct ON cc.id = ct.chunk_db_id
            WHERE ct.tag = ?
        """, (tag_string,))
        total_chunks = cursor.fetchone()[0]
        
        # Chunks by type
        cursor = conn.execute("""
            SELECT cc.chunk_type, COUNT(*) FROM code_chunks cc
            JOIN chunk_tags ct ON cc.id = ct.chunk_db_id
            WHERE ct.tag = ?
            GROUP BY cc.chunk_type
        """, (tag_string,))
        chunks_by_type = dict(cursor.fetchall())
        
        # Chunks by language
        cursor = conn.execute("""
            SELECT cc.language, COUNT(*) FROM code_chunks cc
            JOIN chunk_tags ct ON cc.id = ct.chunk_db_id
            WHERE ct.tag = ?
            GROUP BY cc.language
        """, (tag_string,))
        chunks_by_language = dict(cursor.fetchall())
        
        # Average scores
        cursor = conn.execute("""
            SELECT AVG(cc.quality_score), AVG(cc.complexity_score) 
            FROM code_chunks cc
            JOIN chunk_tags ct ON cc.id = ct.chunk_db_id
            WHERE ct.tag = ?
        """, (tag_string,))
        avg_scores = cursor.fetchone()
        
        return {
            "total_chunks": total_chunks,
            "chunks_by_type": chunks_by_type,
            "chunks_by_language": chunks_by_language,
            "average_quality": avg_scores[0] or 0.0,
            "average_complexity": avg_scores[1] or 0.0,
        }
