"""
Enhanced vector database integration for semantic code search.

This module provides comprehensive vector database support with multiple providers,
efficient batch processing, and optimized retrieval algorithms. It extends beyond
Continue's LanceDB-only approach to support multiple vector database backends.

Classes:
    VectorDatabase: Abstract base for vector database implementations
    LanceDBProvider: LanceDB vector database implementation
    QdrantProvider: Qdrant vector database implementation
    ChromaProvider: Chroma vector database implementation
    EmbeddingProvider: Abstract base for embedding providers
    VectorIndexManager: Manages vector indexing operations

Features:
    - Multiple vector database backends (LanceDB, Qdrant, Chroma)
    - Multiple embedding providers (OpenAI, HuggingFace, local models)
    - Efficient batch processing for large codebases
    - Optimized retrieval with filtering and ranking
    - Automatic embedding caching and reuse
    - Vector index optimization and maintenance
    - Similarity search with metadata filtering

Example:
    Basic vector indexing:
        >>> from pysearch.storage.vector_db import VectorIndexManager, EmbeddingConfig
        >>> config = EmbeddingConfig(provider="openai", api_key="...")
        >>> manager = VectorIndexManager(Path("./db"), config, provider="lancedb")
        >>> await manager.index_chunks(chunks, "my_collection")

    Advanced retrieval:
        >>> results = await manager.search(
        ...     query="database connection",
        ...     collection_name="my_collection",
        ...     limit=10,
        ...     filters={"language": "python"}
        ... )
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.logging_config import get_logger

if TYPE_CHECKING:
    from ..analysis.content_addressing import IndexTag
    from ..indexing.advanced.chunking import MetadataCodeChunk

logger = get_logger()

# Vector database availability checks
try:
    import lancedb  # noqa: F401

    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

try:
    import qdrant_client  # noqa: F401

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    import chromadb  # noqa: F401

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""

    chunk_id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    similarity_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    provider: str = "openai"  # "openai", "huggingface", "local"
    model_name: str = "text-embedding-ada-002"
    batch_size: int = 100
    max_tokens: int = 8192
    dimensions: int = 1536
    api_key: str | None = None
    base_url: str | None = None


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        try:
            import openai  # type: ignore[import-not-found]

            self.client = openai.AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
            self._available = True
        except ImportError:
            logger.warning("OpenAI library not available")
            self._available = False

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI API."""
        if not self._available:
            raise RuntimeError("OpenAI client not available")

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            try:
                response = await self.client.embeddings.create(
                    model=self.config.model_name, input=batch
                )

                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i}: {e}")
                # Add zero vectors for failed batch
                embeddings.extend([[0.0] * self.dimensions] * len(batch))

        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for search query."""
        embeddings = await self.embed_texts([query])
        return embeddings[0] if embeddings else [0.0] * self.dimensions

    @property
    def dimensions(self) -> int:
        return self.config.dimensions


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace sentence-transformers embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._model = None
        self._available = False
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            self._model = SentenceTransformer(config.model_name)
            self._available = True
            # Auto-detect dimensions from the model
            self._dimensions = self._model.get_sentence_embedding_dimension() or config.dimensions
        except ImportError:
            logger.warning("sentence-transformers library not available")
            self._dimensions = config.dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using HuggingFace sentence-transformers."""
        if not self._available or self._model is None:
            raise RuntimeError("HuggingFace sentence-transformers not available")

        embeddings: list[list[float]] = []

        # Process in batches using asyncio to avoid blocking
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            try:
                # Run synchronous encode in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(
                    None, lambda b=batch: self._model.encode(b, show_progress_bar=False).tolist()
                )
                embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Error generating HuggingFace embeddings for batch {i}: {e}")
                embeddings.extend([[0.0] * self.dimensions] * len(batch))

        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for search query."""
        embeddings = await self.embed_texts([query])
        return embeddings[0] if embeddings else [0.0] * self.dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions


class VectorDatabase(ABC):
    """Abstract base for vector database implementations."""

    def __init__(self, db_path: Path, embedding_provider: EmbeddingProvider):
        self.db_path = db_path
        self.embedding_provider = embedding_provider

    @abstractmethod
    async def create_collection(self, collection_name: str) -> None:
        """Create a new collection/table."""
        pass

    @abstractmethod
    async def insert_vectors(
        self,
        collection_name: str,
        vectors: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None:
        """Insert vectors with metadata."""
        pass

    @abstractmethod
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete_vectors(
        self,
        collection_name: str,
        vector_ids: list[str],
    ) -> None:
        """Delete vectors by ID."""
        pass

    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        pass

    async def batch_search_vectors(
        self,
        collection_name: str,
        query_vectors: list[list[float]],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[list[VectorSearchResult]]:
        """Search for similar vectors for multiple queries in a single batch.

        Default implementation falls back to sequential search_vectors calls.
        Subclasses may override this for more efficient batch operations.
        """
        results = []
        for qv in query_vectors:
            result = await self.search_vectors(collection_name, qv, limit, filters)
            results.append(result)
        return results


class LanceDBProvider(VectorDatabase):
    """LanceDB vector database implementation."""

    def __init__(self, db_path: Path, embedding_provider: EmbeddingProvider):
        super().__init__(db_path, embedding_provider)
        self.db = None

        if not LANCEDB_AVAILABLE:
            raise RuntimeError("LanceDB not available. Install with: pip install lancedb")

    async def _get_db(self) -> Any:
        """Get or create database connection."""
        if self.db is None:
            # Use the globally imported lancedb
            import lancedb  # noqa: F811

            self.db = await lancedb.connect_async(str(self.db_path))
        return self.db

    async def create_collection(self, collection_name: str) -> None:
        """Create a new LanceDB table."""
        db = await self._get_db()

        # Create empty table with schema
        schema = {
            "id": "string",
            "vector": f"vector({self.embedding_provider.dimensions})",
            "content": "string",
            "file_path": "string",
            "start_line": "int32",
            "end_line": "int32",
            "language": "string",
            "chunk_type": "string",
            "metadata": "string",
        }

        try:
            await db.create_table(collection_name, schema=schema)
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise

    async def insert_vectors(
        self,
        collection_name: str,
        vectors: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None:
        """Insert vectors into LanceDB table."""
        if not vectors:
            return

        db = await self._get_db()
        table = await db.open_table(collection_name)

        # Prepare data for insertion
        data = []
        for vector_id, vector, metadata in vectors:
            data.append(
                {
                    "id": vector_id,
                    "vector": vector,
                    "content": metadata.get("content", ""),
                    "file_path": metadata.get("file_path", ""),
                    "start_line": metadata.get("start_line", 0),
                    "end_line": metadata.get("end_line", 0),
                    "language": metadata.get("language", "unknown"),
                    "chunk_type": metadata.get("chunk_type", "code"),
                    "metadata": json.dumps(metadata),
                }
            )

        await table.add(data)

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors in LanceDB."""
        db = await self._get_db()

        try:
            table = await db.open_table(collection_name)

            # Build query
            query = table.search(query_vector).limit(limit)

            # Apply filters if provided (with sanitization to prevent injection)
            if filters:
                for key, value in filters.items():
                    sanitized = self._sanitize_filter_value(str(value))
                    if key == "language":
                        query = query.where(f"language = '{sanitized}'")
                    elif key == "file_path":
                        query = query.where(f"file_path LIKE '%{sanitized}%'")
                    elif key == "chunk_type":
                        query = query.where(f"chunk_type = '{sanitized}'")

            # Execute query
            results = await query.to_list()

            # Convert to VectorSearchResult
            search_results = []
            for result in results:
                metadata = json.loads(result.get("metadata", "{}"))

                search_results.append(
                    VectorSearchResult(
                        chunk_id=result["id"],
                        content=result["content"],
                        file_path=result["file_path"],
                        start_line=result["start_line"],
                        end_line=result["end_line"],
                        similarity_score=1.0 - result.get("_distance", 0.0),
                        metadata=metadata,
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Error searching LanceDB: {e}")
            return []

    @staticmethod
    def _sanitize_filter_value(value: str) -> str:
        """Sanitize a value for use in SQL-like filter expressions."""
        # Escape single quotes and remove dangerous characters
        return value.replace("'", "''").replace(";", "").replace("--", "")

    async def delete_vectors(
        self,
        collection_name: str,
        vector_ids: list[str],
    ) -> None:
        """Delete vectors from LanceDB table."""
        if not vector_ids:
            return

        db = await self._get_db()
        table = await db.open_table(collection_name)

        # Sanitize IDs to prevent injection
        sanitized_ids = [
            self._sanitize_filter_value(vid) for vid in vector_ids
        ]
        id_list = "', '".join(sanitized_ids)
        condition = f"id IN ('{id_list}')"

        await table.delete(condition)

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if LanceDB table exists."""
        db = await self._get_db()
        table_names = await db.table_names()
        return collection_name in table_names


class QdrantProvider(VectorDatabase):
    """Qdrant vector database implementation conforming to VectorDatabase ABC.

    This bridges the standalone QdrantVectorStore (in qdrant_client.py) to
    the unified VectorDatabase interface used by VectorIndexManager.
    """

    def __init__(self, db_path: Path, embedding_provider: EmbeddingProvider):
        super().__init__(db_path, embedding_provider)
        self._store = None
        self._initialized = False

        if not QDRANT_AVAILABLE:
            raise RuntimeError("Qdrant not available. Install with: pip install qdrant-client numpy")

    async def _get_store(self) -> Any:
        """Get or create the underlying QdrantVectorStore."""
        if self._store is None:
            from .qdrant_client import QdrantConfig, QdrantVectorStore

            config = QdrantConfig(
                vector_size=self.embedding_provider.dimensions,
            )
            self._store = QdrantVectorStore(config)
            await self._store.initialize()
            self._initialized = True
        return self._store

    async def create_collection(self, collection_name: str) -> None:
        """Create a new Qdrant collection."""
        store = await self._get_store()
        await store.create_collection(
            collection_name,
            self.embedding_provider.dimensions,
        )

    async def insert_vectors(
        self,
        collection_name: str,
        vectors: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None:
        """Insert vectors into Qdrant collection."""
        if not vectors:
            return

        store = await self._get_store()

        ids = [vid for vid, _, _ in vectors]
        vecs = [vec for _, vec, _ in vectors]
        metadata = [meta for _, _, meta in vectors]

        await store.add_vectors(collection_name, vecs, metadata, ids=ids)

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors in Qdrant."""
        store = await self._get_store()

        try:
            results = await store.search_similar(
                query_vector=query_vector,
                collection_name=collection_name,
                top_k=limit,
                filter_conditions=filters,
            )

            search_results = []
            for result in results:
                payload = result.payload or {}
                search_results.append(
                    VectorSearchResult(
                        chunk_id=result.id,
                        content=payload.get("content", ""),
                        file_path=payload.get("file_path", ""),
                        start_line=payload.get("start_line", 0),
                        end_line=payload.get("end_line", 0),
                        similarity_score=result.score,
                        metadata=payload,
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return []

    async def batch_search_vectors(
        self,
        collection_name: str,
        query_vectors: list[list[float]],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[list[VectorSearchResult]]:
        """Optimized batch search using Qdrant's native batch_search API."""
        store = await self._get_store()

        try:
            batch_results = await store.batch_search(
                query_vectors=query_vectors,
                collection_name=collection_name,
                top_k=limit,
                filter_conditions=filters,
            )

            all_search_results: list[list[VectorSearchResult]] = []
            for single_result in batch_results:
                search_results = []
                for result in single_result:
                    payload = result.payload or {}
                    search_results.append(
                        VectorSearchResult(
                            chunk_id=result.id,
                            content=payload.get("content", ""),
                            file_path=payload.get("file_path", ""),
                            start_line=payload.get("start_line", 0),
                            end_line=payload.get("end_line", 0),
                            similarity_score=result.score,
                            metadata=payload,
                        )
                    )
                all_search_results.append(search_results)

            return all_search_results

        except Exception as e:
            logger.error(f"Error batch searching Qdrant: {e}")
            return [[] for _ in query_vectors]

    async def delete_vectors(
        self,
        collection_name: str,
        vector_ids: list[str],
    ) -> None:
        """Delete vectors from Qdrant collection."""
        if not vector_ids:
            return

        store = await self._get_store()
        await store.delete_vectors(collection_name, vector_ids)

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if Qdrant collection exists."""
        store = await self._get_store()
        try:
            info = await store.get_collection_info(collection_name)
            return bool(info)
        except Exception:
            return False


class ChromaProvider(VectorDatabase):
    """Chroma vector database implementation conforming to VectorDatabase ABC."""

    def __init__(self, db_path: Path, embedding_provider: EmbeddingProvider):
        super().__init__(db_path, embedding_provider)
        self._client = None

        if not CHROMA_AVAILABLE:
            raise RuntimeError("Chroma not available. Install with: pip install chromadb")

    def _get_client(self) -> Any:
        """Get or create Chroma client."""
        if self._client is None:
            import chromadb  # noqa: F811

            self._client = chromadb.PersistentClient(path=str(self.db_path))
        return self._client

    async def create_collection(self, collection_name: str) -> None:
        """Create a new Chroma collection."""
        client = self._get_client()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            ),
        )

    async def insert_vectors(
        self,
        collection_name: str,
        vectors: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None:
        """Insert vectors into Chroma collection."""
        if not vectors:
            return

        client = self._get_client()
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [vid for vid, _, _ in vectors]
        embeddings = [vec for _, vec, _ in vectors]
        documents = [meta.get("content", "") for _, _, meta in vectors]
        # Chroma requires metadata values to be str/int/float/bool
        metadatas = []
        for _, _, meta in vectors:
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif v is not None:
                    clean_meta[k] = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
            metadatas.append(clean_meta)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            ),
        )

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors in Chroma."""
        client = self._get_client()

        try:
            collection = client.get_collection(name=collection_name)

            # Build Chroma where filter
            where_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append({key: {"$eq": value}})
                if len(conditions) == 1:
                    where_filter = conditions[0]
                elif len(conditions) > 1:
                    where_filter = {"$and": conditions}

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: collection.query(
                    query_embeddings=[query_vector],
                    n_results=limit,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                ),
            )

            search_results = []
            if results and results.get("ids") and results["ids"][0]:
                ids = results["ids"][0]
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]

                for i, doc_id in enumerate(ids):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    distance = distances[i] if i < len(distances) else 1.0
                    # Chroma returns distances; convert to similarity (cosine: 1 - distance)
                    similarity = max(0.0, 1.0 - distance)

                    search_results.append(
                        VectorSearchResult(
                            chunk_id=doc_id,
                            content=documents[i] if i < len(documents) else "",
                            file_path=meta.get("file_path", ""),
                            start_line=int(meta.get("start_line", 0)),
                            end_line=int(meta.get("end_line", 0)),
                            similarity_score=similarity,
                            metadata=meta,
                        )
                    )

            return search_results

        except Exception as e:
            logger.error(f"Error searching Chroma: {e}")
            return []

    async def delete_vectors(
        self,
        collection_name: str,
        vector_ids: list[str],
    ) -> None:
        """Delete vectors from Chroma collection."""
        if not vector_ids:
            return

        client = self._get_client()

        try:
            collection = client.get_collection(name=collection_name)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: collection.delete(ids=vector_ids),
            )
        except Exception as e:
            logger.error(f"Error deleting vectors from Chroma: {e}")

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if Chroma collection exists."""
        client = self._get_client()
        try:
            existing = client.list_collections()
            # chromadb >= 0.5 returns list of Collection objects with .name
            names = set()
            for c in existing:
                if hasattr(c, "name"):
                    names.add(c.name)
                elif isinstance(c, str):
                    names.add(c)
            return collection_name in names
        except Exception:
            return False


class VectorIndexManager:
    """
    Manages vector indexing operations across multiple databases and providers.

    This manager coordinates embedding generation, vector storage, and retrieval
    operations, providing a unified interface for vector-based semantic search.
    """

    def __init__(
        self,
        db_path: Path,
        embedding_config: EmbeddingConfig,
        provider: str = "lancedb",
    ):
        self.db_path = db_path
        self.embedding_config = embedding_config
        self.provider_name = provider

        # Initialize embedding provider
        self.embedding_provider = self._create_embedding_provider(embedding_config)

        # Initialize vector database
        self.vector_db = self._create_vector_db(provider, db_path, self.embedding_provider)

    @staticmethod
    def _create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
        """Create the appropriate embedding provider based on config."""
        if config.provider == "openai":
            return OpenAIEmbeddingProvider(config)
        elif config.provider in ("huggingface", "sentence-transformers"):
            return HuggingFaceEmbeddingProvider(config)
        else:
            raise ValueError(
                f"Unsupported embedding provider: {config.provider}. "
                f"Supported: openai, huggingface"
            )

    @staticmethod
    def _create_vector_db(
        provider: str, db_path: Path, embedding_provider: EmbeddingProvider
    ) -> VectorDatabase:
        """Create the appropriate vector database provider."""
        if provider == "lancedb":
            if not LANCEDB_AVAILABLE:
                raise RuntimeError("LanceDB not available. Install with: pip install lancedb")
            return LanceDBProvider(db_path, embedding_provider)
        elif provider == "qdrant":
            if not QDRANT_AVAILABLE:
                raise RuntimeError("Qdrant not available. Install with: pip install qdrant-client numpy")
            return QdrantProvider(db_path, embedding_provider)
        elif provider == "chroma":
            if not CHROMA_AVAILABLE:
                raise RuntimeError("Chroma not available. Install with: pip install chromadb")
            return ChromaProvider(db_path, embedding_provider)
        else:
            raise ValueError(
                f"Unsupported vector database: {provider}. "
                f"Supported: lancedb, qdrant, chroma"
            )

    async def index_chunks(
        self,
        chunks: list[MetadataCodeChunk],
        collection_name: str,
    ) -> None:
        """Index code chunks in the vector database."""
        if not chunks:
            return

        logger.info(f"Indexing {len(chunks)} chunks in collection {collection_name}")

        # Create collection if it doesn't exist
        if not await self.vector_db.collection_exists(collection_name):
            await self.vector_db.create_collection(collection_name)

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding_provider.embed_texts(texts)

        # Prepare vectors for insertion
        vectors = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            metadata = {
                "content": chunk.content,
                "file_path": chunk.chunk_id.split(":")[0] if ":" in chunk.chunk_id else "",
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "language": chunk.language.value,
                "chunk_type": chunk.chunk_type,
                "entity_name": chunk.entity_name,
                "entity_type": chunk.entity_type.value if chunk.entity_type else None,
                "complexity_score": chunk.complexity_score,
                "quality_score": chunk.quality_score,
                "dependencies": chunk.dependencies,
            }

            vectors.append((chunk.chunk_id, embedding, metadata))

        # Insert vectors
        await self.vector_db.insert_vectors(collection_name, vectors)
        logger.info(f"Successfully indexed {len(vectors)} vectors")

    async def search(
        self,
        query: str,
        collection_name: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float = 0.0,
    ) -> list[VectorSearchResult]:
        """
        Search for semantically similar code chunks.

        Args:
            query: Search query text
            collection_name: Vector collection to search
            limit: Maximum number of results
            filters: Optional metadata filters
            similarity_threshold: Minimum similarity score

        Returns:
            List of search results sorted by similarity
        """
        # Generate query embedding
        query_vector = await self.embedding_provider.embed_query(query)

        # Search vector database
        results = await self.vector_db.search_vectors(collection_name, query_vector, limit, filters)

        # Filter by similarity threshold
        filtered_results = [
            result for result in results if result.similarity_score >= similarity_threshold
        ]

        return filtered_results

    async def batch_search(
        self,
        queries: list[str],
        collection_name: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float = 0.0,
    ) -> list[list[VectorSearchResult]]:
        """
        Batch search for semantically similar code chunks across multiple queries.

        Uses optimized batch operations when the underlying provider supports it
        (e.g., Qdrant's native batch search API), falling back to sequential
        search for other providers.

        Args:
            queries: List of search query texts
            collection_name: Vector collection to search
            limit: Maximum number of results per query
            filters: Optional metadata filters
            similarity_threshold: Minimum similarity score

        Returns:
            List of result lists, one per query
        """
        if not queries:
            return []

        # Generate query embeddings in batch
        query_vectors = await self.embedding_provider.embed_texts(queries)

        # Use batch search on the vector database
        all_results = await self.vector_db.batch_search_vectors(
            collection_name, query_vectors, limit, filters
        )

        # Filter by similarity threshold
        if similarity_threshold > 0.0:
            all_results = [
                [r for r in results if r.similarity_score >= similarity_threshold]
                for results in all_results
            ]

        return all_results

    async def delete_chunks(
        self,
        chunk_ids: list[str],
        collection_name: str,
    ) -> None:
        """Delete chunks from vector database."""
        await self.vector_db.delete_vectors(collection_name, chunk_ids)

    async def update_chunks(
        self,
        chunks: list[MetadataCodeChunk],
        collection_name: str,
    ) -> None:
        """Update existing chunks in vector database."""
        # Delete old versions
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        await self.delete_chunks(chunk_ids, collection_name)

        # Insert new versions
        await self.index_chunks(chunks, collection_name)

    def get_collection_name(self, tag: IndexTag) -> str:
        """Generate collection name from index tag."""
        # Clean collection name for database compatibility
        name = f"{tag.directory}_{tag.branch}_{tag.artifact_id}"
        # Replace invalid characters
        name = re.sub(r"[^\w\-_]", "_", name)
        return name

    async def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """Get statistics for a vector collection."""
        try:
            exists = await self.vector_db.collection_exists(collection_name)
            if not exists:
                return {
                    "total_vectors": 0,
                    "provider": self.provider_name,
                    "dimensions": self.embedding_provider.dimensions,
                }

            if isinstance(self.vector_db, LanceDBProvider):
                db = await self.vector_db._get_db()
                table = await db.open_table(collection_name)
                count = await table.count_rows()
                return {
                    "total_vectors": count,
                    "provider": "lancedb",
                    "dimensions": self.embedding_provider.dimensions,
                }
            elif isinstance(self.vector_db, QdrantProvider):
                store = await self.vector_db._get_store()
                info = await store.get_collection_info(collection_name)
                return {
                    "total_vectors": info.get("points_count", 0),
                    "provider": "qdrant",
                    "dimensions": self.embedding_provider.dimensions,
                    "indexed_vectors": info.get("indexed_vectors_count", 0),
                }
            elif isinstance(self.vector_db, ChromaProvider):
                client = self.vector_db._get_client()
                collection = client.get_collection(name=collection_name)
                count = collection.count()
                return {
                    "total_vectors": count,
                    "provider": "chroma",
                    "dimensions": self.embedding_provider.dimensions,
                }

            return {
                "total_vectors": 0,
                "provider": self.provider_name,
                "dimensions": self.embedding_provider.dimensions,
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    async def optimize_collection(self, collection_name: str) -> None:
        """Optimize vector collection for better performance."""
        try:
            if isinstance(self.vector_db, LanceDBProvider):
                db = await self.vector_db._get_db()
                if await self.vector_db.collection_exists(collection_name):
                    table = await db.open_table(collection_name)
                    await table.optimize()
                    logger.info(f"Optimized LanceDB collection: {collection_name}")
            elif isinstance(self.vector_db, QdrantProvider):
                # Qdrant automatically optimizes indexes; no explicit action needed
                logger.info(f"Qdrant collection {collection_name} uses automatic optimization")
            elif isinstance(self.vector_db, ChromaProvider):
                # Chroma auto-manages HNSW index; no explicit optimization
                logger.info(f"Chroma collection {collection_name} uses automatic HNSW optimization")
        except Exception as e:
            logger.error(f"Error optimizing collection {collection_name}: {e}")

    async def cleanup_orphaned_vectors(
        self, collection_name: str, valid_file_paths: set[str] | None = None
    ) -> int:
        """Remove orphaned vectors that no longer have corresponding files.

        Args:
            collection_name: The collection to clean up.
            valid_file_paths: Set of currently valid file paths. If provided,
                vectors whose file_path is not in this set will be removed.
                If None, checks file existence on disk.

        Returns:
            Number of orphaned vectors removed.
        """
        removed_count = 0

        try:
            if not await self.vector_db.collection_exists(collection_name):
                return 0

            # For LanceDB, scan all rows and check file_path validity
            if isinstance(self.vector_db, LanceDBProvider):
                db = await self.vector_db._get_db()
                table = await db.open_table(collection_name)
                rows = await table.to_list()

                orphaned_ids = []
                for row in rows:
                    file_path = row.get("file_path", "")
                    if not file_path:
                        continue
                    if valid_file_paths is not None:
                        if file_path not in valid_file_paths:
                            orphaned_ids.append(row["id"])
                    elif not Path(file_path).exists():
                        orphaned_ids.append(row["id"])

                if orphaned_ids:
                    await self.vector_db.delete_vectors(collection_name, orphaned_ids)
                    removed_count = len(orphaned_ids)

            elif isinstance(self.vector_db, QdrantProvider):
                store = await self.vector_db._get_store()
                # Qdrant scroll to iterate over all points
                scroll_result = await store._retry_operation(
                    lambda: store.client.scroll(
                        collection_name=collection_name,
                        limit=1000,
                        with_payload=True,
                        with_vectors=False,
                    )
                )
                if scroll_result:
                    points, _next_offset = scroll_result
                    orphaned_ids = []
                    for point in points:
                        file_path = (point.payload or {}).get("file_path", "")
                        if not file_path:
                            continue
                        if valid_file_paths is not None:
                            if file_path not in valid_file_paths:
                                orphaned_ids.append(str(point.id))
                        elif not Path(file_path).exists():
                            orphaned_ids.append(str(point.id))

                    if orphaned_ids:
                        await self.vector_db.delete_vectors(collection_name, orphaned_ids)
                        removed_count = len(orphaned_ids)

            elif isinstance(self.vector_db, ChromaProvider):
                client = self.vector_db._get_client()
                collection = client.get_collection(name=collection_name)
                all_data = collection.get(include=["metadatas"])

                if all_data and all_data.get("ids"):
                    orphaned_ids = []
                    for i, doc_id in enumerate(all_data["ids"]):
                        meta = all_data["metadatas"][i] if all_data.get("metadatas") else {}
                        file_path = (meta or {}).get("file_path", "")
                        if not file_path:
                            continue
                        if valid_file_paths is not None:
                            if file_path not in valid_file_paths:
                                orphaned_ids.append(doc_id)
                        elif not Path(file_path).exists():
                            orphaned_ids.append(doc_id)

                    if orphaned_ids:
                        await self.vector_db.delete_vectors(collection_name, orphaned_ids)
                        removed_count = len(orphaned_ids)

            if removed_count > 0:
                logger.info(f"Removed {removed_count} orphaned vectors from {collection_name}")

        except Exception as e:
            logger.error(f"Error cleaning up orphaned vectors in {collection_name}: {e}")

        return removed_count


class MultiProviderVectorManager:
    """
    Manages multiple vector database providers for redundancy and performance.

    This manager can coordinate multiple vector databases to provide redundancy,
    load balancing, and specialized indexing strategies.
    """

    def __init__(self, db_path: Path, embedding_config: EmbeddingConfig):
        self.db_path = db_path
        self.embedding_config = embedding_config
        self.providers: dict[str, VectorIndexManager] = {}
        self.primary_provider = "lancedb"

    async def add_provider(self, name: str, provider_type: str) -> None:
        """Add a vector database provider."""
        try:
            manager = VectorIndexManager(self.db_path / name, self.embedding_config, provider_type)
            self.providers[name] = manager
            logger.info(f"Added vector provider: {name} ({provider_type})")
        except Exception as e:
            logger.error(f"Error adding provider {name}: {e}")

    async def index_chunks_all_providers(
        self,
        chunks: list[MetadataCodeChunk],
        collection_name: str,
    ) -> None:
        """Index chunks across all providers."""
        tasks = []
        for _name, provider in self.providers.items():
            task = provider.index_chunks(chunks, collection_name)
            tasks.append(task)

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                provider_name = list(self.providers.keys())[i]
                logger.error(f"Error indexing with provider {provider_name}: {result}")

    async def search_best_provider(
        self,
        query: str,
        collection_name: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search using the best available provider."""
        # Try primary provider first
        if self.primary_provider in self.providers:
            try:
                return await self.providers[self.primary_provider].search(
                    query, collection_name, limit, filters
                )
            except Exception as e:
                logger.warning(f"Primary provider {self.primary_provider} failed: {e}")

        # Try other providers
        for name, provider in self.providers.items():
            if name != self.primary_provider:
                try:
                    return await provider.search(query, collection_name, limit, filters)
                except Exception as e:
                    logger.warning(f"Provider {name} failed: {e}")

        logger.error("All vector providers failed")
        return []

    async def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics from all providers."""
        stats = {}
        for name, provider in self.providers.items():
            try:
                provider_stats = await provider.get_collection_stats("default")
                stats[name] = provider_stats
            except Exception as e:
                logger.error(f"Error getting stats from provider {name}: {e}")
                stats[name] = {"error": str(e)}

        return stats
