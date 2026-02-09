"""
Qdrant vector database integration for pysearch.

This module provides comprehensive integration with Qdrant vector database
for enhanced semantic search capabilities, supporting vector embeddings,
similarity search, and efficient storage/retrieval operations.

Classes:
    QdrantConfig: Configuration for Qdrant connection and operations
    QdrantVectorStore: Main interface for Qdrant vector operations
    VectorSearchResult: Results from vector similarity searches

Features:
    - Qdrant client connection management with retry logic
    - Vector collection creation and management
    - Batch vector operations for efficient indexing
    - Similarity search with filtering capabilities
    - Error handling and connection recovery
    - Support for multiple vector configurations
    - Integration with existing semantic search

Example:
    Basic Qdrant integration:
        >>> from pysearch.storage.qdrant_client import QdrantVectorStore, QdrantConfig
        >>> config = QdrantConfig(host="localhost", port=6333)
        >>> vector_store = QdrantVectorStore(config)
        >>> await vector_store.initialize()
        >>>
        >>> # Add vectors
        >>> vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        >>> metadata = [{"file": "a.py", "entity": "func1"}, {"file": "b.py", "entity": "func2"}]
        >>> await vector_store.add_vectors("code_entities", vectors, metadata)
        >>>
        >>> # Search similar vectors
        >>> results = await vector_store.search_similar([0.1, 0.2, 0.3], top_k=5)

    Advanced usage with filtering:
        >>> from pysearch.analysis.graphrag.schema import EntityType
        >>> filter_conditions = {"entity_type": EntityType.FUNCTION.value}
        >>> results = await vector_store.search_similar(
        ...     query_vector=[0.1, 0.2, 0.3],
        ...     top_k=10,
        ...     filter_conditions=filter_conditions
        ... )
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

try:
    import numpy as np
    from qdrant_client import AsyncQdrantClient, QdrantClient, models
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

    QDRANT_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when qdrant not installed
    QDRANT_AVAILABLE = False
    np = None

    class _FallbackResponseHandlingException(Exception):
        pass

    class _FallbackUnexpectedResponse(Exception):
        pass

    # Assign fallback classes to the expected names
    ResponseHandlingException = _FallbackResponseHandlingException
    UnexpectedResponse = _FallbackUnexpectedResponse

    # Lightweight stubs to satisfy type checkers; runtime prevented before use
    class QdrantClient:  # type: ignore
        ...

    class AsyncQdrantClient:  # type: ignore
        ...

    class _StubDistance:
        COSINE = "Cosine"
        DOT = "Dot"
        EUCLID = "Euclid"

    class _StubModels:
        Distance = _StubDistance

        class VectorParams:
            def __init__(self, size: int, distance: Any) -> None:
                self.size = size
                self.distance = distance

        class PointStruct:
            def __init__(self, id: str, vector: list[float], payload: dict[str, Any]) -> None:
                self.id = id
                self.vector = vector
                self.payload = payload

        class FieldCondition:
            def __init__(self, key: str, match: Any) -> None:
                self.key = key
                self.match = match

        class MatchAny:
            def __init__(self, any: list[Any]) -> None:
                self.any = any

        class MatchValue:
            def __init__(self, value: Any) -> None:
                self.value = value

        class Filter:
            def __init__(self, must: list[Any]) -> None:
                self.must = must

        class PointIdsList:
            def __init__(self, points: list[str]) -> None:
                self.points = points

        class ScalarQuantizationConfig:
            def __init__(self, type: str = "int8") -> None:
                self.type = type

        class ScalarQuantization:
            def __init__(self, scalar: Any = None) -> None:
                self.scalar = scalar

    models = _StubModels()

if TYPE_CHECKING:  # Hints only; actual runtime handled above
    from qdrant_client import AsyncQdrantClient as _RealAsyncQdrantClient  # noqa: F401
    from qdrant_client import QdrantClient as _RealQdrantClient  # noqa: F401
    from qdrant_client import models as _real_models  # noqa: F401
    from qdrant_client.http.exceptions import ResponseHandlingException as _RHE  # noqa: F401
    from qdrant_client.http.exceptions import UnexpectedResponse as _UR  # noqa: F401

from ..utils.error_handling import SearchError

logger = logging.getLogger(__name__)


@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    api_key: str | None = None
    https: bool = False
    timeout: float = 30.0  # user-facing float; cast to int for client
    collection_name: str = "pysearch_vectors"
    vector_size: int = 384
    distance_metric: str = "Cosine"  # Cosine, Dot, Euclid
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100
    enable_compression: bool = False


@dataclass
class VectorSearchResult:
    id: str
    score: float
    payload: dict[str, Any]
    vector: list[float] | None = None


class QdrantVectorStore:
    def __init__(self, config: QdrantConfig) -> None:
        if not QDRANT_AVAILABLE:
            raise SearchError(
                "Qdrant dependencies not available. Install with: pip install qdrant-client numpy"
            )
        self.config = config
        self.client: QdrantClient | None = None
        self._async_client: AsyncQdrantClient | None = None
        self._initialized = False
        self._collections: set[str] = set()

    async def initialize(self) -> None:
        """Initialize the Qdrant client and create default collection."""
        try:
            timeout_int = int(self.config.timeout)

            # Primary: use AsyncQdrantClient for true async operations
            try:
                self._async_client = AsyncQdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                    api_key=self.config.api_key,
                    https=self.config.https,
                    timeout=timeout_int,
                )
                # Test async connection
                await self._async_client.get_collections()
                logger.debug("Using AsyncQdrantClient for async operations")
            except Exception:
                logger.debug("AsyncQdrantClient unavailable, falling back to sync client")
                self._async_client = None

            # Fallback: sync client (used when async client is not available)
            self.client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key,
                https=self.config.https,
                timeout=timeout_int,
            )

            # Test sync connection
            if self._async_client is None:
                client = self.client
                await self._retry_operation(lambda: client.get_collections())

            # Create default collection if it doesn't exist
            await self.create_collection(
                self.config.collection_name,
                self.config.vector_size,
                self.config.distance_metric,
            )

            self._initialized = True
            logger.info(
                f"Qdrant client initialized successfully: {self.config.host}:{self.config.port}"
            )

        except Exception as e:  # pragma: no cover - initialization error path
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise SearchError(f"Qdrant initialization failed: {e}") from e

    async def __aenter__(self) -> QdrantVectorStore:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def _retry_operation(self, operation: Callable[[], Any]) -> Any:
        """Execute operation with retry logic."""
        last_exception: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                result = operation()
                if asyncio.iscoroutine(result):
                    return await result
                return result
            except (ResponseHandlingException, UnexpectedResponse, ConnectionError) as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2**attempt))
                    logger.warning(
                        f"Qdrant operation failed, retrying ({attempt + 1}/{self.config.max_retries}): {e}"
                    )
                else:
                    logger.error(
                        f"Qdrant operation failed after {self.config.max_retries} attempts: {e}"
                    )

        raise SearchError(f"Qdrant operation failed: {last_exception}")

    def _get_effective_client(self) -> Any:
        """Return the async client if available, otherwise the sync client."""
        if self._async_client is not None:
            return self._async_client
        if self.client is not None:
            return self.client
        raise SearchError("Qdrant client not initialized")

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str = "Cosine",
    ) -> None:
        """Create a new vector collection."""
        effective_client = self._get_effective_client()

        try:
            if self._async_client is not None:
                collections = await self._async_client.get_collections()
            else:
                collections = await self._retry_operation(
                    lambda: effective_client.get_collections()
                )

            existing_names = {col.name for col in collections.collections}

            if collection_name not in existing_names:
                assert models is not None
                distance_map = {
                    "Cosine": models.Distance.COSINE,
                    "Dot": models.Distance.DOT,
                    "Euclid": models.Distance.EUCLID,
                }

                vectors_config = models.VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance_metric, models.Distance.COSINE),
                )

                # Build quantization config if compression is enabled
                quantization_config = None
                if self.config.enable_compression:
                    try:
                        quantization_config = models.ScalarQuantization(
                            scalar=models.ScalarQuantizationConfig(
                                type=models.ScalarType.INT8,
                                always_ram=True,
                            )
                        )
                    except (AttributeError, TypeError):
                        logger.debug(
                            "Scalar quantization not supported in this qdrant-client version"
                        )

                if self._async_client is not None:
                    await self._async_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=vectors_config,
                        quantization_config=quantization_config,
                    )
                else:
                    await self._retry_operation(
                        lambda: effective_client.create_collection(
                            collection_name=collection_name,
                            vectors_config=vectors_config,
                            quantization_config=quantization_config,
                        )
                    )
                logger.info(f"Created Qdrant collection: {collection_name}")

            self._collections.add(collection_name)

        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise SearchError(f"Collection creation failed: {e}") from e

    async def add_vectors(
        self,
        collection_name: str,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add vectors to a collection with associated metadata."""
        effective_client = self._get_effective_client()

        if len(vectors) != len(metadata):
            raise SearchError("Number of vectors must match number of metadata items")

        if ids and len(ids) != len(vectors):
            raise SearchError("Number of IDs must match number of vectors")

        if not ids:
            ids = [str(uuid4()) for _ in range(len(vectors))]

        try:
            batch_size = self.config.batch_size
            all_ids: list[str] = []

            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i : i + batch_size]
                batch_metadata = metadata[i : i + batch_size]
                batch_ids = ids[i : i + batch_size]

                assert models is not None
                points = [
                    models.PointStruct(id=point_id, vector=vector, payload=payload)
                    for point_id, vector, payload in zip(
                        batch_ids, batch_vectors, batch_metadata, strict=False
                    )
                ]

                # Capture points in lambda default arg to avoid closure issue
                if self._async_client is not None:
                    await self._async_client.upsert(
                        collection_name=collection_name,
                        points=points,
                    )
                else:
                    await self._retry_operation(
                        lambda _pts=points: effective_client.upsert(  # type: ignore[misc]
                            collection_name=collection_name,
                            points=_pts,
                        )
                    )

                all_ids.extend(batch_ids)
                logger.debug(f"Added batch of {len(points)} vectors to {collection_name}")

            logger.info(f"Successfully added {len(vectors)} vectors to {collection_name}")
            return all_ids

        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to add vectors to {collection_name}: {e}")
            raise SearchError(f"Vector addition failed: {e}") from e

    def _build_search_filter(self, filter_conditions: dict[str, Any]) -> Any:
        """Build a Qdrant filter from a dict of conditions."""
        assert models is not None
        conditions: list[Any] = []
        for key, value in filter_conditions.items():
            if isinstance(value, list):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )
        return models.Filter(must=conditions) if conditions else None

    async def search_similar(
        self,
        query_vector: list[float],
        collection_name: str | None = None,
        top_k: int = 10,
        filter_conditions: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors in the collection."""
        effective_client = self._get_effective_client()
        collection = collection_name or self.config.collection_name

        try:
            search_filter = None
            if filter_conditions:
                search_filter = self._build_search_filter(filter_conditions)

            if self._async_client is not None:
                search_result = await self._async_client.search(
                    collection_name=collection,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=search_filter,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
            else:
                search_result = await self._retry_operation(
                    lambda: effective_client.search(
                        collection_name=collection,
                        query_vector=query_vector,
                        limit=top_k,
                        query_filter=search_filter,
                        score_threshold=score_threshold,
                        with_payload=True,
                        with_vectors=False,
                    )
                )

            results: list[VectorSearchResult] = []
            for point in search_result:
                results.append(
                    VectorSearchResult(
                        id=str(point.id),
                        score=point.score,
                        payload=point.payload or {},
                    )
                )

            logger.debug(f"Found {len(results)} similar vectors in {collection}")
            return results

        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to search vectors in {collection}: {e}")
            raise SearchError(f"Vector search failed: {e}") from e

    async def batch_search(
        self,
        query_vectors: list[list[float]],
        collection_name: str | None = None,
        top_k: int = 10,
        filter_conditions: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[list[VectorSearchResult]]:
        """Search for similar vectors for multiple queries in a single batch call.

        This leverages Qdrant's batch search API to reduce network overhead
        when performing multiple queries.

        Args:
            query_vectors: List of query vectors.
            collection_name: Collection to search.
            top_k: Max results per query.
            filter_conditions: Optional shared filters.
            score_threshold: Minimum similarity score.

        Returns:
            List of result lists, one per query vector.
        """
        effective_client = self._get_effective_client()
        collection = collection_name or self.config.collection_name

        try:
            search_filter = None
            if filter_conditions:
                search_filter = self._build_search_filter(filter_conditions)

            assert models is not None
            search_requests = [
                models.SearchRequest(
                    vector=qv,
                    limit=top_k,
                    filter=search_filter,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vector=False,
                )
                for qv in query_vectors
            ]

            if self._async_client is not None:
                batch_results = await self._async_client.search_batch(
                    collection_name=collection,
                    requests=search_requests,
                )
            else:
                batch_results = await self._retry_operation(
                    lambda: effective_client.search_batch(
                        collection_name=collection,
                        requests=search_requests,
                    )
                )

            all_results: list[list[VectorSearchResult]] = []
            for single_result in batch_results:
                results: list[VectorSearchResult] = []
                for point in single_result:
                    results.append(
                        VectorSearchResult(
                            id=str(point.id),
                            score=point.score,
                            payload=point.payload or {},
                        )
                    )
                all_results.append(results)

            logger.debug(
                f"Batch search: {len(query_vectors)} queries, "
                f"{sum(len(r) for r in all_results)} total results in {collection}"
            )
            return all_results

        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to batch search in {collection}: {e}")
            raise SearchError(f"Batch vector search failed: {e}") from e

    async def delete_vectors(self, collection_name: str, vector_ids: list[str]) -> None:
        """Delete vectors from a collection."""
        effective_client = self._get_effective_client()

        try:
            assert models is not None
            selector = models.PointIdsList(points=list(vector_ids))

            if self._async_client is not None:
                await self._async_client.delete(
                    collection_name=collection_name,
                    points_selector=selector,
                )
            else:
                await self._retry_operation(
                    lambda: effective_client.delete(
                        collection_name=collection_name,
                        points_selector=selector,
                    )
                )
            logger.info(f"Deleted {len(vector_ids)} vectors from {collection_name}")

        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to delete vectors from {collection_name}: {e}")
            raise SearchError(f"Vector deletion failed: {e}") from e

    async def update_vector_metadata(
        self,
        collection_name: str,
        vector_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Update metadata for a specific vector."""
        effective_client = self._get_effective_client()

        try:
            if self._async_client is not None:
                await self._async_client.set_payload(
                    collection_name=collection_name,
                    payload=metadata,
                    points=[vector_id],
                )
            else:
                await self._retry_operation(
                    lambda: effective_client.set_payload(
                        collection_name=collection_name,
                        payload=metadata,
                        points=[vector_id],
                    )
                )
            logger.debug(f"Updated metadata for vector {vector_id} in {collection_name}")

        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to update metadata for vector {vector_id}: {e}")
            raise SearchError(f"Metadata update failed: {e}") from e

    async def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get information about a collection."""
        effective_client = self._get_effective_client()

        try:
            if self._async_client is not None:
                info = await self._async_client.get_collection(collection_name)
            else:
                info = await self._retry_operation(
                    lambda: effective_client.get_collection(collection_name)
                )

            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.value,
                },
            }

        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to get collection info for {collection_name}: {e}")
            raise SearchError(f"Collection info retrieval failed: {e}") from e

    async def list_collections(self) -> list[str]:
        """List all collections in the Qdrant instance."""
        effective_client = self._get_effective_client()

        try:
            if self._async_client is not None:
                collections = await self._async_client.get_collections()
            else:
                collections = await self._retry_operation(
                    lambda: effective_client.get_collections()
                )
            return [col.name for col in collections.collections]

        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to list collections: {e}")
            raise SearchError(f"Collection listing failed: {e}") from e

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from Qdrant."""
        effective_client = self._get_effective_client()

        try:
            if self._async_client is not None:
                await self._async_client.delete_collection(collection_name)
            else:
                await self._retry_operation(
                    lambda: effective_client.delete_collection(collection_name)
                )
            self._collections.discard(collection_name)
            logger.info(f"Deleted Qdrant collection: {collection_name}")

        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise SearchError(f"Collection deletion failed: {e}") from e

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        try:
            if self._async_client is not None:
                await self._async_client.close()
                self._async_client = None
            self.client = None
            self._initialized = False
            logger.info("Qdrant client connection closed")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Error closing Qdrant client: {e}")

    def is_available(self) -> bool:
        """Check if Qdrant is available and client is initialized."""
        return bool(QDRANT_AVAILABLE and self._initialized and (self.client or self._async_client))


def normalize_vector(vector: list[float]) -> list[float]:
    """Normalize a vector to unit length."""
    if not QDRANT_AVAILABLE or np is None:
        magnitude = sum(x * x for x in vector) ** 0.5
        if magnitude == 0:
            return vector
        return [x / magnitude for x in vector]

    np_vector = np.array(vector)
    norm = np.linalg.norm(np_vector)
    if norm == 0:
        return vector
    return (np_vector / norm).tolist()


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension")

    if not QDRANT_AVAILABLE or np is None:
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))
