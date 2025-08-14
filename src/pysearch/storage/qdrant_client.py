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
        >>> from pysearch.qdrant_client import QdrantVectorStore, QdrantConfig
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
        >>> from pysearch.types import EntityType
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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

try:
    import numpy as np
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
    QDRANT_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when qdrant not installed
    QDRANT_AVAILABLE = False
    np = None  # type: ignore

    class ResponseHandlingException(Exception):
        pass

    class UnexpectedResponse(Exception):
        pass

    # Lightweight stubs to satisfy type checkers; runtime prevented before use
    class QdrantClient:  # type: ignore
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
            def __init__(self, id: str, vector: List[float], payload: Dict[str, Any]) -> None:
                self.id = id
                self.vector = vector
                self.payload = payload

        class FieldCondition:
            def __init__(self, key: str, match: Any) -> None:
                self.key = key
                self.match = match

        class MatchAny:
            def __init__(self, any: List[Any]) -> None:
                self.any = any

        class MatchValue:
            def __init__(self, value: Any) -> None:
                self.value = value

        class Filter:
            def __init__(self, must: List[Any]) -> None:
                self.must = must

        class PointIdsList:
            def __init__(self, points: List[str]) -> None:
                self.points = points

    models = _StubModels()  # type: ignore

if TYPE_CHECKING:  # Hints only; actual runtime handled above
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
    api_key: Optional[str] = None
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
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


class QdrantVectorStore:
    def __init__(self, config: QdrantConfig) -> None:
        if not QDRANT_AVAILABLE:
            raise SearchError(
                "Qdrant dependencies not available. Install with: pip install qdrant-client numpy"
            )
        self.config = config
        self.client: "QdrantClient | None" = None
        self._initialized = False
        self._collections: set[str] = set()

    async def initialize(self) -> None:
        """Initialize the Qdrant client and create default collection."""
        try:
            # Cast timeout to int to satisfy qdrant-client typing (int | None expected)
            timeout_int = int(self.config.timeout)
            self.client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key,
                https=self.config.https,
                timeout=timeout_int,
            )

            # Test connection
            if self.client:
                await self._retry_operation(lambda: self.client.get_collections())

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
            raise SearchError(f"Qdrant initialization failed: {e}")

    async def _retry_operation(
        self, operation: Callable[[], Any]
    ) -> Any:
        """Execute operation with retry logic."""
        last_exception: Optional[Exception] = None

        for attempt in range(self.config.max_retries):
            try:
                result = operation()
                if asyncio.iscoroutine(result):
                    return await result
                return result
            except (ResponseHandlingException, UnexpectedResponse, ConnectionError) as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    logger.warning(
                        f"Qdrant operation failed, retrying ({attempt + 1}/{self.config.max_retries}): {e}"
                    )
                else:
                    logger.error(
                        f"Qdrant operation failed after {self.config.max_retries} attempts: {e}"
                    )

        raise SearchError(f"Qdrant operation failed: {last_exception}")

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str = "Cosine",
    ) -> None:
        """Create a new vector collection."""
        client = self.client
        if client is None:
            raise SearchError("Qdrant client not initialized")

        try:
            collections = await self._retry_operation(lambda: client.get_collections())
            existing_names = {col.name for col in collections.collections}

            if collection_name not in existing_names:
                # Ensure models is available (runtime guard already passed in __init__)
                assert models is not None  # for type checker
                distance_map = {
                    "Cosine": models.Distance.COSINE,
                    "Dot": models.Distance.DOT,
                    "Euclid": models.Distance.EUCLID,
                }

                await self._retry_operation(
                    lambda: client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=vector_size,
                            distance=distance_map.get(
                                distance_metric, models.Distance.COSINE
                            ),
                        ),
                    )
                )
                logger.info(f"Created Qdrant collection: {collection_name}")

            self._collections.add(collection_name)

        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise SearchError(f"Collection creation failed: {e}")

    async def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add vectors to a collection with associated metadata."""
        client = self.client
        if client is None:
            raise SearchError("Qdrant client not initialized")

        if len(vectors) != len(metadata):
            raise SearchError(
                "Number of vectors must match number of metadata items"
            )

        if ids and len(ids) != len(vectors):
            raise SearchError("Number of IDs must match number of vectors")

        if not ids:
            ids = [str(uuid4()) for _ in range(len(vectors))]

        try:
            batch_size = self.config.batch_size
            all_ids: List[str] = []

            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i : i + batch_size]
                batch_metadata = metadata[i : i + batch_size]
                batch_ids = ids[i : i + batch_size]

                assert models is not None
                points = [
                    models.PointStruct(
                        id=point_id, vector=vector, payload=payload
                    )
                    for point_id, vector, payload in zip(
                        batch_ids, batch_vectors, batch_metadata
                    )
                ]

                await self._retry_operation(
                    lambda: client.upsert(
                        collection_name=collection_name,
                        points=points,
                    )
                )

                all_ids.extend(batch_ids)
                logger.debug(
                    f"Added batch of {len(points)} vectors to {collection_name}"
                )

            logger.info(
                f"Successfully added {len(vectors)} vectors to {collection_name}"
            )
            return all_ids

        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to add vectors to {collection_name}: {e}")
            raise SearchError(f"Vector addition failed: {e}")

    async def search_similar(
        self,
        query_vector: List[float],
        collection_name: Optional[str] = None,
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors in the collection."""
        client = self.client
        if client is None:
            raise SearchError("Qdrant client not initialized")

        collection = collection_name or self.config.collection_name

        try:
            search_filter = None
            if filter_conditions:
                assert models is not None
                conditions: List[Any] = []
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
                if conditions:
                    search_filter = models.Filter(
                        must=conditions
                    )

            search_result = await self._retry_operation(
                lambda: client.search(
                    collection_name=collection,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=search_filter,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
            )

            results: List[VectorSearchResult] = []
            for point in search_result:
                results.append(
                    VectorSearchResult(
                        id=str(point.id),
                        score=point.score,
                        payload=point.payload or {},
                    )
                )

            logger.debug(
                f"Found {len(results)} similar vectors in {collection}"
            )
            return results

        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to search vectors in {collection}: {e}")
            raise SearchError(f"Vector search failed: {e}")

    async def delete_vectors(
        self, collection_name: str, vector_ids: List[str]
    ) -> None:
        """Delete vectors from a collection."""
        client = self.client
        if client is None:
            raise SearchError("Qdrant client not initialized")

        try:
            assert models is not None
            await self._retry_operation(
                lambda: client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(
                        points=vector_ids
                    ),
                )
            )
            logger.info(
                f"Deleted {len(vector_ids)} vectors from {collection_name}"
            )

        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to delete vectors from {collection_name}: {e}")
            raise SearchError(f"Vector deletion failed: {e}")

    async def update_vector_metadata(
        self,
        collection_name: str,
        vector_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Update metadata for a specific vector."""
        client = self.client
        if client is None:
            raise SearchError("Qdrant client not initialized")

        try:
            await self._retry_operation(
                lambda: client.set_payload(
                    collection_name=collection_name,
                    payload=metadata,
                    points=[vector_id],
                )
            )
            logger.debug(
                f"Updated metadata for vector {vector_id} in {collection_name}"
            )

        except Exception as e:  # pragma: no cover
            logger.error(
                f"Failed to update metadata for vector {vector_id}: {e}"
            )
            raise SearchError(f"Metadata update failed: {e}")

    async def get_collection_info(
        self, collection_name: str
    ) -> Dict[str, Any]:
        """Get information about a collection."""
        client = self.client
        if client is None:
            raise SearchError("Qdrant client not initialized")

        try:
            info = await self._retry_operation(
                lambda: client.get_collection(collection_name)
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
            logger.error(
                f"Failed to get collection info for {collection_name}: {e}"
            )
            raise SearchError(f"Collection info retrieval failed: {e}")

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        if self.client:
            try:
                self.client = None
                self._initialized = False
                logger.info("Qdrant client connection closed")
            except Exception as e:  # pragma: no cover
                logger.warning(f"Error closing Qdrant client: {e}")

    def is_available(self) -> bool:
        """Check if Qdrant is available and client is initialized."""
        return bool(QDRANT_AVAILABLE and self._initialized and self.client)


def normalize_vector(vector: List[float]) -> List[float]:
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
    return (np_vector / norm).tolist()  # type: ignore[no-any-return]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension")

    if not QDRANT_AVAILABLE or np is None:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))
