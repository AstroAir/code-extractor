"""
Tests for Qdrant vector database integration.

This module contains tests for the Qdrant vector database integration including
configuration, connection management, vector operations, and error handling.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pysearch.qdrant_client import (
    QdrantConfig, QdrantVectorStore, VectorSearchResult,
    normalize_vector, cosine_similarity
)


class TestQdrantConfig:
    """Test QdrantConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = QdrantConfig()
        
        assert config.host == "localhost"
        assert config.port == 6333
        assert config.api_key is None
        assert config.https is False
        assert config.timeout == 30.0
        assert config.collection_name == "pysearch_vectors"
        assert config.vector_size == 384
        assert config.distance_metric == "Cosine"
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.batch_size == 100
        assert config.enable_compression is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = QdrantConfig(
            host="remote-host",
            port=6334,
            api_key="test-key",
            https=True,
            timeout=60.0,
            collection_name="custom_collection",
            vector_size=512,
            distance_metric="Dot",
            max_retries=5,
            retry_delay=2.0,
            batch_size=200,
            enable_compression=True
        )
        
        assert config.host == "remote-host"
        assert config.port == 6334
        assert config.api_key == "test-key"
        assert config.https is True
        assert config.timeout == 60.0
        assert config.collection_name == "custom_collection"
        assert config.vector_size == 512
        assert config.distance_metric == "Dot"
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.batch_size == 200
        assert config.enable_compression is True


class TestVectorSearchResult:
    """Test VectorSearchResult functionality."""
    
    def test_init(self):
        """Test VectorSearchResult initialization."""
        result = VectorSearchResult(
            id="test_id",
            score=0.95,
            payload={"entity_type": "function", "name": "test_func"},
            vector=[0.1, 0.2, 0.3]
        )
        
        assert result.id == "test_id"
        assert result.score == 0.95
        assert result.payload == {"entity_type": "function", "name": "test_func"}
        assert result.vector == [0.1, 0.2, 0.3]
    
    def test_init_without_vector(self):
        """Test VectorSearchResult initialization without vector."""
        result = VectorSearchResult(
            id="test_id",
            score=0.85,
            payload={"entity_type": "class"}
        )
        
        assert result.id == "test_id"
        assert result.score == 0.85
        assert result.payload == {"entity_type": "class"}
        assert result.vector is None


class TestQdrantVectorStore:
    """Test QdrantVectorStore functionality."""
    
    def test_init_without_qdrant(self):
        """Test initialization when Qdrant is not available."""
        config = QdrantConfig()
        
        with patch('pysearch.qdrant_client.QDRANT_AVAILABLE', False):
            with pytest.raises(Exception) as exc_info:
                QdrantVectorStore(config)
            
            assert "Qdrant dependencies not available" in str(exc_info.value)
    
    @patch('pysearch.qdrant_client.QDRANT_AVAILABLE', True)
    @patch('pysearch.qdrant_client.QdrantClient')
    def test_init_with_qdrant(self, mock_qdrant_client):
        """Test initialization when Qdrant is available."""
        config = QdrantConfig()
        vector_store = QdrantVectorStore(config)
        
        assert vector_store.config == config
        assert vector_store.client is None
        assert vector_store._initialized is False
        assert len(vector_store._collections) == 0
    
    @patch('pysearch.qdrant_client.QDRANT_AVAILABLE', True)
    @patch('pysearch.qdrant_client.QdrantClient')
    @patch('pysearch.qdrant_client.models')
    @pytest.mark.asyncio
    async def test_initialize(self, mock_models, mock_qdrant_client):
        """Test vector store initialization."""
        config = QdrantConfig()
        vector_store = QdrantVectorStore(config)
        
        # Mock client and methods
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client.create_collection = MagicMock()
        
        await vector_store.initialize()
        
        assert vector_store.client is not None
        assert vector_store._initialized is True
        mock_qdrant_client.assert_called_once()
        mock_client.get_collections.assert_called_once()
    
    @patch('pysearch.qdrant_client.QDRANT_AVAILABLE', True)
    @patch('pysearch.qdrant_client.QdrantClient')
    @patch('pysearch.qdrant_client.models')
    @pytest.mark.asyncio
    async def test_add_vectors(self, mock_models, mock_qdrant_client):
        """Test adding vectors to collection."""
        config = QdrantConfig()
        vector_store = QdrantVectorStore(config)
        
        # Mock client
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        vector_store.client = mock_client
        vector_store._initialized = True
        
        # Test data
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadata = [{"name": "func1"}, {"name": "func2"}]
        
        # Mock upsert
        mock_client.upsert = MagicMock()
        
        result_ids = await vector_store.add_vectors("test_collection", vectors, metadata)
        
        assert len(result_ids) == 2
        mock_client.upsert.assert_called_once()
    
    @patch('pysearch.qdrant_client.QDRANT_AVAILABLE', True)
    @patch('pysearch.qdrant_client.QdrantClient')
    @patch('pysearch.qdrant_client.models')
    @pytest.mark.asyncio
    async def test_search_similar(self, mock_models, mock_qdrant_client):
        """Test similarity search."""
        config = QdrantConfig()
        vector_store = QdrantVectorStore(config)
        
        # Mock client
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        vector_store.client = mock_client
        vector_store._initialized = True
        
        # Mock search results
        mock_point = MagicMock()
        mock_point.id = "test_id"
        mock_point.score = 0.95
        mock_point.payload = {"name": "test_func"}
        mock_client.search.return_value = [mock_point]
        
        query_vector = [0.1, 0.2, 0.3]
        results = await vector_store.search_similar(query_vector)
        
        assert len(results) == 1
        assert results[0].id == "test_id"
        assert results[0].score == 0.95
        assert results[0].payload == {"name": "test_func"}
        mock_client.search.assert_called_once()
    
    @patch('pysearch.qdrant_client.QDRANT_AVAILABLE', True)
    @patch('pysearch.qdrant_client.QdrantClient')
    @pytest.mark.asyncio
    async def test_close(self, mock_qdrant_client):
        """Test closing the vector store."""
        config = QdrantConfig()
        vector_store = QdrantVectorStore(config)
        vector_store.client = MagicMock()
        vector_store._initialized = True
        
        await vector_store.close()
        
        assert vector_store.client is None
        assert vector_store._initialized is False
    
    @patch('pysearch.qdrant_client.QDRANT_AVAILABLE', True)
    def test_is_available(self):
        """Test availability check."""
        config = QdrantConfig()
        vector_store = QdrantVectorStore(config)
        
        # Not initialized
        assert vector_store.is_available() is False
        
        # Initialized
        vector_store._initialized = True
        vector_store.client = MagicMock()
        assert vector_store.is_available() is True


class TestVectorUtilities:
    """Test vector utility functions."""
    
    def test_normalize_vector_without_numpy(self):
        """Test vector normalization without numpy."""
        with patch('pysearch.qdrant_client.QDRANT_AVAILABLE', False):
            vector = [3.0, 4.0]
            normalized = normalize_vector(vector)
            
            # Should be unit vector
            magnitude = sum(x * x for x in normalized) ** 0.5
            assert abs(magnitude - 1.0) < 1e-6
    
    def test_normalize_vector_zero_vector(self):
        """Test normalization of zero vector."""
        with patch('pysearch.qdrant_client.QDRANT_AVAILABLE', False):
            vector = [0.0, 0.0, 0.0]
            normalized = normalize_vector(vector)
            
            # Should return original vector
            assert normalized == vector
    
    @patch('pysearch.qdrant_client.QDRANT_AVAILABLE', True)
    @patch('pysearch.qdrant_client.np')
    def test_normalize_vector_with_numpy(self, mock_np):
        """Test vector normalization with numpy."""
        # Mock numpy operations
        mock_array = MagicMock()
        mock_np.array.return_value = mock_array
        mock_np.linalg.norm.return_value = 5.0
        mock_array.__truediv__.return_value.tolist.return_value = [0.6, 0.8]
        
        vector = [3.0, 4.0]
        normalized = normalize_vector(vector)
        
        assert normalized == [0.6, 0.8]
        mock_np.array.assert_called_once_with(vector)
        mock_np.linalg.norm.assert_called_once()
    
    def test_cosine_similarity_without_numpy(self):
        """Test cosine similarity without numpy."""
        with patch('pysearch.qdrant_client.QDRANT_AVAILABLE', False):
            vec1 = [1.0, 0.0]
            vec2 = [0.0, 1.0]
            similarity = cosine_similarity(vec1, vec2)
            
            # Orthogonal vectors should have similarity 0
            assert abs(similarity - 0.0) < 1e-6
            
            # Identical vectors should have similarity 1
            similarity = cosine_similarity(vec1, vec1)
            assert abs(similarity - 1.0) < 1e-6
    
    def test_cosine_similarity_different_dimensions(self):
        """Test cosine similarity with different vector dimensions."""
        vec1 = [1.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        
        with pytest.raises(ValueError) as exc_info:
            cosine_similarity(vec1, vec2)
        
        assert "same dimension" in str(exc_info.value)
    
    def test_cosine_similarity_zero_vectors(self):
        """Test cosine similarity with zero vectors."""
        with patch('pysearch.qdrant_client.QDRANT_AVAILABLE', False):
            vec1 = [0.0, 0.0]
            vec2 = [1.0, 0.0]
            similarity = cosine_similarity(vec1, vec2)
            
            # Zero vector should have similarity 0 with any vector
            assert similarity == 0.0
    
    @patch('pysearch.qdrant_client.QDRANT_AVAILABLE', True)
    @patch('pysearch.qdrant_client.np')
    def test_cosine_similarity_with_numpy(self, mock_np):
        """Test cosine similarity with numpy."""
        # Mock numpy operations
        mock_v1 = MagicMock()
        mock_v2 = MagicMock()
        mock_np.array.side_effect = [mock_v1, mock_v2]
        mock_np.dot.return_value = 0.8
        mock_np.linalg.norm.side_effect = [1.0, 1.0]
        
        vec1 = [0.8, 0.6]
        vec2 = [0.6, 0.8]
        similarity = cosine_similarity(vec1, vec2)
        
        assert similarity == 0.8
        assert mock_np.array.call_count == 2
        mock_np.dot.assert_called_once()
        assert mock_np.linalg.norm.call_count == 2


@pytest.mark.integration
class TestQdrantIntegration:
    """Integration tests for Qdrant functionality."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("qdrant_client", reason="Qdrant client not available"),
        reason="Qdrant not available"
    )
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete Qdrant workflow (requires running Qdrant instance)."""
        # This test would require a running Qdrant instance
        # For now, we'll skip it in CI/CD environments
        pytest.skip("Requires running Qdrant instance")
        
        config = QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="test_collection"
        )
        
        vector_store = QdrantVectorStore(config)
        
        try:
            # Initialize
            await vector_store.initialize()
            
            # Add vectors
            vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            metadata = [{"name": "func1"}, {"name": "func2"}]
            ids = await vector_store.add_vectors("test_collection", vectors, metadata)
            
            assert len(ids) == 2
            
            # Search
            results = await vector_store.search_similar([0.1, 0.2, 0.3], top_k=1)
            assert len(results) >= 1
            
        finally:
            await vector_store.close()
