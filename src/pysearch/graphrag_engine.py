"""
GraphRAG Engine for pysearch - Main orchestrator for GraphRAG operations.

This module provides the main GraphRAG engine that coordinates entity extraction,
relationship mapping, knowledge graph building, and graph-based retrieval operations.

Classes:
    KnowledgeGraphBuilder: Builds and maintains knowledge graphs
    GraphRAGEngine: Main engine for GraphRAG operations

Features:
    - Incremental knowledge graph building
    - Vector embedding integration
    - Graph-based query processing
    - Semantic similarity search with graph context
    - Efficient graph storage and retrieval
    - Integration with existing search infrastructure

Example:
    Complete GraphRAG workflow:
        >>> from pysearch.graphrag_engine import GraphRAGEngine
        >>> from pysearch.config import SearchConfig
        >>> from pysearch.types import GraphRAGQuery
        >>> 
        >>> # Initialize engine
        >>> config = SearchConfig(paths=["./src"])
        >>> engine = GraphRAGEngine(config)
        >>> await engine.initialize()
        >>> 
        >>> # Build knowledge graph
        >>> await engine.build_knowledge_graph()
        >>> 
        >>> # Query the graph
        >>> query = GraphRAGQuery(
        ...     pattern="database connection handling",
        ...     entity_types=[EntityType.FUNCTION, EntityType.CLASS],
        ...     max_hops=2
        ... )
        >>> results = await engine.query_graph(query)
        >>> 
        >>> # Process results
        >>> for entity in results.entities:
        ...     print(f"Found: {entity.name} ({entity.entity_type})")
        ...     for related_entity, relationship in results.relationships:
        ...         print(f"  -> {relationship.relation_type}: {related_entity.name}")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import SearchConfig
from .graphrag import EntityExtractor, RelationshipMapper
from .qdrant_client import QdrantVectorStore, QdrantConfig
from .semantic_advanced import SemanticEmbedding
from .types import (
    CodeEntity, EntityRelationship, EntityType, GraphRAGQuery, GraphRAGResult,
    KnowledgeGraph, RelationType
)
from .utils import read_text_safely

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Builds and maintains knowledge graphs from code analysis.
    
    This class coordinates the entity extraction and relationship mapping
    processes to build comprehensive knowledge graphs of codebases.
    """
    
    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.entity_extractor = EntityExtractor()
        self.relationship_mapper = RelationshipMapper()
        self.semantic_embedding = SemanticEmbedding()
        self.knowledge_graph = KnowledgeGraph()
        self._cache_path = config.resolve_cache_dir() / "knowledge_graph.json"
    
    async def build_graph(self, force_rebuild: bool = False) -> KnowledgeGraph:
        """Build the knowledge graph from the configured paths."""
        if not force_rebuild and await self._load_cached_graph():
            logger.info("Loaded knowledge graph from cache")
            return self.knowledge_graph
        
        logger.info("Building knowledge graph from source code...")
        start_time = time.time()
        
        # Extract entities from all files
        all_entities = []
        file_contents = {}
        
        for path_str in self.config.paths:
            path = Path(path_str)
            if path.is_file():
                entities = await self.entity_extractor.extract_from_file(path)
                all_entities.extend(entities)
                file_contents[path] = read_text_safely(path) or ""
            else:
                entities = await self.entity_extractor.extract_from_directory(path, self.config)
                all_entities.extend(entities)
                
                # Load file contents for relationship mapping
                from .indexer import Indexer
                indexer = Indexer(self.config)
                changed_files, removed_files, total_files = indexer.scan()
                for file_path in changed_files:
                    file_contents[file_path] = read_text_safely(file_path) or ""
        
        logger.info(f"Extracted {len(all_entities)} entities")
        
        # Generate embeddings for entities
        await self._generate_embeddings(all_entities)
        
        # Map relationships between entities
        relationships = await self.relationship_mapper.map_relationships(all_entities, file_contents)
        
        # Build the knowledge graph
        self.knowledge_graph = KnowledgeGraph()
        self.knowledge_graph.created_at = time.time()
        
        for entity in all_entities:
            self.knowledge_graph.add_entity(entity)
        
        for relationship in relationships:
            self.knowledge_graph.add_relationship(relationship)
        
        # Update metadata
        self.knowledge_graph.metadata = {
            "total_entities": len(all_entities),
            "total_relationships": len(relationships),
            "entity_types": {
                entity_type.value: len(self.knowledge_graph.get_entities_by_type(entity_type))
                for entity_type in EntityType
            },
            "build_time_seconds": time.time() - start_time,
            "source_paths": self.config.paths
        }
        
        # Cache the graph
        await self._cache_graph()
        
        elapsed = time.time() - start_time
        logger.info(f"Built knowledge graph with {len(all_entities)} entities and {len(relationships)} relationships in {elapsed:.2f}s")
        
        return self.knowledge_graph
    
    async def _generate_embeddings(self, entities: List[CodeEntity]) -> None:
        """Generate vector embeddings for entities."""
        if not entities:
            return
        
        # Prepare documents for embedding
        documents = []
        for entity in entities:
            # Create a text representation of the entity
            text_parts = [entity.name]
            
            if entity.signature:
                text_parts.append(entity.signature)
            
            if entity.docstring:
                text_parts.append(entity.docstring)
            
            # Add context from properties
            if entity.properties:
                for key, value in entity.properties.items():
                    if isinstance(value, str):
                        text_parts.append(f"{key}: {value}")
                    elif isinstance(value, list):
                        text_parts.append(f"{key}: {', '.join(str(v) for v in value)}")
            
            document = " ".join(text_parts)
            documents.append(document)
        
        # Fit the embedding model if not already fitted
        if not self.semantic_embedding.is_fitted:
            self.semantic_embedding.fit(documents)
        
        # Generate embeddings
        for entity, document in zip(entities, documents):
            embedding_vector = self.semantic_embedding.transform(document)
            # Convert sparse vector to dense list
            if embedding_vector:
                max_dim = max(embedding_vector.keys()) + 1 if embedding_vector else 0
                dense_vector = [0.0] * max_dim
                for dim, value in embedding_vector.items():
                    dense_vector[dim] = value
                entity.embedding = dense_vector
    
    async def _load_cached_graph(self) -> bool:
        """Load knowledge graph from cache if available."""
        if not self._cache_path.exists():
            return False
        
        try:
            with open(self._cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct knowledge graph from JSON
            self.knowledge_graph = KnowledgeGraph()
            
            # Load entities
            for entity_data in data.get('entities', []):
                entity = CodeEntity(
                    id=entity_data['id'],
                    name=entity_data['name'],
                    entity_type=EntityType(entity_data['entity_type']),
                    file_path=Path(entity_data['file_path']),
                    start_line=entity_data['start_line'],
                    end_line=entity_data['end_line'],
                    signature=entity_data.get('signature'),
                    docstring=entity_data.get('docstring'),
                    properties=entity_data.get('properties', {}),
                    embedding=entity_data.get('embedding'),
                    confidence=entity_data.get('confidence', 1.0),
                    language=entity_data.get('language', 'unknown'),
                    scope=entity_data.get('scope'),
                    access_modifier=entity_data.get('access_modifier')
                )
                self.knowledge_graph.add_entity(entity)
            
            # Load relationships
            for rel_data in data.get('relationships', []):
                relationship = EntityRelationship(
                    id=rel_data['id'],
                    source_entity_id=rel_data['source_entity_id'],
                    target_entity_id=rel_data['target_entity_id'],
                    relation_type=RelationType(rel_data['relation_type']),
                    properties=rel_data.get('properties', {}),
                    confidence=rel_data.get('confidence', 1.0),
                    weight=rel_data.get('weight', 1.0),
                    context=rel_data.get('context'),
                    file_path=Path(rel_data['file_path']) if rel_data.get('file_path') else None,
                    line_number=rel_data.get('line_number')
                )
                self.knowledge_graph.add_relationship(relationship)
            
            # Load metadata
            self.knowledge_graph.metadata = data.get('metadata', {})
            self.knowledge_graph.version = data.get('version', '1.0')
            self.knowledge_graph.created_at = data.get('created_at')
            self.knowledge_graph.updated_at = data.get('updated_at')
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load cached knowledge graph: {e}")
            return False
    
    async def _cache_graph(self) -> None:
        """Cache the knowledge graph to disk."""
        try:
            # Convert knowledge graph to JSON-serializable format
            data: Dict[str, Any] = {
                'version': self.knowledge_graph.version,
                'created_at': self.knowledge_graph.created_at,
                'updated_at': time.time(),
                'metadata': self.knowledge_graph.metadata,
                'entities': [],
                'relationships': []
            }
            
            # Serialize entities
            for entity in self.knowledge_graph.entities.values():
                entity_data = {
                    'id': entity.id,
                    'name': entity.name,
                    'entity_type': entity.entity_type.value,
                    'file_path': str(entity.file_path),
                    'start_line': entity.start_line,
                    'end_line': entity.end_line,
                    'signature': entity.signature,
                    'docstring': entity.docstring,
                    'properties': entity.properties,
                    'embedding': entity.embedding,
                    'confidence': entity.confidence,
                    'language': entity.language.value if hasattr(entity.language, 'value') else str(entity.language),
                    'scope': entity.scope,
                    'access_modifier': entity.access_modifier
                }
                data['entities'].append(entity_data)
            
            # Serialize relationships
            for relationship in self.knowledge_graph.relationships:
                rel_data = {
                    'id': relationship.id,
                    'source_entity_id': relationship.source_entity_id,
                    'target_entity_id': relationship.target_entity_id,
                    'relation_type': relationship.relation_type.value,
                    'properties': relationship.properties,
                    'confidence': relationship.confidence,
                    'weight': relationship.weight,
                    'context': relationship.context,
                    'file_path': str(relationship.file_path) if relationship.file_path else None,
                    'line_number': relationship.line_number
                }
                data['relationships'].append(rel_data)
            
            # Write to cache file
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Cached knowledge graph to {self._cache_path}")
            
        except Exception as e:
            logger.warning(f"Failed to cache knowledge graph: {e}")


class GraphRAGEngine:
    """
    Main engine for GraphRAG operations.

    This class provides the primary interface for GraphRAG functionality,
    coordinating knowledge graph building, vector storage, and graph-based
    query processing.
    """

    def __init__(
        self,
        config: SearchConfig,
        qdrant_config: Optional[QdrantConfig] = None
    ) -> None:
        self.config = config
        self.qdrant_config = qdrant_config
        self.graph_builder = KnowledgeGraphBuilder(config)
        self.vector_store: Optional[QdrantVectorStore] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the GraphRAG engine."""
        if self.qdrant_config:
            try:
                self.vector_store = QdrantVectorStore(self.qdrant_config)
                await self.vector_store.initialize()
                logger.info("Qdrant vector store initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Qdrant: {e}")
                self.vector_store = None

        self._initialized = True
        logger.info("GraphRAG engine initialized")

    async def build_knowledge_graph(self, force_rebuild: bool = False) -> KnowledgeGraph:
        """Build the knowledge graph and optionally store vectors."""
        if not self._initialized:
            await self.initialize()

        # Build the knowledge graph
        knowledge_graph = await self.graph_builder.build_graph(force_rebuild)

        # Store entity vectors in Qdrant if available
        if self.vector_store and self.vector_store.is_available():
            await self._store_entity_vectors(knowledge_graph)

        return knowledge_graph

    async def query_graph(self, query: GraphRAGQuery) -> GraphRAGResult:
        """Query the knowledge graph using GraphRAG techniques."""
        if not self._initialized:
            await self.initialize()

        knowledge_graph = self.graph_builder.knowledge_graph
        if not knowledge_graph.entities:
            # Build graph if not available
            knowledge_graph = await self.build_knowledge_graph()

        # Start with semantic/vector search if available
        candidate_entities: List[CodeEntity] = []
        similarity_scores: Dict[str, float] = {}

        if self.vector_store and self.vector_store.is_available():
            candidate_entities, similarity_scores = await self._vector_search(query, knowledge_graph)
        else:
            # Fallback to text-based search
            candidate_entities, similarity_scores = await self._text_search(query, knowledge_graph)

        # Filter by entity types if specified
        if query.entity_types:
            candidate_entities = [
                entity for entity in candidate_entities
                if entity.entity_type in query.entity_types
            ]

        # Expand search using graph relationships
        expanded_entities = await self._expand_with_relationships(
            candidate_entities,
            knowledge_graph,
            query
        )

        # Find relevant relationships
        relevant_relationships = await self._find_relevant_relationships(
            expanded_entities,
            knowledge_graph,
            query
        )

        # Rank and filter results
        final_entities = await self._rank_and_filter_results(
            expanded_entities,
            similarity_scores,
            query
        )

        return GraphRAGResult(
            entities=final_entities,
            relationships=relevant_relationships,
            similarity_scores=similarity_scores,
            query=query,
            metadata={
                "total_candidates": len(candidate_entities),
                "expanded_entities": len(expanded_entities),
                "final_entities": len(final_entities),
                "relationships": len(relevant_relationships)
            }
        )

    async def _store_entity_vectors(self, knowledge_graph: KnowledgeGraph) -> None:
        """Store entity vectors in Qdrant."""
        if not self.vector_store:
            return

        entities_with_embeddings = [
            entity for entity in knowledge_graph.entities.values()
            if entity.embedding
        ]

        if not entities_with_embeddings:
            logger.warning("No entities with embeddings found")
            return

        vectors = [entity.embedding for entity in entities_with_embeddings if entity.embedding is not None]
        metadata = []
        ids = []

        for entity in entities_with_embeddings:
            metadata.append({
                "entity_id": entity.id,
                "name": entity.name,
                "entity_type": entity.entity_type.value,
                "file_path": str(entity.file_path),
                "start_line": entity.start_line,
                "language": entity.language.value if hasattr(entity.language, 'value') else str(entity.language),
                "signature": entity.signature or "",
                "docstring": entity.docstring or ""
            })
            ids.append(entity.id)

        try:
            await self.vector_store.add_vectors(
                collection_name="code_entities",
                vectors=vectors,
                metadata=metadata,
                ids=ids
            )
            logger.info(f"Stored {len(vectors)} entity vectors in Qdrant")
        except Exception as e:
            logger.error(f"Failed to store entity vectors: {e}")

    async def _vector_search(
        self,
        query: GraphRAGQuery,
        knowledge_graph: KnowledgeGraph
    ) -> Tuple[List[CodeEntity], Dict[str, float]]:
        """Perform vector-based search for entities."""
        if not self.vector_store:
            return [], {}

        # Generate query vector
        query_vector = self.graph_builder.semantic_embedding.transform(query.pattern)
        if not query_vector:
            return [], {}

        # Convert sparse to dense vector
        max_dim = max(query_vector.keys()) + 1 if query_vector else 0
        dense_query_vector = [0.0] * max_dim
        for dim, value in query_vector.items():
            dense_query_vector[dim] = value

        # Build filter conditions
        filter_conditions = {}
        if query.entity_types:
            filter_conditions["entity_type"] = [et.value for et in query.entity_types]

        # Search similar vectors
        try:
            search_results = await self.vector_store.search_similar(
                query_vector=dense_query_vector,
                collection_name="code_entities",
                top_k=50,  # Get more candidates for graph expansion
                filter_conditions=filter_conditions,
                score_threshold=query.semantic_threshold
            )

            # Convert to entities
            entities = []
            similarity_scores = {}

            for result in search_results:
                entity = knowledge_graph.get_entity(result.id)
                if entity:
                    entities.append(entity)
                    similarity_scores[entity.id] = result.score

            return entities, similarity_scores

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return [], {}

    async def _text_search(
        self,
        query: GraphRAGQuery,
        knowledge_graph: KnowledgeGraph
    ) -> Tuple[List[CodeEntity], Dict[str, float]]:
        """Fallback text-based search for entities."""
        entities = []
        similarity_scores = {}

        query_lower = query.pattern.lower()

        for entity in knowledge_graph.entities.values():
            # Simple text matching
            score = 0.0

            if query_lower in entity.name.lower():
                score += 0.8

            if entity.docstring and query_lower in entity.docstring.lower():
                score += 0.6

            if entity.signature and query_lower in entity.signature.lower():
                score += 0.4

            # Check properties
            for value in entity.properties.values():
                if isinstance(value, str) and query_lower in value.lower():
                    score += 0.2
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and query_lower in item.lower():
                            score += 0.1

            if score >= query.semantic_threshold:
                entities.append(entity)
                similarity_scores[entity.id] = score

        # Sort by score
        entities.sort(key=lambda e: similarity_scores[e.id], reverse=True)

        return entities[:50], similarity_scores  # Limit to top 50

    async def _expand_with_relationships(
        self,
        entities: List[CodeEntity],
        knowledge_graph: KnowledgeGraph,
        query: GraphRAGQuery
    ) -> List[CodeEntity]:
        """Expand entity set using graph relationships."""
        expanded = set(entities)

        for hop in range(query.max_hops):
            current_entities = list(expanded)

            for entity in current_entities:
                related = knowledge_graph.get_related_entities(
                    entity.id,
                    relation_types=query.relation_types,
                    max_hops=1
                )

                for related_entity, relationship in related:
                    if relationship.confidence >= query.min_confidence:
                        expanded.add(related_entity)

        return list(expanded)

    async def _find_relevant_relationships(
        self,
        entities: List[CodeEntity],
        knowledge_graph: KnowledgeGraph,
        query: GraphRAGQuery
    ) -> List[EntityRelationship]:
        """Find relationships relevant to the query entities."""
        if not query.include_relationships:
            return []

        entity_ids = {entity.id for entity in entities}
        relevant_relationships = []

        for relationship in knowledge_graph.relationships:
            if (relationship.source_entity_id in entity_ids or
                relationship.target_entity_id in entity_ids):

                if relationship.confidence >= query.min_confidence:
                    if not query.relation_types or relationship.relation_type in query.relation_types:
                        relevant_relationships.append(relationship)

        return relevant_relationships

    async def _rank_and_filter_results(
        self,
        entities: List[CodeEntity],
        similarity_scores: Dict[str, float],
        query: GraphRAGQuery
    ) -> List[CodeEntity]:
        """Rank and filter final results."""
        # Sort by similarity score
        scored_entities = [
            (entity, similarity_scores.get(entity.id, 0.0))
            for entity in entities
        ]
        scored_entities.sort(key=lambda x: x[1], reverse=True)

        # Apply context window limit
        max_results = query.context_window * 2  # Allow some expansion
        return [entity for entity, score in scored_entities[:max_results]]

    async def get_entity_context(
        self,
        entity_id: str,
        context_hops: int = 2
    ) -> Dict[str, Any]:
        """Get contextual information about an entity."""
        knowledge_graph = self.graph_builder.knowledge_graph
        entity = knowledge_graph.get_entity(entity_id)

        if not entity:
            return {}

        # Get related entities
        related = knowledge_graph.get_related_entities(entity_id, max_hops=context_hops)

        context: Dict[str, Any] = {
            "entity": {
                "id": entity.id,
                "name": entity.name,
                "type": entity.entity_type.value,
                "file": str(entity.file_path),
                "line": entity.start_line,
                "signature": entity.signature,
                "docstring": entity.docstring
            },
            "relationships": [],
            "related_entities": []
        }

        for related_entity, relationship in related:
            context["relationships"].append({
                "type": relationship.relation_type.value,
                "target": related_entity.name,
                "confidence": relationship.confidence,
                "context": relationship.context
            })

            context["related_entities"].append({
                "id": related_entity.id,
                "name": related_entity.name,
                "type": related_entity.entity_type.value,
                "file": str(related_entity.file_path)
            })

        return context

    async def close(self) -> None:
        """Close the GraphRAG engine and cleanup resources."""
        if self.vector_store:
            await self.vector_store.close()

        logger.info("GraphRAG engine closed")
