#!/usr/bin/env python3
"""
GraphRAG Examples for pysearch.

This module demonstrates comprehensive usage of GraphRAG (Graph Retrieval-Augmented
Generation) capabilities including knowledge graph building, entity extraction,
relationship mapping, and graph-based querying.

Examples covered:
1. Basic GraphRAG setup and initialization
2. Building knowledge graphs from codebases
3. Entity extraction and relationship mapping
4. Graph-based search and retrieval
5. Hybrid search combining traditional and GraphRAG methods
6. Qdrant vector database integration
7. Enhanced metadata indexing
8. Advanced GraphRAG queries with filtering

Requirements:
    - pysearch with GraphRAG features enabled
    - Optional: Qdrant vector database for enhanced semantic search
    - Optional: numpy for vector operations
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.indexer_metadata import IndexQuery
from pysearch.graphrag_engine import GraphRAGEngine
from pysearch.qdrant_client import QdrantConfig
from pysearch.types import (
    EntityType, GraphRAGQuery, GraphRAGResult, Language, RelationType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_1_basic_graphrag_setup():
    """
    Example 1: Basic GraphRAG setup and initialization.
    
    This example shows how to set up GraphRAG with basic configuration
    and initialize the necessary components.
    """
    print("\n" + "="*60)
    print("Example 1: Basic GraphRAG Setup")
    print("="*60)
    
    # Create configuration with GraphRAG enabled
    config = SearchConfig(
        paths=["./src"],  # Analyze source code directory
        include=["**/*.py", "**/*.js", "**/*.ts"],
        enable_graphrag=True,
        enable_enhanced_indexing=True,
        graphrag_max_hops=2,
        graphrag_min_confidence=0.7,
        graphrag_semantic_threshold=0.8
    )
    
    # Initialize PySearch with GraphRAG capabilities
    search_engine = PySearch(
        config=config,
        enable_graphrag=True,
        enable_enhanced_indexing=True
    )
    
    # Initialize GraphRAG components
    await search_engine.initialize_graphrag()
    await search_engine.initialize_enhanced_indexing()
    
    print("✓ GraphRAG engine initialized successfully")
    print("✓ Enhanced indexing initialized successfully")
    
    # Clean up
    await search_engine.close_async_components()
    
    return search_engine


async def example_2_build_knowledge_graph():
    """
    Example 2: Building knowledge graphs from codebases.
    
    This example demonstrates how to build comprehensive knowledge graphs
    from source code, including entity extraction and relationship mapping.
    """
    print("\n" + "="*60)
    print("Example 2: Building Knowledge Graph")
    print("="*60)
    
    config = SearchConfig(
        paths=["./src"],
        include=["**/*.py"],
        enable_graphrag=True,
        enable_enhanced_indexing=True
    )
    
    search_engine = PySearch(
        config=config,
        enable_graphrag=True,
        enable_enhanced_indexing=True
    )
    
    # Build the knowledge graph
    print("Building knowledge graph from source code...")
    success = await search_engine.build_knowledge_graph(force_rebuild=False)
    
    if success:
        print("✓ Knowledge graph built successfully")
        
        # Get graph statistics
        if search_engine._graphrag_engine:
            graph = search_engine._graphrag_engine.graph_builder.knowledge_graph
            print(f"  - Total entities: {len(graph.entities)}")
            print(f"  - Total relationships: {len(graph.relationships)}")
            
            # Show entity type distribution
            entity_types = {}
            for entity in graph.entities.values():
                entity_type = entity.entity_type.value
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            print("  - Entity types:")
            for entity_type, count in sorted(entity_types.items()):
                print(f"    {entity_type}: {count}")
    else:
        print("✗ Failed to build knowledge graph")
    
    await search_engine.close_async_components()
    return success


async def example_3_entity_extraction():
    """
    Example 3: Entity extraction and relationship mapping.
    
    This example shows how to extract entities from specific files
    and understand the relationships between them.
    """
    print("\n" + "="*60)
    print("Example 3: Entity Extraction and Relationships")
    print("="*60)
    
    from pysearch.graphrag import EntityExtractor, RelationshipMapper
    
    # Initialize entity extractor
    extractor = EntityExtractor()
    
    # Extract entities from a specific file
    test_file = Path("./src/pysearch/api.py")
    if test_file.exists():
        print(f"Extracting entities from {test_file}...")
        entities = await extractor.extract_from_file(test_file)
        
        print(f"✓ Extracted {len(entities)} entities")
        
        # Show some example entities
        for i, entity in enumerate(entities[:5]):  # Show first 5
            print(f"  {i+1}. {entity.entity_type.value}: {entity.name}")
            if entity.signature:
                print(f"     Signature: {entity.signature[:80]}...")
            if entity.docstring:
                print(f"     Doc: {entity.docstring[:60]}...")
            print()
        
        # Map relationships between entities
        print("Mapping relationships...")
        mapper = RelationshipMapper()
        
        # Read file content for relationship analysis
        file_contents = {test_file: test_file.read_text(encoding='utf-8')}
        relationships = await mapper.map_relationships(entities, file_contents)
        
        print(f"✓ Found {len(relationships)} relationships")
        
        # Show some example relationships
        for i, rel in enumerate(relationships[:3]):  # Show first 3
            source_entity = next((e for e in entities if e.id == rel.source_entity_id), None)
            target_entity = next((e for e in entities if e.id == rel.target_entity_id), None)
            
            if source_entity and target_entity:
                print(f"  {i+1}. {source_entity.name} --{rel.relation_type.value}--> {target_entity.name}")
                if rel.context:
                    print(f"     Context: {rel.context}")
    else:
        print(f"✗ Test file {test_file} not found")


async def example_4_graphrag_search():
    """
    Example 4: Graph-based search and retrieval.
    
    This example demonstrates how to perform GraphRAG queries
    to find related code entities using graph traversal.
    """
    print("\n" + "="*60)
    print("Example 4: GraphRAG Search and Retrieval")
    print("="*60)
    
    config = SearchConfig(
        paths=["./src"],
        include=["**/*.py"],
        enable_graphrag=True,
        enable_enhanced_indexing=True
    )
    
    search_engine = PySearch(
        config=config,
        enable_graphrag=True,
        enable_enhanced_indexing=True
    )
    
    # Build knowledge graph first
    await search_engine.build_knowledge_graph()
    
    # Perform GraphRAG search
    print("Performing GraphRAG search for 'search functionality'...")
    
    graphrag_query = GraphRAGQuery(
        pattern="search functionality",
        entity_types=[EntityType.FUNCTION, EntityType.CLASS],
        relation_types=[RelationType.CALLS, RelationType.CONTAINS],
        max_hops=2,
        include_relationships=True,
        min_confidence=0.6,
        semantic_threshold=0.7
    )
    
    results = await search_engine.graphrag_search(graphrag_query)
    
    if results:
        print(f"✓ Found {len(results.entities)} relevant entities")
        
        # Display results
        for i, entity in enumerate(results.entities[:5]):  # Show top 5
            similarity_score = results.similarity_scores.get(entity.id, 0.0)
            print(f"  {i+1}. {entity.entity_type.value}: {entity.name}")
            print(f"     File: {entity.file_path}")
            print(f"     Line: {entity.start_line}")
            print(f"     Similarity: {similarity_score:.3f}")
            if entity.docstring:
                print(f"     Doc: {entity.docstring[:80]}...")
            print()
        
        # Show relationships
        if results.relationships:
            print(f"Found {len(results.relationships)} relationships:")
            for i, rel in enumerate(results.relationships[:3]):  # Show first 3
                print(f"  {i+1}. {rel.relation_type.value} (confidence: {rel.confidence:.2f})")
                if rel.context:
                    print(f"     Context: {rel.context}")
    else:
        print("✗ No GraphRAG results found")
    
    await search_engine.close_async_components()


async def example_5_hybrid_search():
    """
    Example 5: Hybrid search combining traditional and GraphRAG methods.
    
    This example shows how to use the hybrid search functionality
    that combines multiple search approaches for comprehensive results.
    """
    print("\n" + "="*60)
    print("Example 5: Hybrid Search")
    print("="*60)
    
    config = SearchConfig(
        paths=["./src"],
        include=["**/*.py"],
        enable_graphrag=True,
        enable_enhanced_indexing=True
    )
    
    search_engine = PySearch(
        config=config,
        enable_graphrag=True,
        enable_enhanced_indexing=True
    )
    
    # Build necessary indexes
    await search_engine.build_knowledge_graph()
    await search_engine.build_enhanced_index()
    
    # Perform hybrid search
    print("Performing hybrid search for 'file indexing'...")
    
    results = await search_engine.hybrid_search(
        pattern="file indexing",
        use_graphrag=True,
        use_enhanced_index=True,
        graphrag_max_hops=2,
        use_regex=False,
        context=3
    )
    
    print("Hybrid search results:")
    print(f"Methods used: {results['metadata']['methods_used']}")
    
    # Traditional search results
    if results["traditional"]:
        traditional = results["traditional"]
        print(f"\nTraditional search: {len(traditional['items'])} matches")
        for i, item in enumerate(traditional["items"][:3]):  # Show first 3
            print(f"  {i+1}. {item['file']} (lines {item['start_line']}-{item['end_line']})")
    
    # GraphRAG results
    if results["graphrag"]:
        graphrag = results["graphrag"]
        print(f"\nGraphRAG search: {len(graphrag['entities'])} entities")
        for i, entity in enumerate(graphrag["entities"][:3]):  # Show first 3
            print(f"  {i+1}. {entity['type']}: {entity['name']} in {entity['file']}")
    
    # Enhanced index results
    if results["enhanced_index"]:
        enhanced = results["enhanced_index"]
        print(f"\nEnhanced index: {len(enhanced['files'])} files, {len(enhanced['entities'])} entities")
    
    await search_engine.close_async_components()


async def example_6_qdrant_integration():
    """
    Example 6: Qdrant vector database integration.
    
    This example demonstrates how to use Qdrant for enhanced
    semantic search capabilities with vector embeddings.
    """
    print("\n" + "="*60)
    print("Example 6: Qdrant Vector Database Integration")
    print("="*60)
    
    # Configure Qdrant (assumes Qdrant is running locally)
    qdrant_config = QdrantConfig(
        host="localhost",
        port=6333,
        collection_name="pysearch_demo",
        vector_size=384,
        distance_metric="Cosine"
    )
    
    config = SearchConfig(
        paths=["./src"],
        include=["**/*.py"],
        enable_graphrag=True,
        enable_enhanced_indexing=True,
        qdrant_enabled=True
    )
    
    try:
        search_engine = PySearch(
            config=config,
            qdrant_config=qdrant_config,
            enable_graphrag=True,
            enable_enhanced_indexing=True
        )
        
        # Initialize with Qdrant
        await search_engine.initialize_graphrag()
        
        # Build knowledge graph with vector storage
        print("Building knowledge graph with vector embeddings...")
        success = await search_engine.build_knowledge_graph()
        
        if success:
            print("✓ Knowledge graph with vectors built successfully")
            
            # Perform vector-enhanced GraphRAG search
            graphrag_query = GraphRAGQuery(
                pattern="database connection management",
                use_vector_search=True,
                semantic_threshold=0.75,
                max_hops=2
            )
            
            results = await search_engine.graphrag_search(graphrag_query)
            
            if results:
                print(f"✓ Vector-enhanced search found {len(results.entities)} entities")
                for entity in results.entities[:3]:
                    score = results.similarity_scores.get(entity.id, 0.0)
                    print(f"  - {entity.name}: {score:.3f}")
        
        await search_engine.close_async_components()
        
    except Exception as e:
        print(f"✗ Qdrant integration failed: {e}")
        print("  Make sure Qdrant is running on localhost:6333")


async def example_7_enhanced_indexing():
    """
    Example 7: Enhanced metadata indexing.
    
    This example shows how to use the enhanced indexing system
    for comprehensive metadata-based queries.
    """
    print("\n" + "="*60)
    print("Example 7: Enhanced Metadata Indexing")
    print("="*60)
    
    config = SearchConfig(
        paths=["./src"],
        include=["**/*.py"],
        enable_enhanced_indexing=True,
        enhanced_indexing_include_semantic=True,
        enhanced_indexing_complexity_analysis=True
    )
    
    search_engine = PySearch(
        config=config,
        enable_enhanced_indexing=True
    )
    
    # Build enhanced index
    print("Building enhanced metadata index...")
    success = await search_engine.build_enhanced_index(include_semantic=True)
    
    if success:
        print("✓ Enhanced index built successfully")
        
        # Query by entity type
        print("\nQuerying functions with high complexity...")
        query = IndexQuery(
            entity_types=["function"],
            min_complexity=10.0,
            has_docstring=True,
            limit=5
        )
        
        results = await search_engine.enhanced_index_search(query)
        
        if results:
            print(f"Found {len(results['entities'])} complex functions:")
            for entity in results["entities"]:
                print(f"  - {entity['name']} (complexity: {entity['complexity']:.1f})")
        
        # Query by file characteristics
        print("\nQuerying large Python files...")
        query = IndexQuery(
            languages=["python"],
            min_lines=100,
            min_size=5000,
            limit=5
        )
        
        results = await search_engine.enhanced_index_search(query)
        
        if results:
            print(f"Found {len(results['files'])} large files:")
            for file_info in results["files"]:
                print(f"  - {file_info['path']} ({file_info['line_count']} lines)")
    
    await search_engine.close_async_components()


async def example_8_advanced_queries():
    """
    Example 8: Advanced GraphRAG queries with filtering.
    
    This example demonstrates advanced GraphRAG query capabilities
    including complex filtering and multi-hop traversal.
    """
    print("\n" + "="*60)
    print("Example 8: Advanced GraphRAG Queries")
    print("="*60)
    
    config = SearchConfig(
        paths=["./src"],
        include=["**/*.py"],
        enable_graphrag=True,
        enable_enhanced_indexing=True
    )
    
    search_engine = PySearch(
        config=config,
        enable_graphrag=True,
        enable_enhanced_indexing=True
    )
    
    await search_engine.build_knowledge_graph()
    
    # Advanced query 1: Find test-related functions and their dependencies
    print("Query 1: Finding test functions and their dependencies...")
    
    query = GraphRAGQuery(
        pattern="test",
        entity_types=[EntityType.FUNCTION],
        relation_types=[RelationType.CALLS, RelationType.USES],
        max_hops=3,
        min_confidence=0.8,
        context_window=10
    )
    
    results = await search_engine.graphrag_search(query)
    
    if results:
        print(f"✓ Found {len(results.entities)} test-related entities")
        
        # Analyze the graph structure
        test_functions = [e for e in results.entities if "test" in e.name.lower()]
        print(f"  - Test functions: {len(test_functions)}")
        
        call_relationships = [r for r in results.relationships if r.relation_type == RelationType.CALLS]
        print(f"  - Call relationships: {len(call_relationships)}")
    
    # Advanced query 2: Find classes and their inheritance hierarchy
    print("\nQuery 2: Finding class inheritance hierarchies...")
    
    query = GraphRAGQuery(
        pattern="class",
        entity_types=[EntityType.CLASS],
        relation_types=[RelationType.INHERITS, RelationType.EXTENDS],
        max_hops=2,
        include_relationships=True
    )
    
    results = await search_engine.graphrag_search(query)
    
    if results:
        print(f"✓ Found {len(results.entities)} classes")
        
        inheritance_rels = [r for r in results.relationships 
                          if r.relation_type in [RelationType.INHERITS, RelationType.EXTENDS]]
        print(f"  - Inheritance relationships: {len(inheritance_rels)}")
        
        for rel in inheritance_rels[:3]:  # Show first 3
            source = next((e for e in results.entities if e.id == rel.source_entity_id), None)
            target = next((e for e in results.entities if e.id == rel.target_entity_id), None)
            if source and target:
                print(f"    {source.name} inherits from {target.name}")
    
    await search_engine.close_async_components()


async def main():
    """
    Main function to run all GraphRAG examples.
    
    This function executes all the examples in sequence,
    demonstrating the full range of GraphRAG capabilities.
    """
    print("GraphRAG Examples for pysearch")
    print("=" * 60)
    print("This demo showcases comprehensive GraphRAG functionality")
    print("including knowledge graphs, entity extraction, and semantic search.")
    print()
    
    examples = [
        example_1_basic_graphrag_setup,
        example_2_build_knowledge_graph,
        example_3_entity_extraction,
        example_4_graphrag_search,
        example_5_hybrid_search,
        example_6_qdrant_integration,
        example_7_enhanced_indexing,
        example_8_advanced_queries
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\nRunning Example {i}...")
            await example_func()
            print(f"✓ Example {i} completed successfully")
        except Exception as e:
            print(f"✗ Example {i} failed: {e}")
            logger.exception(f"Error in example {i}")
    
    print("\n" + "="*60)
    print("All GraphRAG examples completed!")
    print("="*60)


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
