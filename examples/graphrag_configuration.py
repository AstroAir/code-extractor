#!/usr/bin/env python3
"""
GraphRAG Configuration Examples for pysearch.

This module demonstrates various configuration options for GraphRAG functionality,
including Qdrant integration, enhanced indexing, and performance tuning.

Configuration topics covered:
1. Basic GraphRAG configuration
2. Qdrant vector database setup
3. Enhanced indexing configuration
4. Performance optimization settings
5. Multi-repository GraphRAG setup
6. Custom entity and relationship filtering
7. Semantic search tuning
8. Production deployment configuration

Requirements:
    - pysearch with GraphRAG features
    - Optional: Qdrant vector database
    - Optional: Custom embedding models
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional

from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.qdrant_client import QdrantConfig
from pysearch.types import EntityType, RelationType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_graphrag_config():
    """
    Example 1: Basic GraphRAG configuration.
    
    This example shows the minimal configuration needed to enable
    GraphRAG functionality with default settings.
    """
    print("\n" + "="*60)
    print("Example 1: Basic GraphRAG Configuration")
    print("="*60)
    
    # Minimal GraphRAG configuration
    config = SearchConfig(
        # Basic search paths
        paths=["./src", "./lib"],
        include=["**/*.py", "**/*.js", "**/*.ts"],
        exclude=["**/node_modules/**", "**/__pycache__/**"],
        
        # Enable GraphRAG features
        enable_graphrag=True,
        enable_enhanced_indexing=True,
        
        # Basic GraphRAG settings (using defaults)
        graphrag_max_hops=2,
        graphrag_min_confidence=0.5,
        graphrag_semantic_threshold=0.7,
        graphrag_context_window=5
    )
    
    print("Basic GraphRAG configuration:")
    print(f"  - GraphRAG enabled: {config.enable_graphrag}")
    print(f"  - Enhanced indexing: {config.enable_enhanced_indexing}")
    print(f"  - Max hops: {config.graphrag_max_hops}")
    print(f"  - Min confidence: {config.graphrag_min_confidence}")
    print(f"  - Semantic threshold: {config.graphrag_semantic_threshold}")
    
    # Validate configuration
    issues = config.validate_advanced_config()
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("✓ Configuration is valid")
    
    return config


def example_2_qdrant_configuration():
    """
    Example 2: Qdrant vector database configuration.
    
    This example demonstrates how to configure Qdrant for enhanced
    semantic search capabilities with vector embeddings.
    """
    print("\n" + "="*60)
    print("Example 2: Qdrant Vector Database Configuration")
    print("="*60)
    
    # Qdrant configuration for local development
    qdrant_config_local = QdrantConfig(
        host="localhost",
        port=6333,
        api_key=None,  # No authentication for local
        https=False,
        timeout=30.0,
        collection_name="pysearch_dev",
        vector_size=384,  # For sentence-transformers models
        distance_metric="Cosine",
        max_retries=3,
        retry_delay=1.0,
        batch_size=100,
        enable_compression=False
    )
    
    # Qdrant configuration for production
    qdrant_config_prod = QdrantConfig(
        host="qdrant.example.com",
        port=6333,
        api_key="your-api-key-here",
        https=True,
        timeout=60.0,
        collection_name="pysearch_prod",
        vector_size=768,  # For larger embedding models
        distance_metric="Cosine",
        max_retries=5,
        retry_delay=2.0,
        batch_size=200,
        enable_compression=True
    )
    
    # Search configuration with Qdrant
    config = SearchConfig(
        paths=["./src"],
        include=["**/*.py"],
        
        # Enable all advanced features
        enable_graphrag=True,
        enable_enhanced_indexing=True,
        
        # Qdrant settings
        qdrant_enabled=True,
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_collection_name="pysearch_vectors",
        qdrant_vector_size=384,
        qdrant_distance_metric="Cosine",
        qdrant_batch_size=100,
        
        # Optimized GraphRAG settings for vector search
        graphrag_semantic_threshold=0.8,  # Higher threshold with vectors
        graphrag_max_hops=3,  # Can afford more hops with fast vector search
        graphrag_context_window=10
    )
    
    print("Qdrant configurations:")
    print("\nLocal development:")
    print(f"  - Host: {qdrant_config_local.host}:{qdrant_config_local.port}")
    print(f"  - Collection: {qdrant_config_local.collection_name}")
    print(f"  - Vector size: {qdrant_config_local.vector_size}")
    print(f"  - Distance metric: {qdrant_config_local.distance_metric}")
    
    print("\nProduction:")
    print(f"  - Host: {qdrant_config_prod.host}:{qdrant_config_prod.port}")
    print(f"  - HTTPS: {qdrant_config_prod.https}")
    print(f"  - API Key: {'***' if qdrant_config_prod.api_key else 'None'}")
    print(f"  - Compression: {qdrant_config_prod.enable_compression}")
    
    return config, qdrant_config_local, qdrant_config_prod


def example_3_enhanced_indexing_config():
    """
    Example 3: Enhanced indexing configuration.
    
    This example shows how to configure the enhanced indexing system
    for comprehensive metadata analysis and storage.
    """
    print("\n" + "="*60)
    print("Example 3: Enhanced Indexing Configuration")
    print("="*60)
    
    config = SearchConfig(
        paths=["./src", "./tests", "./docs"],
        include=["**/*.py", "**/*.js", "**/*.ts", "**/*.md"],
        exclude=["**/node_modules/**", "**/.git/**"],
        
        # Enhanced indexing settings
        enable_enhanced_indexing=True,
        enhanced_indexing_include_semantic=True,
        enhanced_indexing_complexity_analysis=True,
        enhanced_indexing_dependency_tracking=True,
        
        # GraphRAG integration
        enable_graphrag=True,
        graphrag_max_hops=2,
        graphrag_min_confidence=0.6,
        
        # Performance settings
        max_file_size=10 * 1024 * 1024,  # 10MB limit
        max_line_length=1000,
        timeout=60.0
    )
    
    print("Enhanced indexing configuration:")
    print(f"  - Semantic analysis: {config.enhanced_indexing_include_semantic}")
    print(f"  - Complexity analysis: {config.enhanced_indexing_complexity_analysis}")
    print(f"  - Dependency tracking: {config.enhanced_indexing_dependency_tracking}")
    print(f"  - Max file size: {config.max_file_size / (1024*1024):.1f}MB")
    
    return config


def example_4_performance_optimization():
    """
    Example 4: Performance optimization configuration.
    
    This example demonstrates configuration settings for optimizing
    GraphRAG performance in different scenarios.
    """
    print("\n" + "="*60)
    print("Example 4: Performance Optimization Configuration")
    print("="*60)
    
    # Configuration for large codebases
    config_large = SearchConfig(
        paths=["./src"],
        include=["**/*.py"],
        
        # GraphRAG settings optimized for large codebases
        enable_graphrag=True,
        enable_enhanced_indexing=True,
        graphrag_max_hops=2,  # Limit hops to reduce computation
        graphrag_min_confidence=0.7,  # Higher confidence to filter results
        graphrag_semantic_threshold=0.8,  # Higher threshold for precision
        graphrag_context_window=5,  # Smaller context window
        
        # Qdrant optimization
        qdrant_enabled=True,
        qdrant_batch_size=500,  # Larger batches for efficiency
        qdrant_vector_size=256,  # Smaller vectors for speed
        
        # Enhanced indexing optimization
        enhanced_indexing_include_semantic=True,
        enhanced_indexing_complexity_analysis=False,  # Disable for speed
        
        # General performance settings
        max_file_size=5 * 1024 * 1024,  # 5MB limit
        timeout=30.0,
        cache_ttl=3600,  # 1 hour cache
        
        # Parallel processing
        max_workers=8
    )
    
    # Configuration for real-time search
    config_realtime = SearchConfig(
        paths=["./src"],
        include=["**/*.py"],
        
        # GraphRAG settings for real-time performance
        enable_graphrag=True,
        enable_enhanced_indexing=True,
        graphrag_max_hops=1,  # Single hop for speed
        graphrag_min_confidence=0.8,
        graphrag_semantic_threshold=0.9,  # Very high threshold
        graphrag_context_window=3,  # Minimal context
        
        # Fast vector search
        qdrant_enabled=True,
        qdrant_batch_size=50,  # Smaller batches for responsiveness
        
        # Minimal indexing for speed
        enhanced_indexing_include_semantic=False,
        enhanced_indexing_complexity_analysis=False,
        enhanced_indexing_dependency_tracking=False,
        
        # Aggressive caching
        cache_ttl=7200,  # 2 hours
        max_workers=4
    )
    
    print("Performance configurations:")
    print("\nLarge codebase optimization:")
    print(f"  - Max hops: {config_large.graphrag_max_hops}")
    print(f"  - Batch size: {config_large.qdrant_batch_size}")
    print(f"  - Vector size: {config_large.qdrant_vector_size}")
    print(f"  - Workers: {config_large.max_workers}")
    
    print("\nReal-time search optimization:")
    print(f"  - Max hops: {config_realtime.graphrag_max_hops}")
    print(f"  - Semantic threshold: {config_realtime.graphrag_semantic_threshold}")
    print(f"  - Context window: {config_realtime.graphrag_context_window}")
    print(f"  - Cache TTL: {config_realtime.cache_ttl}s")
    
    return config_large, config_realtime


def example_5_multi_repo_config():
    """
    Example 5: Multi-repository GraphRAG configuration.
    
    This example shows how to configure GraphRAG for analyzing
    multiple repositories with different settings.
    """
    print("\n" + "="*60)
    print("Example 5: Multi-Repository Configuration")
    print("="*60)
    
    # Configuration for main application repository
    config_main = SearchConfig(
        paths=["./main-app/src"],
        include=["**/*.py", "**/*.js"],
        exclude=["**/tests/**"],
        
        enable_graphrag=True,
        enable_enhanced_indexing=True,
        graphrag_max_hops=3,  # Deep analysis for main app
        graphrag_min_confidence=0.6,
        
        qdrant_enabled=True,
        qdrant_collection_name="main_app_vectors"
    )
    
    # Configuration for library dependencies
    config_libs = SearchConfig(
        paths=["./libs", "./vendor"],
        include=["**/*.py", "**/*.js"],
        exclude=["**/node_modules/**", "**/dist/**"],
        
        enable_graphrag=True,
        enable_enhanced_indexing=True,
        graphrag_max_hops=2,  # Moderate analysis for libs
        graphrag_min_confidence=0.7,  # Higher confidence for external code
        
        qdrant_enabled=True,
        qdrant_collection_name="libs_vectors"
    )
    
    # Configuration for test suites
    config_tests = SearchConfig(
        paths=["./tests", "./integration-tests"],
        include=["**/*.py", "**/*.js"],
        
        enable_graphrag=True,
        enable_enhanced_indexing=True,
        graphrag_max_hops=2,
        graphrag_min_confidence=0.5,  # Lower confidence for test patterns
        
        qdrant_enabled=True,
        qdrant_collection_name="tests_vectors"
    )
    
    print("Multi-repository configurations:")
    print(f"\nMain application:")
    print(f"  - Paths: {config_main.paths}")
    print(f"  - Max hops: {config_main.graphrag_max_hops}")
    print(f"  - Collection: {config_main.qdrant_collection_name}")
    
    print(f"\nLibraries:")
    print(f"  - Paths: {config_libs.paths}")
    print(f"  - Min confidence: {config_libs.graphrag_min_confidence}")
    print(f"  - Collection: {config_libs.qdrant_collection_name}")
    
    print(f"\nTests:")
    print(f"  - Paths: {config_tests.paths}")
    print(f"  - Min confidence: {config_tests.graphrag_min_confidence}")
    print(f"  - Collection: {config_tests.qdrant_collection_name}")
    
    return config_main, config_libs, config_tests


def example_6_custom_filtering():
    """
    Example 6: Custom entity and relationship filtering.
    
    This example demonstrates how to configure custom filtering
    for entities and relationships in GraphRAG queries.
    """
    print("\n" + "="*60)
    print("Example 6: Custom Filtering Configuration")
    print("="*60)
    
    config = SearchConfig(
        paths=["./src"],
        include=["**/*.py"],
        
        enable_graphrag=True,
        enable_enhanced_indexing=True,
        
        # Custom GraphRAG settings
        graphrag_max_hops=3,
        graphrag_min_confidence=0.6,
        graphrag_semantic_threshold=0.7,
        graphrag_context_window=8
    )
    
    # Define custom entity type preferences
    preferred_entity_types = [
        EntityType.FUNCTION,
        EntityType.CLASS,
        EntityType.METHOD
    ]
    
    # Define custom relationship type preferences
    preferred_relation_types = [
        RelationType.CALLS,
        RelationType.INHERITS,
        RelationType.CONTAINS,
        RelationType.USES
    ]
    
    # Custom filtering rules
    filtering_rules = {
        "min_entity_complexity": 5.0,
        "require_docstring": True,
        "exclude_test_entities": True,
        "min_relationship_confidence": 0.8,
        "max_relationship_distance": 2
    }
    
    print("Custom filtering configuration:")
    print(f"  - Preferred entity types: {[et.value for et in preferred_entity_types]}")
    print(f"  - Preferred relation types: {[rt.value for rt in preferred_relation_types]}")
    print(f"  - Filtering rules:")
    for rule, value in filtering_rules.items():
        print(f"    {rule}: {value}")
    
    return config, preferred_entity_types, preferred_relation_types, filtering_rules


def example_7_semantic_tuning():
    """
    Example 7: Semantic search tuning configuration.
    
    This example shows how to fine-tune semantic search parameters
    for different types of code analysis tasks.
    """
    print("\n" + "="*60)
    print("Example 7: Semantic Search Tuning")
    print("="*60)
    
    # Configuration for API discovery
    config_api = SearchConfig(
        paths=["./src/api"],
        include=["**/*.py"],
        
        enable_graphrag=True,
        enable_enhanced_indexing=True,
        
        # Tuned for finding API endpoints and handlers
        graphrag_semantic_threshold=0.85,  # High precision for API matching
        graphrag_max_hops=2,
        graphrag_context_window=15,  # Larger context for API docs
        
        qdrant_enabled=True,
        qdrant_vector_size=512,  # Larger vectors for better API semantics
        qdrant_distance_metric="Cosine"
    )
    
    # Configuration for bug pattern detection
    config_bugs = SearchConfig(
        paths=["./src"],
        include=["**/*.py"],
        
        enable_graphrag=True,
        enable_enhanced_indexing=True,
        
        # Tuned for finding potential issues and patterns
        graphrag_semantic_threshold=0.6,  # Lower threshold for broader matching
        graphrag_max_hops=4,  # Deeper traversal for bug patterns
        graphrag_min_confidence=0.5,  # Include uncertain relationships
        
        enhanced_indexing_complexity_analysis=True,  # Important for bug detection
        
        qdrant_enabled=True,
        qdrant_distance_metric="Dot"  # Different metric for pattern matching
    )
    
    # Configuration for documentation generation
    config_docs = SearchConfig(
        paths=["./src"],
        include=["**/*.py"],
        
        enable_graphrag=True,
        enable_enhanced_indexing=True,
        
        # Tuned for comprehensive documentation
        graphrag_semantic_threshold=0.7,
        graphrag_max_hops=3,
        graphrag_context_window=20,  # Large context for documentation
        
        enhanced_indexing_include_semantic=True,
        enhanced_indexing_dependency_tracking=True,
        
        qdrant_enabled=True,
        qdrant_vector_size=768  # Large vectors for rich semantic content
    )
    
    print("Semantic tuning configurations:")
    print(f"\nAPI discovery:")
    print(f"  - Semantic threshold: {config_api.graphrag_semantic_threshold}")
    print(f"  - Context window: {config_api.graphrag_context_window}")
    print(f"  - Vector size: {config_api.qdrant_vector_size}")
    
    print(f"\nBug pattern detection:")
    print(f"  - Semantic threshold: {config_bugs.graphrag_semantic_threshold}")
    print(f"  - Max hops: {config_bugs.graphrag_max_hops}")
    print(f"  - Distance metric: {config_bugs.qdrant_distance_metric}")
    
    print(f"\nDocumentation generation:")
    print(f"  - Context window: {config_docs.graphrag_context_window}")
    print(f"  - Vector size: {config_docs.qdrant_vector_size}")
    print(f"  - Dependency tracking: {config_docs.enhanced_indexing_dependency_tracking}")
    
    return config_api, config_bugs, config_docs


def example_8_production_deployment():
    """
    Example 8: Production deployment configuration.
    
    This example provides a complete configuration suitable
    for production deployment with monitoring and reliability.
    """
    print("\n" + "="*60)
    print("Example 8: Production Deployment Configuration")
    print("="*60)
    
    config = SearchConfig(
        # Production paths
        paths=["/app/src", "/app/lib"],
        include=["**/*.py", "**/*.js", "**/*.ts"],
        exclude=[
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**",
            "**/.git/**"
        ],
        
        # Production GraphRAG settings
        enable_graphrag=True,
        enable_enhanced_indexing=True,
        graphrag_max_hops=2,  # Conservative for production
        graphrag_min_confidence=0.75,  # High confidence for reliability
        graphrag_semantic_threshold=0.8,  # High precision
        graphrag_context_window=8,
        
        # Production Qdrant settings
        qdrant_enabled=True,
        qdrant_host="qdrant-cluster.internal",
        qdrant_port=6333,
        qdrant_https=True,
        qdrant_timeout=60.0,
        qdrant_collection_name="pysearch_prod",
        qdrant_vector_size=384,
        qdrant_distance_metric="Cosine",
        qdrant_batch_size=200,
        
        # Production indexing settings
        enhanced_indexing_include_semantic=True,
        enhanced_indexing_complexity_analysis=True,
        enhanced_indexing_dependency_tracking=True,
        
        # Production performance settings
        max_file_size=20 * 1024 * 1024,  # 20MB limit
        max_line_length=2000,
        timeout=120.0,  # Generous timeout
        cache_ttl=3600,  # 1 hour cache
        max_workers=16,  # High parallelism
        
        # Production reliability settings
        max_retries=5,
        retry_delay=2.0,
        
        # Logging and monitoring
        log_level="INFO",
        enable_metrics=True,
        metrics_port=9090
    )
    
    # Production Qdrant configuration
    qdrant_config = QdrantConfig(
        host="qdrant-cluster.internal",
        port=6333,
        api_key="${QDRANT_API_KEY}",  # From environment
        https=True,
        timeout=120.0,
        collection_name="pysearch_prod",
        vector_size=384,
        distance_metric="Cosine",
        max_retries=5,
        retry_delay=2.0,
        batch_size=200,
        enable_compression=True
    )
    
    print("Production deployment configuration:")
    print(f"  - Paths: {config.paths}")
    print(f"  - Max workers: {config.max_workers}")
    print(f"  - Cache TTL: {config.cache_ttl}s")
    print(f"  - Timeout: {config.timeout}s")
    print(f"  - Max retries: {config.max_retries}")
    print(f"  - Qdrant host: {qdrant_config.host}")
    print(f"  - Qdrant HTTPS: {qdrant_config.https}")
    print(f"  - Compression: {qdrant_config.enable_compression}")
    
    # Validate production configuration
    issues = config.validate_advanced_config()
    if issues:
        print("\n⚠️  Production configuration issues:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("\n✓ Production configuration is valid")
    
    return config, qdrant_config


def main():
    """
    Main function to demonstrate all configuration examples.
    """
    print("GraphRAG Configuration Examples for pysearch")
    print("=" * 60)
    print("This demo showcases various GraphRAG configuration options")
    print("for different use cases and deployment scenarios.")
    print()
    
    examples = [
        example_1_basic_graphrag_config,
        example_2_qdrant_configuration,
        example_3_enhanced_indexing_config,
        example_4_performance_optimization,
        example_5_multi_repo_config,
        example_6_custom_filtering,
        example_7_semantic_tuning,
        example_8_production_deployment
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\nRunning Configuration Example {i}...")
            result = example_func()
            print(f"✓ Configuration Example {i} completed successfully")
        except Exception as e:
            print(f"✗ Configuration Example {i} failed: {e}")
            logger.exception(f"Error in configuration example {i}")
    
    print("\n" + "="*60)
    print("All configuration examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
