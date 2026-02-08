"""
Internal manager modules for pysearch core functionality.

This package contains specialized manager modules that extend the core
PySearch functionality with advanced features. Each module focuses on a
specific manager or advanced capability:

- hybrid_search: Semantic and hybrid search capabilities
- graphrag_integration: GraphRAG knowledge graph integration
- indexing_integration: Metadata indexing
- dependency_integration: Code dependency analysis
- file_watching: Real-time file monitoring and auto-updates
- cache_integration: Caching management
- distributed_indexing_integration: Distributed parallel indexing
- ide_integration: IDE hooks and features (jump-to-definition, references, etc.)
- multi_repo_integration: Multi-repository search support
- parallel_processing: Parallel search execution strategies

These modules are used internally by the PySearch class to provide
advanced functionality while keeping the core API clean and focused.
"""

# This package contains integration modules used internally by PySearch
# No public exports needed as these are implementation details
__all__: list[str] = []
