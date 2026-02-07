"""
GraphRAG (Graph Retrieval-Augmented Generation) functionality.

This module implements GraphRAG capabilities for enhanced code understanding:
- Graph-based code representation and analysis
- RAG-powered code search and retrieval
- Entity relationship mapping
- Semantic code understanding

GraphRAG combines graph-based analysis with retrieval-augmented generation
to provide intelligent code search and understanding capabilities.
"""

# Import main classes
from .core import EntityExtractor, RelationshipMapper

__all__ = ["EntityExtractor", "RelationshipMapper"]
