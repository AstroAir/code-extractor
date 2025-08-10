"""
Backward compatibility module for enhanced_indexer.

This module provides backward compatibility for the old enhanced_indexer module.
The functionality has been moved to indexer_metadata.py.

DEPRECATED: Use pysearch.indexer_metadata instead.
"""

from __future__ import annotations

import warnings

# Import everything from the new module
from .indexer_metadata import (
    EntityMetadata,
    FileMetadata,
    IndexQuery,
    IndexStats,
    MetadataIndex,
    MetadataIndexer,
)

# Backward compatibility alias
EnhancedIndexer = MetadataIndexer

# Issue deprecation warning
warnings.warn(
    "pysearch.enhanced_indexer is deprecated. Use pysearch.indexer_metadata instead.",
    DeprecationWarning,
    stacklevel=2
)
