"""
Configuration module for pysearch.

This module defines the SearchConfig class which serves as the central configuration
object for all search operations. It provides comprehensive settings for search scope,
behavior, performance, and output formatting.

Classes:
    RankStrategy: Enumeration of available ranking strategies
    SearchConfig: Main configuration class with all search parameters

Key Configuration Areas:
    - Search scope: paths, include/exclude patterns, language filtering
    - Search behavior: context lines, content toggles, symlink handling
    - Performance: parallel execution, caching, file size limits
    - Output: format selection, ranking strategies
    - Advanced: strict hash checking, directory pruning

Example:
    Basic configuration:
        >>> from pysearch.config import SearchConfig
        >>> from pysearch.types import OutputFormat
        >>>
        >>> config = SearchConfig(
        ...     paths=["."],
        ...     include=["**/*.py"],
        ...     exclude=["**/.venv/**", "**/__pycache__/**"],
        ...     context=3,
        ...     output_format=OutputFormat.JSON
        ... )

    Performance-optimized configuration:
        >>> config = SearchConfig(
        ...     paths=["./src", "./tests"],
        ...     parallel=True,
        ...     workers=4,
        ...     strict_hash_check=False,
        ...     dir_prune_exclude=True
        ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..utils.error_handling import ConfigurationError
from .types import Language, OutputFormat


class RankStrategy(str, Enum):
    DEFAULT = "default"


@dataclass(slots=True)
class SearchConfig:
    # Scope
    paths: list[str] = field(default_factory=lambda: ["."], metadata={"help": "Root search paths."})
    include: list[str] | None = None  # None = auto-detect based on languages
    exclude: list[str] | None = None  # None = use defaults

    # Language filtering
    # None = auto-detect all supported languages
    languages: set[Language] | None = None
    file_size_limit: int = 2_000_000  # 2MB default limit

    # Behavior
    context: int = 2
    output_format: OutputFormat = OutputFormat.TEXT
    follow_symlinks: bool = False
    # 2MB safeguard (kept for backward compatibility)
    max_file_bytes: int = 2_000_000

    # Content toggles
    enable_docstrings: bool = True
    enable_comments: bool = True
    enable_strings: bool = True

    # Performance
    parallel: bool = True
    workers: int = 0  # 0 = auto(cpu_count)
    cache_dir: Path | None = None  # default: .pysearch-cache under first path
    # New toggles
    # if True, compute sha1 on scan for exact change detection
    strict_hash_check: bool = False
    # if True, prune excluded directories during traversal
    dir_prune_exclude: bool = True

    # Ranking
    rank_strategy: RankStrategy = RankStrategy.DEFAULT
    ast_weight: float = 2.0
    text_weight: float = 1.0

    # GraphRAG Configuration
    enable_graphrag: bool = False
    graphrag_max_hops: int = 2
    graphrag_min_confidence: float = 0.5
    graphrag_semantic_threshold: float = 0.7
    graphrag_context_window: int = 5

    # Metadata Indexing Configuration
    enable_metadata_indexing: bool = False
    metadata_indexing_include_semantic: bool = True
    metadata_indexing_complexity_analysis: bool = True
    metadata_indexing_dependency_tracking: bool = True

    # Indexing Engine Configuration
    embedding_provider: str = "openai"  # "openai", "huggingface", "local"
    embedding_model: str = "text-embedding-ada-002"
    embedding_batch_size: int = 100
    embedding_api_key: str | None = None
    vector_db_provider: str = "lancedb"  # "lancedb", "qdrant", "chroma"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    enable_parallel_processing: bool = False
    max_workers: int = 4
    chunking_strategy: str = "hybrid"  # "structural", "semantic", "hybrid"
    quality_threshold: float = 0.7

    # Qdrant Vector Database Configuration
    qdrant_enabled: bool = False
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_https: bool = False
    qdrant_timeout: float = 30.0
    qdrant_collection_name: str = "pysearch_vectors"
    qdrant_vector_size: int = 384
    qdrant_distance_metric: str = "Cosine"
    qdrant_batch_size: int = 100

    def resolve_cache_dir(self) -> Path:
        base = Path(self.paths[0]).resolve() if self.paths else Path(".").resolve()
        return self.cache_dir or (base / ".pysearch-cache")

    def get_include_patterns(self) -> list[str]:
        """Get include patterns, using defaults if not specified."""
        if self.include is not None:
            return self.include

        # Generate default patterns based on enabled languages
        if self.languages is None:
            # Include all common text file types
            return [
                "**/*.py",
                "**/*.js",
                "**/*.ts",
                "**/*.java",
                "**/*.c",
                "**/*.cpp",
                "**/*.h",
                "**/*.hpp",
                "**/*.cs",
                "**/*.go",
                "**/*.rs",
                "**/*.php",
                "**/*.rb",
                "**/*.swift",
                "**/*.kt",
                "**/*.scala",
                "**/*.r",
                "**/*.m",
                "**/*.sh",
                "**/*.ps1",
                "**/*.sql",
                "**/*.html",
                "**/*.css",
                "**/*.xml",
                "**/*.json",
                "**/*.yaml",
                "**/*.yml",
                "**/*.toml",
                "**/*.md",
                "**/*.txt",
                "**/Dockerfile",
                "**/Makefile",
            ]

        # Generate patterns for specific languages
        from ..analysis.language_detection import get_language_extensions

        patterns = []
        for lang in self.languages:
            extensions = get_language_extensions(lang)
            for ext in extensions:
                patterns.append(f"**/*{ext}")

        # Add special filename patterns
        special_files = ["**/Dockerfile", "**/Makefile", "**/Rakefile", "**/Gemfile"]
        patterns.extend(special_files)

        return patterns

    def get_qdrant_config(self) -> Any:
        """Get Qdrant configuration object from search config."""
        if not self.qdrant_enabled:
            return None

        from ..storage.qdrant_client import QdrantConfig

        return QdrantConfig(
            host=self.qdrant_host,
            port=self.qdrant_port,
            api_key=self.qdrant_api_key,
            https=self.qdrant_https,
            timeout=self.qdrant_timeout,
            collection_name=self.qdrant_collection_name,
            vector_size=self.qdrant_vector_size,
            distance_metric=self.qdrant_distance_metric,
            batch_size=self.qdrant_batch_size,
        )

    def get_graphrag_query_defaults(self) -> dict[str, Any]:
        """Get default GraphRAG query parameters."""
        return {
            "max_hops": self.graphrag_max_hops,
            "min_confidence": self.graphrag_min_confidence,
            "semantic_threshold": self.graphrag_semantic_threshold,
            "context_window": self.graphrag_context_window,
        }

    def is_optional_features_enabled(self) -> bool:
        """Check if any optional features are enabled."""
        return self.enable_graphrag or self.enable_metadata_indexing or self.qdrant_enabled

    def validate(self) -> None:
        """Validate core configuration and raise ConfigurationError on issues.

        Raises:
            ConfigurationError: If the configuration is invalid.
        """
        if not self.paths:
            raise ConfigurationError(
                "At least one search path must be specified",
                context={"field": "paths"},
            )

        if self.context < 0:
            raise ConfigurationError(
                "Context lines must be non-negative",
                context={"field": "context", "value": self.context},
            )

        if self.workers < 0:
            raise ConfigurationError(
                "Worker count must be non-negative (0 = auto-detect CPU count)",
                context={"field": "workers", "value": self.workers},
            )

        if self.file_size_limit <= 0:
            raise ConfigurationError(
                "File size limit must be positive",
                context={"field": "file_size_limit", "value": self.file_size_limit},
            )

        # Also validate optional features
        issues = self.validate_optional_config()
        if issues:
            raise ConfigurationError(
                f"Optional feature configuration issues: {'; '.join(issues)}",
                context={"issues": issues},
            )

    def validate_optional_config(self) -> list[str]:
        """Validate optional feature configuration and return any issues."""
        issues = []

        if self.enable_graphrag and not self.enable_metadata_indexing:
            issues.append("GraphRAG requires metadata indexing to be enabled")

        if self.enable_graphrag and not self.qdrant_enabled:
            issues.append("GraphRAG works best with Qdrant vector database enabled")

        if self.qdrant_enabled:
            if self.qdrant_vector_size <= 0:
                issues.append("Qdrant vector size must be positive")

            if self.qdrant_distance_metric not in ["Cosine", "Dot", "Euclid"]:
                issues.append("Qdrant distance metric must be one of: Cosine, Dot, Euclid")

            if self.qdrant_batch_size <= 0:
                issues.append("Qdrant batch size must be positive")

        if self.graphrag_max_hops < 1:
            issues.append("GraphRAG max hops must be at least 1")

        if not (0.0 <= self.graphrag_min_confidence <= 1.0):
            issues.append("GraphRAG min confidence must be between 0.0 and 1.0")

        if not (0.0 <= self.graphrag_semantic_threshold <= 1.0):
            issues.append("GraphRAG semantic threshold must be between 0.0 and 1.0")

        return issues

    def get_exclude_patterns(self) -> list[str]:
        """Get exclude patterns, using defaults if not specified."""
        if self.exclude is not None:
            return self.exclude

        return [
            "**/.venv/**",
            "**/.git/**",
            "**/build/**",
            "**/dist/**",
            "**/__pycache__/**",
            "**/.svn/**",
            "**/.hg/**",
            "**/node_modules/**",
            "**/target/**",
            "**/bin/**",
            "**/obj/**",
            "**/.pysearch-cache/**",
        ]

    def should_include_language(self, language: Language) -> bool:
        """Check if a language should be included in search."""
        if self.languages is None:
            return language != Language.UNKNOWN
        return language in self.languages
