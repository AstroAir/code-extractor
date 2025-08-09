#!/usr/bin/env python3
"""
MCP Composition Support for PySearch

This module implements composition support for chaining multiple search operations
and combining results from different search modes with sophisticated merging strategies.

Features:
- Search operation chaining and pipelining
- Result merging and deduplication
- Cross-search correlation and analysis
- Workflow automation for complex search scenarios
- Result transformation and filtering pipelines
- Performance optimization for composed operations
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Import MCP server components
from ..servers.mcp_server import (
    FuzzySearchConfig,
    MultiPatternQuery,
    PySearchMCPServer,
    SearchFilter,
    SearchResponse,
)


class CompositionStrategy(Enum):
    """Strategies for combining search results."""

    UNION = "union"  # Combine all results
    INTERSECTION = "intersection"  # Only results that appear in all searches
    DIFFERENCE = "difference"  # Results from first search minus subsequent
    WEIGHTED_MERGE = "weighted_merge"  # Merge with weighted scoring
    SEQUENTIAL_FILTER = "sequential_filter"  # Apply searches as filters


class ResultMergeMode(Enum):
    """Modes for merging overlapping results."""

    DEDUPLICATE = "deduplicate"  # Remove exact duplicates
    MERGE_OVERLAPPING = "merge_overlapping"  # Merge overlapping line ranges
    KEEP_ALL = "keep_all"  # Keep all results including duplicates
    BEST_SCORE = "best_score"  # Keep result with best score for duplicates


@dataclass
class SearchOperation:
    """Definition of a single search operation in a composition."""

    operation_type: str  # "text", "regex", "fuzzy", "ast", "semantic", "ranked", "filtered"
    parameters: dict[str, Any]
    weight: float = 1.0
    filter_previous: bool = False  # Whether to filter previous results


@dataclass
class CompositionPipeline:
    """Definition of a search composition pipeline."""

    name: str
    description: str
    operations: list[SearchOperation]
    strategy: CompositionStrategy = CompositionStrategy.UNION
    merge_mode: ResultMergeMode = ResultMergeMode.DEDUPLICATE
    max_results: int | None = None
    session_id: str | None = None


@dataclass
class CompositionResult:
    """Result of a composed search operation."""

    pipeline_name: str
    total_operations: int
    successful_operations: int
    final_results: list[dict[str, Any]]
    intermediate_results: list[SearchResponse]
    execution_time_ms: float
    composition_stats: dict[str, Any]
    session_id: str | None = None


class SearchComposer:
    """
    Handles composition and chaining of multiple search operations.

    Provides sophisticated result merging, deduplication, and correlation
    capabilities for complex search workflows.
    """

    def __init__(self, server: PySearchMCPServer):
        self.server = server
        self.predefined_pipelines: dict[str, CompositionPipeline] = {}
        self._initialize_predefined_pipelines()

    def _initialize_predefined_pipelines(self) -> None:
        """Initialize common predefined search pipelines."""

        # Security analysis pipeline
        self.predefined_pipelines["security_analysis"] = CompositionPipeline(
            name="Security Analysis",
            description="Comprehensive security vulnerability detection",
            operations=[
                SearchOperation(
                    operation_type="multi_pattern",
                    parameters={
                        "patterns": [
                            r"password\s*=\s*[\"'][^\"']+[\"']",
                            r"api_key\s*=\s*[\"'][^\"']+[\"']",
                            r"secret\s*=\s*[\"'][^\"']+[\"']",
                        ],
                        "operator": "OR",
                        "use_regex": True,
                    },
                    weight=2.0,
                ),
                SearchOperation(
                    operation_type="regex",
                    parameters={"pattern": r"eval\s*\(|exec\s*\(", "use_regex": True},
                    weight=1.5,
                ),
                SearchOperation(
                    operation_type="semantic", parameters={"concept": "sql injection"}, weight=1.0
                ),
            ],
            strategy=CompositionStrategy.WEIGHTED_MERGE,
            merge_mode=ResultMergeMode.MERGE_OVERLAPPING,
        )

        # Performance analysis pipeline
        self.predefined_pipelines["performance_analysis"] = CompositionPipeline(
            name="Performance Analysis",
            description="Identify performance bottlenecks and optimization opportunities",
            operations=[
                SearchOperation(
                    operation_type="ast",
                    parameters={
                        "pattern": "for",
                        "func_name": ".*",
                    },
                    weight=1.0,
                ),
                SearchOperation(
                    operation_type="filtered",
                    parameters={
                        "pattern": "query|select|insert|update|delete",
                        "min_complexity": 10.0,
                        "use_regex": True,
                    },
                    weight=1.5,
                ),
                SearchOperation(
                    operation_type="semantic", parameters={"concept": "performance"}, weight=1.0
                ),
            ],
            strategy=CompositionStrategy.WEIGHTED_MERGE,
            merge_mode=ResultMergeMode.BEST_SCORE,
        )

        # Code quality pipeline
        self.predefined_pipelines["code_quality"] = CompositionPipeline(
            name="Code Quality Assessment",
            description="Comprehensive code quality analysis",
            operations=[
                SearchOperation(
                    operation_type="filtered",
                    parameters={"pattern": "TODO|FIXME|HACK", "use_regex": True},
                    weight=0.5,
                ),
                SearchOperation(
                    operation_type="filtered",
                    parameters={"pattern": ".*", "min_complexity": 15.0, "use_regex": True},
                    weight=1.5,
                ),
                SearchOperation(
                    operation_type="multi_pattern",
                    parameters={
                        "patterns": [r"def .{50,}", r"class .{50,}"],
                        "operator": "OR",
                        "use_regex": True,
                    },
                    weight=1.0,
                ),
            ],
            strategy=CompositionStrategy.UNION,
            merge_mode=ResultMergeMode.DEDUPLICATE,
        )

    async def execute_pipeline(self, pipeline: CompositionPipeline) -> CompositionResult:
        """
        Execute a complete search composition pipeline.

        Args:
            pipeline: The pipeline definition to execute

        Returns:
            CompositionResult with combined results and metadata
        """
        start_time = asyncio.get_event_loop().time()
        intermediate_results = []
        successful_operations = 0

        try:
            # Execute each operation in the pipeline
            for i, operation in enumerate(pipeline.operations):
                try:
                    result = await self._execute_single_operation(operation, pipeline.session_id)
                    intermediate_results.append(result)
                    successful_operations += 1
                except Exception as e:
                    # Log error but continue with other operations
                    print(f"Operation {i} failed: {e}")
                    continue

            # Combine results based on strategy
            final_results = self._combine_results(
                intermediate_results,
                pipeline.strategy,
                pipeline.merge_mode,
                [op.weight for op in pipeline.operations[: len(intermediate_results)]],
            )

            # Apply result limit if specified
            if pipeline.max_results:
                final_results = final_results[: pipeline.max_results]

            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

            # Calculate composition statistics
            composition_stats = self._calculate_composition_stats(
                intermediate_results, final_results, pipeline
            )

            return CompositionResult(
                pipeline_name=pipeline.name,
                total_operations=len(pipeline.operations),
                successful_operations=successful_operations,
                final_results=final_results,
                intermediate_results=intermediate_results,
                execution_time_ms=execution_time,
                composition_stats=composition_stats,
                session_id=pipeline.session_id,
            )

        except Exception as e:
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            return CompositionResult(
                pipeline_name=pipeline.name,
                total_operations=len(pipeline.operations),
                successful_operations=successful_operations,
                final_results=[],
                intermediate_results=intermediate_results,
                execution_time_ms=execution_time,
                composition_stats={"error": str(e)},
                session_id=pipeline.session_id,
            )

    async def _execute_single_operation(
        self, operation: SearchOperation, session_id: str | None
    ) -> SearchResponse:
        """Execute a single search operation."""

        if operation.operation_type == "text":
            return await self.server.search_text(
                pattern=operation.parameters["pattern"],
                paths=operation.parameters.get("paths"),
                context=operation.parameters.get("context", 3),
                case_sensitive=operation.parameters.get("case_sensitive", False),
            )

        elif operation.operation_type == "regex":
            return await self.server.search_regex(
                pattern=operation.parameters["pattern"],
                paths=operation.parameters.get("paths"),
                context=operation.parameters.get("context", 3),
                case_sensitive=operation.parameters.get("case_sensitive", False),
            )

        elif operation.operation_type == "fuzzy":
            config = FuzzySearchConfig(
                similarity_threshold=operation.parameters.get("similarity_threshold", 0.6),
                max_results=operation.parameters.get("max_results", 100),
                algorithm=operation.parameters.get("algorithm", "ratio"),
                case_sensitive=operation.parameters.get("case_sensitive", False),
            )
            return await self.server.search_fuzzy(
                pattern=operation.parameters["pattern"],
                paths=operation.parameters.get("paths"),
                config=config,
                context=operation.parameters.get("context", 3),
                session_id=session_id,
            )

        elif operation.operation_type == "ast":
            return await self.server.search_ast(
                pattern=operation.parameters["pattern"],
                func_name=operation.parameters.get("func_name"),
                class_name=operation.parameters.get("class_name"),
                decorator=operation.parameters.get("decorator"),
                imported=operation.parameters.get("imported"),
                paths=operation.parameters.get("paths"),
                context=operation.parameters.get("context", 3),
            )

        elif operation.operation_type == "semantic":
            return await self.server.search_semantic(
                concept=operation.parameters["concept"],
                paths=operation.parameters.get("paths"),
                context=operation.parameters.get("context", 3),
            )

        elif operation.operation_type == "multi_pattern":
            from ..servers.mcp_server import SearchOperator

            query = MultiPatternQuery(
                patterns=operation.parameters["patterns"],
                operator=SearchOperator(operation.parameters.get("operator", "OR")),
                use_regex=operation.parameters.get("use_regex", False),
                use_fuzzy=operation.parameters.get("use_fuzzy", False),
            )
            return await self.server.search_multi_pattern(
                query=query,
                paths=operation.parameters.get("paths"),
                context=operation.parameters.get("context", 3),
                session_id=session_id,
            )

        elif operation.operation_type == "filtered":
            search_filter = SearchFilter(
                min_file_size=operation.parameters.get("min_file_size"),
                max_file_size=operation.parameters.get("max_file_size"),
                languages=operation.parameters.get("languages"),
                file_extensions=operation.parameters.get("file_extensions"),
                min_complexity=operation.parameters.get("min_complexity"),
                max_complexity=operation.parameters.get("max_complexity"),
            )
            return await self.server.search_with_filters(
                pattern=operation.parameters["pattern"],
                search_filter=search_filter,
                paths=operation.parameters.get("paths"),
                context=operation.parameters.get("context", 3),
                use_regex=operation.parameters.get("use_regex", False),
                session_id=session_id,
            )

        else:
            raise ValueError(f"Unknown operation type: {operation.operation_type}")

    def _combine_results(
        self,
        results: list[SearchResponse],
        strategy: CompositionStrategy,
        merge_mode: ResultMergeMode,
        weights: list[float],
    ) -> list[dict[str, Any]]:
        """Combine multiple search results based on strategy and merge mode."""

        if not results:
            return []

        if strategy == CompositionStrategy.UNION:
            return self._union_results(results, merge_mode)

        elif strategy == CompositionStrategy.INTERSECTION:
            return self._intersection_results(results, merge_mode)

        elif strategy == CompositionStrategy.DIFFERENCE:
            return self._difference_results(results, merge_mode)

        elif strategy == CompositionStrategy.WEIGHTED_MERGE:
            return self._weighted_merge_results(results, weights, merge_mode)

        elif strategy == CompositionStrategy.SEQUENTIAL_FILTER:
            return self._sequential_filter_results(results, merge_mode)

        else:
            return self._union_results(results, merge_mode)

    def _union_results(
        self, results: list[SearchResponse], merge_mode: ResultMergeMode
    ) -> list[dict[str, Any]]:
        """Combine all results using union strategy."""
        all_items = []
        for result in results:
            all_items.extend(result.items)

        return self._apply_merge_mode(all_items, merge_mode)

    def _intersection_results(
        self, results: list[SearchResponse], merge_mode: ResultMergeMode
    ) -> list[dict[str, Any]]:
        """Find intersection of all results."""
        if not results:
            return []

        # Start with first result
        common_items = results[0].items.copy()

        # Find intersection with each subsequent result
        for result in results[1:]:
            common_items = self._find_overlapping_items(common_items, result.items)

        return self._apply_merge_mode(common_items, merge_mode)

    def _difference_results(
        self, results: list[SearchResponse], merge_mode: ResultMergeMode
    ) -> list[dict[str, Any]]:
        """Subtract subsequent results from first result."""
        if not results:
            return []

        base_items = results[0].items.copy()

        # Remove items that appear in subsequent results
        for result in results[1:]:
            base_items = self._subtract_items(base_items, result.items)

        return self._apply_merge_mode(base_items, merge_mode)

    def _weighted_merge_results(
        self, results: list[SearchResponse], weights: list[float], merge_mode: ResultMergeMode
    ) -> list[dict[str, Any]]:
        """Merge results with weighted scoring."""
        weighted_items = []

        for i, result in enumerate(results):
            weight = weights[i] if i < len(weights) else 1.0
            for item in result.items:
                # Add weight to item score
                weighted_item = item.copy()
                current_score = weighted_item.get("score", 1.0) or 1.0
                weighted_item["score"] = current_score * weight
                weighted_item["composition_weight"] = weight
                weighted_items.append(weighted_item)

        # Sort by weighted score
        weighted_items.sort(key=lambda x: x.get("score", 0), reverse=True)

        return self._apply_merge_mode(weighted_items, merge_mode)

    def _sequential_filter_results(
        self, results: list[SearchResponse], merge_mode: ResultMergeMode
    ) -> list[dict[str, Any]]:
        """Apply each search as a filter on previous results."""
        if not results:
            return []

        filtered_items = results[0].items.copy()

        # Apply each subsequent result as a filter
        for result in results[1:]:
            filtered_items = self._filter_items_by_overlap(filtered_items, result.items)

        return self._apply_merge_mode(filtered_items, merge_mode)

    def _apply_merge_mode(
        self, items: list[dict[str, Any]], merge_mode: ResultMergeMode
    ) -> list[dict[str, Any]]:
        """Apply merge mode to handle overlapping results."""

        if merge_mode == ResultMergeMode.KEEP_ALL:
            return items

        elif merge_mode == ResultMergeMode.DEDUPLICATE:
            return self._deduplicate_items(items)

        elif merge_mode == ResultMergeMode.MERGE_OVERLAPPING:
            return self._merge_overlapping_items(items)

        elif merge_mode == ResultMergeMode.BEST_SCORE:
            return self._keep_best_score_items(items)

        else:
            return self._deduplicate_items(items)

    def _deduplicate_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove exact duplicate items."""
        seen = set()
        unique_items = []

        for item in items:
            key = f"{item['file']}:{item['start_line']}:{item['end_line']}"
            if key not in seen:
                unique_items.append(item)
                seen.add(key)

        return unique_items

    def _merge_overlapping_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge items with overlapping line ranges."""
        # Group by file
        file_groups: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            file_path = item["file"]
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(item)

        merged_items = []

        # Merge overlapping items within each file
        for file_path, file_items in file_groups.items():
            # Sort by start line
            file_items.sort(key=lambda x: x["start_line"])

            current_item = None
            for item in file_items:
                if current_item is None:
                    current_item = item.copy()
                elif self._items_overlap(current_item, item):
                    # Merge the items
                    current_item = self._merge_two_items(current_item, item)
                else:
                    # No overlap, add current and start new
                    merged_items.append(current_item)
                    current_item = item.copy()

            if current_item:
                merged_items.append(current_item)

        return merged_items

    def _keep_best_score_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only the best scoring item for each location."""
        location_items: dict[str, dict[str, Any]] = {}

        for item in items:
            key = f"{item['file']}:{item['start_line']}:{item['end_line']}"
            score = item.get("score", 0) or 0

            if key not in location_items or score > location_items[key].get("score", 0):
                location_items[key] = item

        return list(location_items.values())

    def _find_overlapping_items(
        self, items1: list[dict[str, Any]], items2: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find items that overlap between two lists."""
        overlapping = []

        for item1 in items1:
            for item2 in items2:
                if self._items_overlap(item1, item2):
                    overlapping.append(item1)
                    break

        return overlapping

    def _subtract_items(
        self, base_items: list[dict[str, Any]], subtract_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove items from base_items that overlap with subtract_items."""
        remaining = []

        for base_item in base_items:
            has_overlap = False
            for subtract_item in subtract_items:
                if self._items_overlap(base_item, subtract_item):
                    has_overlap = True
                    break

            if not has_overlap:
                remaining.append(base_item)

        return remaining

    def _filter_items_by_overlap(
        self, items: list[dict[str, Any]], filter_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Keep only items that overlap with filter_items."""
        filtered = []

        for item in items:
            for filter_item in filter_items:
                if self._items_overlap(item, filter_item):
                    filtered.append(item)
                    break

        return filtered

    def _items_overlap(self, item1: dict[str, Any], item2: dict[str, Any]) -> bool:
        """Check if two items overlap (same file and overlapping line ranges)."""
        if item1["file"] != item2["file"]:
            return False

        return bool(item1["start_line"] <= item2["end_line"] and item2["start_line"] <= item1["end_line"])

    def _merge_two_items(self, item1: dict[str, Any], item2: dict[str, Any]) -> dict[str, Any]:
        """Merge two overlapping items."""
        merged = item1.copy()

        # Extend line range
        merged["start_line"] = min(item1["start_line"], item2["start_line"])
        merged["end_line"] = max(item1["end_line"], item2["end_line"])

        # Combine lines
        all_lines = item1.get("lines", []) + item2.get("lines", [])
        merged["lines"] = list(dict.fromkeys(all_lines))  # Remove duplicates while preserving order

        # Combine scores
        score1 = item1.get("score", 0) or 0
        score2 = item2.get("score", 0) or 0
        merged["score"] = max(score1, score2)

        return merged

    def _calculate_composition_stats(
        self,
        intermediate_results: list[SearchResponse],
        final_results: list[dict[str, Any]],
        pipeline: CompositionPipeline,
    ) -> dict[str, Any]:
        """Calculate statistics for the composition operation."""

        total_intermediate_results = sum(len(result.items) for result in intermediate_results)

        return {
            "intermediate_result_count": total_intermediate_results,
            "final_result_count": len(final_results),
            "reduction_ratio": 1 - (len(final_results) / max(total_intermediate_results, 1)),
            "strategy_used": pipeline.strategy.value,
            "merge_mode_used": pipeline.merge_mode.value,
            "operations_executed": len(intermediate_results),
            "operations_planned": len(pipeline.operations),
            "success_rate": len(intermediate_results) / len(pipeline.operations),
        }

    def get_predefined_pipelines(self) -> list[dict[str, Any]]:
        """Get list of predefined composition pipelines."""
        return [
            {
                "id": pipeline_id,
                "name": pipeline.name,
                "description": pipeline.description,
                "operation_count": len(pipeline.operations),
                "strategy": pipeline.strategy.value,
                "merge_mode": pipeline.merge_mode.value,
            }
            for pipeline_id, pipeline in self.predefined_pipelines.items()
        ]

    def get_pipeline_by_id(self, pipeline_id: str) -> CompositionPipeline | None:
        """Get a predefined pipeline by ID."""
        return self.predefined_pipelines.get(pipeline_id)


def create_search_composer(server: PySearchMCPServer) -> SearchComposer:
    """Create and configure a search composer instance."""
    return SearchComposer(server)
