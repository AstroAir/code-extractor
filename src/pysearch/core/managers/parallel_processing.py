"""
Parallel processing strategies for search operations.

This module provides adaptive parallelization strategies for search operations,
optimizing performance based on workload characteristics and system resources.

Classes:
    ParallelSearchManager: Manages parallel search execution strategies

Key Features:
    - Adaptive parallelization based on workload size and complexity
    - Thread pool optimization for I/O-bound operations
    - Process pool support for CPU-intensive operations
    - Intelligent worker count selection
    - Fallback strategies for error handling

Example:
    Using parallel search:
        >>> from pysearch.core.managers.parallel_processing import ParallelSearchManager
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig(parallel=True, workers=4)
        >>> manager = ParallelSearchManager(config)
        >>> results = manager.search_files(file_paths, query, search_function)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from ..config import SearchConfig
from ..types import Query, SearchItem


def _search_file_batch(file_batch: list[Path], query: Query) -> list[list[SearchItem]]:
    """
    Search a batch of files in a separate process.

    This function is used for process pool execution and must be defined
    at module level for pickling.
    """
    from ..api import PySearch  # Import here to avoid circular imports

    # Create a temporary PySearch instance for this process
    temp_engine = PySearch(SearchConfig())
    results = []

    for file_path in file_batch:
        try:
            file_results = temp_engine._search_file(file_path, query)
            results.append(file_results)
        except Exception:
            results.append([])  # Empty results on error

    return results


class ParallelSearchManager:
    """Manages parallel search execution strategies."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.cpu_count = os.cpu_count() or 4

    def search_files(
        self,
        file_paths: list[Path],
        query: Query,
        search_function: Callable[[Path, Query], list[SearchItem]],
    ) -> list[SearchItem]:
        """
        Execute search across multiple files using adaptive parallelization.

        Args:
            file_paths: List of file paths to search
            query: Search query to execute
            search_function: Function to search individual files

        Returns:
            Combined list of search results from all files
        """
        if not self.config.parallel or len(file_paths) < 10:
            return self._search_sequential(file_paths, query, search_function)

        return self._search_with_adaptive_parallelism(file_paths, query, search_function)

    def _search_sequential(
        self,
        file_paths: list[Path],
        query: Query,
        search_function: Callable[[Path, Query], list[SearchItem]],
    ) -> list[SearchItem]:
        """Sequential search for small workloads."""
        items: list[SearchItem] = []

        for file_path in file_paths:
            try:
                results = search_function(file_path, query)
                if results:
                    items.extend(results)
            except Exception:
                # Continue processing other files on error
                continue

        return items

    def _search_with_adaptive_parallelism(
        self,
        file_paths: list[Path],
        query: Query,
        search_function: Callable[[Path, Query], list[SearchItem]],
    ) -> list[SearchItem]:
        """Adaptive parallelization strategy based on workload size and complexity."""
        items: list[SearchItem] = []
        num_files = len(file_paths)

        # Choose between thread and process pool based on workload
        if num_files > 1000 and not query.use_ast:  # Heavy I/O workload
            # Use process pool for CPU-intensive work with many files
            workers = min(self.cpu_count, self.config.workers or self.cpu_count)
            try:
                items = self._search_with_process_pool(file_paths, query, workers)
            except Exception:
                # Fallback to thread pool on process pool failure
                items = self._search_with_thread_pool(file_paths, query, search_function)
        else:
            # Use thread pool for I/O bound or smaller workloads
            items = self._search_with_thread_pool(file_paths, query, search_function)

        return items

    def _search_with_process_pool(
        self, file_paths: list[Path], query: Query, workers: int
    ) -> list[SearchItem]:
        """Process pool-based parallel search for CPU-intensive workloads."""
        items: list[SearchItem] = []

        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Batch files to reduce overhead
                batch_size = max(1, len(file_paths) // (workers * 4))
                batches = [
                    file_paths[i : i + batch_size] for i in range(0, len(file_paths), batch_size)
                ]

                futures = {
                    executor.submit(_search_file_batch, batch, query): batch for batch in batches
                }

                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        for result in batch_results:
                            if result:
                                items.extend(result)
                    except Exception:
                        # Continue processing other batches on error
                        continue

        except Exception:
            # If process pool fails entirely, fall back to sequential
            from ..api import PySearch  # Import here to avoid circular imports

            temp_engine = PySearch(self.config)
            for file_path in file_paths:
                try:
                    results = temp_engine._search_file(file_path, query)
                    if results:
                        items.extend(results)
                except Exception:
                    continue

        return items

    def _search_with_thread_pool(
        self,
        file_paths: list[Path],
        query: Query,
        search_function: Callable[[Path, Query], list[SearchItem]],
    ) -> list[SearchItem]:
        """Thread-based parallel search with optimized worker count."""
        items: list[SearchItem] = []

        # Optimize worker count based on I/O vs CPU ratio
        if query.use_ast or query.use_regex:
            workers = min(self.cpu_count * 2, self.config.workers or self.cpu_count * 2)
        else:
            # I/O bound
            workers = min(self.cpu_count * 4, self.config.workers or self.cpu_count * 4)

        workers = min(workers, len(file_paths))  # Don't over-provision

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(search_function, file_path, query): file_path
                    for file_path in file_paths
                }

                for future in as_completed(futures):
                    try:
                        results = future.result()
                        if results:
                            items.extend(results)
                    except Exception:
                        # Log error but continue processing
                        continue

        except Exception:
            # If thread pool fails, fall back to sequential
            items = self._search_sequential(file_paths, query, search_function)

        return items

    def get_optimal_worker_count(self, file_count: int, query: Query) -> int:
        """Calculate optimal worker count based on workload characteristics."""
        if not self.config.parallel:
            return 1

        base_workers = self.config.workers or self.cpu_count

        # Adjust based on query complexity
        if query.use_ast:
            # AST parsing is CPU-intensive
            return min(base_workers, self.cpu_count)
        elif query.use_regex:
            # Regex is moderately CPU-intensive
            return min(base_workers * 2, self.cpu_count * 2)
        else:
            # Simple text search is I/O bound
            return min(base_workers * 4, self.cpu_count * 4, file_count)

    def should_use_process_pool(self, file_count: int, query: Query) -> bool:
        """Determine if process pool should be used over thread pool."""
        # Use process pool for large workloads without AST parsing
        return file_count > 1000 and not query.use_ast and self.config.parallel

    def get_batch_size(self, file_count: int, worker_count: int) -> int:
        """Calculate optimal batch size for process pool execution."""
        if worker_count <= 0:
            return file_count

        # Aim for 4 batches per worker to balance load
        return max(1, file_count // (worker_count * 4))

    def estimate_search_time(self, file_count: int, query: Query) -> float:
        """Estimate search time based on file count and query complexity."""
        # Base time per file in milliseconds
        base_time_per_file = 1.0

        # Adjust based on query complexity
        if query.use_ast:
            base_time_per_file *= 5.0  # AST parsing is expensive
        elif query.use_regex:
            base_time_per_file *= 2.0  # Regex is moderately expensive

        # Adjust for parallelization
        if self.config.parallel and file_count > 10:
            worker_count = self.get_optimal_worker_count(file_count, query)
            parallelization_factor = min(worker_count, file_count) * 0.8  # 80% efficiency
            base_time_per_file /= parallelization_factor

        return file_count * base_time_per_file
