"""Tests for pysearch.core.integrations.parallel_processing module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pysearch.core.config import SearchConfig
from pysearch.core.integrations.parallel_processing import ParallelSearchManager
from pysearch.core.types import Query, SearchItem


class TestParallelSearchManager:
    """Tests for ParallelSearchManager class."""

    def test_init(self):
        cfg = SearchConfig(parallel=True, workers=4)
        mgr = ParallelSearchManager(cfg)
        assert mgr.config is cfg
        assert mgr.cpu_count >= 1

    def test_init_default_workers(self):
        cfg = SearchConfig()
        mgr = ParallelSearchManager(cfg)
        assert mgr.config.workers == 0

    def test_get_optimal_worker_count_parallel_disabled(self):
        cfg = SearchConfig(parallel=False)
        mgr = ParallelSearchManager(cfg)
        q = Query(pattern="test")
        assert mgr.get_optimal_worker_count(100, q) == 1

    def test_get_optimal_worker_count_ast(self):
        cfg = SearchConfig(parallel=True, workers=4)
        mgr = ParallelSearchManager(cfg)
        q = Query(pattern="test", use_ast=True)
        count = mgr.get_optimal_worker_count(100, q)
        assert count <= mgr.cpu_count

    def test_get_optimal_worker_count_regex(self):
        cfg = SearchConfig(parallel=True, workers=4)
        mgr = ParallelSearchManager(cfg)
        q = Query(pattern="test", use_regex=True)
        count = mgr.get_optimal_worker_count(100, q)
        assert count >= 1

    def test_get_optimal_worker_count_simple(self):
        cfg = SearchConfig(parallel=True, workers=4)
        mgr = ParallelSearchManager(cfg)
        q = Query(pattern="test")
        count = mgr.get_optimal_worker_count(100, q)
        assert count >= 1

    def test_should_use_process_pool_small(self):
        cfg = SearchConfig(parallel=True)
        mgr = ParallelSearchManager(cfg)
        q = Query(pattern="test")
        assert mgr.should_use_process_pool(100, q) is False

    def test_should_use_process_pool_large(self):
        cfg = SearchConfig(parallel=True)
        mgr = ParallelSearchManager(cfg)
        q = Query(pattern="test")
        assert mgr.should_use_process_pool(2000, q) is True

    def test_should_use_process_pool_ast(self):
        cfg = SearchConfig(parallel=True)
        mgr = ParallelSearchManager(cfg)
        q = Query(pattern="test", use_ast=True)
        assert mgr.should_use_process_pool(2000, q) is False

    def test_should_use_process_pool_parallel_disabled(self):
        cfg = SearchConfig(parallel=False)
        mgr = ParallelSearchManager(cfg)
        q = Query(pattern="test")
        assert mgr.should_use_process_pool(2000, q) is False

    def test_get_batch_size_normal(self):
        cfg = SearchConfig()
        mgr = ParallelSearchManager(cfg)
        size = mgr.get_batch_size(1000, 4)
        assert size == 1000 // (4 * 4)

    def test_get_batch_size_zero_workers(self):
        cfg = SearchConfig()
        mgr = ParallelSearchManager(cfg)
        assert mgr.get_batch_size(100, 0) == 100

    def test_get_batch_size_min_one(self):
        cfg = SearchConfig()
        mgr = ParallelSearchManager(cfg)
        size = mgr.get_batch_size(1, 100)
        assert size == 1

    def test_estimate_search_time_simple(self):
        cfg = SearchConfig(parallel=False)
        mgr = ParallelSearchManager(cfg)
        q = Query(pattern="test")
        est = mgr.estimate_search_time(100, q)
        assert est == 100.0

    def test_estimate_search_time_ast(self):
        cfg = SearchConfig(parallel=False)
        mgr = ParallelSearchManager(cfg)
        q = Query(pattern="test", use_ast=True)
        est = mgr.estimate_search_time(100, q)
        assert est == 500.0

    def test_estimate_search_time_regex(self):
        cfg = SearchConfig(parallel=False)
        mgr = ParallelSearchManager(cfg)
        q = Query(pattern="test", use_regex=True)
        est = mgr.estimate_search_time(100, q)
        assert est == 200.0

    def test_estimate_search_time_parallel(self):
        cfg = SearchConfig(parallel=True, workers=4)
        mgr = ParallelSearchManager(cfg)
        q = Query(pattern="test")
        est_parallel = mgr.estimate_search_time(100, q)
        cfg2 = SearchConfig(parallel=False)
        mgr2 = ParallelSearchManager(cfg2)
        est_sequential = mgr2.estimate_search_time(100, q)
        assert est_parallel < est_sequential

    def test_search_files_sequential_small(self, tmp_path: Path):
        cfg = SearchConfig(parallel=True)
        mgr = ParallelSearchManager(cfg)
        files = [tmp_path / f"f{i}.py" for i in range(5)]
        q = Query(pattern="test")

        def mock_search(path, query):
            return [SearchItem(file=path, start_line=1, end_line=1, lines=["test"])]

        results = mgr.search_files(files, q, mock_search)
        assert len(results) == 5

    def test_search_files_sequential_parallel_disabled(self, tmp_path: Path):
        cfg = SearchConfig(parallel=False)
        mgr = ParallelSearchManager(cfg)
        files = [tmp_path / f"f{i}.py" for i in range(50)]
        q = Query(pattern="test")

        def mock_search(path, query):
            return [SearchItem(file=path, start_line=1, end_line=1, lines=["test"])]

        results = mgr.search_files(files, q, mock_search)
        assert len(results) == 50

    def test_search_files_handles_errors(self, tmp_path: Path):
        cfg = SearchConfig(parallel=False)
        mgr = ParallelSearchManager(cfg)
        files = [tmp_path / f"f{i}.py" for i in range(3)]
        q = Query(pattern="test")
        call_count = 0

        def flaky_search(path, query):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("fail")
            return [SearchItem(file=path, start_line=1, end_line=1, lines=["ok"])]

        results = mgr.search_files(files, q, flaky_search)
        assert len(results) == 2

    def test_search_files_empty_results(self, tmp_path: Path):
        cfg = SearchConfig(parallel=False)
        mgr = ParallelSearchManager(cfg)
        files = [tmp_path / f"f{i}.py" for i in range(3)]
        q = Query(pattern="test")

        def no_results(path, query):
            return []

        results = mgr.search_files(files, q, no_results)
        assert results == []

    def test_search_files_thread_pool(self, tmp_path: Path):
        cfg = SearchConfig(parallel=True, workers=2)
        mgr = ParallelSearchManager(cfg)
        files = [tmp_path / f"f{i}.py" for i in range(20)]
        q = Query(pattern="test")

        def mock_search(path, query):
            return [SearchItem(file=path, start_line=1, end_line=1, lines=["test"])]

        results = mgr.search_files(files, q, mock_search)
        assert len(results) == 20
