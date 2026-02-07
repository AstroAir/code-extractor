"""Tests for pysearch.utils.performance_monitoring module."""

from __future__ import annotations

import time

import pytest

from pysearch.utils.performance_monitoring import (
    MetricType,
    MetricsCollector,
    PerformanceMetric,
    PerformanceMonitor,
    PerformanceProfiler,
)


class TestPerformanceMetric:
    """Tests for PerformanceMetric dataclass."""

    def test_creation(self):
        m = PerformanceMetric(
            name="indexing_time",
            metric_type=MetricType.GAUGE,
            value=1.5,
            unit="seconds",
        )
        assert m.name == "indexing_time"
        assert m.value == 1.5
        assert m.unit == "seconds"


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_init(self):
        collector = MetricsCollector()
        assert collector is not None

    @pytest.mark.asyncio
    async def test_record_metric(self):
        collector = MetricsCollector()
        await collector.record_metric("test_metric", 42.0)
        metrics = await collector.get_metrics()
        assert len(metrics) >= 1

    @pytest.mark.asyncio
    async def test_record_multiple(self):
        collector = MetricsCollector()
        await collector.record_metric("m1", 1.0)
        await collector.record_metric("m2", 2.0)
        metrics = await collector.get_metrics()
        assert len(metrics) >= 2

    @pytest.mark.asyncio
    async def test_get_metric_by_name(self):
        collector = MetricsCollector()
        await collector.record_metric("target", 99.0)
        metrics = await collector.get_metrics(name_pattern="target")
        assert len(metrics) >= 1

    def test_get_all_aggregates(self):
        collector = MetricsCollector()
        agg = collector.get_all_aggregates()
        assert isinstance(agg, dict)


class TestPerformanceProfiler:
    """Tests for PerformanceProfiler class."""

    def test_init(self):
        collector = MetricsCollector()
        profiler = PerformanceProfiler(collector)
        assert profiler is not None


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""

    def test_init(self, tmp_path):
        from unittest.mock import MagicMock
        monitor = PerformanceMonitor(config=MagicMock(), cache_dir=tmp_path)
        assert monitor is not None
