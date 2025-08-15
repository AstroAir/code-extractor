"""
Performance monitoring and optimization system for enhanced indexing.

This module implements comprehensive performance monitoring, metrics collection,
and optimization tools to ensure the indexing system operates efficiently
and can scale to handle large codebases.

Classes:
    PerformanceMetric: Individual performance measurement
    MetricsCollector: Collects and aggregates performance metrics
    PerformanceProfiler: Profiles indexing operations
    OptimizationEngine: Suggests and applies optimizations
    PerformanceMonitor: Main monitoring coordinator

Features:
    - Real-time performance metrics collection
    - Memory usage tracking and optimization
    - CPU utilization monitoring
    - I/O performance analysis
    - Indexing throughput measurement
    - Bottleneck identification and resolution
    - Automatic optimization suggestions
    - Performance trend analysis

Example:
    Basic performance monitoring:
        >>> from pysearch.performance_monitoring import PerformanceMonitor
        >>> monitor = PerformanceMonitor()
        >>> await monitor.start_monitoring()

    Advanced profiling:
        >>> from pysearch.performance_monitoring import PerformanceProfiler
        >>> profiler = PerformanceProfiler()
        >>> async with profiler.profile_operation("indexing"):
        ...     await index_files()
"""

from __future__ import annotations

import asyncio
import json
import psutil
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from .logging_config import get_logger

logger = get_logger()


class MetricType(str, Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class PerformanceMetric:
    """Individual performance measurement."""
    name: str
    metric_type: MetricType
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class OperationProfile:
    """Profile data for an indexing operation."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_start: float = 0.0
    memory_peak: float = 0.0
    memory_end: float = 0.0
    cpu_usage: float = 0.0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    files_processed: int = 0
    errors_encountered: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates performance metrics.

    Provides a centralized system for collecting various performance
    metrics during indexing operations with efficient storage and retrieval.
    """

    def __init__(self, max_metrics: int = 100000):
        self.max_metrics = max_metrics
        self.metrics: List[PerformanceMetric] = []
        self.metric_aggregates: Dict[str, Dict[str, float]] = {}
        self._lock = asyncio.Lock()

    async def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        unit: str = "",
    ) -> None:
        """Record a performance metric."""
        async with self._lock:
            metric = PerformanceMetric(
                name=name,
                metric_type=metric_type,
                value=value,
                tags=tags or {},
                unit=unit,
            )

            self.metrics.append(metric)

            # Update aggregates
            if name not in self.metric_aggregates:
                self.metric_aggregates[name] = {
                    "count": 0,
                    "sum": 0.0,
                    "min": float('inf'),
                    "max": float('-inf'),
                    "avg": 0.0,
                }

            agg = self.metric_aggregates[name]
            agg["count"] += 1
            agg["sum"] += value
            agg["min"] = min(agg["min"], value)
            agg["max"] = max(agg["max"], value)
            agg["avg"] = agg["sum"] / agg["count"]

            # Trim old metrics if needed
            if len(self.metrics) > self.max_metrics:
                self.metrics = self.metrics[-self.max_metrics:]

    async def get_metrics(
        self,
        name_pattern: Optional[str] = None,
        since: Optional[float] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[PerformanceMetric]:
        """Get metrics matching the criteria."""
        async with self._lock:
            filtered_metrics = self.metrics

            # Filter by name pattern
            if name_pattern:
                filtered_metrics = [
                    m for m in filtered_metrics
                    if name_pattern in m.name
                ]

            # Filter by timestamp
            if since:
                filtered_metrics = [
                    m for m in filtered_metrics
                    if m.timestamp >= since
                ]

            # Filter by tags
            if tags:
                filtered_metrics = [
                    m for m in filtered_metrics
                    if all(m.tags.get(k) == v for k, v in tags.items())
                ]

            return filtered_metrics

    def get_aggregate_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get aggregate statistics for a metric."""
        return self.metric_aggregates.get(name)

    def get_all_aggregates(self) -> Dict[str, Dict[str, float]]:
        """Get all metric aggregates."""
        return self.metric_aggregates.copy()


class PerformanceProfiler:
    """
    Profiles indexing operations for detailed performance analysis.

    Provides detailed profiling of indexing operations including memory usage,
    CPU utilization, I/O patterns, and operation timing.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_profiles: Dict[str, OperationProfile] = {}
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def profile_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """Context manager for profiling an operation."""
        profile_id = f"{operation_name}_{int(time.time() * 1000)}"

        # Start profiling
        await self._start_profile(profile_id, operation_name, metadata or {})

        try:
            yield profile_id
        finally:
            # End profiling
            await self._end_profile(profile_id)

    async def _start_profile(
        self,
        profile_id: str,
        operation_name: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Start profiling an operation."""
        async with self._lock:
            # Get initial system stats
            process = psutil.Process()
            memory_info = process.memory_info()

            profile = OperationProfile(
                operation_name=operation_name,
                start_time=time.time(),
                memory_start=memory_info.rss / 1024 / 1024,  # MB
                memory_peak=memory_info.rss / 1024 / 1024,
                metadata=metadata,
            )

            self.active_profiles[profile_id] = profile

            # Record start metric
            await self.metrics_collector.record_metric(
                f"operation_started",
                1.0,
                MetricType.COUNTER,
                tags={"operation": operation_name},
            )

    async def _end_profile(self, profile_id: str) -> None:
        """End profiling an operation."""
        async with self._lock:
            if profile_id not in self.active_profiles:
                return

            profile = self.active_profiles.pop(profile_id)

            # Get final system stats
            process = psutil.Process()
            memory_info = process.memory_info()
            io_counters = process.io_counters()

            profile.end_time = time.time()
            profile.duration = profile.end_time - profile.start_time
            profile.memory_end = memory_info.rss / 1024 / 1024  # MB
            profile.cpu_usage = process.cpu_percent()
            profile.io_read_bytes = io_counters.read_bytes
            profile.io_write_bytes = io_counters.write_bytes

            # Record completion metrics
            await self.metrics_collector.record_metric(
                f"operation_duration",
                profile.duration,
                MetricType.TIMER,
                tags={"operation": profile.operation_name},
                unit="seconds",
            )

            await self.metrics_collector.record_metric(
                f"operation_memory_peak",
                profile.memory_peak,
                MetricType.GAUGE,
                tags={"operation": profile.operation_name},
                unit="MB",
            )

            await self.metrics_collector.record_metric(
                f"operation_completed",
                1.0,
                MetricType.COUNTER,
                tags={"operation": profile.operation_name},
            )

            logger.info(
                f"Operation {profile.operation_name} completed in {profile.duration:.2f}s")

    async def update_profile_stats(
        self,
        profile_id: str,
        files_processed: int = 0,
        errors_encountered: int = 0,
    ) -> None:
        """Update profile statistics during operation."""
        async with self._lock:
            if profile_id in self.active_profiles:
                profile = self.active_profiles[profile_id]
                profile.files_processed += files_processed
                profile.errors_encountered += errors_encountered

                # Update peak memory usage
                try:
                    process = psutil.Process()
                    current_memory = process.memory_info().rss / 1024 / 1024
                    profile.memory_peak = max(
                        profile.memory_peak, current_memory)
                except Exception:
                    pass


class OptimizationEngine:
    """
    Analyzes performance metrics and suggests optimizations.

    Uses collected performance data to identify bottlenecks and
    automatically apply or suggest optimizations for better performance.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.optimization_history: List[Dict[str, Any]] = []

    async def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance and identify issues."""
        analysis: dict[str, Any] = {
            "bottlenecks": [],
            "recommendations": [],
            "optimization_opportunities": [],
        }

        # Analyze memory usage
        memory_metrics = await self.metrics_collector.get_metrics("memory")
        if memory_metrics:
            avg_memory = sum(m.value for m in memory_metrics) / \
                len(memory_metrics)
            if avg_memory > 1000:  # > 1GB
                analysis["bottlenecks"].append("High memory usage")
                analysis["recommendations"].append(
                    "Consider reducing batch sizes or enabling streaming")

        # Analyze operation durations
        duration_aggregates = self.metrics_collector.get_all_aggregates()
        for metric_name, stats in duration_aggregates.items():
            if "duration" in metric_name and stats["avg"] > 60:  # > 1 minute
                analysis["bottlenecks"].append(
                    f"Slow operation: {metric_name}")
                analysis["recommendations"].append(
                    f"Optimize {metric_name} operation")

        # Analyze error rates
        error_metrics = await self.metrics_collector.get_metrics("error")
        if error_metrics:
            recent_errors = [
                m for m in error_metrics if time.time() - m.timestamp < 3600]
            if len(recent_errors) > 10:
                analysis["bottlenecks"].append("High error rate")
                analysis["recommendations"].append(
                    "Review error logs and improve error handling")

        return analysis

    async def suggest_optimizations(self, config: Any) -> List[Dict[str, Any]]:
        """Suggest specific optimizations based on performance data."""
        suggestions = []

        # Analyze current configuration
        analysis = await self.analyze_performance()

        # Memory optimization suggestions
        if "High memory usage" in analysis["bottlenecks"]:
            suggestions.append({
                "type": "memory",
                "description": "Reduce batch size for embedding generation",
                "current_value": getattr(config, "embedding_batch_size", 100),
                "suggested_value": max(10, getattr(config, "embedding_batch_size", 100) // 2),
                "expected_improvement": "50% memory reduction",
            })

        # Performance optimization suggestions
        duration_aggregates = self.metrics_collector.get_all_aggregates()
        for metric_name, stats in duration_aggregates.items():
            if "indexing" in metric_name and stats["avg"] > 30:
                suggestions.append({
                    "type": "performance",
                    "description": f"Enable parallel processing for {metric_name}",
                    "current_value": "sequential",
                    "suggested_value": "parallel",
                    "expected_improvement": "2-4x speed improvement",
                })

        return suggestions

    async def apply_optimization(
        self,
        optimization: Dict[str, Any],
        config: Any,
    ) -> bool:
        """Apply an optimization to the configuration."""
        try:
            opt_type = optimization["type"]

            if opt_type == "memory":
                # Apply memory optimization
                if "batch_size" in optimization["description"]:
                    setattr(config, "embedding_batch_size",
                            optimization["suggested_value"])
                    logger.info(
                        f"Applied memory optimization: batch_size = {optimization['suggested_value']}")
                    return True

            elif opt_type == "performance":
                # Apply performance optimization
                if "parallel" in optimization["suggested_value"]:
                    setattr(config, "enable_parallel_processing", True)
                    logger.info(
                        "Applied performance optimization: enabled parallel processing")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error applying optimization: {e}")
            return False


class PerformanceMonitor:
    """
    Main performance monitoring coordinator.

    Integrates metrics collection, profiling, and optimization to provide
    comprehensive performance monitoring for the indexing system.
    """

    def __init__(self, config: Any, cache_dir: Path) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.metrics_collector = MetricsCollector()
        self.profiler = PerformanceProfiler(self.metrics_collector)
        self.optimization_engine = OptimizationEngine(self.metrics_collector)

        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.metrics_file = cache_dir / "performance_metrics.jsonl"

    async def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics()

                # Save metrics to file periodically
                await self._save_metrics_to_file()

                # Wait before next collection
                await asyncio.sleep(10.0)  # Collect every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5.0)

    async def _collect_system_metrics(self) -> None:
        """Collect system-level performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.metrics_collector.record_metric(
                "system_cpu_usage",
                cpu_percent,
                MetricType.GAUGE,
                unit="percent"
            )

            # Memory usage
            memory = psutil.virtual_memory()
            await self.metrics_collector.record_metric(
                "system_memory_usage",
                memory.percent,
                MetricType.GAUGE,
                unit="percent"
            )

            await self.metrics_collector.record_metric(
                "system_memory_available",
                memory.available / 1024 / 1024,  # MB
                MetricType.GAUGE,
                unit="MB"
            )

            # Disk usage
            disk = psutil.disk_usage(str(self.cache_dir))
            await self.metrics_collector.record_metric(
                "disk_usage",
                disk.percent,
                MetricType.GAUGE,
                unit="percent"
            )

            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()

            await self.metrics_collector.record_metric(
                "process_memory_rss",
                process_memory.rss / 1024 / 1024,  # MB
                MetricType.GAUGE,
                unit="MB"
            )

            await self.metrics_collector.record_metric(
                "process_cpu_usage",
                process.cpu_percent(),
                MetricType.GAUGE,
                unit="percent"
            )

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    async def _save_metrics_to_file(self) -> None:
        """Save metrics to persistent file."""
        try:
            # Get recent metrics (last 5 minutes)
            recent_time = time.time() - 300
            recent_metrics = await self.metrics_collector.get_metrics(since=recent_time)

            # Save to file
            with open(self.metrics_file, "a", encoding="utf-8") as f:
                for metric in recent_metrics:
                    metric_data = {
                        "name": metric.name,
                        "type": metric.metric_type.value,
                        "value": metric.value,
                        "timestamp": metric.timestamp,
                        "tags": metric.tags,
                        "unit": metric.unit,
                    }
                    f.write(json.dumps(metric_data) + "\n")

        except Exception as e:
            logger.error(f"Error saving metrics to file: {e}")

    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        # Get system metrics
        system_metrics = await self._get_current_system_metrics()

        # Get operation metrics
        operation_metrics = self.metrics_collector.get_all_aggregates()

        # Get optimization suggestions
        optimizations = await self.optimization_engine.suggest_optimizations(self.config)

        # Analyze performance trends
        trends = await self._analyze_performance_trends()

        return {
            "timestamp": time.time(),
            "system": system_metrics,
            "operations": operation_metrics,
            "optimizations": optimizations,
            "trends": trends,
            "health_score": self._calculate_health_score(system_metrics, operation_metrics),
        }

    async def _get_current_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(str(self.cache_dir))

            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()

            return {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_mb": memory.available / 1024 / 1024,
                "disk_usage_percent": disk.percent,
                "process_memory_mb": process_memory.rss / 1024 / 1024,
                "process_cpu_percent": process.cpu_percent(),
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}

    async def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        # Get metrics from last hour
        recent_time = time.time() - 3600
        recent_metrics = await self.metrics_collector.get_metrics(since=recent_time)

        if not recent_metrics:
            return {"trend": "no_data"}

        # Group by metric name
        metric_groups: dict[str, list[float]] = {}
        for metric in recent_metrics:
            if metric.name not in metric_groups:
                metric_groups[metric.name] = []
            metric_groups[metric.name].append(metric.value)

        # Analyze trends
        trends = {}
        for name, values in metric_groups.items():
            if len(values) >= 2:
                # Simple trend analysis
                first_half = values[:len(values)//2]
                second_half = values[len(values)//2:]

                avg_first = sum(first_half) / len(first_half)
                avg_second = sum(second_half) / len(second_half)

                if avg_second > avg_first * 1.1:
                    trends[name] = "increasing"
                elif avg_second < avg_first * 0.9:
                    trends[name] = "decreasing"
                else:
                    trends[name] = "stable"

        return trends

    def _calculate_health_score(
        self,
        system_metrics: Dict[str, Any],
        operation_metrics: Dict[str, Dict[str, float]],
    ) -> float:
        """Calculate overall system health score (0.0 to 1.0)."""
        health_score = 1.0

        # System health factors
        cpu_usage = system_metrics.get("cpu_usage_percent", 0)
        memory_usage = system_metrics.get("memory_usage_percent", 0)
        disk_usage = system_metrics.get("disk_usage_percent", 0)

        # Penalize high resource usage
        if cpu_usage > 80:
            health_score -= 0.3
        elif cpu_usage > 60:
            health_score -= 0.1

        if memory_usage > 90:
            health_score -= 0.4
        elif memory_usage > 70:
            health_score -= 0.2

        if disk_usage > 95:
            health_score -= 0.3
        elif disk_usage > 85:
            health_score -= 0.1

        # Operation health factors
        for metric_name, stats in operation_metrics.items():
            if "error" in metric_name and stats["count"] > 0:
                error_rate = stats["sum"] / stats["count"]
                health_score -= min(0.2, error_rate * 0.1)

        return max(0.0, health_score)
