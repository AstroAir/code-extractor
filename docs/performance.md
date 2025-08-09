# Performance Tuning Guide

This guide helps you optimize pysearch performance for different use cases and environments.

## Table of Contents

- [Performance Overview](#performance-overview)
- [Quick Optimization](#quick-optimization)
- [Configuration Tuning](#configuration-tuning)
- [Hardware Considerations](#hardware-considerations)
- [Use Case Optimization](#use-case-optimization)
- [Monitoring & Profiling](#monitoring--profiling)
- [Troubleshooting Performance](#troubleshooting-performance)

---

## Performance Overview

### Performance Factors

pysearch performance depends on several factors:

1. **Search Scope**: Number and size of files to search
2. **Search Type**: Text vs. regex vs. AST vs. semantic
3. **Hardware**: CPU cores, memory, storage type
4. **Configuration**: Parallel processing, caching, filters
5. **Codebase Characteristics**: File sizes, directory structure

### Performance Metrics

Key metrics to monitor:

- **Search Time**: Total time to complete search
- **Files/Second**: Throughput of file processing
- **Memory Usage**: Peak memory consumption
- **Cache Hit Rate**: Percentage of cached results used
- **CPU Utilization**: Processor usage during search

### Baseline Performance

Typical performance on modern hardware:

| Codebase Size | Files | Search Time | Memory Usage |
|---------------|-------|-------------|--------------|
| Small (1K files) | 1,000 | 0.1-0.5s | 50-100 MB |
| Medium (10K files) | 10,000 | 0.5-2s | 100-300 MB |
| Large (100K files) | 100,000 | 2-10s | 300-800 MB |
| Very Large (1M files) | 1,000,000 | 10-60s | 800-2000 MB |

---

## Quick Optimization

### Immediate Performance Gains

Apply these optimizations for instant performance improvements:

```python
from pysearch import PySearch, SearchConfig

# High-performance configuration
config = SearchConfig(
    paths=["./src"],                    # Specific paths only
    exclude=[                           # Comprehensive exclusions
        "**/.venv/**", "**/.git/**", 
        "**/node_modules/**", "**/__pycache__/**",
        "**/build/**", "**/dist/**"
    ],
    parallel=True,                      # Enable parallel processing
    workers=0,                          # Auto-detect CPU cores
    strict_hash_check=False,            # Faster file change detection
    dir_prune_exclude=True,             # Skip excluded directories
    file_size_limit=2_000_000,          # 2MB file limit
    context=2,                          # Minimal context
    enable_docstrings=False,            # Skip docstrings if not needed
    enable_comments=False               # Skip comments if not needed
)

engine = PySearch(config)
engine.enable_caching(ttl=3600)         # Enable 1-hour caching
```

### CLI Quick Optimization

```bash
# Fast search command
pysearch find \
  --path ./src \
  --exclude "**/.venv/**" "**/.git/**" "**/__pycache__/**" \
  --pattern "your_pattern" \
  --parallel \
  --workers 8 \
  --no-docstrings \
  --no-comments \
  --context 1
```

---

## Configuration Tuning

### Parallel Processing

Optimize worker configuration based on your hardware:

```python
import os

# Conservative (for limited resources)
config = SearchConfig(
    parallel=True,
    workers=2  # Fewer workers
)

# Balanced (recommended)
config = SearchConfig(
    parallel=True,
    workers=0  # Auto-detect (usually cpu_count())
)

# Aggressive (for high-end systems)
config = SearchConfig(
    parallel=True,
    workers=min(16, os.cpu_count() * 2)  # Up to 16 workers
)
```

### Memory Optimization

Control memory usage for different environments:

```python
# Memory-constrained environment
config = SearchConfig(
    file_size_limit=500_000,    # 500KB limit
    workers=2,                  # Fewer workers
    context=1,                  # Minimal context
    strict_hash_check=False     # Less memory for hashing
)

# Memory-rich environment
config = SearchConfig(
    file_size_limit=10_000_000, # 10MB limit
    workers=12,                 # More workers
    context=5,                  # More context
    strict_hash_check=True      # More accurate but uses more memory
)
```

### I/O Optimization

Optimize for different storage types:

```python
# SSD optimization (fast random access)
config = SearchConfig(
    parallel=True,
    workers=8,                  # More workers for parallel I/O
    strict_hash_check=False,    # Faster file checking
    dir_prune_exclude=True      # Skip directories early
)

# HDD optimization (sequential access preferred)
config = SearchConfig(
    parallel=True,
    workers=4,                  # Fewer workers to reduce seeking
    strict_hash_check=False,    # Minimize file reads
    dir_prune_exclude=True      # Essential for HDDs
)

# Network storage optimization
config = SearchConfig(
    parallel=True,
    workers=2,                  # Limit network connections
    file_size_limit=1_000_000,  # Smaller files only
    strict_hash_check=False     # Minimize network I/O
)
```

### Caching Configuration

Optimize caching for different usage patterns:

```python
# Development (frequent searches, changing files)
engine.enable_caching(
    ttl=1800,                   # 30 minutes
    cache_dir="./dev-cache"
)

# CI/CD (stable files, repeated searches)
engine.enable_caching(
    ttl=7200,                   # 2 hours
    cache_dir="/tmp/ci-cache"
)

# Production (stable codebase)
engine.enable_caching(
    ttl=86400,                  # 24 hours
    cache_dir="/var/cache/pysearch"
)
```

---

## Hardware Considerations

### CPU Optimization

Optimize for different CPU configurations:

```python
import psutil

cpu_count = psutil.cpu_count(logical=False)  # Physical cores
logical_count = psutil.cpu_count(logical=True)  # Logical cores

# Single-core systems
if cpu_count == 1:
    config = SearchConfig(
        parallel=False,  # Disable parallelism
        workers=1
    )

# Multi-core systems
elif cpu_count <= 4:
    config = SearchConfig(
        parallel=True,
        workers=cpu_count  # One worker per core
    )

# High-core systems
else:
    config = SearchConfig(
        parallel=True,
        workers=min(cpu_count * 2, 16)  # Up to 2x cores, max 16
    )
```

### Memory Optimization

Adapt to available memory:

```python
import psutil

memory_gb = psutil.virtual_memory().total // (1024**3)

# Low memory systems (< 4GB)
if memory_gb < 4:
    config = SearchConfig(
        file_size_limit=200_000,    # 200KB limit
        workers=2,
        context=1
    )

# Medium memory systems (4-16GB)
elif memory_gb < 16:
    config = SearchConfig(
        file_size_limit=1_000_000,  # 1MB limit
        workers=4,
        context=3
    )

# High memory systems (16GB+)
else:
    config = SearchConfig(
        file_size_limit=5_000_000,  # 5MB limit
        workers=8,
        context=5
    )
```

### Storage Optimization

Optimize for storage characteristics:

```python
import shutil

# Detect storage type (Linux)
def get_storage_type(path):
    try:
        # Check if path is on SSD
        device = shutil.disk_usage(path)
        # This is a simplified check - real implementation would
        # check /sys/block/*/queue/rotational
        return "ssd"  # or "hdd"
    except:
        return "unknown"

storage_type = get_storage_type(".")

if storage_type == "ssd":
    config = SearchConfig(
        parallel=True,
        workers=8,              # More workers for SSD
        strict_hash_check=False # Fast random access
    )
elif storage_type == "hdd":
    config = SearchConfig(
        parallel=True,
        workers=4,              # Fewer workers for HDD
        strict_hash_check=False # Minimize seeks
    )
```

---

## Use Case Optimization

### Development Environment

Optimize for frequent, interactive searches:

```python
# Development configuration
dev_config = SearchConfig(
    paths=["./src"],            # Focus on source code
    exclude=[
        "**/.venv/**", "**/.git/**", "**/__pycache__/**",
        "**/node_modules/**", "**/build/**", "**/dist/**"
    ],
    parallel=True,
    workers=4,                  # Moderate parallelism
    strict_hash_check=False,    # Fast iteration
    dir_prune_exclude=True,
    file_size_limit=1_000_000,  # 1MB limit
    context=3,                  # Useful context
    enable_docstrings=False,    # Focus on code
    enable_comments=False,
    enable_strings=True
)

# Enable short-term caching
engine = PySearch(dev_config)
engine.enable_caching(ttl=1800)  # 30 minutes
```

### CI/CD Environment

Optimize for automated, batch processing:

```python
# CI/CD configuration
ci_config = SearchConfig(
    paths=["./src", "./tests"],
    exclude=[
        "**/.venv/**", "**/.git/**", "**/__pycache__/**",
        "**/node_modules/**", "**/build/**", "**/dist/**",
        "**/htmlcov/**", "**/.pytest_cache/**"
    ],
    parallel=True,
    workers=2,                  # Limited CI resources
    strict_hash_check=True,     # Consistency important
    dir_prune_exclude=True,
    file_size_limit=2_000_000,  # 2MB limit
    context=2,                  # Minimal context for logs
    enable_docstrings=True,     # Include all content
    enable_comments=True,
    enable_strings=True
)

# Longer caching for stable builds
engine = PySearch(ci_config)
engine.enable_caching(ttl=7200)  # 2 hours
```

### Production Analysis

Optimize for comprehensive, accurate analysis:

```python
# Production configuration
prod_config = SearchConfig(
    paths=["./src", "./tests", "./docs"],
    exclude=[
        "**/.venv/**", "**/.git/**", "**/__pycache__/**",
        "**/node_modules/**", "**/build/**", "**/dist/**"
    ],
    parallel=True,
    workers=8,                  # Full parallelism
    strict_hash_check=True,     # Maximum accuracy
    dir_prune_exclude=True,
    file_size_limit=5_000_000,  # 5MB limit
    context=5,                  # Full context
    enable_docstrings=True,     # Include everything
    enable_comments=True,
    enable_strings=True
)

# Long-term caching
engine = PySearch(prod_config)
engine.enable_caching(ttl=86400)  # 24 hours
```

### Large Codebase

Optimize for very large repositories:

```python
# Large codebase configuration
large_config = SearchConfig(
    paths=["./src"],            # Limit scope
    exclude=[
        "**/.venv/**", "**/.git/**", "**/__pycache__/**",
        "**/node_modules/**", "**/build/**", "**/dist/**",
        "**/vendor/**", "**/third_party/**", "**/external/**"
    ],
    parallel=True,
    workers=12,                 # High parallelism
    strict_hash_check=False,    # Performance over precision
    dir_prune_exclude=True,     # Essential for large repos
    file_size_limit=1_000_000,  # 1MB limit
    context=2,                  # Minimal context
    enable_docstrings=False,    # Skip non-essential content
    enable_comments=False,
    enable_strings=True
)

# Aggressive caching
engine = PySearch(large_config)
engine.enable_caching(ttl=3600)  # 1 hour
```

---

## Monitoring & Profiling

### Performance Monitoring

Monitor search performance:

```python
import time
import psutil
import os

def monitor_search(engine, pattern):
    """Monitor search performance metrics."""
    process = psutil.Process(os.getpid())
    
    # Initial state
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_cpu = process.cpu_percent()
    
    # Perform search
    results = engine.search(pattern)
    
    # Final state
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    end_cpu = process.cpu_percent()
    
    # Calculate metrics
    elapsed = end_time - start_time
    memory_used = end_memory - start_memory
    
    print(f"Performance Metrics:")
    print(f"  Search time: {elapsed:.2f}s")
    print(f"  Memory used: {memory_used:.1f} MB")
    print(f"  CPU usage: {end_cpu:.1f}%")
    print(f"  Files scanned: {results.stats.files_scanned}")
    print(f"  Files/second: {results.stats.files_scanned / elapsed:.1f}")
    print(f"  Results found: {len(results.items)}")
    
    if hasattr(results.stats, 'cache_hits'):
        total_cache = results.stats.cache_hits + results.stats.cache_misses
        if total_cache > 0:
            cache_rate = results.stats.cache_hits / total_cache
            print(f"  Cache hit rate: {cache_rate:.1%}")
    
    return results

# Usage
engine = PySearch(SearchConfig(paths=["./src"]))
results = monitor_search(engine, "def main")
```

### Profiling Search Operations

Profile different search types:

```python
import cProfile
import pstats
from pysearch import PySearch, SearchConfig

def profile_search(pattern, search_type="text"):
    """Profile search performance."""
    config = SearchConfig(paths=["./src"])
    engine = PySearch(config)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    if search_type == "text":
        results = engine.search(pattern)
    elif search_type == "regex":
        results = engine.search(pattern, regex=True)
    elif search_type == "ast":
        from pysearch.types import ASTFilters
        filters = ASTFilters(func_name=".*")
        results = engine.search(pattern, use_ast=True, filters=filters)
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return results

# Profile different search types
print("=== Text Search Profile ===")
profile_search("def main", "text")

print("\n=== Regex Search Profile ===")
profile_search("def.*main", "regex")

print("\n=== AST Search Profile ===")
profile_search("def", "ast")
```

### Benchmarking

Create benchmarks for performance regression testing:

```python
import time
import statistics
from pysearch import PySearch, SearchConfig

def benchmark_search(pattern, iterations=5):
    """Benchmark search performance."""
    config = SearchConfig(paths=["./src"])
    engine = PySearch(config)
    
    times = []
    
    for i in range(iterations):
        start = time.time()
        results = engine.search(pattern)
        end = time.time()
        times.append(end - start)
        
        print(f"Iteration {i+1}: {end - start:.3f}s ({len(results.items)} results)")
    
    # Statistics
    mean_time = statistics.mean(times)
    median_time = statistics.median(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0
    
    print(f"\nBenchmark Results:")
    print(f"  Mean time: {mean_time:.3f}s")
    print(f"  Median time: {median_time:.3f}s")
    print(f"  Std deviation: {stdev_time:.3f}s")
    print(f"  Min time: {min(times):.3f}s")
    print(f"  Max time: {max(times):.3f}s")
    
    return {
        'mean': mean_time,
        'median': median_time,
        'stdev': stdev_time,
        'min': min(times),
        'max': max(times)
    }

# Run benchmark
benchmark_results = benchmark_search("def main")
```

---

## Troubleshooting Performance

### Common Performance Issues

#### Slow Initial Search

**Problem**: First search is much slower than subsequent searches

**Cause**: Index building overhead

**Solutions**:

1. **Pre-build index**:

   ```python
   engine = PySearch(config)
   engine.indexer.build_index()  # Pre-build
   ```

2. **Enable persistent caching**:

   ```python
   engine.enable_caching(ttl=86400)  # 24-hour cache
   ```

#### High Memory Usage

**Problem**: Memory usage grows during search

**Cause**: Large files or too many results

**Solutions**:

1. **Limit file sizes**:

   ```python
   config = SearchConfig(file_size_limit=1_000_000)  # 1MB
   ```

2. **Reduce context**:

   ```python
   config = SearchConfig(context=1)  # Minimal context
   ```

3. **Process results in batches**:

   ```python
   for i in range(0, len(results.items), 100):
       batch = results.items[i:i+100]
       process_batch(batch)
   ```

#### Poor Parallel Performance

**Problem**: Parallel search isn't faster than sequential

**Cause**: I/O bottleneck or overhead

**Solutions**:

1. **Adjust worker count**:

   ```python
   # Try different worker counts
   for workers in [1, 2, 4, 8]:
       config = SearchConfig(workers=workers)
       # Benchmark each configuration
   ```

2. **Check storage type**:

   ```python
   # HDDs may not benefit from high parallelism
   config = SearchConfig(workers=2)  # For HDD
   ```

#### Cache Inefficiency

**Problem**: Low cache hit rates

**Cause**: Files changing frequently or cache TTL too short

**Solutions**:

1. **Increase cache TTL**:

   ```python
   engine.enable_caching(ttl=7200)  # 2 hours
   ```

2. **Check file modification patterns**:

   ```python
   # Monitor which files are changing
   results = engine.search("pattern")
   print(f"Cache hits: {results.stats.cache_hits}")
   print(f"Cache misses: {results.stats.cache_misses}")
   ```

### Performance Regression Testing

Create automated performance tests:

```python
import pytest
import time
from pysearch import PySearch, SearchConfig

class TestPerformance:
    def setup_method(self):
        self.config = SearchConfig(paths=["./src"])
        self.engine = PySearch(self.config)
    
    def test_search_performance(self):
        """Test that search completes within time limit."""
        start = time.time()
        results = self.engine.search("def main")
        elapsed = time.time() - start
        
        # Assert performance requirements
        assert elapsed < 5.0, f"Search took {elapsed:.2f}s, expected < 5.0s"
        assert len(results.items) > 0, "No results found"
    
    def test_memory_usage(self):
        """Test memory usage stays within limits."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        results = self.engine.search("class")
        
        final_memory = process.memory_info().rss
        memory_used = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        assert memory_used < 500, f"Used {memory_used:.1f}MB, expected < 500MB"

# Run with: pytest test_performance.py -v
```

---

## Best Practices Summary

### Configuration Best Practices

1. **Start with defaults** and optimize incrementally
2. **Profile before optimizing** to identify bottlenecks
3. **Match configuration to use case** (dev vs. CI vs. prod)
4. **Monitor performance metrics** regularly
5. **Test configuration changes** with benchmarks

### Hardware Best Practices

1. **Use SSDs** for better I/O performance
2. **Ensure adequate RAM** for your codebase size
3. **Utilize multiple CPU cores** with parallel processing
4. **Consider network latency** for remote storage

### Usage Best Practices

1. **Limit search scope** to relevant directories
2. **Use appropriate search types** for your needs
3. **Enable caching** for repeated searches
4. **Exclude unnecessary files** early
5. **Monitor resource usage** in production

### Development Best Practices

1. **Create performance tests** for regression detection
2. **Profile different configurations** for your codebase
3. **Document optimal settings** for your team
4. **Monitor performance trends** over time
5. **Share configurations** across team members
