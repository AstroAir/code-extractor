# Utils Module

[根目录](../../../CLAUDE.md) > **utils**

---

## Change Log (Changelog)

### 2026-02-08 - Module Documentation Update
- Added performance monitoring documentation
- Enhanced error handling documentation
- Updated file watching documentation
- Synchronized with current project structure

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive utils module documentation

---

## Module Responsibility

The **Utils** module provides utility functions and helpers for error handling, output formatting, file watching, logging, metadata filtering, and performance monitoring.

### Key Responsibilities
1. **Error Handling**: Comprehensive error collection and reporting
2. **Output Formatting**: Multiple output formats with syntax highlighting
3. **File Watching**: Real-time file system monitoring
4. **Logging Configuration**: Structured logging with multiple formats
5. **Metadata Filtering**: File metadata-based filtering
6. **Performance Monitoring**: Profiling and metrics collection
7. **Helper Functions**: Common utilities

---

## Entry and Startup

### Main Components
- **`error_handling.py`** - Error handling system
  - `ErrorCollector` - Collect and aggregate errors
  - `SearchError` hierarchy - Custom exception types

- **`formatter.py`** - Output formatting
  - `format_result()` - Format search results
  - `render_highlight_console()` - Syntax-highlighted output

- **`file_watcher.py`** - File watching
  - `FileWatcher` - Monitor file changes

- **`logging_config.py`** - Logging configuration
  - `configure_logging()` - Setup logging
  - `get_logger()` - Get logger instance

- **`metadata_filters.py`** - Metadata filtering
  - `create_metadata_filters()` - Create filters

- **`performance_monitoring.py`** - Performance monitoring
  - `PerformanceMonitor` - Track performance

- **`helpers.py`** - Helper functions
  - Common utility functions

---

## Public API

### Error Handling

```python
from pysearch.utils.error_handling import ErrorCollector, SearchError

collector = ErrorCollector()

try:
    # Some operation
    pass
except SearchError as e:
    collector.add_error(e)

# Get error summary
summary = collector.get_summary()
print(f"Total errors: {summary['total_errors']}")
```

### Output Formatting

```python
from pysearch.utils.formatter import format_result, render_highlight_console
from pysearch.core.types import OutputFormat

# Format as text
text_output = format_result(results, OutputFormat.TEXT)

# Format as JSON
json_output = format_result(results, OutputFormat.JSON)

# Render with highlighting
render_highlight_console(results)
```

### File Watching

```python
from pysearch.utils.file_watcher import FileWatcher

def callback(path, event_type):
    print(f"File {path} {event_type}")

watcher = FileWatcher(callback)
watcher.watch_directory("./src", recursive=True)
watcher.start()
```

### Logging

```python
from pysearch.utils.logging_config import configure_logging, get_logger, LogLevel, LogFormat

# Configure logging
configure_logging(
    level=LogLevel.DEBUG,
    format_type=LogFormat.DETAILED,
    log_file="pysearch.log"
)

# Get logger
logger = get_logger(__name__)
logger.info("Search completed")
```

### Metadata Filtering

```python
from pysearch.utils.metadata_filters import create_metadata_filters

filters = create_metadata_filters(
    min_size="1KB",
    max_size="10MB",
    modified_after="1d",
    author="John.*"
)
```

### Performance Monitoring

```python
from pysearch.utils.performance_monitoring import PerformanceMonitor

monitor = PerformanceMonitor()

with monitor.track_operation("search"):
    # Perform search
    results = engine.search("pattern")

metrics = monitor.get_metrics()
print(f"Search took {metrics['duration_ms']}ms")
```

---

## Key Dependencies and Configuration

### Internal Dependencies
- `pysearch.core.types` - Result types for formatting

### External Dependencies
- `rich>=13.7.1` - Terminal formatting
- `pygments>=2.0` - Syntax highlighting
- `watchdog>=4.0.0` - File watching (optional)
- `psutil>=5.9.0` - System monitoring (optional)

---

## Data Models

### Error Types
- `SearchError` - Base search exception
- `FileAccessError` - File access errors
- `PermissionError` - Permission errors
- `EncodingError` - Encoding errors
- `ParsingError` - Parsing errors
- `ConfigurationError` - Configuration errors

### Logging Types
- `LogLevel` - DEBUG, INFO, WARNING, ERROR
- `LogFormat` - SIMPLE, DETAILED, JSON, STRUCTURED

---

## Testing

### Test Directory
- `tests/unit/utils/` - Utils module tests
  - `test_error_handling.py` - Error handling tests
  - `test_formatter.py` - Formatter tests
  - `test_file_watcher.py` - File watcher tests
  - `test_logging_config.py` - Logging tests
  - `test_metadata_filters.py` - Metadata filter tests
  - `test_performance_monitoring.py` - Performance tests
  - `test_helpers.py` - Helper tests

### Running Tests
```bash
pytest tests/unit/utils/ -v
```

---

## Common Issues and Solutions

### Issue 1: File watcher not detecting changes
**Symptoms**: File changes not detected
**Solution**: Check file system and permissions:
```python
watcher = FileWatcher(callback, debounce_delay=1.0)
```

### Issue 2: Logging too verbose
**Symptoms**: Too much log output
**Solution**: Adjust log level:
```python
configure_logging(level=LogLevel.WARNING)
```

### Issue 3: Performance monitoring overhead
**Symptoms**: Monitoring slowing down searches
**Solution**: Disable monitoring or sample:
```python
monitor = PerformanceMonitor(sample_rate=0.1)  # 10% sampling
```

---

## Related Files

### Utils Module Files
- `src/pysearch/utils/__init__.py`
- `src/pysearch/utils/error_handling.py` - Error handling
- `src/pysearch/utils/formatter.py` - Output formatting
- `src/pysearch/utils/file_watcher.py` - File watching
- `src/pysearch/utils/logging_config.py` - Logging configuration
- `src/pysearch/utils/metadata_filters.py` - Metadata filters
- `src/pysearch/utils/performance_monitoring.py` - Performance monitoring
- `src/pysearch/utils/helpers.py` - Helper functions

---

## Module Structure

```
utils/
├── __init__.py
├── error_handling.py          # Error handling system
├── formatter.py               # Output formatting
├── file_watcher.py            # File system monitoring
├── logging_config.py          # Logging configuration
├── metadata_filters.py        # Metadata-based filtering
├── performance_monitoring.py  # Performance tracking
└── helpers.py                 # Common utilities
```
