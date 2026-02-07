# Utils Module

[根目录](../../../CLAUDE.md) > [src](../../) > [pysearch](../) > **utils**

---

## Change Log (Changelog)

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive Utils module documentation

---

## Module Responsibility

The **Utils** module contains utility functions and helper classes used throughout the application:

1. **Output Formatting**: Result formatting and syntax highlighting
2. **Metadata Processing**: File metadata filtering and processing
3. **File Monitoring**: File system watching and change detection
4. **Error Handling**: Comprehensive error handling and reporting
5. **Performance Monitoring**: Performance metrics and monitoring
6. **General Utilities**: Common utility functions

---

## Key Files

| File | Purpose | Description |
|------|---------|-------------|
| `formatter.py` | Output Formatting | Result formatting and syntax highlighting |
| `metadata_filters.py` | Metadata Filtering | File metadata filtering and processing |
| `file_watcher.py` | File Watching | File system monitoring and change detection |
| `error_handling.py` | Error Handling | Basic error handling and reporting |
| `advanced_error_handling.py` | Advanced Errors | Advanced error handling with recovery strategies |
| `logging_config.py` | Logging | Logging configuration and management |
| `performance_monitoring.py` | Performance | Performance metrics and monitoring |
| `utils.py` | General Utilities | Common utility functions |

---

## Output Formatting (formatter.py)

### Overview
Provides result formatting and syntax highlighting for multiple output formats.

### Key Functions
```python
def format_result(
    result: SearchResult,
    output_format: OutputFormat,
    config: SearchConfig
) -> str

def format_highlight(
    text: str,
    language: Language = Language.PYTHON
) -> str

def format_json(result: SearchResult) -> str
```

### Supported Formats
- **TEXT**: Plain text output
- **JSON**: Structured JSON output
- **HIGHLIGHT**: Syntax-highlighted terminal output

---

## Metadata Filters (metadata_filters.py)

### Overview
Provides file metadata filtering based on size, date, language, and author.

### Key Functions
```python
def create_metadata_filters(
    min_size: str | None = None,
    max_size: str | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
    languages: set[Language] | None = None,
    author_pattern: str | None = None
) -> MetadataFilters

def apply_metadata_filters(
    metadata: FileMetadata,
    filters: MetadataFilters
) -> bool
```

### Filter Types
- **Size Filters**: Minimum/maximum file size
- **Date Filters**: Minimum/maximum modification date
- **Language Filters**: Programming language filtering
- **Author Filters**: Author/creator pattern matching

---

## File Watching (file_watcher.py)

### Overview
Provides real-time file system monitoring with change detection.

### Key Classes
```python
class FileWatcher:
    def __init__(self, path: Path, **kwargs)
    def start(self) -> None
    def stop(self) -> None
    def add_handler(self, handler: Callable) -> None

class FileEvent:
    event_type: FileEventType
    path: Path
    timestamp: float
```

### Features
- **Cross-platform**: Works on Windows, Linux, macOS
- **Debouncing**: Configurable debounce delay
- **Batch Processing**: Efficient batch change processing
- **Recursive**: Recursive directory watching

---

## Error Handling (error_handling.py & advanced_error_handling.py)

### Overview
Comprehensive error handling with collection, reporting, and recovery strategies.

### Key Classes
```python
class ErrorCollector:
    def add_error(self, error: Exception) -> None
    def get_errors(self) -> list[ErrorEntry]
    def get_summary(self) -> dict[str, Any]
    def clear(self) -> None
    def has_critical_errors(self) -> bool

class ErrorEntry:
    error: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    context: dict[str, Any]
    timestamp: float
```

### Error Categories
- **FILE_ACCESS**: File read/write errors
- **PARSING**: Code parsing errors
- **ENCODING**: Text encoding errors
- **PERMISSION**: Permission denied errors
- **NETWORK**: Network-related errors
- **UNKNOWN**: Uncategorized errors

### Recovery Strategies
- **RETRY**: Retry the operation
- **SKIP**: Skip and continue
- **FALLBACK**: Use fallback method
- **ABORT**: Abort the operation

---

## Logging Configuration (logging_config.py)

### Overview
Provides logging configuration and management for PySearch.

### Key Functions
```python
def configure_logging(
    level: LogLevel = LogLevel.INFO,
    format_type: LogFormat = LogFormat.SIMPLE,
    log_file: Path | None = None,
    enable_file: bool = False
) -> SearchLogger

def enable_debug_logging() -> None
def disable_logging() -> None
def get_logger(name: str = "pysearch") -> SearchLogger
```

### Log Levels
- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical errors

### Log Formats
- **SIMPLE**: Simple text format
- **DETAILED**: Detailed format with timestamps
- **JSON**: Structured JSON format

---

## Performance Monitoring (performance_monitoring.py)

### Overview
Provides performance metrics and monitoring for search operations.

### Key Classes
```python
class PerformanceMonitor:
    def start_operation(self, name: str) -> None
    def end_operation(self, name: str) -> None
    def get_metrics(self) -> dict[str, Any]
    def reset(self) -> None

class PerformanceMetrics:
    operation_name: str
    duration_ms: float
    memory_used_mb: float
    cpu_percent: float
```

### Metrics Tracked
- **Duration**: Operation execution time
- **Memory**: Memory usage
- **CPU**: CPU utilization
- **I/O**: Disk I/O operations

---

## General Utilities (utils.py)

### Overview
Common utility functions used throughout the application.

### Key Functions
```python
def read_text_safely(path: Path, encoding: str = "utf-8") -> str | None
def create_file_metadata(path: Path, content: str) -> FileMetadata | None
def matches_patterns(path: Path, patterns: list[str]) -> bool
def get_file_author(path: Path) -> str | None
```

---

## Dependencies

### Internal Dependencies
- `pysearch.core`: Types and configuration
- `pysearch.analysis`: Language detection

### External Dependencies
- `rich`: Terminal output formatting
- `pygments`: Syntax highlighting
- `watchdog`: File system monitoring
- `orjson`: Fast JSON serialization

---

## Testing

### Unit Tests
Located in `tests/unit/core/`:
- `test_formatter.py` - Formatter tests
- `test_metadata_filters.py` - Metadata filter tests
- `test_file_watcher_*.py` - File watcher tests
- `test_utils_*.py` - Utility function tests

---

## Common Usage Patterns

### Output Formatting
```python
from pysearch.utils import format_result
from pysearch.types import OutputFormat

# Format as JSON
json_output = format_result(results, OutputFormat.JSON, config)

# Format with highlighting
highlighted = format_result(results, OutputFormat.HIGHLIGHT, config)
```

### Metadata Filtering
```python
from pysearch.utils import create_metadata_filters, apply_metadata_filters

# Create filters
filters = create_metadata_filters(
    min_size="1KB",
    max_size="1MB",
    languages={Language.PYTHON}
)

# Apply filters
if apply_metadata_filters(metadata, filters):
    # Process file
    pass
```

### File Watching
```python
from pysearch.utils import FileWatcher, FileEventType

def handler(events):
    for event in events:
        if event.event_type == FileEventType.CREATED:
            print(f"Created: {event.path}")

watcher = FileWatcher("./src", change_handler=handler)
watcher.start()
```

### Error Handling
```python
from pysearch.utils.error_handling import ErrorCollector, ErrorCategory

collector = ErrorCollector()

try:
    # Operation that might fail
    pass
except Exception as e:
    collector.add_error(e)

# Get summary
summary = collector.get_summary()
```

### Logging
```python
from pysearch.utils.logging_config import configure_logging, get_logger

logger = configure_logging(level="DEBUG", log_file="app.log")
logger.info("Search started")
logger.debug(f"Processing file: {file_path}")
```

---

## Related Files
- `README.md` - Module overview
- `docs/architecture.md` - Architecture details
- `docs/api/utils.md` - Utils API reference
