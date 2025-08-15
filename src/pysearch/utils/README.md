# Utils Module

The utils module contains utility functions and helper classes used throughout the application.

## Responsibilities

- **Output Formatting**: Result formatting and syntax highlighting
- **Metadata Processing**: File metadata filtering and processing
- **File Monitoring**: File system watching and change detection
- **Error Handling**: Comprehensive error handling and reporting
- **Performance Monitoring**: Performance metrics and monitoring
- **General Utilities**: Common utility functions

## Key Files

- `formatter.py` - Output formatting and syntax highlighting
- `metadata_filters.py` - Metadata filtering and processing
- `file_watcher.py` - File system monitoring and change detection
- `error_handling.py` - Basic error handling and reporting
- `advanced_error_handling.py` - Advanced error handling with recovery strategies
- `logging_config.py` - Logging configuration and management
- `performance_monitoring.py` - Performance metrics and monitoring
- `utils.py` - General utility functions

## Utility Categories

1. **Formatting**: Rich console output, JSON formatting, syntax highlighting
2. **Filtering**: Metadata-based filtering, pattern matching
3. **Monitoring**: File system changes, performance metrics
4. **Error Management**: Error collection, reporting, recovery strategies, and circuit breakers

## Usage

```python
from pysearch.utils import format_result, create_metadata_filters
from pysearch.utils.error_handling import ErrorCollector

# Format search results
formatted = format_result(results, OutputFormat.JSON)

# Create metadata filters
filters = create_metadata_filters(max_size="1MB", languages=["python"])
```
