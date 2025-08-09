# Error Handling API

Comprehensive error handling and reporting system for robust search operations.

## Error Classes

### SearchError

::: pysearch.error_handling.SearchError
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source
      group_by_category: true
      show_bases: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### FileAccessError

::: pysearch.error_handling.FileAccessError
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source
      group_by_category: true
      show_bases: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### ConfigurationError

::: pysearch.error_handling.ConfigurationError
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source
      group_by_category: true
      show_bases: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Error Information

### ErrorInfo

::: pysearch.error_handling.ErrorInfo
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source
      group_by_category: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### ErrorSeverity

::: pysearch.error_handling.ErrorSeverity
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### ErrorCategory

::: pysearch.error_handling.ErrorCategory
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Error Collection

### ErrorCollector

::: pysearch.error_handling.ErrorCollector
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source
      group_by_category: true
      show_bases: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Utility Functions

### handle_file_error

::: pysearch.error_handling.handle_file_error
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### create_error_report

::: pysearch.error_handling.create_error_report
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Usage Examples

### Basic Error Handling

```python
from pysearch.error_handling import PySearchError, FileAccessError
from pysearch.api import PySearch

try:
    engine = PySearch(config)
    results = engine.search("pattern")
except FileAccessError as e:
    print(f"File access error: {e}")
    print(f"Affected file: {e.file_path}")
except PySearchError as e:
    print(f"Search error: {e}")
```

### Error Collection

```python
from pysearch.error_handling import ErrorCollector

# Create error collector
collector = ErrorCollector()

# Collect errors during search
try:
    # Perform search operations
    pass
except Exception as e:
    collector.add_error(e, file_path="example.py")

# Generate error report
report = collector.generate_report()
print(f"Total errors: {len(report.errors)}")
print(f"Critical errors: {len(report.critical_errors)}")
```

### File Error Handling

```python
from pysearch.error_handling import handle_file_error
from pathlib import Path

def safe_file_operation(file_path):
    try:
        # Attempt file operation
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        error_info = handle_file_error(e, Path(file_path))
        print(f"Error category: {error_info.category}")
        print(f"Severity: {error_info.severity}")
        return None
```

### Custom Error Handling

```python
from pysearch.error_handling import ErrorInfo, ErrorCategory, ErrorSeverity

def custom_error_handler(exception, context=None):
    """Custom error handler with context information."""
    
    error_info = ErrorInfo(
        category=ErrorCategory.UNKNOWN,
        severity=ErrorSeverity.MEDIUM,
        message=str(exception),
        file_path=context.get('file_path') if context else None,
        line_number=context.get('line_number') if context else None,
        exception_type=type(exception).__name__,
        traceback_str=traceback.format_exc()
    )
    
    # Log error based on severity
    if error_info.severity == ErrorSeverity.CRITICAL:
        logger.critical(f"Critical error: {error_info.message}")
    elif error_info.severity == ErrorSeverity.HIGH:
        logger.error(f"High severity error: {error_info.message}")
    else:
        logger.warning(f"Error: {error_info.message}")
    
    return error_info
```

## Error Categories

### File Access Errors
- Permission denied
- File not found
- Directory access issues
- Symlink resolution problems

```python
# Handle file access errors
try:
    content = read_file(path)
except PermissionError as e:
    error_info = handle_file_error(e, path)
    # error_info.category == ErrorCategory.PERMISSION
```

### Encoding Errors
- Invalid file encoding
- Unicode decode errors
- Binary file detection

```python
# Handle encoding errors
try:
    content = path.read_text(encoding='utf-8')
except UnicodeDecodeError as e:
    error_info = handle_file_error(e, path)
    # error_info.category == ErrorCategory.ENCODING
```

### Parsing Errors
- Invalid regex patterns
- AST parsing failures
- Malformed configuration

```python
# Handle parsing errors
try:
    tree = ast.parse(code)
except SyntaxError as e:
    error_info = ErrorInfo(
        category=ErrorCategory.PARSING,
        severity=ErrorSeverity.MEDIUM,
        message=f"Syntax error: {e}",
        line_number=e.lineno
    )
```

## Error Recovery

### Graceful Degradation

```python
def search_with_fallback(engine, pattern):
    """Search with fallback strategies."""
    
    try:
        # Try full search
        return engine.search(pattern)
    except FileAccessError:
        # Fallback to accessible files only
        return engine.search_accessible_only(pattern)
    except ConfigurationError:
        # Fallback to default configuration
        return engine.search_with_defaults(pattern)
```

### Retry Logic

```python
import time
from pysearch.error_handling import PySearchError

def search_with_retry(engine, pattern, max_retries=3):
    """Search with exponential backoff retry."""
    
    for attempt in range(max_retries):
        try:
            return engine.search(pattern)
        except PySearchError as e:
            if attempt == max_retries - 1:
                raise
            
            wait_time = 2 ** attempt
            print(f"Search failed, retrying in {wait_time}s...")
            time.sleep(wait_time)
```

## Error Reporting

### Detailed Error Reports

```python
from pysearch.error_handling import create_error_report

def generate_search_report(errors):
    """Generate comprehensive error report."""
    
    report = create_error_report(errors)
    
    print("=== PySearch Error Report ===")
    print(f"Total errors: {len(report.errors)}")
    print(f"Critical: {len(report.critical_errors)}")
    print(f"High: {len(report.high_severity_errors)}")
    print(f"Medium: {len(report.medium_severity_errors)}")
    print(f"Low: {len(report.low_severity_errors)}")
    
    # Group by category
    by_category = report.group_by_category()
    for category, category_errors in by_category.items():
        print(f"\n{category}: {len(category_errors)} errors")
        for error in category_errors[:3]:  # Show first 3
            print(f"  - {error.message}")
```

### Integration with Logging

```python
import logging
from pysearch.error_handling import ErrorCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create error collector with logging
collector = ErrorCollector(logger=logger)

# Errors are automatically logged
collector.add_error(Exception("Test error"))
```

## Best Practices

### Error Handling Strategy

1. **Catch specific exceptions** rather than generic Exception
2. **Provide context** in error messages
3. **Use appropriate severity levels**
4. **Log errors for debugging**
5. **Implement graceful degradation**

### Performance Considerations

```python
# Efficient error handling
def efficient_search(engine, pattern):
    """Efficient search with minimal error overhead."""
    
    collector = ErrorCollector(max_errors=100)  # Limit error collection
    
    try:
        return engine.search(pattern, error_collector=collector)
    finally:
        # Only generate report if there are errors
        if collector.has_errors():
            report = collector.generate_report()
            logger.warning(f"Search completed with {len(report.errors)} errors")
```

## Related

- [PySearch API](pysearch.md) - Main API with error handling
- [Utils](utils.md) - Utility functions with error handling
- [Configuration](config.md) - Configuration validation and errors
