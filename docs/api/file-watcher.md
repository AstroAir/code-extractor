# File Watcher API

Real-time file system monitoring for automatic index updates and change detection.

## WatchManager

::: pysearch.file_watcher.WatchManager
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
      members_order: source
      group_by_category: true
      show_bases: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## FileEvent

::: pysearch.file_watcher.FileEvent
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
      members_order: source
      group_by_category: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Usage Examples

### Basic File Watching

```python
from pysearch import SearchConfig
from pysearch.utils.file_watcher import WatchManager

# Create configuration
config = SearchConfig(paths=["./src"], include=["**/*.py"])

# Initialize watch manager
watcher = WatchManager(config)

# Start watching for changes
watcher.start()

# Handle file events
def on_file_changed(event):
    print(f"File {event.path} was {event.event_type}")

watcher.add_handler(on_file_changed)
```

### Integration with Search Engine

```python
from pysearch import PySearch
from pysearch.utils.file_watcher import WatchManager

# Create search engine
engine = PySearch(config)

# Create watcher with automatic index updates
watcher = WatchManager(config)

def update_index(event):
    """Update search index when files change."""
    if event.event_type in ['created', 'modified']:
        engine.indexer.invalidate_file(event.path)
    elif event.event_type == 'deleted':
        engine.indexer.remove_file(event.path)

watcher.add_handler(update_index)
watcher.start()
```

### Batch Processing

```python
# Enable batch processing for performance
watcher = WatchManager(config, batch_size=10, batch_timeout=1.0)

def process_batch(events):
    """Process multiple file events together."""
    modified_files = [e.path for e in events if e.event_type == 'modified']
    if modified_files:
        engine.indexer.update_files(modified_files)

watcher.add_batch_handler(process_batch)
```

## Event Types

The file watcher recognizes several event types:

- **created**: New file or directory created
- **modified**: File content or metadata changed
- **deleted**: File or directory removed
- **moved**: File or directory moved/renamed

## Performance Considerations

### Filtering Events

```python
# Filter events by file type
def should_process(event):
    return event.path.suffix in ['.py', '.js', '.ts']

watcher.add_filter(should_process)
```

### Debouncing

```python
# Debounce rapid file changes
watcher = WatchManager(config, debounce_delay=0.5)
```

### Resource Management

```python
# Properly stop watching
try:
    watcher.start()
    # ... do work ...
finally:
    watcher.stop()
```

## Related

- [Indexer](indexer.md) - File indexing and caching
- [Configuration](config.md) - Watch configuration options
- [PySearch API](pysearch.md) - Main search engine integration
