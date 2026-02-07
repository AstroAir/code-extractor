# Search History API

Comprehensive search history tracking, analytics, and query management system.

## SearchHistory

::: pysearch.history.SearchHistory
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

## SearchHistoryEntry

::: pysearch.history.SearchHistoryEntry
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

## SearchCategory

::: pysearch.history.SearchCategory
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## Usage Examples

### Basic History Tracking

```python
from pysearch import SearchConfig, Query
from pysearch.core.history import SearchHistory

# Initialize history
config = SearchConfig()
history = SearchHistory(config)

# Record a search
query = Query(pattern="def main", use_regex=True)
results = engine.run(query)
history.record_search(query, results)

# Get recent searches
recent = history.get_recent_searches(limit=10)
for entry in recent:
    print(f"{entry.query_pattern}: {entry.items_count} results")
```

### Search Analytics

```python
# Get comprehensive analytics
analytics = history.get_search_analytics()
print(f"Total searches: {analytics.total_searches}")
print(f"Average results: {analytics.avg_results_per_search:.1f}")
print(f"Most common patterns: {analytics.top_patterns}")

# Get performance metrics
perf_stats = history.get_performance_stats()
print(f"Average search time: {perf_stats.avg_search_time_ms:.1f}ms")
print(f"Slowest searches: {perf_stats.slowest_searches}")
```

### Query Suggestions

```python
# Get suggestions based on history
suggestions = history.get_query_suggestions("def")
print("Suggested queries:", suggestions)

# Get similar searches
similar = history.find_similar_searches("async def handler")
for entry in similar:
    print(f"Similar: {entry.query_pattern} (score: {entry.similarity:.2f})")
```

### Search Categories

```python
from pysearch.history import SearchCategory

# Categorize searches automatically
history.enable_auto_categorization()

# Get searches by category
function_searches = history.get_searches_by_category(SearchCategory.FUNCTION)
class_searches = history.get_searches_by_category(SearchCategory.CLASS)

# Manual categorization
history.categorize_search(search_id, SearchCategory.REFACTORING)
```

### Bookmarks and Favorites

```python
# Bookmark useful searches
bookmark_id = history.bookmark_search(
    pattern="def.*handler",
    name="Handler Functions",
    description="Find all handler function definitions"
)

# Get bookmarked searches
bookmarks = history.get_bookmarks()
for bookmark in bookmarks:
    print(f"{bookmark.name}: {bookmark.pattern}")

# Execute bookmarked search
results = history.execute_bookmark(bookmark_id)
```

### Session Management

```python
# Start a new search session
session_id = history.start_session("refactoring-session")

# All searches in this session will be grouped
query1 = Query(pattern="old_function_name")
history.record_search(query1, results1, session_id=session_id)

query2 = Query(pattern="new_function_name")
history.record_search(query2, results2, session_id=session_id)

# Get session summary
session = history.get_session(session_id)
print(f"Session: {session.name}")
print(f"Searches: {len(session.searches)}")
print(f"Duration: {session.duration_minutes:.1f} minutes")
```

## Advanced Features

### Search Pattern Analysis

```python
# Analyze search patterns
patterns = history.analyze_search_patterns()
print(f"Most common regex patterns: {patterns.regex_patterns}")
print(f"Most searched file types: {patterns.file_types}")
print(f"Peak search times: {patterns.peak_hours}")
```

### Export and Import

```python
# Export history
history_data = history.export_history(format="json")
with open("search_history.json", "w") as f:
    f.write(history_data)

# Import history
with open("search_history.json", "r") as f:
    history_data = f.read()
history.import_history(history_data)
```

### Performance Optimization

```python
# Configure history limits
history = SearchHistory(
    config,
    max_entries=10000,
    cleanup_interval_days=30,
    enable_analytics=True
)

# Periodic cleanup
history.cleanup_old_entries(days=90)
```

## Integration

### CLI Integration

The history system integrates with the CLI:

```bash
# Show recent searches
pysearch history --recent 10

# Search history
pysearch history --search "def.*handler"

# Show analytics
pysearch history --analytics

# Execute bookmarked search
pysearch history --bookmark "handler-functions"
```

### API Integration

```python
# Automatic history tracking
engine = PySearch(config, enable_history=True)

# All searches are automatically recorded
results = engine.search("def main")  # Recorded in history

# Access history
history = engine.history
recent_searches = history.get_recent_searches()
```

## Related

- [PySearch API](pysearch.md) - Main search engine with history integration
- [Types](types.md) - Query and result types
- [Configuration](config.md) - History configuration options
