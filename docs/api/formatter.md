# Formatter API

The formatter module handles output rendering in different formats with syntax highlighting support.

## Main Functions

### format_result

::: pysearch.formatter.format_result
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### to_json_bytes

::: pysearch.formatter.to_json_bytes
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### format_text

::: pysearch.formatter.format_text
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### render_highlight_console

::: pysearch.formatter.render_highlight_console
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Usage Examples

### Basic Formatting

```python
from pysearch.formatter import format_result
from pysearch.types import OutputFormat

# Format as plain text
text_output = format_result(search_results, OutputFormat.TEXT)
print(text_output)

# Format as JSON
json_output = format_result(search_results, OutputFormat.JSON)
print(json_output.decode('utf-8'))

# Format with syntax highlighting
highlighted = format_result(search_results, OutputFormat.HIGHLIGHT)
print(highlighted)
```

### Custom Text Formatting

```python
from pysearch.formatter import format_text

# Format with highlighting enabled
formatted = format_text(search_results, highlight=True)
print(formatted)

# Format without highlighting (faster)
plain = format_text(search_results, highlight=False)
print(plain)
```

### JSON Output

```python
from pysearch.formatter import to_json_bytes
import json

# Get JSON bytes
json_bytes = to_json_bytes(search_results)

# Parse as JSON object
data = json.loads(json_bytes)
print(f"Found {len(data['items'])} results")

# Pretty print JSON
print(json.dumps(data, indent=2))
```

### Console Highlighting

```python
from pysearch.formatter import render_highlight_console
from rich.console import Console

console = Console()

# Render with rich highlighting
render_highlight_console(search_results, console)
```

## Output Formats

### TEXT Format

Plain text output with line numbers and file paths:

```
example.py:10-12
    10 | def main():
    11 |     print("Hello, world!")
    12 |     return 0
```

### JSON Format

Structured JSON output for programmatic processing:

```json
{
  "items": [
    {
      "file": "example.py",
      "start_line": 10,
      "end_line": 12,
      "lines": [
        "def main():",
        "    print(\"Hello, world!\")",
        "    return 0"
      ],
      "match_spans": [[0, [0, 8]]]
    }
  ],
  "stats": {
    "files_scanned": 100,
    "files_matched": 5,
    "items": 10,
    "elapsed_ms": 45.2,
    "indexed_files": 95
  }
}
```

### HIGHLIGHT Format

Rich console output with syntax highlighting and match emphasis:

- Syntax highlighting based on file type
- Match spans highlighted in different colors
- Line numbers and file paths styled
- Progress indicators and statistics

## Customization

### Custom Formatters

```python
def custom_formatter(results):
    """Custom formatter for specific output needs."""
    output = []
    
    for item in results.items:
        # Custom header format
        header = f"📁 {item.file} (lines {item.start_line}-{item.end_line})"
        output.append(header)
        
        # Custom line format
        for i, line in enumerate(item.lines):
            line_num = item.start_line + i
            output.append(f"  {line_num:4d} │ {line}")
        
        output.append("")  # Blank line between results
    
    return "\n".join(output)

# Use custom formatter
custom_output = custom_formatter(search_results)
print(custom_output)
```

### Highlighting Customization

```python
from rich.console import Console
from rich.theme import Theme

# Custom theme for highlighting
custom_theme = Theme({
    "match": "bold red",
    "line_number": "dim blue",
    "file_path": "bold green",
    "context": "dim white"
})

console = Console(theme=custom_theme)
render_highlight_console(search_results, console)
```

## Performance Considerations

### Format Performance
- **TEXT**: Fastest format, minimal processing
- **JSON**: Fast serialization with orjson
- **HIGHLIGHT**: Slower due to syntax highlighting

### Memory Usage
- Large result sets may require streaming for JSON output
- Highlighting keeps syntax trees in memory
- Text format has minimal memory overhead

### Streaming Output

```python
def stream_results(results, format_type):
    """Stream results for large datasets."""
    if format_type == OutputFormat.TEXT:
        for item in results.items:
            yield format_text_item(item)
    elif format_type == OutputFormat.JSON:
        yield '{"items": ['
        for i, item in enumerate(results.items):
            if i > 0:
                yield ','
            yield json.dumps(format_json_item(item))
        yield '], "stats": ' + json.dumps(results.stats) + '}'
```

## Integration

### CLI Integration

The formatter integrates with the CLI to provide consistent output:

```bash
# Plain text output
pysearch find --pattern "def main" --format text

# JSON output for scripting
pysearch find --pattern "def main" --format json | jq '.items[].file'

# Highlighted console output
pysearch find --pattern "def main" --format highlight
```

### API Integration

```python
from pysearch import PySearch, OutputFormat
from pysearch.utils.formatter import format_result

engine = PySearch(config)
results = engine.search("def main")

# Format results
formatted = format_result(results, OutputFormat.HIGHLIGHT)
print(formatted)
```

## Related

- [Types](types.md) - Result and output types
- [CLI Reference](../guide/cli-reference.md) - Command-line formatting options
- [Utils](utils.md) - Utility functions for formatting
