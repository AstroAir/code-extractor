# Utils API

Utility functions and helpers used throughout the PySearch system.

## File Operations

### read_text_safely

::: pysearch.utils.helpers.read_text_safely
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### file_sha1

::: pysearch.utils.helpers.file_sha1
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### create_file_metadata

::: pysearch.utils.helpers.create_file_metadata
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Text Processing

### split_lines_keepends

::: pysearch.utils.helpers.split_lines_keepends
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### extract_context

::: pysearch.utils.helpers.extract_context
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### highlight_spans

::: pysearch.utils.helpers.highlight_spans
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## AST Utilities

### iter_python_ast_nodes

::: pysearch.utils.helpers.iter_python_ast_nodes
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Path Utilities

### iter_files

::: pysearch.utils.helpers.iter_files
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Data Classes

### FileMeta

::: pysearch.utils.helpers.FileMeta
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

## Usage Examples

### Safe File Reading

```python
from pysearch.utils import read_text_safely
from pathlib import Path

# Read file with encoding detection and size limits
content = read_text_safely(
    Path("example.py"),
    max_bytes=1_000_000  # 1MB limit
)

if content is not None:
    print(f"File content: {len(content)} characters")
else:
    print("Failed to read file or file too large")
```

### File Hashing

```python
from pysearch.utils import file_sha1
from pathlib import Path

# Compute SHA1 hash of file
hash_value = file_sha1(Path("example.py"))
if hash_value:
    print(f"File hash: {hash_value}")
```

### Text Processing

```python
from pysearch.utils import split_lines_keepends, extract_context

# Split text preserving line endings
text = "line1\nline2\nline3\n"
lines = split_lines_keepends(text)
print(lines)  # ['line1\n', 'line2\n', 'line3\n']

# Extract context around specific lines
context_lines = extract_context(
    lines,
    start_line=1,  # 0-based
    end_line=1,
    context=1
)
print(context_lines)  # Lines with 1 line of context before/after
```

### Highlighting

```python
from pysearch.utils import highlight_spans

# Highlight specific spans in text
text = "def main():"
spans = [(0, 3), (4, 8)]  # Highlight "def" and "main"

highlighted = highlight_spans(text, spans, start_tag="<mark>", end_tag="</mark>")
print(highlighted)  # "<mark>def</mark> <mark>main</mark>():"
```

### AST Processing

```python
from pysearch.utils import iter_python_ast_nodes
import ast

# Parse Python code and iterate over AST nodes
code = """
def example():
    return 42

class MyClass:
    pass
"""

tree = ast.parse(code)
for node in iter_python_ast_nodes(tree):
    print(f"{type(node).__name__}: {getattr(node, 'name', 'N/A')}")
```

### File Iteration with Pruning

```python
from pysearch.utils import iter_files_prune
from pathlib import Path
import pathspec

# Create pathspec for exclusions
exclude_spec = pathspec.PathSpec.from_lines('gitwildmatch', [
    '**/__pycache__/**',
    '**/.venv/**',
    '*.pyc'
])

# Iterate files with directory pruning
for file_path in iter_files_prune(
    Path("."),
    include_spec=None,
    exclude_spec=exclude_spec,
    follow_symlinks=False
):
    print(file_path)
```

### File Metadata Creation

```python
from pysearch.utils import create_file_metadata
from pathlib import Path

# Create comprehensive file metadata
metadata = create_file_metadata(Path("example.py"))
print(f"File: {metadata.path}")
print(f"Size: {metadata.size} bytes")
print(f"Language: {metadata.language}")
print(f"Encoding: {metadata.encoding}")
print(f"Lines: {metadata.line_count}")
```

## Performance Utilities

### Chunked File Reading

```python
def read_large_file_chunked(path, chunk_size=1024*1024):
    """Read large files in chunks to manage memory."""
    try:
        with path.open('rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    except Exception as e:
        print(f"Error reading {path}: {e}")
```

### Efficient Line Processing

```python
def process_lines_efficiently(content):
    """Process lines without loading entire file into memory."""
    lines = split_lines_keepends(content)
    
    for i, line in enumerate(lines):
        # Process line by line
        if line.strip():  # Skip empty lines
            yield i, line.rstrip('\n\r')
```

## Error Handling

### Safe Operations

```python
from pysearch.utils import read_text_safely

def safe_file_operation(path):
    """Safely perform file operations with error handling."""
    try:
        content = read_text_safely(path)
        if content is None:
            return None, "Failed to read file"
        
        # Process content
        return content, None
        
    except Exception as e:
        return None, f"Error: {e}"
```

## Related

- [Language Detection](language-detection.md) - File type detection
- [Error Handling](error-handling.md) - Error management utilities
- [Indexer](indexer.md) - File indexing utilities
- [Types](types.md) - Data types and structures
