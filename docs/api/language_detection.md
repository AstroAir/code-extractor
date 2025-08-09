# Language Detection API

Automatic programming language detection and file type classification system.

## Main Functions

### detect_language

::: pysearch.language_detection.detect_language
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### is_text_file

::: pysearch.language_detection.is_text_file
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### get_language_extensions

::: pysearch.language_detection.get_language_extensions
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### get_language_extensions

::: pysearch.language_detection.get_language_extensions
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Usage Examples

### Basic Language Detection

```python
from pysearch.language_detection import detect_language
from pysearch.types import Language
from pathlib import Path

# Detect language from file extension
lang = detect_language(Path("example.py"))
print(f"Language: {lang}")  # Language.PYTHON

# Detect from various file types
files = [
    "script.js",
    "component.tsx", 
    "style.css",
    "config.toml",
    "README.md"
]

for filename in files:
    lang = detect_language(Path(filename))
    print(f"{filename}: {lang}")
```

### Content-Based Detection

```python
# Detect language from file content
def detect_from_content(path):
    """Detect language using both filename and content."""
    
    # First try filename-based detection
    lang = detect_language(path)
    
    if lang == Language.UNKNOWN:
        # Fallback to content-based detection
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read(1024)  # Read first 1KB
                lang = detect_language_from_content(content, path)
        except Exception:
            pass
    
    return lang
```

### File Type Filtering

```python
from pysearch.language_detection import is_text_file, is_binary_file

def filter_text_files(file_paths):
    """Filter to only include text files."""
    text_files = []
    
    for path in file_paths:
        if is_text_file(path):
            text_files.append(path)
        elif is_binary_file(path):
            print(f"Skipping binary file: {path}")
    
    return text_files
```

### Language-Specific Processing

```python
from pysearch.types import Language

def process_by_language(file_path):
    """Process file based on detected language."""
    
    lang = detect_language(file_path)
    
    if lang == Language.PYTHON:
        return process_python_file(file_path)
    elif lang in {Language.JAVASCRIPT, Language.TYPESCRIPT}:
        return process_js_file(file_path)
    elif lang == Language.JAVA:
        return process_java_file(file_path)
    else:
        return process_generic_file(file_path)
```

## Supported Languages

### Programming Languages

The system supports detection of major programming languages:

```python
from pysearch.types import Language

# Core languages
CORE_LANGUAGES = {
    Language.PYTHON,      # .py, .pyw
    Language.JAVASCRIPT,  # .js, .mjs
    Language.TYPESCRIPT,  # .ts, .tsx
    Language.JAVA,        # .java
    Language.C,           # .c, .h
    Language.CPP,         # .cpp, .cxx, .hpp
    Language.CSHARP,      # .cs
    Language.GO,          # .go
    Language.RUST,        # .rs
    Language.PHP,         # .php
    Language.RUBY,        # .rb
    Language.KOTLIN,      # .kt, .kts
    Language.SWIFT,       # .swift
    Language.SCALA,       # .scala
    Language.R,           # .r, .R
    Language.MATLAB,      # .m
}
```

### Scripting and Shell

```python
# Scripting languages
SCRIPT_LANGUAGES = {
    Language.SHELL,       # .sh, .bash, .zsh
    Language.POWERSHELL,  # .ps1, .psm1
    Language.SQL,         # .sql
}
```

### Markup and Data

```python
# Markup and data formats
MARKUP_LANGUAGES = {
    Language.HTML,        # .html, .htm
    Language.CSS,         # .css, .scss, .sass
    Language.XML,         # .xml, .xsd, .xsl
    Language.JSON,        # .json
    Language.YAML,        # .yaml, .yml
    Language.TOML,        # .toml
    Language.MARKDOWN,    # .md, .markdown
}
```

### Configuration and Build

```python
# Configuration files
CONFIG_LANGUAGES = {
    Language.DOCKERFILE,  # Dockerfile, .dockerfile
    Language.MAKEFILE,    # Makefile, .mk
}
```

## Extension Mapping

### Get Language Extensions

```python
from pysearch.language_detection import get_language_extensions

# Get all extensions for a language
py_extensions = get_language_extensions(Language.PYTHON)
print(py_extensions)  # ['.py', '.pyw', '.pyi']

js_extensions = get_language_extensions(Language.JAVASCRIPT)
print(js_extensions)  # ['.js', '.mjs', '.jsx']
```

### Custom Extension Mapping

```python
# Add custom extension mappings
CUSTOM_EXTENSIONS = {
    '.myext': Language.PYTHON,
    '.custom': Language.JAVASCRIPT,
}

def detect_with_custom(path):
    """Detect language with custom extensions."""
    
    # Check custom mappings first
    suffix = path.suffix.lower()
    if suffix in CUSTOM_EXTENSIONS:
        return CUSTOM_EXTENSIONS[suffix]
    
    # Fallback to standard detection
    return detect_language(path)
```

## Content-Based Detection

### Shebang Detection

```python
def detect_from_shebang(content):
    """Detect language from shebang line."""
    
    if not content.startswith('#!'):
        return Language.UNKNOWN
    
    first_line = content.split('\n')[0].lower()
    
    if 'python' in first_line:
        return Language.PYTHON
    elif 'node' in first_line or 'javascript' in first_line:
        return Language.JAVASCRIPT
    elif 'bash' in first_line or 'sh' in first_line:
        return Language.SHELL
    
    return Language.UNKNOWN
```

### Pattern-Based Detection

```python
import re

def detect_from_patterns(content):
    """Detect language from code patterns."""
    
    # Python patterns
    if re.search(r'^\s*def\s+\w+\s*\(', content, re.MULTILINE):
        return Language.PYTHON
    
    # JavaScript patterns
    if re.search(r'^\s*function\s+\w+\s*\(', content, re.MULTILINE):
        return Language.JAVASCRIPT
    
    # Java patterns
    if re.search(r'^\s*public\s+class\s+\w+', content, re.MULTILINE):
        return Language.JAVA
    
    return Language.UNKNOWN
```

## Binary File Detection

### Binary Detection Methods

```python
from pysearch.language_detection import is_binary_file

def comprehensive_binary_check(path):
    """Comprehensive binary file detection."""
    
    # Quick extension-based check
    binary_extensions = {'.exe', '.dll', '.so', '.dylib', '.bin', 
                        '.jpg', '.png', '.gif', '.pdf', '.zip'}
    
    if path.suffix.lower() in binary_extensions:
        return True
    
    # Content-based check
    return is_binary_file(path)
```

### Text File Validation

```python
def is_searchable_text_file(path):
    """Check if file is searchable text."""
    
    # Must be a text file
    if not is_text_file(path):
        return False
    
    # Must have supported language
    lang = detect_language(path)
    if lang == Language.UNKNOWN:
        return False
    
    # Check file size (avoid huge files)
    try:
        if path.stat().st_size > 10 * 1024 * 1024:  # 10MB
            return False
    except OSError:
        return False
    
    return True
```

## Integration with Search

### Language-Aware Search Configuration

```python
from pysearch.config import SearchConfig
from pysearch.types import Language

def create_language_config(languages):
    """Create search config for specific languages."""
    
    extensions = []
    for lang in languages:
        extensions.extend(get_language_extensions(lang))
    
    # Create include patterns
    include_patterns = [f"**/*{ext}" for ext in extensions]
    
    return SearchConfig(
        include=include_patterns,
        languages=set(languages)
    )

# Example: Search only Python and JavaScript files
config = create_language_config([Language.PYTHON, Language.JAVASCRIPT])
```

### Language-Specific Search Strategies

```python
def get_search_strategy(language):
    """Get optimal search strategy for language."""
    
    if language == Language.PYTHON:
        return {
            'enable_docstrings': True,
            'enable_comments': True,
            'ast_weight': 2.0,
            'context': 3
        }
    elif language in {Language.JAVASCRIPT, Language.TYPESCRIPT}:
        return {
            'enable_comments': True,
            'enable_strings': False,
            'text_weight': 1.5,
            'context': 2
        }
    else:
        return {
            'enable_comments': True,
            'text_weight': 1.0,
            'context': 2
        }
```

## Related

- [Types](types.md) - Language enumeration and types
- [Utils](utils.md) - File processing utilities
- [Configuration](config.md) - Language-specific configuration
- [Indexer](indexer.md) - Language-aware file indexing
