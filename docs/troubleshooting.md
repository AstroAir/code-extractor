# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with pysearch.

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Installation Issues](#installation-issues)
- [Search Issues](#search-issues)
- [Performance Issues](#performance-issues)
- [Configuration Issues](#configuration-issues)
- [Error Messages](#error-messages)
- [Debug Mode](#debug-mode)
- [Getting Help](#getting-help)

---

## Quick Diagnostics

### Health Check

Run this quick health check to identify common issues:

```bash
# Check pysearch installation
pysearch --version

# Test basic functionality
pysearch find --pattern "import" --path . --stats

# Check configuration
python -c "
from pysearch import SearchConfig
config = SearchConfig()
print('✅ Configuration loaded successfully')
print(f'Paths: {config.paths}')
print(f'Include: {config.get_include_patterns()[:3]}...')
print(f'Exclude: {config.get_exclude_patterns()[:3]}...')
"

# Check dependencies
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import regex
    print('✅ regex library available')
except ImportError:
    print('❌ regex library missing')

try:
    import rich
    print('✅ rich library available')
except ImportError:
    print('❌ rich library missing')

try:
    import orjson
    print('✅ orjson library available')
except ImportError:
    print('❌ orjson library missing')
"
```

### System Information

Gather system information for troubleshooting:

```bash
# System info script
python -c "
import platform
import sys
import os
from pathlib import Path

print('=== System Information ===')
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'Architecture: {platform.machine()}')
print(f'CPU cores: {os.cpu_count()}')

try:
    import psutil
    memory = psutil.virtual_memory()
    print(f'Memory: {memory.total // (1024**3)} GB total, {memory.available // (1024**3)} GB available')
except ImportError:
    print('Memory: psutil not available')

print(f'Current directory: {Path.cwd()}')
print(f'Python executable: {sys.executable}')

try:
    import pysearch
    print(f'pysearch version: {pysearch.__version__}')
    print(f'pysearch location: {pysearch.__file__}')
except ImportError as e:
    print(f'pysearch import error: {e}')
"
```

---

## Installation Issues

### Python Version Problems

**Issue**: `ERROR: Python 3.10 or higher is required`

**Diagnosis**:

```bash
python --version
python3 --version
which python
which python3
```

**Solutions**:

1. **Install correct Python version**:

   ```bash
   # Using pyenv (recommended)
   curl https://pyenv.run | bash
   pyenv install 3.11.0
   pyenv global 3.11.0
   
   # Using system package manager
   # Ubuntu/Debian:
   sudo apt update && sudo apt install python3.11
   
   # macOS with Homebrew:
   brew install python@3.11
   
   # Windows: Download from python.org
   ```

2. **Use specific Python version**:

   ```bash
   python3.11 -m pip install pysearch
   ```

### Permission Errors

**Issue**: `Permission denied` or `Access is denied`

**Diagnosis**:

```bash
pip install --user --dry-run pysearch
ls -la $(python -m site --user-base)/bin/
```

**Solutions**:

1. **User installation**:

   ```bash
   pip install --user pysearch
   ```

2. **Virtual environment**:

   ```bash
   python -m venv pysearch-env
   source pysearch-env/bin/activate  # Linux/macOS
   # pysearch-env\Scripts\activate   # Windows
   pip install pysearch
   ```

3. **Fix permissions** (Linux/macOS):

   ```bash
   sudo chown -R $USER:$USER ~/.local
   ```

### Dependency Conflicts

**Issue**: `ERROR: pip's dependency resolver does not currently consider all the packages`

**Diagnosis**:

```bash
pip check
pip list --outdated
```

**Solutions**:

1. **Fresh environment**:

   ```bash
   python -m venv fresh-env
   source fresh-env/bin/activate
   pip install --upgrade pip
   pip install pysearch
   ```

2. **Dependency resolution**:

   ```bash
   pip install --upgrade pip
   pip install --force-reinstall pysearch
   ```

3. **Manual resolution**:

   ```bash
   pip uninstall conflicting-package
   pip install pysearch
   pip install conflicting-package
   ```

### Build Failures

**Issue**: `Failed building wheel` or compilation errors

**Diagnosis**:

```bash
pip install --verbose pysearch
gcc --version  # Linux/macOS
```

**Solutions**:

1. **Update build tools**:

   ```bash
   pip install --upgrade pip setuptools wheel build
   ```

2. **Install system dependencies**:

   ```bash
   # Ubuntu/Debian:
   sudo apt install python3-dev build-essential libffi-dev
   
   # CentOS/RHEL:
   sudo yum install python3-devel gcc libffi-devel
   
   # macOS:
   xcode-select --install
   ```

3. **Use pre-built wheels**:

   ```bash
   pip install --only-binary=all pysearch
   ```

---

## Search Issues

### No Results Found

**Issue**: Search returns no results when matches should exist

**Diagnosis**:

```bash
# Check if files are being scanned
pysearch find --pattern "import" --path . --stats

# Test with very broad pattern
pysearch find --pattern "." --regex --path . --stats

# Check include/exclude patterns
pysearch find --pattern "pattern" --include "**/*" --stats
```

**Solutions**:

1. **Verify paths**:

   ```bash
   # Use absolute paths
   pysearch find --pattern "pattern" --path /absolute/path/to/code
   
   # Check current directory
   ls -la
   ```

2. **Check patterns**:

   ```bash
   # Test include patterns
   pysearch find --pattern "pattern" --include "**/*.py" --include "**/*.txt"
   
   # Remove exclude patterns temporarily
   pysearch find --pattern "pattern" --exclude ""
   ```

3. **Content type issues**:

   ```bash
   # Enable all content types
   pysearch find --pattern "pattern" --docstrings --comments --strings
   ```

4. **File size limits**:

   ```bash
   # Increase file size limit
   pysearch find --pattern "pattern" --max-file-size 10485760  # 10MB
   ```

### Incorrect Results

**Issue**: Search returns unexpected or irrelevant results

**Diagnosis**:

```bash
# Check exact pattern matching
pysearch find --pattern "exact_pattern" --path .

# Test regex escaping
pysearch find --pattern "\\." --regex --path .  # Literal dot
```

**Solutions**:

1. **Pattern escaping**:

   ```bash
   # Escape special characters
   pysearch find --pattern "\\[\\]" --regex  # Literal brackets
   
   # Use non-regex search for literals
   pysearch find --pattern "[literal]"  # No --regex flag
   ```

2. **AST filtering**:

   ```bash
   # Use AST filters for precise matching
   pysearch find --pattern "def" --ast --filter-func-name "^exact_name$"
   ```

3. **Context adjustment**:

   ```bash
   # Reduce context to see exact matches
   pysearch find --pattern "pattern" --context 0
   ```

### Encoding Issues

**Issue**: `UnicodeDecodeError` or garbled text in results

**Diagnosis**:

```bash
# Check file encodings
python -c "
import chardet
import os

for root, dirs, files in os.walk('.'):
    for file in files[:5]:  # Check first 5 files
        if file.endswith('.py'):
            path = os.path.join(root, file)
            try:
                with open(path, 'rb') as f:
                    raw = f.read(1024)
                    result = chardet.detect(raw)
                    print(f'{path}: {result[\"encoding\"]} ({result[\"confidence\"]:.2f})')
            except Exception as e:
                print(f'{path}: Error - {e}')
"
```

**Solutions**:

1. **Skip problematic files**:

   ```bash
   pysearch find --pattern "pattern" --exclude "**/problematic_file.py"
   ```

2. **Convert file encodings**:

   ```bash
   # Convert to UTF-8
   iconv -f ISO-8859-1 -t UTF-8 file.py > file_utf8.py
   ```

3. **Handle encoding in API**:

   ```python
   from pysearch.utils import read_text_safely
   
   # This function handles encoding detection automatically
   content = read_text_safely(file_path)
   ```

---

## Performance Issues

### Slow Search Performance

**Issue**: Searches take too long to complete

**Diagnosis**:

```bash
# Profile search performance
time pysearch find --pattern "pattern" --path . --stats

# Check file count
find . -name "*.py" | wc -l

# Monitor resource usage
top -p $(pgrep -f pysearch)
```

**Solutions**:

1. **Optimize search scope**:

   ```bash
   # Limit to specific directories
   pysearch find --pattern "pattern" --path ./src --path ./tests
   
   # Add more exclusions
   pysearch find --pattern "pattern" \
     --exclude "**/.venv/**" \
     --exclude "**/.git/**" \
     --exclude "**/node_modules/**" \
     --exclude "**/__pycache__/**" \
     --exclude "**/.mypy_cache/**" \
     --exclude "**/.pytest_cache/**"
   ```

2. **Enable parallel processing**:

   ```bash
   pysearch find --pattern "pattern" --parallel --workers 8
   ```

3. **Disable unnecessary features**:

   ```bash
   # Skip docstrings and comments
   pysearch find --pattern "pattern" --no-docstrings --no-comments
   
   # Reduce context
   pysearch find --pattern "pattern" --context 1
   ```

4. **Use caching**:

   ```python
   from pysearch import PySearch, SearchConfig
   
   engine = PySearch(SearchConfig(paths=["./src"]))
   engine.enable_caching(ttl=3600)  # 1 hour cache
   ```

### High Memory Usage

**Issue**: pysearch uses too much memory

**Diagnosis**:

```bash
# Monitor memory usage
python -c "
import psutil
import os
from pysearch import PySearch, SearchConfig

process = psutil.Process(os.getpid())
print(f'Initial memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')

config = SearchConfig(paths=['.'])
engine = PySearch(config)
print(f'After init: {process.memory_info().rss / 1024 / 1024:.1f} MB')

results = engine.search('def')
print(f'After search: {process.memory_info().rss / 1024 / 1024:.1f} MB')
print(f'Results: {len(results.items)} items')
"
```

**Solutions**:

1. **Limit file sizes**:

   ```python
   config = SearchConfig(
       file_size_limit=1_000_000,  # 1MB limit
       max_file_bytes=1_000_000
   )
   ```

2. **Reduce parallelism**:

   ```python
   config = SearchConfig(
       workers=2,  # Fewer workers
       parallel=True
   )
   ```

3. **Limit context**:

   ```python
   config = SearchConfig(context=1)  # Minimal context
   ```

4. **Process in batches**:

   ```python
   # Process large result sets in chunks
   for i in range(0, len(results.items), 100):
       batch = results.items[i:i+100]
       process_batch(batch)
   ```

---

## Configuration Issues

### Configuration Not Loading

**Issue**: Configuration file is ignored

**Diagnosis**:

```bash
# Check configuration file locations
ls -la pysearch.toml
ls -la ~/.config/pysearch/config.toml
ls -la /etc/pysearch/config.toml

# Test configuration loading
python -c "
import tomllib
try:
    with open('pysearch.toml', 'rb') as f:
        config = tomllib.load(f)
    print('✅ Configuration loaded successfully')
    print(config)
except FileNotFoundError:
    print('❌ Configuration file not found')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
```

**Solutions**:

1. **Check file format**:

   ```toml
   # Ensure proper TOML format
   [search]
   paths = ["./src"]  # Array syntax
   parallel = true    # Boolean syntax
   ```

2. **Verify file location**:

   ```bash
   # Create in correct location
   mkdir -p ~/.config/pysearch
   cp pysearch.toml ~/.config/pysearch/config.toml
   ```

3. **Test configuration**:

   ```python
   from pysearch import SearchConfig

   # Load with explicit validation
   config = SearchConfig()
   print(f"Loaded paths: {config.paths}")
   ```

### Environment Variables Not Working

**Issue**: Environment variables are ignored

**Diagnosis**:

```bash
# Check environment variables
env | grep PYSEARCH

# Test variable loading
python -c "
import os
print('PYSEARCH_PATHS:', os.environ.get('PYSEARCH_PATHS'))
print('PYSEARCH_PARALLEL:', os.environ.get('PYSEARCH_PARALLEL'))
"
```

**Solutions**:

1. **Set variables correctly**:

   ```bash
   export PYSEARCH_PATHS="./src:./tests"
   export PYSEARCH_PARALLEL="true"
   export PYSEARCH_WORKERS="4"
   ```

2. **Check variable names**:

   ```bash
   # Use correct variable names (see configuration.md)
   export PYSEARCH_CONTEXT="5"  # Not PYSEARCH_CONTEXT_LINES
   ```

---

## Error Messages

### Common Error Messages and Solutions

#### `ModuleNotFoundError: No module named 'pysearch'`

**Cause**: pysearch not installed or not in Python path

**Solution**:

```bash
pip install pysearch
# Or check virtual environment activation
```

#### `FileNotFoundError: [Errno 2] No such file or directory`

**Cause**: Invalid search path

**Solution**:

```bash
# Use absolute paths or verify current directory
pysearch find --pattern "pattern" --path $(pwd)/src
```

#### `PermissionError: [Errno 13] Permission denied`

**Cause**: Insufficient permissions to read files

**Solution**:

```bash
# Check file permissions
ls -la problematic_file.py

# Skip inaccessible files
pysearch find --pattern "pattern" --exclude "**/restricted/**"
```

#### `regex.error: bad character range`

**Cause**: Invalid regex pattern

**Solution**:

```bash
# Escape special characters
pysearch find --pattern "\\[a-z\\]" --regex

# Or use literal search
pysearch find --pattern "[a-z]"  # No --regex flag
```

#### `OSError: [Errno 24] Too many open files`

**Cause**: File descriptor limit exceeded

**Solution**:

```bash
# Increase file descriptor limit
ulimit -n 4096

# Or reduce parallel workers
pysearch find --pattern "pattern" --workers 2
```

---

## Debug Mode

### Enabling Debug Output

```bash
# Environment variable
export PYSEARCH_DEBUG=1
pysearch find --pattern "pattern"

# Python API
from pysearch.logging_config import enable_debug_logging
enable_debug_logging()
```

### Debug Information

Debug mode provides:

- File scanning progress
- Pattern matching details
- Performance metrics
- Error stack traces
- Cache hit/miss statistics

### Collecting Debug Information

```bash
# Comprehensive debug output
PYSEARCH_DEBUG=1 pysearch find --pattern "pattern" --stats 2>&1 | tee debug.log

# Python debug session
python -c "
from pysearch.logging_config import enable_debug_logging
from pysearch import PySearch, SearchConfig

enable_debug_logging()

config = SearchConfig(paths=['.'])
engine = PySearch(config)
results = engine.search('pattern')

print(f'Debug complete. Results: {len(results.items)}')
"
```

---

## Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Review the FAQ**: [FAQ](faq.md)
3. **Search existing issues** on GitHub
4. **Try the quick diagnostics** above

### When Reporting Issues

Include this information:

```bash
# System information
python --version
pysearch --version
uname -a  # Linux/macOS
systeminfo  # Windows

# Error reproduction
PYSEARCH_DEBUG=1 pysearch find --pattern "your_pattern" --path . 2>&1

# Configuration
cat pysearch.toml  # If using config file
env | grep PYSEARCH  # Environment variables
```

### Where to Get Help

1. **GitHub Issues**: For bugs and feature requests
2. **GitHub Discussions**: For questions and community help
3. **Documentation**: Check all docs in `docs/` directory
4. **Stack Overflow**: Tag questions with `pysearch`

### Creating Good Bug Reports

Include:

- **Clear description** of the problem
- **Steps to reproduce** the issue
- **Expected behavior** vs. actual behavior
- **System information** (OS, Python version, pysearch version)
- **Minimal example** that demonstrates the issue
- **Debug output** if available

### Feature Requests

For feature requests:

- **Check the roadmap**: [roadmap.md](roadmap.md)
- **Describe the use case** clearly
- **Explain the benefits** to other users
- **Suggest implementation** if you have ideas

---

## Contributing to Troubleshooting

### Cache and Cleanup Issues

**Issue**: Unexpected behavior after project cleanup or cache corruption

**Diagnosis**:

```bash
# Check for remaining cache directories
ls -la | grep -E "\.(mypy|pytest|pysearch)"

# Check cache sizes
du -sh .mypy_cache .pytest_cache .pysearch-cache 2>/dev/null || echo "No cache directories found"
```

**Solutions**:

1. **Clean all caches**:

   ```bash
   # Use project's clean command
   make clean

   # Or manually clean
   rm -rf .mypy_cache .pytest_cache .pysearch-cache
   rm -rf .coverage coverage.xml
   find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
   ```

2. **Regenerate caches**:

   ```bash
   # Regenerate type checking cache
   make type

   # Regenerate test cache
   make test

   # Let pysearch regenerate its cache
   pysearch find --pattern "import" --path .
   ```

3. **Check virtual environment**:

   ```bash
   # Ensure using correct virtual environment
   which python
   which pysearch

   # Should point to .venv/bin/ if using project environment
   ```

**Note**: All cache directories are automatically regenerated when needed. The project maintains a clean structure with no unnecessary cache files.

---

## Contributing to This Guide

Help improve this guide by:

- **Adding solutions** for issues you've encountered
- **Improving existing solutions** with better approaches
- **Adding diagnostic commands** for common problems
- **Updating error messages** as they change

Submit improvements via pull requests or GitHub issues.
