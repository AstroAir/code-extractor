# Installation Guide

This guide covers all installation methods for pysearch, from basic installation to advanced development setups.

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Installation](#quick-installation)
- [Installation Methods](#installation-methods)
- [Development Installation](#development-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Uninstallation](#uninstallation)

---

## System Requirements

### Minimum Requirements

- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 512 MB RAM (2 GB recommended for large codebases)
- **Disk Space**: 50 MB for installation (additional space for cache)

### Recommended Requirements

- **Python**: 3.11 or 3.12 (latest stable)
- **Memory**: 4 GB RAM or more
- **CPU**: Multi-core processor for parallel processing
- **Disk**: SSD for better I/O performance

### Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| Linux (Ubuntu 20.04+) | ✅ Fully Supported | Primary development platform |
| Linux (CentOS 8+) | ✅ Fully Supported | Enterprise Linux support |
| macOS (10.15+) | ✅ Fully Supported | Intel and Apple Silicon |
| Windows 10+ | ✅ Fully Supported | PowerShell and Command Prompt |
| Windows WSL2 | ✅ Fully Supported | Recommended for Windows users |

---

## Quick Installation

### For End Users

```bash
# Install from PyPI (when published)
pip install pysearch

# Or install from source
pip install git+https://github.com/your-org/pysearch.git
```

### For Developers

```bash
# Clone and install in development mode
git clone https://github.com/your-org/pysearch.git
cd pysearch
pip install -e ".[dev]"
```

---

## Installation Methods

### Method 1: PyPI Installation (Recommended)

When pysearch is published to PyPI:

```bash
# Install latest stable version
pip install pysearch

# Install specific version
pip install pysearch==1.0.0

# Upgrade to latest version
pip install --upgrade pysearch
```

### Method 2: Source Installation

Install directly from the source repository:

```bash
# Install from GitHub main branch
pip install git+https://github.com/your-org/pysearch.git

# Install from specific branch or tag
pip install git+https://github.com/your-org/pysearch.git@v1.0.0

# Install from local clone
git clone https://github.com/your-org/pysearch.git
cd pysearch
pip install .
```

### Method 3: Editable Installation

For development or customization:

```bash
# Clone repository
git clone https://github.com/your-org/pysearch.git
cd pysearch

# Install in editable mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Method 4: Virtual Environment Installation

Recommended for isolated environments:

```bash
# Create virtual environment
python -m venv pysearch-env

# Activate virtual environment
# On Linux/macOS:
source pysearch-env/bin/activate
# On Windows:
pysearch-env\Scripts\activate

# Install pysearch
pip install pysearch

# Deactivate when done
deactivate
```

### Method 5: Conda Installation

For Conda users:

```bash
# Create conda environment
conda create -n pysearch python=3.11
conda activate pysearch

# Install from PyPI
pip install pysearch

# Or install from conda-forge (if available)
conda install -c conda-forge pysearch
```

---

## Development Installation

### Complete Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/pysearch.git
cd pysearch

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify installation
pytest

# Run linting
make lint

# Run type checking
make type

# Build documentation
make docs
```

### Development Dependencies

The `[dev]` extra includes:

- **Testing**: pytest, pytest-cov, pytest-benchmark
- **Linting**: ruff, black, mypy
- **Documentation**: mkdocs, mkdocs-material
- **Development**: pre-commit, build tools

### Optional Dependencies

Install additional features as needed:

```bash
# Semantic search features
pip install pysearch[semantic]

# Advanced caching
pip install pysearch[cache]

# All optional features
pip install pysearch[all]
```

---

## Verification

### Basic Verification

```bash
# Check installation
pysearch --version

# Run help command
pysearch --help

# Test basic search
pysearch find --pattern "def main" --path .
```

### Comprehensive Verification

```bash
# Run validation script
./scripts/validate-project.sh

# Or use make command
make validate

# Run test suite
pytest

# Check all components
python -c "
import pysearch
from pysearch import PySearch, SearchConfig
from pysearch.types import Query

print('✅ pysearch imported successfully')
print(f'📦 Version: {pysearch.__version__}')

# Test basic functionality
config = SearchConfig(paths=['.'])
engine = PySearch(config)
print('✅ PySearch engine created successfully')

# Test search
try:
    results = engine.search('import')
    print(f'✅ Search completed: {len(results.items)} results')
except Exception as e:
    print(f'❌ Search failed: {e}')
"
```

### Performance Verification

```bash
# Run benchmarks
pytest tests/benchmarks -k benchmark

# Test with large codebase
pysearch find --pattern "class" --path /large/codebase --stats

# Memory usage test
python -c "
import psutil
import os
from pysearch import PySearch, SearchConfig

process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024

config = SearchConfig(paths=['.'])
engine = PySearch(config)
results = engine.search('def')

final_memory = process.memory_info().rss / 1024 / 1024
print(f'Memory usage: {final_memory - initial_memory:.1f} MB')
"
```

---

## Troubleshooting

### Common Installation Issues

#### Python Version Issues

**Problem**: `ERROR: Python 3.10 or higher is required`

**Solution**:

```bash
# Check Python version
python --version

# Install Python 3.10+ using pyenv
pyenv install 3.11.0
pyenv global 3.11.0

# Or use system package manager
# Ubuntu/Debian:
sudo apt update && sudo apt install python3.11

# macOS with Homebrew:
brew install python@3.11
```

#### Permission Issues

**Problem**: `Permission denied` during installation

**Solution**:

```bash
# Use user installation
pip install --user pysearch

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install pysearch
```

#### Dependency Conflicts

**Problem**: `ERROR: pip's dependency resolver does not currently consider all the packages that are installed`

**Solution**:

```bash
# Create fresh virtual environment
python -m venv fresh-env
source fresh-env/bin/activate
pip install --upgrade pip
pip install pysearch

# Or use pip-tools for dependency management
pip install pip-tools
pip-compile requirements.in
pip-sync requirements.txt
```

#### Build Issues

**Problem**: `Failed building wheel` or compilation errors

**Solution**:

```bash
# Update build tools
pip install --upgrade pip setuptools wheel

# Install build dependencies
pip install build

# For C extension issues, install development headers
# Ubuntu/Debian:
sudo apt install python3-dev build-essential

# CentOS/RHEL:
sudo yum install python3-devel gcc

# macOS:
xcode-select --install
```

### Platform-Specific Issues

#### Windows Issues

**Problem**: `'pysearch' is not recognized as an internal or external command`

**Solution**:

```cmd
# Add Python Scripts to PATH
# Or use full path
python -m pysearch find --pattern "def main"

# Or reinstall with --force-reinstall
pip install --force-reinstall pysearch
```

#### macOS Issues

**Problem**: `command not found: pysearch` after installation

**Solution**:

```bash
# Check if installed in user directory
ls ~/.local/bin/pysearch

# Add to PATH if needed
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Or use Homebrew Python
brew install python
pip3 install pysearch
```

#### Linux Issues

**Problem**: `ImportError: No module named '_ctypes'`

**Solution**:

```bash
# Install libffi development package
# Ubuntu/Debian:
sudo apt install libffi-dev

# CentOS/RHEL:
sudo yum install libffi-devel

# Reinstall Python if needed
pyenv install 3.11.0
```

### Network Issues

**Problem**: `Could not fetch URL` or timeout errors

**Solution**:

```bash
# Use different index
pip install -i https://pypi.org/simple/ pysearch

# Configure proxy if needed
pip install --proxy http://proxy.company.com:8080 pysearch

# Use offline installation
pip download pysearch
pip install pysearch-*.whl --no-index --find-links .
```

### Verification Failures

**Problem**: `pysearch --version` fails

**Solution**:

```bash
# Check installation location
pip show pysearch

# Reinstall if corrupted
pip uninstall pysearch
pip install pysearch

# Check for conflicting installations
pip list | grep pysearch
```

---

## Uninstallation

### Complete Removal

```bash
# Uninstall pysearch
pip uninstall pysearch

# Remove cache directories
rm -rf ~/.cache/pysearch
rm -rf ./.pysearch-cache

# Remove configuration files (optional)
rm -rf ~/.config/pysearch

# Remove virtual environment (if used)
rm -rf pysearch-env
```

### Clean Development Environment

```bash
# Remove development installation
pip uninstall pysearch

# Remove pre-commit hooks
pre-commit uninstall

# Clean build artifacts
make clean

# Remove development dependencies
pip uninstall -r requirements-dev.txt
```

---

## Next Steps

After successful installation:

1. **Read the Usage Guide**: [Usage Guide](usage.md)
2. **Configure pysearch**: [Configuration Guide](configuration.md)
3. **Explore Examples**: Check the `examples/` directory
4. **Join the Community**: Contribute to the project

## Support

If you encounter issues not covered in this guide:

1. **Check the FAQ**: [FAQ](faq.md)
2. **Search Issues**: GitHub Issues page
3. **Ask for Help**: Community discussions
4. **Report Bugs**: Create a new issue with installation details
