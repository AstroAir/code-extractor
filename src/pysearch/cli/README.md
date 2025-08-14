# CLI Module

The CLI module provides the command-line interface for pysearch.

## Responsibilities

- **Command Parsing**: Command-line argument parsing and validation
- **Interactive Interface**: User-friendly command-line interaction
- **Output Formatting**: Terminal-optimized output formatting
- **Help System**: Comprehensive help and documentation

## Key Files

- `main.py` - Main CLI implementation (renamed from `cli.py`)

## CLI Features

1. **Rich Commands**: Comprehensive command set with intuitive syntax
2. **Interactive Mode**: Interactive search and exploration
3. **Output Formats**: Multiple output formats (plain, JSON, highlighted)
4. **Configuration**: Command-line configuration options

## Usage

```bash
# Basic search
pysearch find --pattern "def main" --path . --regex

# Advanced search with filters
pysearch find --pattern "class.*Test" --regex --ast --filter-decorator "pytest.*"

# Interactive mode
pysearch interactive
```

## Command Structure

- `find` - Primary search command
- `index` - Indexing management commands
- `config` - Configuration management
- `history` - Search history management
