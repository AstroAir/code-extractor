#!/bin/bash

# CLI Examples for pysearch
# This script demonstrates various command-line usage patterns for pysearch
# 
# Usage:
#   chmod +x examples/cli_examples.sh
#   ./examples/cli_examples.sh

echo "pysearch CLI Examples"
echo "===================="

# Basic text search
echo -e "\n1. Basic text search for function definitions:"
echo "Command: pysearch find --pattern 'def main' --path src --context 2"
pysearch find --pattern "def main" --path src --context 2

# Regex search
echo -e "\n2. Regex search for handler functions:"
echo "Command: pysearch find --pattern 'def .*_handler' --regex --path src --context 1"
pysearch find --pattern "def .*_handler" --regex --path src --context 1

# AST search with filters
echo -e "\n3. AST search for cached functions:"
echo "Command: pysearch find --pattern 'def' --ast --filter-decorator 'lru_cache' --path src"
pysearch find --pattern "def" --ast --filter-decorator "lru_cache" --path src

# Search with multiple paths
echo -e "\n4. Search across multiple directories:"
echo "Command: pysearch find --pattern 'class.*Test' --regex --path src tests --format json"
pysearch find --pattern "class.*Test" --regex --path src tests --format json

# Search with include/exclude patterns
echo -e "\n5. Search with file filtering:"
echo "Command: pysearch find --pattern 'import' --include '**/*.py' --exclude '**/test_*' --path ."
pysearch find --pattern "import" --include "**/*.py" --exclude "**/test_*" --path .

# Search in specific content types
echo -e "\n6. Search only in docstrings and comments:"
echo "Command: pysearch find --pattern 'TODO' --no-strings --path src --context 1"
pysearch find --pattern "TODO" --no-strings --path src --context 1

# Complex AST filtering
echo -e "\n7. Complex AST filtering:"
echo "Command: pysearch find --pattern 'def' --ast --filter-func-name '.*search.*' --filter-class-name '.*Engine.*' --path src"
pysearch find --pattern "def" --ast --filter-func-name ".*search.*" --filter-class-name ".*Engine.*" --path src

# Search with statistics
echo -e "\n8. Search with performance statistics:"
echo "Command: pysearch find --pattern 'from.*import' --regex --path src --stats --format text"
pysearch find --pattern "from.*import" --regex --path src --stats --format text

# Highlighted output (if terminal supports it)
echo -e "\n9. Search with highlighted output:"
echo "Command: pysearch find --pattern 'class' --path src --format highlight --context 2"
pysearch find --pattern "class" --path src --format highlight --context 2

# Search for imports
echo -e "\n10. Search for specific imports:"
echo "Command: pysearch find --pattern 'from pathlib import' --path src --context 0"
pysearch find --pattern "from pathlib import" --path src --context 0

echo -e "\n===================="
echo "CLI examples completed!"
echo ""
echo "Additional useful patterns:"
echo "  # Find all TODO comments:"
echo "  pysearch find --pattern 'TODO|FIXME|XXX' --regex --path src"
echo ""
echo "  # Find all class definitions:"
echo "  pysearch find --pattern '^class \\w+' --regex --path src"
echo ""
echo "  # Find functions with specific decorators:"
echo "  pysearch find --pattern 'def' --ast --filter-decorator 'property|staticmethod|classmethod' --path src"
echo ""
echo "  # Search for error handling:"
echo "  pysearch find --pattern 'except|raise|try:' --regex --path src"
echo ""
echo "  # Find configuration or constants:"
echo "  pysearch find --pattern '[A-Z_]{3,}' --regex --path src"
