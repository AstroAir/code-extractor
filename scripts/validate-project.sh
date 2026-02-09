#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Validating PySearch project structure and functionality..."

# Auto-detect uv for consistent environment
if command -v uv &>/dev/null; then
    PY="uv run python"
else
    PY="python"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILED=0

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
        FAILED=1
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check project structure (inline, no make dependency)
echo "📁 Checking project structure..."
STRUCT_OK=0
for dir in src/pysearch mcp/servers mcp/shared tests docs scripts; do
    if [ ! -d "$dir" ]; then
        echo -e "${RED}❌ Directory missing: $dir${NC}"
        STRUCT_OK=1
    fi
done
for file in pyproject.toml README.md mcp/README.md; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}❌ File missing: $file${NC}"
        STRUCT_OK=1
    fi
done
print_status $STRUCT_OK "Project structure validation"

# Check Python syntax
echo "🐍 Checking Python syntax..."
$PY -m py_compile src/pysearch/*.py
print_status $? "Python syntax check"

# Check imports
echo "📦 Checking package imports..."
$PY -c "import pysearch; print('Core package imports successfully')"
print_status $? "Core package import"

$PY -c "from mcp.servers import pysearch_mcp_server; print('MCP server imports successfully')"
print_status $? "MCP server import"

# Run linting
echo "🔧 Running linting..."
$PY -m ruff check . && $PY -m black --check .
print_status $? "Linting checks"

# Run type checking
echo "🔍 Running type checking..."
$PY -m mypy
print_status $? "Type checking"

# Run tests
echo "🧪 Running tests..."
$PY -m pytest -q
print_status $? "Test suite"

# Check documentation
echo "📚 Checking documentation..."
if [ -f "mkdocs.yml" ]; then
    mkdocs build --quiet
    print_status $? "Documentation build"
else
    print_warning "mkdocs.yml not found, skipping documentation build"
fi

echo ""
if [ "$FAILED" -ne 0 ]; then
    echo -e "${RED}❌ Some validation checks failed!${NC}"
    exit 1
fi
echo -e "${GREEN}🎉 All validation checks passed!${NC}"
echo "Project is ready for development and deployment."
