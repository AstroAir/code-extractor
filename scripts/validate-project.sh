#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Validating PySearch project structure and functionality..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
        exit 1
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check project structure
echo "📁 Checking project structure..."
make check-structure
print_status $? "Project structure validation"

# Check Python syntax
echo "🐍 Checking Python syntax..."
python -m py_compile src/pysearch/*.py
print_status $? "Python syntax check"

# Check imports
echo "📦 Checking package imports..."
python -c "import pysearch; print('Core package imports successfully')"
print_status $? "Core package import"

python -c "import mcp; print('MCP package imports successfully')"
print_status $? "MCP package import"

# Run linting
echo "🔧 Running linting..."
make lint
print_status $? "Linting checks"

# Run type checking
echo "🔍 Running type checking..."
make type
print_status $? "Type checking"

# Run tests
echo "🧪 Running tests..."
make test
print_status $? "Test suite"

# Check MCP servers
echo "🔌 Checking MCP servers..."
make mcp-servers
print_status $? "MCP server validation"

# Check documentation
echo "📚 Checking documentation..."
if [ -f "mkdocs.yml" ]; then
    mkdocs build --quiet
    print_status $? "Documentation build"
else
    print_warning "mkdocs.yml not found, skipping documentation build"
fi

echo ""
echo -e "${GREEN}🎉 All validation checks passed!${NC}"
echo "Project is ready for development and deployment."
