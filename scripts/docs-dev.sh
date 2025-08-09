#!/bin/bash
# Development documentation workflow script
# Provides hot-reloading documentation server for development

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if mkdocs is available
if ! command -v mkdocs >/dev/null 2>&1; then
    echo "❌ mkdocs not found. Installing development dependencies..."
    pip install -e ".[dev]"
fi

# Check if we're in the right directory
if [[ ! -f "mkdocs.yml" ]]; then
    echo "❌ mkdocs.yml not found. Please run this script from the project root."
    exit 1
fi

log_info "Starting development documentation server..."
log_info "Features enabled:"
echo "  📝 Hot reloading on file changes"
echo "  🔍 API documentation from docstrings"
echo "  🎨 Material theme with syntax highlighting"
echo "  🔗 Cross-references and auto-linking"
echo ""

log_success "Documentation server starting..."
log_info "Open your browser to: http://localhost:8000"
log_info "Press Ctrl+C to stop the server"
echo ""

# Start the development server with hot reloading
mkdocs serve \
    --dev-addr=localhost:8000 \
    --livereload \
    --watch=src/pysearch \
    --watch=docs \
    --watch=examples \
    --verbose
