#!/bin/bash
# Documentation build script for pysearch
# This script provides comprehensive documentation building with validation and deployment options

set -e  # Exit on any error

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
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCS_DIR="docs"
BUILD_DIR="site"
CACHE_DIR=".mkdocs_cache"

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v mkdocs >/dev/null 2>&1; then
        log_error "mkdocs not found. Install with: pip install mkdocs-material"
        exit 1
    fi
    
    if ! $PY -c "import mkdocstrings" >/dev/null 2>&1; then
        log_error "mkdocstrings not found. Install with: pip install mkdocstrings[python]"
        exit 1
    fi
    
    log_success "All dependencies found"
}

clean_build() {
    log_info "Cleaning previous build artifacts..."
    rm -rf "$BUILD_DIR"
    rm -rf "$CACHE_DIR"
    rm -rf "docs_build"
    rm -rf ".docstring_cache"
    rm -rf ".api_docs_cache"
    log_success "Build artifacts cleaned"
}

validate_docs() {
    log_info "Validating documentation structure..."
    
    # Check required files (paths must match actual project structure)
    required_files=(
        "docs/index.md"
        "docs/getting-started/installation.md"
        "docs/guide/usage.md"
        "docs/guide/configuration.md"
        "mkdocs.yml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file missing: $file"
            exit 1
        fi
    done
    
    # Check API documentation
    if [[ ! -d "docs/api" ]]; then
        log_error "API documentation directory missing: docs/api"
        exit 1
    fi
    
    log_success "Documentation structure validated"
}

build_docs() {
    log_info "Building documentation..."
    
    # Build with strict mode to catch errors
    if mkdocs build --clean --strict --verbose; then
        log_success "Documentation built successfully"
    else
        log_error "Documentation build failed"
        exit 1
    fi
}

check_links() {
    log_info "Checking for broken links..."
    
    if command -v linkchecker >/dev/null 2>&1; then
        if linkchecker "$BUILD_DIR/index.html" --check-extern; then
            log_success "All links are valid"
        else
            log_warning "Some links may be broken (check output above)"
        fi
    else
        log_warning "linkchecker not installed, skipping link check"
        log_info "Install with: pip install linkchecker"
    fi
}

serve_docs() {
    log_info "Starting documentation server..."
    log_info "Documentation will be available at http://localhost:8000"
    log_info "Press Ctrl+C to stop the server"
    
    mkdocs serve -a 0.0.0.0:8000 --dev-addr=localhost:8000
}

deploy_docs() {
    log_info "Deploying documentation to GitHub Pages..."
    
    if [[ -z "$GITHUB_TOKEN" ]]; then
        log_error "GITHUB_TOKEN environment variable required for deployment"
        exit 1
    fi
    
    if mkdocs gh-deploy --clean --message "Deploy documentation [skip ci]"; then
        log_success "Documentation deployed successfully"
    else
        log_error "Documentation deployment failed"
        exit 1
    fi
}

show_help() {
    echo "Documentation build script for pysearch"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     Build documentation (default)"
    echo "  serve     Build and serve documentation locally"
    echo "  clean     Clean build artifacts"
    echo "  check     Build and validate documentation"
    echo "  deploy    Deploy to GitHub Pages"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build          # Build documentation"
    echo "  $0 serve          # Serve documentation locally"
    echo "  $0 check          # Build and validate"
    echo "  $0 deploy         # Deploy to GitHub Pages"
}

# Main script logic
main() {
    local command="${1:-build}"
    
    case "$command" in
        "build")
            check_dependencies
            validate_docs
            clean_build
            build_docs
            log_success "Documentation build completed successfully!"
            log_info "Built documentation is available in: $BUILD_DIR/"
            ;;
        "serve")
            check_dependencies
            validate_docs
            build_docs
            serve_docs
            ;;
        "clean")
            clean_build
            ;;
        "check")
            check_dependencies
            validate_docs
            clean_build
            build_docs
            check_links
            log_success "Documentation validation completed!"
            ;;
        "deploy")
            check_dependencies
            validate_docs
            clean_build
            build_docs
            check_links
            deploy_docs
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
