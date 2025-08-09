#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Setting up PySearch development environment..."

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install -U pip

# Install package in editable mode
echo "📦 Installing PySearch in editable mode..."
python -m pip install -e .

# Install development dependencies
echo "📦 Installing development dependencies..."
python -m pip install -e ".[dev]"

# Install pre-commit
echo "🔧 Installing pre-commit..."
python -m pip install pre-commit
pre-commit install

# Validate installation
echo "✅ Validating installation..."
python -c "import pysearch; print(f'PySearch version: {pysearch.__version__}')"
python -c "import mcp; print('MCP package available')"

echo "✅ Dev install completed successfully!"
echo ""
echo "Next steps:"
echo "  - Run 'make test' to run tests"
echo "  - Run 'make validate' to run all checks"
echo "  - Run 'make docs' to build documentation"