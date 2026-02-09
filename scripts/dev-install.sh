#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Setting up PySearch development environment..."

# Detect uv for faster installation
USE_UV=0
if command -v uv &>/dev/null; then
    echo "⚡ uv detected, using it for faster installation"
    USE_UV=1
fi

pip_install() {
    if [ "$USE_UV" -eq 1 ]; then
        uv pip install "$@"
    else
        python -m pip install "$@"
    fi
}

# Check/create virtual environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment at $VENV_DIR..."
    if [ "$USE_UV" -eq 1 ]; then
        uv venv "$VENV_DIR"
    else
        python -m venv "$VENV_DIR"
    fi
    echo "⚠️  Virtual environment created. Please activate it:"
    echo "    source $VENV_DIR/bin/activate"
    echo "  Then re-run this script."
    exit 0
else
    echo "✅ Virtual environment found at $VENV_DIR/"
fi

# Upgrade pip
echo "📦 Upgrading pip..."
pip_install -U pip

# Install package in editable mode
echo "📦 Installing PySearch in editable mode..."
pip_install -e .

# Install development dependencies
echo "📦 Installing development dependencies..."
pip_install -e ".[dev]"

# Install pre-commit
echo "🔧 Installing pre-commit..."
if ! command -v pre-commit &>/dev/null; then
    pip_install pre-commit
fi
pre-commit install

# Validate installation
echo "✅ Validating installation..."
PY="python"
if [ "$USE_UV" -eq 1 ]; then PY="uv run python"; fi
$PY -c "import pysearch; print(f'PySearch version: {pysearch.__version__}')" || echo "⚠️  PySearch import failed (may need re-activation)"
$PY -c "from mcp.servers import pysearch_mcp_server; print('MCP server available')" || echo "⚠️  MCP import failed (non-critical)"

echo ""
echo "✅ Dev install completed successfully!"
echo ""
if [ "$USE_UV" -eq 1 ]; then
    echo "Next steps (uv detected):"
    echo "  uv run scripts/tasks.py test        # Run tests"
    echo "  uv run scripts/tasks.py lint        # Run linting"
    echo "  uv run scripts/tasks.py validate    # Validate project"
    echo "  uv run scripts/tasks.py docs serve  # Serve docs locally"
else
    echo "Next steps:"
    echo "  python scripts/tasks.py test        # Run tests"
    echo "  python scripts/tasks.py lint        # Run linting"
    echo "  python scripts/tasks.py validate    # Validate project"
    echo "  python scripts/tasks.py docs serve  # Serve docs locally"
fi