#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/format.sh [--check]
#   --check    Check formatting without modifying files (dry-run)

# Auto-detect uv for consistent environment
if command -v uv &>/dev/null; then
    PY="uv run python"
else
    PY="python"
fi

if [[ "${1:-}" == "--check" ]]; then
    echo "🔍 Checking code formatting (dry-run)..."
    FAILED=0
    $PY -m black --check --diff . || FAILED=1
    $PY -m ruff check --diff . || FAILED=1
    $PY -m ruff format --check . || FAILED=1
    if [ "$FAILED" -ne 0 ]; then
        echo "❌ Code is not properly formatted. Run './scripts/format.sh' to fix."
        exit 1
    fi
    echo "✅ Code is properly formatted."
else
    echo "🎨 Formatting code..."
    $PY -m black .
    $PY -m ruff check . --fix
    $PY -m ruff format .
    echo "✅ Code formatted."
fi