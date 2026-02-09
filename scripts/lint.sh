#!/usr/bin/env bash
set -euo pipefail

# Auto-detect uv for consistent environment
if command -v uv &>/dev/null; then
    PY="uv run python"
else
    PY="python"
fi

FAILED=0

echo "🔍 Running ruff check..."
$PY -m ruff check . || FAILED=1

echo "🎨 Running black format check..."
$PY -m black --check . || FAILED=1

echo "🔎 Running mypy type check..."
# Uses [tool.mypy] config from pyproject.toml (which defines files and excludes)
$PY -m mypy || FAILED=1

if [ "$FAILED" -ne 0 ]; then
    echo "❌ Lint checks failed."
    exit 1
fi
echo "✅ Lint, format check and type check passed."