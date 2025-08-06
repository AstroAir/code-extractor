#!/usr/bin/env bash
set -euo pipefail
python -m ruff check .
python -m black --check .
python -m mypy src tests
echo "✅ Lint, format check and type check passed."