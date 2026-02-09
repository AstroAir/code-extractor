#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/test.sh [options]
#   --cov          Enable coverage reporting
#   --cov-html     Generate HTML coverage report
#   --cov-xml      Generate XML coverage report
#   -m MARKER      Run tests matching marker (e.g. unit, integration, e2e)
#   -k KEYWORD     Run tests matching keyword expression
#   -x             Stop on first failure
#   -v             Verbose output
#   Any extra args are passed directly to pytest.

# Auto-detect uv for consistent environment
if command -v uv &>/dev/null; then
    PY="uv run python"
else
    PY="python"
fi

PYTEST_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --cov)
            PYTEST_ARGS+=("--cov=src/pysearch" "--cov-report=term-missing")
            ;;
        --cov-html)
            PYTEST_ARGS+=("--cov=src/pysearch" "--cov-report=html")
            ;;
        --cov-xml)
            PYTEST_ARGS+=("--cov=src/pysearch" "--cov-report=xml")
            ;;
        --help|-h)
            head -n 13 "$0" | tail -n 11
            exit 0
            ;;
        *)
            PYTEST_ARGS+=("$arg")
            ;;
    esac
done

$PY -m pytest "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"
echo "✅ Tests passed."