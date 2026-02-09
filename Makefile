# Common variables - auto-detect uv for faster operations
UV := $(shell command -v uv 2>/dev/null)
ifdef UV
  RUN := uv run
  PYTHON := uv run python3
  PIP_INSTALL := uv pip install
  TASK := uv run scripts/tasks.py
else
  RUN :=
  PYTHON := python3
  PIP_INSTALL := python3 -m pip install
  TASK := python3 scripts/tasks.py
endif
PKG := pysearch

.PHONY: help install dev lint format type test cov htmlcov bench clean pre-commit hooks docs docs-serve docs-clean docs-check docs-deploy serve release mcp-servers check-structure validate tasks

help:
	@echo "Targets:"
	@echo "  install     - Install package"
	@echo "  dev         - Install dev dependencies"
	@echo "  lint        - Run ruff lint and format check"
	@echo "  format      - Run black and ruff --fix"
	@echo "  type        - Run mypy"
	@echo "  test        - Run pytest with coverage"
	@echo "  cov         - Show coverage summary"
	@echo "  htmlcov     - Open HTML coverage"
	@echo "  bench       - Run pytest benchmarks"
	@echo "  clean       - Cleanup build/test artifacts and cache files"
	@echo "  pre-commit  - Run pre-commit on all files"
	@echo "  hooks       - Install pre-commit hooks"
	@echo "  docs        - Build docs with mkdocs"
	@echo "  docs-serve  - Serve docs locally for development"
	@echo "  docs-clean  - Clean documentation build artifacts"
	@echo "  docs-check  - Check documentation for issues"
	@echo "  docs-deploy - Deploy documentation (requires setup)"
	@echo "  release     - Build and publish (requires env vars)"
	@echo "  mcp-servers - Test MCP servers"
	@echo "  check-structure - Validate project structure"
	@echo "  validate    - Run all validation checks"
	@echo "  tasks       - Show cross-platform task runner usage"
ifdef UV
	@echo ""
	@echo "  [uv detected - using uv for faster operations]"
endif

install:
	$(PIP_INSTALL) -U pip
	$(PIP_INSTALL) -e .

dev:
	$(PIP_INSTALL) -U pip
	$(PIP_INSTALL) -e ".[dev]"
	$(PIP_INSTALL) pre-commit

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

format:
	$(PYTHON) -m black .
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m ruff format .

type:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest

cov:
	@grep -E "TOTAL|TOTAL.*[0-9]+\.[0-9]+%|^" coverage.xml >/dev/null 2>&1 || true
	@echo "See coverage.xml and terminal output for details."

htmlcov:
	$(PYTHON) -m coverage html
	@echo "Open htmlcov/index.html"

bench:
	$(PYTHON) -m pytest -q -k benchmark

clean:
	rm -rf build/ dist/ sdist/ wheels/
	rm -rf .pytest_cache/ .mypy_cache/ .mypy_cache_temp/ .ruff_cache/ .coverage/ htmlcov/ .benchmarks/
	rm -rf .pysearch-cache/ coverage.xml
	rm -rf site/ .mkdocs_cache/ docs_build/ .docstring_cache/ .api_docs_cache/
	rm -rf src/*.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

pre-commit:
	pre-commit run --all-files

hooks:
	pre-commit install

docs:
	@echo "Building documentation..."
	mkdocs build --clean --strict
	@echo "✅ Documentation built successfully in site/"

docs-serve:
	@echo "Starting documentation server..."
	@echo "📖 Documentation will be available at http://localhost:8000"
	mkdocs serve -a 0.0.0.0:8000 --dev-addr=localhost:8000

docs-clean:
	@echo "Cleaning documentation build artifacts..."
	rm -rf site/
	rm -rf .mkdocs_cache/
	rm -rf docs_build/
	@echo "✅ Documentation artifacts cleaned"

docs-check:
	@echo "Checking documentation for issues..."
	mkdocs build --clean --strict --verbose
	@echo "Checking for broken links..."
	@if command -v linkchecker >/dev/null 2>&1; then \
		linkchecker site/index.html; \
	else \
		echo "⚠️  linkchecker not installed, skipping link check"; \
		echo "   Install with: pip install linkchecker"; \
	fi
	@echo "✅ Documentation check completed"

docs-deploy:
	@echo "Deploying documentation..."
	@if [ -z "$$GITHUB_TOKEN" ]; then \
		echo "❌ GITHUB_TOKEN environment variable required for deployment"; \
		exit 1; \
	fi
	mkdocs gh-deploy --clean --message "Deploy documentation [skip ci]"
	@echo "✅ Documentation deployed to GitHub Pages"

# Legacy alias for backwards compatibility
serve: docs-serve

release:
	$(PYTHON) -m pip install -U build twine
	rm -rf dist/ build/
	$(PYTHON) -m build
	twine check dist/*
	@echo "To upload: TWINE_USERNAME=__token__ TWINE_PASSWORD=***** twine upload dist/*"

mcp-servers:
	@echo "Testing MCP servers..."
	$(PYTHON) -c "import mcp.servers.pysearch_mcp_server; print('✅ PySearch MCP server imports successfully')"
	@echo "✅ All MCP servers validated"

check-structure:
	@echo "Checking project structure..."
	@test -d src/pysearch || (echo "❌ src/pysearch directory missing" && exit 1)
	@test -d mcp/servers || (echo "❌ mcp/servers directory missing" && exit 1)
	@test -d mcp/shared || (echo "❌ mcp/shared directory missing" && exit 1)
	@test -d tests || (echo "❌ tests directory missing" && exit 1)
	@test -d docs || (echo "❌ docs directory missing" && exit 1)
	@test -d scripts || (echo "❌ scripts directory missing" && exit 1)
	@test -f pyproject.toml || (echo "❌ pyproject.toml missing" && exit 1)
	@test -f README.md || (echo "❌ README.md missing" && exit 1)
	@test -f mcp/README.md || (echo "❌ mcp/README.md missing" && exit 1)
	@echo "✅ Project structure is valid"

validate: lint type test mcp-servers check-structure
	@echo "✅ All validation checks passed"

tasks:
	@echo "Cross-platform task runner (works on Windows, Linux, macOS):"
	@echo ""
	@echo "  With uv (recommended):"
	@echo "    uv run scripts/tasks.py help"
	@echo "    uv run scripts/tasks.py lint"
	@echo "    uv run scripts/tasks.py test --coverage"
	@echo "    uv run scripts/tasks.py docs serve"
	@echo "    uv run scripts/tasks.py clean"
	@echo "    uv run scripts/tasks.py validate --full"
	@echo ""
	@echo "  Without uv:"
	@echo "    python scripts/tasks.py help"
	@echo ""
	@echo "  Current TASK = $(TASK)"