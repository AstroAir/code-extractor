# Common variables
PYTHON := python3
PKG := pysearch

.PHONY: help install dev lint format type test cov htmlcov bench clean pre-commit hooks docs serve release

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
	@echo "  clean       - Cleanup build/test artifacts"
	@echo "  pre-commit  - Run pre-commit on all files"
	@echo "  hooks       - Install pre-commit hooks"
	@echo "  docs        - Build docs with mkdocs"
	@echo "  serve       - Serve docs locally"
	@echo "  release     - Build and publish (requires env vars)"

install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pip install pre-commit

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

format:
	$(PYTHON) -m black .
	$(PYTHON) -m ruff check . --fix

type:
	$(PYTHON) -m mypy src tests

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
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage/ htmlcov/ .benchmarks/
	find . -type d -name "__pycache__" -exec rm -rf {} +

pre-commit:
	pre-commit run --all-files

hooks:
	pre-commit install

docs:
	mkdocs build

serve:
	mkdocs serve -a 0.0.0.0:8000

release:
	$(PYTHON) -m pip install -U build twine
	rm -rf dist/ build/
	$(PYTHON) -m build
	twine check dist/*
	@echo "To upload: TWINE_USERNAME=__token__ TWINE_PASSWORD=***** twine upload dist/*"