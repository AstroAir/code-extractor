.PHONY: run test lint clean

run:
	uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache
