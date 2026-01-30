.PHONY: setup test run clean help

help:
	@echo "Procedural LTM MVP - Available commands:"
	@echo "  make setup    - Set up virtual environment and dependencies"
	@echo "  make test     - Run test suite with coverage"
	@echo "  make run      - Start FastAPI server"
	@echo "  make clean    - Clean up generated files"
	@echo "  make format   - Format code with black"
	@echo "  make lint     - Lint code with ruff"

setup:
	@bash scripts/setup.sh

test:
	@bash scripts/run_tests.sh

run:
	@bash scripts/start_api.sh

clean:
	@echo "🧹 Cleaning up..."
	@rm -rf __pycache__ .pytest_cache .coverage htmlcov
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

format:
	@echo "🎨 Formatting code..."
	@black src/ tests/
	@echo "✅ Formatting complete"

lint:
	@echo "🔍 Linting code..."
	@ruff check src/ tests/
	@echo "✅ Linting complete"
