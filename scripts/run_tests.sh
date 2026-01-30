#!/bin/bash
# Run test suite

set -e

echo "🧪 Running Procedural LTM test suite..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

echo ""
echo "✅ Tests complete!"
echo "📊 Coverage report generated in htmlcov/index.html"
