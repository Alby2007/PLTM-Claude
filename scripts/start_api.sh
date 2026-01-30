#!/bin/bash
# Start FastAPI server

set -e

echo "🚀 Starting Procedural LTM API..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start uvicorn
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
