"""Initialize the database schema"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings


async def init_database() -> None:
    """Create database directory and initialize schema"""
    # Create data directory if it doesn't exist
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Database will be created at: {settings.DB_PATH}")
    print("Database initialization will be implemented in storage layer")
    print("✓ Data directory created")


if __name__ == "__main__":
    asyncio.run(init_database())
