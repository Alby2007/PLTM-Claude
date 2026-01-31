#!/usr/bin/env python3
"""
PLTM Self-Improvement CLI Tool

Run recursive self-improvement cycles directly.
Usage: python tools/self_improve.py [cycle|meta|history]
"""

import asyncio
import sys
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.sqlite_store import SQLiteGraphStore
from src.meta.recursive_improvement import RecursiveSelfImprovement


async def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            "usage": "python self_improve.py [cycle|meta|history]",
            "commands": {
                "cycle": "Run one improvement cycle",
                "meta": "Meta-learn from improvements", 
                "history": "Show improvement history"
            }
        }))
        return
    
    command = sys.argv[1]
    
    # Connect to same DB as MCP server
    db_path = Path(__file__).parent.parent / "mcp_server" / "pltm_mcp.db"
    store = SQLiteGraphStore(str(db_path))
    await store.connect()
    
    improver = RecursiveSelfImprovement(store)
    
    if command == "cycle":
        result = await improver.run_improvement_cycle()
        print(json.dumps(result))
    
    elif command == "meta":
        result = await improver.meta_learn()
        print(json.dumps(result))
    
    elif command == "history":
        result = await improver.get_improvement_history()
        print(json.dumps(result))
    
    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))


if __name__ == "__main__":
    asyncio.run(main())
