"""Test MMR with actual database"""
import asyncio
import sys
sys.path.insert(0, '.')

from src.storage.sqlite_store import SQLiteGraphStore

async def test():
    store = SQLiteGraphStore('mcp_server/pltm_mcp.db')
    await store.connect()
    
    # Count atoms
    cursor = await store._conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE subject = ? AND graph = ?", 
        ('alby', 'substantiated')
    )
    row = await cursor.fetchone()
    print(f"Atoms for alby (substantiated): {row[0]}")
    
    # Test the actual query
    cursor = await store._conn.execute(
        """SELECT id, predicate, object, confidence 
           FROM atoms WHERE subject = ? AND graph = 'substantiated'
           ORDER BY confidence DESC LIMIT 50""",
        ('alby',)
    )
    rows = await cursor.fetchall()
    print(f"Fetched {len(rows)} rows")
    
    if rows:
        print(f"First row: {rows[0]}")
    
    await store.close()
    print("Test passed!")

asyncio.run(test())
