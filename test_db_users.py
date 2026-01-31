"""Check what users exist in the database"""
import asyncio
import sys
sys.path.insert(0, '.')

from src.storage.sqlite_store import SQLiteGraphStore

async def test():
    store = SQLiteGraphStore('mcp_server/pltm_mcp.db')
    await store.connect()
    
    cursor = await store._conn.execute(
        "SELECT DISTINCT subject, graph, COUNT(*) FROM atoms GROUP BY subject, graph"
    )
    rows = await cursor.fetchall()
    print("Users in database:")
    for row in rows:
        print(f"  {row[0]} ({row[1]}): {row[2]} atoms")
    
    await store.close()

asyncio.run(test())
