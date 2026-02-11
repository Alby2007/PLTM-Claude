"""
Embedding-based Semantic Search for PLTM Typed Memories

Uses sentence-transformers (all-MiniLM-L6-v2) to generate embeddings
and stores them in SQLite as binary blobs. Cosine similarity search
finds semantically related memories even without keyword overlap.

Example: searching "coding style" will also find memories about
"programming preferences" and "development workflow".
"""

import json
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite
import numpy as np
from loguru import logger


# Lazy-load the model to avoid slow import on startup
_model = None


def _get_model():
    """Lazy-load sentence-transformers model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded embedding model: all-MiniLM-L6-v2 (384 dims)")
    return _model


def embed_text(text: str) -> np.ndarray:
    """Generate embedding for a text string. Returns 384-dim float32 vector."""
    model = _get_model()
    return model.encode(text, normalize_embeddings=True)


def embed_batch(texts: List[str]) -> np.ndarray:
    """Generate embeddings for multiple texts. Returns (N, 384) array."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, batch_size=64)


def _vec_to_bytes(vec: np.ndarray) -> bytes:
    """Serialize float32 vector to bytes for SQLite storage."""
    return vec.astype(np.float32).tobytes()


def _bytes_to_vec(data: bytes) -> np.ndarray:
    """Deserialize bytes back to float32 vector."""
    return np.frombuffer(data, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (already normalized → dot product)."""
    return float(np.dot(a, b))


async def embed_text_async(text: str) -> np.ndarray:
    """Non-blocking version of embed_text — runs model in thread pool."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_text, text)


async def embed_batch_async(texts: list) -> np.ndarray:
    """Non-blocking version of embed_batch — runs model in thread pool."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_batch, texts)


class EmbeddingStore:
    """
    Manages embeddings for typed memories in SQLite.
    
    Schema: memory_embeddings(memory_id TEXT PK, embedding BLOB, content_hash TEXT)
    
    Embeddings are generated on store and searched via brute-force cosine
    similarity (fast enough for <100K memories). For larger scale, swap
    to FAISS or Annoy.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
    
    async def connect(self):
        """Connect and create schema."""
        self._conn = await aiosqlite.connect(self.db_path)
        await self._setup_schema()
    
    async def close(self):
        if self._conn:
            await self._conn.close()
    
    async def _setup_schema(self):
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                content_hash TEXT NOT NULL,
                indexed_at REAL NOT NULL
            )
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_emb_content_hash
            ON memory_embeddings(content_hash)
        """)
        await self._conn.commit()
        logger.info("EmbeddingStore schema initialized")
    
    # ========== INDEX ==========
    
    async def index_memory(self, memory_id: str, content: str) -> bool:
        """
        Generate and store embedding for a memory.
        Skips if content hasn't changed (same hash).
        Returns True if newly indexed.
        """
        content_hash = str(hash(content))
        
        # Check if already indexed with same content
        cursor = await self._conn.execute(
            "SELECT content_hash FROM memory_embeddings WHERE memory_id = ?",
            (memory_id,)
        )
        row = await cursor.fetchone()
        if row and row[0] == content_hash:
            return False  # Already indexed, content unchanged
        
        # Generate embedding (non-blocking — runs in thread pool)
        vec = await embed_text_async(content)
        blob = _vec_to_bytes(vec)
        
        await self._conn.execute(
            """INSERT OR REPLACE INTO memory_embeddings 
               (memory_id, embedding, content_hash, indexed_at)
               VALUES (?, ?, ?, ?)""",
            (memory_id, blob, content_hash, time.time())
        )
        await self._conn.commit()
        return True
    
    async def index_batch(self, memories: List[Tuple[str, str]]) -> int:
        """
        Batch index multiple memories. Input: [(memory_id, content), ...]
        Returns count of newly indexed.
        """
        if not memories:
            return 0
        
        # Filter out already-indexed with same content
        to_index = []
        for mem_id, content in memories:
            content_hash = str(hash(content))
            cursor = await self._conn.execute(
                "SELECT content_hash FROM memory_embeddings WHERE memory_id = ?",
                (mem_id,)
            )
            row = await cursor.fetchone()
            if not row or row[0] != content_hash:
                to_index.append((mem_id, content, content_hash))
        
        if not to_index:
            return 0
        
        # Batch embed
        texts = [content for _, content, _ in to_index]
        vecs = await embed_batch_async(texts)
        
        for i, (mem_id, content, content_hash) in enumerate(to_index):
            blob = _vec_to_bytes(vecs[i])
            await self._conn.execute(
                """INSERT OR REPLACE INTO memory_embeddings
                   (memory_id, embedding, content_hash, indexed_at)
                   VALUES (?, ?, ?, ?)""",
                (mem_id, blob, content_hash, time.time())
            )
        
        await self._conn.commit()
        return len(to_index)
    
    async def remove_embedding(self, memory_id: str):
        """Remove embedding when memory is deleted."""
        await self._conn.execute(
            "DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
        )
        await self._conn.commit()
    
    # ========== SEARCH ==========
    
    async def search(
        self, query: str, limit: int = 10, min_similarity: float = 0.3,
        memory_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search: find memories most similar to query.
        
        Returns: [{"memory_id": str, "similarity": float}, ...]
        sorted by similarity descending.
        """
        query_vec = await embed_text_async(query)
        
        # Load all embeddings (brute force — fine for <100K)
        if memory_ids:
            placeholders = ",".join("?" for _ in memory_ids)
            cursor = await self._conn.execute(
                f"SELECT memory_id, embedding FROM memory_embeddings WHERE memory_id IN ({placeholders})",
                memory_ids
            )
        else:
            cursor = await self._conn.execute(
                "SELECT memory_id, embedding FROM memory_embeddings"
            )
        
        rows = await cursor.fetchall()
        
        if not rows:
            return []
        
        # Compute similarities
        results = []
        for mem_id, blob in rows:
            vec = _bytes_to_vec(blob)
            sim = cosine_similarity(query_vec, vec)
            if sim >= min_similarity:
                results.append({"memory_id": mem_id, "similarity": round(float(sim), 4)})
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]
    
    async def find_similar(
        self, memory_id: str, limit: int = 5, min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Find memories similar to a given memory."""
        cursor = await self._conn.execute(
            "SELECT embedding FROM memory_embeddings WHERE memory_id = ?",
            (memory_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return []
        
        target_vec = _bytes_to_vec(row[0])
        
        # Compare against all others
        cursor = await self._conn.execute(
            "SELECT memory_id, embedding FROM memory_embeddings WHERE memory_id != ?",
            (memory_id,)
        )
        rows = await cursor.fetchall()
        
        results = []
        for mid, blob in rows:
            vec = _bytes_to_vec(blob)
            sim = cosine_similarity(target_vec, vec)
            if sim >= min_similarity:
                results.append({"memory_id": mid, "similarity": round(float(sim), 4)})
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]
    
    # ========== STATS ==========
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get embedding index statistics."""
        cursor = await self._conn.execute("SELECT COUNT(*) FROM memory_embeddings")
        count = (await cursor.fetchone())[0]
        
        cursor = await self._conn.execute(
            "SELECT MIN(indexed_at), MAX(indexed_at) FROM memory_embeddings"
        )
        row = await cursor.fetchone()
        
        return {
            "indexed_count": count,
            "oldest_index": row[0],
            "newest_index": row[1],
            "model": "all-MiniLM-L6-v2",
            "dimensions": 384,
        }
