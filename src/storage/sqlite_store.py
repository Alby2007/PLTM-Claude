"""SQLite-based graph store with JSON support for memory atoms"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from uuid import UUID

import aiosqlite
from loguru import logger

from src.core.models import GraphType, MemoryAtom, Provenance


class SQLiteGraphStore:
    """
    SQLite with JSON support for graph-like operations.
    
    Features:
    - JSON metadata storage for complex fields
    - FTS5 full-text search on object field
    - Indexes for conflict detection (subject + predicate)
    - Graph-ready schema for future Neo4j migration
    """

    def __init__(self, db_path: Union[Path, str]) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Establish database connection"""
        # Create parent directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        
        await self._setup_schema()
        logger.info(f"Connected to database: {self.db_path}")

    async def close(self) -> None:
        """Close database connection"""
        if self._conn:
            await self._conn.close()
            logger.info("Database connection closed")

    async def _setup_schema(self) -> None:
        """Create tables with JSON support and graph-ready structure"""
        if not self._conn:
            raise RuntimeError("Database not connected")

        # Main atoms table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS atoms (
                id TEXT PRIMARY KEY,
                atom_type TEXT NOT NULL,
                graph TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                
                -- Store complex fields as JSON
                metadata TEXT NOT NULL,
                
                -- For fast queries (denormalized from metadata)
                confidence REAL DEFAULT 0.5,
                first_observed INTEGER DEFAULT 0,
                last_accessed INTEGER DEFAULT 0,
                
                UNIQUE(subject, predicate, object, graph)
            )
        """)

        # Create indexes for common queries
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_atoms_subject_predicate
            ON atoms(subject, predicate)
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_atoms_graph
            ON atoms(graph) WHERE graph = 'substantiated'
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_atoms_confidence
            ON atoms(confidence DESC)
        """)

        # Full-text search for object field
        await self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS atoms_fts
            USING fts5(id UNINDEXED, object, content=atoms, content_rowid=rowid)
        """)

        # Triggers to keep FTS in sync
        await self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS atoms_fts_insert AFTER INSERT ON atoms BEGIN
                INSERT INTO atoms_fts(rowid, id, object) VALUES (new.rowid, new.id, new.object);
            END
        """)

        await self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS atoms_fts_delete AFTER DELETE ON atoms BEGIN
                DELETE FROM atoms_fts WHERE rowid = old.rowid;
            END
        """)

        await self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS atoms_fts_update AFTER UPDATE ON atoms BEGIN
                DELETE FROM atoms_fts WHERE rowid = old.rowid;
                INSERT INTO atoms_fts(rowid, id, object) VALUES (new.rowid, new.id, new.object);
            END
        """)

        await self._conn.commit()
        logger.debug("Database schema initialized")

    async def insert_atom(self, atom: MemoryAtom) -> None:
        """Insert or update atom with JSON serialization"""
        if not self._conn:
            raise RuntimeError("Database not connected")

        # Serialize complex fields to JSON
        metadata = {
            "provenance": atom.provenance.value,
            "assertion_count": atom.assertion_count,
            "explicit_confirms": [dt.isoformat() for dt in atom.explicit_confirms],
            "last_contradicted": (
                atom.last_contradicted.isoformat() if atom.last_contradicted else None
            ),
            "strength": atom.strength,
            "jury_history": [j.model_dump(mode='json') for j in atom.jury_history],
            "contexts": atom.contexts,
            "flags": atom.flags,
            "related_atoms": [str(aid) for aid in atom.related_atoms],
            "supersedes": str(atom.supersedes) if atom.supersedes else None,
            "superseded_by": str(atom.superseded_by) if atom.superseded_by else None,
            "session_id": atom.session_id,
            "topic_cluster": atom.topic_cluster,
            "object_metadata": atom.object_metadata,
            "temporal_validity": atom.temporal_validity,
            "access_count": atom.access_count,
        }

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO atoms 
            (id, atom_type, graph, subject, predicate, object, metadata, 
             confidence, first_observed, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(atom.id),
                atom.atom_type.value,
                atom.graph.value,
                atom.subject,
                atom.predicate,
                atom.object,
                json.dumps(metadata),
                atom.confidence,
                int(atom.first_observed.timestamp()),
                int(atom.last_accessed.timestamp()),
            ),
        )

        await self._conn.commit()
        logger.debug(f"Inserted atom {atom.id}: [{atom.subject}] [{atom.predicate}] [{atom.object}]")

    async def get_atom(self, atom_id: UUID) -> Optional[MemoryAtom]:
        """Retrieve atom by ID"""
        if not self._conn:
            raise RuntimeError("Database not connected")

        cursor = await self._conn.execute(
            "SELECT * FROM atoms WHERE id = ?",
            (str(atom_id),)
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_atom(row)

    async def find_by_triple(
        self, 
        subject: str, 
        predicate: str,
        exclude_historical: bool = True
    ) -> list[MemoryAtom]:
        """
        Find atoms matching subject + predicate (for conflict detection).
        
        This is the critical query for conflict detection.
        """
        if not self._conn:
            raise RuntimeError("Database not connected")

        if exclude_historical:
            query = """
                SELECT * FROM atoms
                WHERE subject = ? AND predicate = ?
                AND graph != 'historical'
                ORDER BY confidence DESC
            """
        else:
            query = """
                SELECT * FROM atoms
                WHERE subject = ? AND predicate = ?
                ORDER BY confidence DESC
            """

        cursor = await self._conn.execute(query, (subject, predicate))
        rows = await cursor.fetchall()

        return [self._row_to_atom(row) for row in rows]

    async def get_substantiated_atoms(
        self, 
        subject: Optional[str] = None,
        limit: Optional[int] = None
    ) -> list[MemoryAtom]:
        """
        Get substantiated knowledge (for prompt context).
        
        Ordered by confidence and recency.
        """
        if not self._conn:
            raise RuntimeError("Database not connected")

        if subject:
            query = """
                SELECT * FROM atoms
                WHERE graph = 'substantiated' AND subject = ?
                ORDER BY confidence DESC, last_accessed DESC
            """
            params = (subject,)
        else:
            query = """
                SELECT * FROM atoms
                WHERE graph = 'substantiated'
                ORDER BY confidence DESC, last_accessed DESC
            """
            params = ()

        if limit:
            query += f" LIMIT {limit}"

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()

        return [self._row_to_atom(row) for row in rows]

    async def get_unsubstantiated_atoms(
        self,
        subject: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> list[MemoryAtom]:
        """Get unsubstantiated atoms (shadow buffer)"""
        if not self._conn:
            raise RuntimeError("Database not connected")

        if subject:
            query = """
                SELECT * FROM atoms
                WHERE graph = 'unsubstantiated' 
                AND subject = ?
                AND confidence >= ?
                ORDER BY confidence DESC
            """
            params = (subject, min_confidence)
        else:
            query = """
                SELECT * FROM atoms
                WHERE graph = 'unsubstantiated'
                AND confidence >= ?
                ORDER BY confidence DESC
            """
            params = (min_confidence,)

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()

        return [self._row_to_atom(row) for row in rows]

    async def promote_atom(self, atom_id: UUID) -> None:
        """Move atom to substantiated graph"""
        if not self._conn:
            raise RuntimeError("Database not connected")

        await self._conn.execute(
            """
            UPDATE atoms 
            SET graph = 'substantiated',
                confidence = 1.0
            WHERE id = ?
            """,
            (str(atom_id),)
        )

        await self._conn.commit()
        logger.info(f"Promoted atom {atom_id} to substantiated graph")

    async def archive_atom(self, atom_id: UUID) -> None:
        """Move atom to historical archive"""
        if not self._conn:
            raise RuntimeError("Database not connected")

        await self._conn.execute(
            "UPDATE atoms SET graph = 'historical' WHERE id = ?",
            (str(atom_id),)
        )

        await self._conn.commit()
        logger.info(f"Archived atom {atom_id}")

    async def delete_atom(self, atom_id: UUID) -> None:
        """Permanently delete atom (use sparingly)"""
        if not self._conn:
            raise RuntimeError("Database not connected")

        await self._conn.execute(
            "DELETE FROM atoms WHERE id = ?",
            (str(atom_id),)
        )

        await self._conn.commit()
        logger.warning(f"Deleted atom {atom_id}")

    async def link_atoms(
        self, 
        source_id: UUID, 
        target_id: UUID
    ) -> None:
        """
        Create bidirectional relationship (cached in metadata).
        
        Updates both atoms' related_atoms lists.
        """
        if not self._conn:
            raise RuntimeError("Database not connected")

        # Update source atom
        await self._conn.execute(
            """
            UPDATE atoms 
            SET metadata = json_set(
                metadata, 
                '$.related_atoms', 
                json_insert(
                    COALESCE(json_extract(metadata, '$.related_atoms'), '[]'),
                    '$[#]',
                    ?
                )
            )
            WHERE id = ?
            """,
            (str(target_id), str(source_id))
        )

        # Update target atom
        await self._conn.execute(
            """
            UPDATE atoms 
            SET metadata = json_set(
                metadata, 
                '$.related_atoms', 
                json_insert(
                    COALESCE(json_extract(metadata, '$.related_atoms'), '[]'),
                    '$[#]',
                    ?
                )
            )
            WHERE id = ?
            """,
            (str(source_id), str(target_id))
        )

        await self._conn.commit()
        logger.debug(f"Linked atoms {source_id} <-> {target_id}")

    async def get_related_atoms(
        self, 
        atom_id: UUID,
        depth: int = 1
    ) -> list[MemoryAtom]:
        """
        Get atoms related to this atom (using cached related_atoms).
        
        For MVP: depth=1 only (direct relationships).
        For full version: Use recursive CTE for depth>1.
        """
        if not self._conn:
            raise RuntimeError("Database not connected")

        if depth > 1:
            logger.warning("Depth > 1 not implemented in MVP, using depth=1")

        query = """
            SELECT a.*
            FROM atoms a
            WHERE a.id IN (
                SELECT value
                FROM json_each(
                    (SELECT json_extract(metadata, '$.related_atoms')
                     FROM atoms WHERE id = ?)
                )
            )
        """

        cursor = await self._conn.execute(query, (str(atom_id),))
        rows = await cursor.fetchall()

        return [self._row_to_atom(row) for row in rows]

    async def search_objects(self, query: str, limit: int = 10) -> list[MemoryAtom]:
        """Full-text search on object field using FTS5"""
        if not self._conn:
            raise RuntimeError("Database not connected")

        cursor = await self._conn.execute(
            """
            SELECT a.*
            FROM atoms a
            JOIN atoms_fts fts ON a.id = fts.id
            WHERE atoms_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit)
        )
        rows = await cursor.fetchall()

        return [self._row_to_atom(row) for row in rows]

    async def get_stats(self) -> dict[str, int]:
        """Get database statistics"""
        if not self._conn:
            raise RuntimeError("Database not connected")

        stats = {}

        # Count by graph type
        for graph_type in ["substantiated", "unsubstantiated", "historical"]:
            cursor = await self._conn.execute(
                "SELECT COUNT(*) FROM atoms WHERE graph = ?",
                (graph_type,)
            )
            row = await cursor.fetchone()
            stats[f"{graph_type}_count"] = row[0] if row else 0

        # Total atoms
        cursor = await self._conn.execute("SELECT COUNT(*) FROM atoms")
        row = await cursor.fetchone()
        stats["total_atoms"] = row[0] if row else 0

        return stats

    def _row_to_atom(self, row: aiosqlite.Row) -> MemoryAtom:
        """Convert database row to MemoryAtom"""
        metadata = json.loads(row["metadata"])

        # Parse datetime fields
        explicit_confirms = [
            datetime.fromisoformat(dt) for dt in metadata.get("explicit_confirms", [])
        ]
        last_contradicted = (
            datetime.fromisoformat(metadata["last_contradicted"])
            if metadata.get("last_contradicted")
            else None
        )

        # Parse UUIDs
        related_atoms = [UUID(aid) for aid in metadata.get("related_atoms", [])]
        supersedes = UUID(metadata["supersedes"]) if metadata.get("supersedes") else None
        superseded_by = UUID(metadata["superseded_by"]) if metadata.get("superseded_by") else None

        # Parse jury history
        from src.core.models import JuryDecision
        jury_history = [JuryDecision(**jd) for jd in metadata.get("jury_history", [])]

        return MemoryAtom(
            id=UUID(row["id"]),
            atom_type=row["atom_type"],
            graph=row["graph"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            object_metadata=metadata.get("object_metadata"),
            contexts=metadata.get("contexts", []),
            temporal_validity=metadata.get("temporal_validity"),
            provenance=Provenance(metadata["provenance"]),
            assertion_count=metadata.get("assertion_count", 1),
            explicit_confirms=explicit_confirms,
            first_observed=datetime.fromtimestamp(row["first_observed"]),
            last_contradicted=last_contradicted,
            strength=metadata.get("strength", 0.5),
            last_accessed=datetime.fromtimestamp(row["last_accessed"]),
            access_count=metadata.get("access_count", 0),
            supersedes=supersedes,
            superseded_by=superseded_by,
            related_atoms=related_atoms,
            confidence=row["confidence"],
            jury_history=jury_history,
            flags=metadata.get("flags", []),
            session_id=metadata.get("session_id", ""),
            topic_cluster=metadata.get("topic_cluster"),
        )
