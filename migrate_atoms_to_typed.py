"""
Migrate 1600+ atoms from the knowledge graph into the TypedMemory system.

Mapping rules:
  - atom_type 'event' → episodic
  - atom_type 'hypothesis' → belief
  - atom_type 'interaction_pattern' → procedural
  - atom_type 'invariant' → semantic
  - atom_type 'state' (default) → mapped by predicate:
      - believes/claims/hypothesizes/suggests → belief
      - uses/requires/enables/allows/achieves → procedural
      - published/discovered/occurred/tested → episodic (research events)
      - everything else → semantic (general facts)

Runs synchronously against the DB — no embedding model needed.
"""

import sqlite3
import json
import time
import uuid
import re

DB_PATH = "data/pltm_mcp.db"

# Predicate patterns for type classification
BELIEF_PREDS = {
    'believes', 'claims', 'hypothesizes', 'suggests', 'proposes',
    'argues', 'assumes', 'predicts', 'estimates', 'speculates',
    'theorizes', 'infers', 'conjectures',
}

PROCEDURAL_PREDS = {
    'uses', 'requires', 'enables', 'allows', 'achieves',
    'used_for', 'used_by', 'used_in', 'implements', 'applies',
    'executes', 'performs', 'operates', 'triggers', 'activates',
    'measures', 'tested_on', 'compared_to',
}

EPISODIC_PREDS = {
    'published', 'published_on', 'published_in', 'discovered',
    'discovered_in', 'occurred', 'occurred_in', 'observed',
    'co-authored', 'co-authored_with', 'written_by', 'created',
    'session_count', 'arxiv_paper',
}


def classify_atom(atom_type, predicate):
    """Map atom to typed memory type."""
    if atom_type == 'event':
        return 'episodic'
    elif atom_type == 'hypothesis':
        return 'belief'
    elif atom_type == 'interaction_pattern':
        return 'procedural'
    elif atom_type == 'invariant':
        return 'semantic'

    # For 'state' type, classify by predicate
    pred_lower = predicate.lower().replace('-', '_')
    if pred_lower in BELIEF_PREDS:
        return 'belief'
    elif pred_lower in PROCEDURAL_PREDS:
        return 'procedural'
    elif pred_lower in EPISODIC_PREDS:
        return 'episodic'
    else:
        return 'semantic'


def atom_to_content(subject, predicate, obj):
    """Convert SPO triple to natural language content."""
    s = subject.replace('_', ' ')
    p = predicate.replace('_', ' ')
    o = obj.replace('_', ' ')
    return f"{s} {p} {o}"


def auto_tag(subject, predicate, obj):
    """Generate tags from the atom."""
    tags = set()
    for part in [subject, predicate]:
        clean = part.replace('_', ' ').lower().strip()
        if len(clean) > 2 and clean not in {'the', 'and', 'for', 'with', 'has', 'have', 'is', 'are'}:
            tags.add(clean)
    if len(tags) > 5:
        tags = set(list(tags)[:5])
    return list(tags)


def migrate():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Check current typed_memories count
    try:
        existing = conn.execute("SELECT COUNT(*) FROM typed_memories").fetchone()[0]
        print(f"Existing typed memories: {existing}")
    except Exception:
        print("typed_memories table doesn't exist yet — will be created by server")
        conn.close()
        return

    # Get all atoms
    atoms = conn.execute(
        "SELECT id, atom_type, subject, predicate, object, confidence, "
        "first_observed, last_accessed, metadata FROM atoms"
    ).fetchall()
    print(f"Atoms to migrate: {len(atoms)}")

    # Check for already-migrated atoms (by content match)
    existing_contents = set()
    try:
        rows = conn.execute("SELECT content FROM typed_memories").fetchall()
        existing_contents = {r[0] for r in rows}
    except Exception:
        pass

    now = time.time()
    migrated = 0
    skipped = 0
    type_counts = {'episodic': 0, 'semantic': 0, 'belief': 0, 'procedural': 0}

    for atom in atoms:
        a = dict(atom)
        content = atom_to_content(a['subject'], a['predicate'], a['object'])

        # Skip if already migrated
        if content in existing_contents:
            skipped += 1
            continue

        mem_type = classify_atom(a['atom_type'], a['predicate'])
        confidence = a['confidence'] or 0.5
        tags = auto_tag(a['subject'], a['predicate'], a['object'])
        created = a['first_observed'] or now
        accessed = a['last_accessed'] or created

        # For beliefs, start with moderate confidence
        if mem_type == 'belief':
            confidence = min(confidence, 0.7)

        mem_id = str(uuid.uuid4())

        conn.execute("""
            INSERT INTO typed_memories
            (id, memory_type, user_id, content, context, source,
             strength, created_at, last_accessed, access_count,
             confidence, evidence_for, evidence_against,
             episode_timestamp, participants, emotional_valence,
             trigger, action, success_count, failure_count,
             consolidated_from, consolidation_count, tags)
            VALUES (?,?,?,?,?,?, ?,?,?,?, ?,?,?, ?,?,?, ?,?,?,?, ?,?,?)
        """, (
            mem_id, mem_type, 'claude', content,
            f"migrated from atom {a['id'][:8]}", 'atom_migration',
            0.8, created, accessed, 0,
            confidence, '[]', '[]',
            created if mem_type == 'episodic' else 0,
            '[]', 0.0,
            '', '', 0, 0,
            '[]', 0,
            json.dumps(tags),
        ))

        type_counts[mem_type] += 1
        migrated += 1

    conn.commit()
    conn.close()

    print(f"\nMigration complete!")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  By type:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t}: {c}")

    # Verify
    conn2 = sqlite3.connect(DB_PATH)
    total = conn2.execute("SELECT COUNT(*) FROM typed_memories").fetchone()[0]
    print(f"\n  Total typed memories now: {total}")
    conn2.close()


if __name__ == "__main__":
    migrate()
