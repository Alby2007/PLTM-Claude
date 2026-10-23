"""One-off: restore typed-memory strengths damaged by the edit-decay bug.

get()/query() used to rewrite mem.strength to the floor-clamped decayed
value, and update_belief/correct_memory persisted it. This script restores
each affected row's strength from the pre-edit backup.

Usage:
    python restore_strength.py            # dry run — list candidates
    python restore_strength.py --apply    # write restored strengths
"""

import sqlite3
import sys
from pathlib import Path

LIVE = Path("data/pltm_mcp.db")
BACKUP = Path("data/pltm_mcp_backup_2026-10-23.db")


def main(apply: bool = False):
    live = sqlite3.connect(str(LIVE), timeout=10)
    backup = sqlite3.connect(str(BACKUP), timeout=10)

    old = {r[0]: r[1] for r in backup.execute(
        "SELECT id, strength FROM typed_memories").fetchall()}
    rows = live.execute(
        "SELECT id, memory_type, strength, content FROM typed_memories").fetchall()

    candidates = [
        (rid, mtype, cur, old[rid], content[:60])
        for rid, mtype, cur, content in rows
        if rid in old and cur < old[rid] - 1e-9
    ]
    candidates.sort(key=lambda r: r[2] - r[3])  # biggest drop first

    print(f"{len(candidates)} memories have lower strength than the backup:")
    for rid, mtype, cur, orig, content in candidates:
        print(f"  {rid[:12]}  {mtype:10s} {cur:.3f} -> {orig:.3f}  {content}")

    if not apply:
        print("\nDry run. Re-run with --apply to restore.")
        return

    with live:
        for rid, *_ in candidates:
            live.execute("UPDATE typed_memories SET strength = ? WHERE id = ?",
                         (old[rid], rid))
    print(f"\nRestored {len(candidates)} strengths from backup.")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    main(apply="--apply" in sys.argv)
