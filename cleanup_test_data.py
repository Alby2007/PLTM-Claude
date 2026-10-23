"""One-off cleanup: remove verification test data from meta_judge_* tables.

Removes the jury events + feedback rows Claude generated while verifying
the jury changes live, then decrements meta_judge_stats by exactly what
those deleted events contributed (totals and verdict counts drop by 8).

Usage:
    python cleanup_test_data.py            # dry run — show what would change
    python cleanup_test_data.py --apply    # delete and decrement

Run while the MCP server is stopped (or accept a brief WAL lock retry).
"""

import sqlite3
import sys

DB_PATH = "data/pltm_mcp.db"

# Test events: sha1 of the two throwaway memories Claude stored
HASH_PREFIX = "885dab04b9%"
HASH_FULL = "a198bcf3e24807a67f07a12fda7f6033b021a276"
EVENT_WHERE = "(content_hash LIKE ? OR content_hash = ?)"
EVENT_PARAMS = (HASH_PREFIX, HASH_FULL)
FEEDBACK_WHERE = "details LIKE 'ZZ-TEST%'"


def out(s):
    sys.stdout.buffer.write((str(s) + "\n").encode("utf-8", "replace"))


def main():
    apply = "--apply" in sys.argv
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")

    rows = conn.execute(
        f"SELECT id, judge_name, verdict, confidence, latency_ms, "
        f"memory_type, ground_truth, is_correct FROM meta_judge_events "
        f"WHERE {EVENT_WHERE}", EVENT_PARAMS).fetchall()
    fb_rows = conn.execute(
        f"SELECT id, memory_id, feedback_type FROM meta_judge_feedback "
        f"WHERE {FEEDBACK_WHERE}").fetchall()

    out(f"{'APPLYING' if apply else 'DRY RUN'} — test data in {DB_PATH}")
    out(f"  events to delete:  {len(rows)} (ids "
        f"{[r[0] for r in rows]})")
    out(f"  feedback to delete: {len(fb_rows)} (ids "
        f"{[r[0] for r in fb_rows]})")

    # Contribution each deleted event made to meta_judge_stats at track()
    # time — per (_all + its memory_type). correct was incremented only for
    # insert-time ground truth matches; feedback-stamped is_correct must NOT
    # be decremented (it was never added to stats.correct).
    deltas = {}
    for _, judge, verdict, conf, lat, mtype, gtruth, _isc in rows:
        for mt in ("_all", mtype or "_all"):
            d = deltas.setdefault((judge, mt), {
                "total": 0, "correct": 0, "conflicts": 0,
                "latency": 0.0, "conf_sum": 0.0,
                "verdict_approve": 0, "verdict_quarantine": 0,
                "verdict_reject": 0})
            d["total"] += 1
            d[f"verdict_{verdict.lower()}"] += 1
            d["conflicts"] += 1 if verdict in ("REJECT", "QUARANTINE") else 0
            d["correct"] += 1 if (gtruth and verdict == gtruth) else 0
            d["latency"] += lat or 0.0
            d["conf_sum"] += conf or 0.0

    out("  stats decrements:")
    for (jn, mt), d in sorted(deltas.items()):
        out(f"    {jn}/{mt}: total -{d['total']}, "
            f"approve -{d['verdict_approve']}, "
            f"quarantine -{d['verdict_quarantine']}, "
            f"reject -{d['verdict_reject']}")

    if not apply:
        out("Re-run with --apply to delete.")
        conn.close()
        return

    conn.execute(f"DELETE FROM meta_judge_events WHERE {EVENT_WHERE}",
                 EVENT_PARAMS)
    conn.execute(f"DELETE FROM meta_judge_feedback WHERE {FEEDBACK_WHERE}")

    for (jn, mt), d in deltas.items():
        conn.execute("""
            UPDATE meta_judge_stats SET
                total = total - ?, correct = correct - ?,
                conflicts_detected = conflicts_detected - ?,
                total_latency_ms = total_latency_ms - ?,
                confidence_sum = confidence_sum - ?,
                verdict_approve = verdict_approve - ?,
                verdict_quarantine = verdict_quarantine - ?,
                verdict_reject = verdict_reject - ?
            WHERE judge_name = ? AND memory_type = ?
        """, (d["total"], d["correct"], d["conflicts"], d["latency"],
              d["conf_sum"], d["verdict_approve"], d["verdict_quarantine"],
              d["verdict_reject"], jn, mt))
    conn.commit()

    # Verify: exactly the 4,160 real events remain, ids 1..4160 untouched
    n, lo, hi = conn.execute(
        "SELECT COUNT(*), MIN(id), MAX(id) FROM meta_judge_events").fetchone()
    gaps = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT id, LAG(id) OVER (ORDER BY id) prev
            FROM meta_judge_events) WHERE id != prev + 1 AND prev IS NOT NULL
    """).fetchone()[0]
    leftover = conn.execute(
        f"SELECT COUNT(*) FROM meta_judge_events WHERE {EVENT_WHERE}",
        EVENT_PARAMS).fetchone()[0]
    fb_left = conn.execute(
        f"SELECT COUNT(*) FROM meta_judge_feedback WHERE {FEEDBACK_WHERE}"
    ).fetchone()[0]
    out(f"  result: {n} events (ids {lo}..{hi}, gaps={gaps}), "
        f"test rows left: events={leftover} feedback={fb_left}")
    out(f"  stats totals now: "
        f"{conn.execute('SELECT judge_name, total FROM meta_judge_stats WHERE memory_type=%s' % repr('_all')).fetchall()}")
    conn.close()


if __name__ == "__main__":
    main()
