"""
Autonomous Learning Engine

Persistent, scheduled knowledge acquisition that runs across sessions:

1. SCHEDULED INGESTION
   - Daily arXiv ingestion (consciousness, AI, complexity, IIT)
   - Weekly GitHub trending analysis (AI/ML repos)
   - News feed monitoring (HN, arXiv RSS)
   - Schedules persist to SQLite — survive restarts

2. CROSS-SESSION ACCUMULATION
   - Φ measurement after every learning run
   - Automated episodic→semantic consolidation triggers
   - Progressive knowledge graph growth metrics
   - Learning velocity tracking (facts/hour, domains/week)

3. LEARNING DIGEST
   - Generates a compact "what I learned since last session" summary
   - Tracks Φ trajectory over time
   - Identifies knowledge gaps and suggests focus areas
"""

import json
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import aiosqlite
from loguru import logger


# ── Focused arXiv topics for PLTM's research interests ──────────────────────

ARXIV_TOPICS = {
    "consciousness": {
        "query": "ti:consciousness+AND+ti:information",
        "keywords": ["consciousness", "IIT", "integrated information", "qualia",
                     "phenomenal", "global workspace", "neural correlates"],
        "priority": 1.0,
    },
    "ai_alignment": {
        "query": "ti:alignment+AND+cat:cs.AI",
        "keywords": ["alignment", "safety", "interpretability", "RLHF",
                     "constitutional AI", "reward model"],
        "priority": 0.9,
    },
    "complexity": {
        "query": "ti:self-organized+criticality+OR+ti:emergence",
        "keywords": ["complexity", "emergence", "criticality", "phase transition",
                     "power law", "scale-free"],
        "priority": 0.85,
    },
    "memory_systems": {
        "query": "ti:memory+AND+ti:augmented+AND+cat:cs.AI",
        "keywords": ["episodic memory", "semantic memory", "memory consolidation",
                     "retrieval augmented", "knowledge graph"],
        "priority": 0.8,
    },
    "llm_advances": {
        "query": "ti:language+model+AND+ti:reasoning",
        "keywords": ["language model", "reasoning", "in-context learning",
                     "chain of thought", "tool use", "agent"],
        "priority": 0.75,
    },
}

# GitHub trending search queries
GITHUB_TOPICS = [
    "consciousness AI",
    "memory augmented language model",
    "knowledge graph LLM",
    "self-improving AI agent",
]


@dataclass
class LearningRun:
    """Record of a single learning run."""
    run_id: str
    task: str
    started_at: float
    ended_at: float = 0.0
    items_fetched: int = 0
    items_stored: int = 0
    phi_before: float = 0.0
    phi_after: float = 0.0
    domains_touched: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    status: str = "running"  # running, completed, failed


@dataclass
class LearningDigest:
    """Summary of learning since last session."""
    since: str
    runs_completed: int
    total_items_learned: int
    phi_trajectory: List[Dict[str, float]]
    top_domains: List[Dict[str, Any]]
    knowledge_gaps: List[str]
    graph_growth: Dict[str, int]
    velocity: Dict[str, float]


class AutonomousLearningEngine:
    """
    Persistent autonomous learning with cross-session accumulation.

    Schedules persist to SQLite. Φ is measured after every run.
    Generates learning digests for session handoff.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(__file__).parent.parent.parent / "data" / "pltm_mcp.db"
        self._conn: Optional[aiosqlite.Connection] = None
        self._store = None  # Set externally
        self._phi_calc = None  # Lazy
        self._arxiv = None  # Lazy
        self._learner = None  # Lazy

    async def connect(self, store=None):
        """Open DB connection and ensure tables."""
        self._conn = await aiosqlite.connect(str(self.db_path))
        await self._conn.execute("PRAGMA journal_mode=WAL")
        if store:
            self._store = store
        await self._ensure_tables()
        logger.info("AutonomousLearningEngine connected")

    async def _ensure_tables(self):
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS learning_schedules (
                task TEXT PRIMARY KEY,
                interval_hours REAL NOT NULL,
                enabled INTEGER DEFAULT 1,
                last_run_at REAL DEFAULT 0,
                next_run_at REAL DEFAULT 0,
                run_count INTEGER DEFAULT 0,
                total_items_learned INTEGER DEFAULT 0,
                config TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS learning_runs (
                run_id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                started_at REAL NOT NULL,
                ended_at REAL DEFAULT 0,
                items_fetched INTEGER DEFAULT 0,
                items_stored INTEGER DEFAULT 0,
                phi_before REAL DEFAULT 0,
                phi_after REAL DEFAULT 0,
                domains TEXT DEFAULT '[]',
                errors TEXT DEFAULT '[]',
                status TEXT DEFAULT 'running'
            );
            CREATE INDEX IF NOT EXISTS idx_lr_task ON learning_runs(task);
            CREATE INDEX IF NOT EXISTS idx_lr_started ON learning_runs(started_at);

            CREATE TABLE IF NOT EXISTS learning_accumulation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                measured_at REAL NOT NULL,
                total_atoms INTEGER DEFAULT 0,
                total_domains INTEGER DEFAULT 0,
                phi_global REAL DEFAULT 0,
                phi_by_domain TEXT DEFAULT '{}',
                graph_nodes INTEGER DEFAULT 0,
                graph_edges INTEGER DEFAULT 0,
                facts_per_hour REAL DEFAULT 0,
                domains_per_week REAL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_la_measured ON learning_accumulation(measured_at);
        """)
        await self._conn.commit()

        # Seed default schedules if empty
        cursor = await self._conn.execute("SELECT COUNT(*) FROM learning_schedules")
        count = (await cursor.fetchone())[0]
        if count == 0:
            await self._seed_default_schedules()

    async def _seed_default_schedules(self):
        defaults = [
            ("arxiv_consciousness", 24.0, 1, json.dumps({"topic": "consciousness"})),
            ("arxiv_ai_alignment", 24.0, 1, json.dumps({"topic": "ai_alignment"})),
            ("arxiv_complexity", 48.0, 1, json.dumps({"topic": "complexity"})),
            ("arxiv_memory_systems", 48.0, 1, json.dumps({"topic": "memory_systems"})),
            ("arxiv_llm_advances", 24.0, 1, json.dumps({"topic": "llm_advances"})),
            ("github_trending", 168.0, 1, json.dumps({"queries": GITHUB_TOPICS})),
            ("news_feed", 24.0, 1, json.dumps({"feeds": ["https://news.ycombinator.com/rss", "https://rss.arxiv.org/rss/cs.AI"]})),
            ("consolidation", 24.0, 1, json.dumps({})),
            ("phi_measurement", 12.0, 1, json.dumps({})),
        ]
        for task, interval, enabled, config in defaults:
            await self._conn.execute(
                "INSERT OR IGNORE INTO learning_schedules (task, interval_hours, enabled, config) VALUES (?,?,?,?)",
                (task, interval, enabled, config),
            )
        await self._conn.commit()
        logger.info("Seeded default learning schedules")

    # ── Schedule Management ──────────────────────────────────────────────────

    async def get_schedules(self) -> Dict[str, Any]:
        """Get all schedules with status."""
        cursor = await self._conn.execute(
            "SELECT task, interval_hours, enabled, last_run_at, next_run_at, run_count, total_items_learned, config FROM learning_schedules ORDER BY task"
        )
        rows = await cursor.fetchall()
        now = time.time()
        schedules = []
        overdue = 0
        for r in rows:
            last = r[3] or 0
            interval_sec = r[1] * 3600
            next_due = last + interval_sec if last > 0 else now
            is_overdue = now >= next_due and r[2]
            if is_overdue:
                overdue += 1
            schedules.append({
                "task": r[0],
                "h": r[1],
                "on": bool(r[2]),
                "last": datetime.fromtimestamp(r[3]).isoformat()[:16] if r[3] else None,
                "due": is_overdue,
                "runs": r[5],
                "learned": r[6],
            })
        return {"n": len(schedules), "overdue": overdue, "schedules": schedules}

    async def update_schedule(self, task: str, interval_hours: Optional[float] = None, enabled: Optional[bool] = None) -> Dict[str, Any]:
        """Update a schedule's interval or enabled state."""
        updates = []
        params = []
        if interval_hours is not None:
            updates.append("interval_hours = ?")
            params.append(interval_hours)
        if enabled is not None:
            updates.append("enabled = ?")
            params.append(1 if enabled else 0)
        if not updates:
            return {"ok": False, "err": "nothing to update"}
        params.append(task)
        await self._conn.execute(
            f"UPDATE learning_schedules SET {', '.join(updates)} WHERE task = ?", params
        )
        await self._conn.commit()
        return {"ok": True, "task": task}

    # ── Run Due Tasks ────────────────────────────────────────────────────────

    async def run_due_tasks(self) -> Dict[str, Any]:
        """Check all schedules and run any that are due. Returns summary."""
        cursor = await self._conn.execute(
            "SELECT task, interval_hours, last_run_at, config FROM learning_schedules WHERE enabled = 1"
        )
        rows = await cursor.fetchall()
        now = time.time()

        results = []
        for task, interval_h, last_run, config_json in rows:
            interval_sec = interval_h * 3600
            next_due = (last_run or 0) + interval_sec
            if now >= next_due:
                result = await self.run_task(task, json.loads(config_json or "{}"))
                results.append(result)

        return {
            "ok": True,
            "checked": len(rows),
            "ran": len(results),
            "results": results,
        }

    async def run_task(self, task: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Run a specific learning task."""
        if config is None:
            cursor = await self._conn.execute(
                "SELECT config FROM learning_schedules WHERE task = ?", (task,)
            )
            row = await cursor.fetchone()
            config = json.loads(row[0]) if row else {}

        run = LearningRun(
            run_id=f"run_{uuid4().hex[:8]}",
            task=task,
            started_at=time.time(),
        )

        # Measure Φ before
        run.phi_before = await self._measure_global_phi()

        try:
            if task.startswith("arxiv_"):
                topic_key = config.get("topic", task.replace("arxiv_", ""))
                result = await self._run_arxiv_ingestion(topic_key)
            elif task == "github_trending":
                result = await self._run_github_trending(config.get("queries", GITHUB_TOPICS))
            elif task == "news_feed":
                result = await self._run_news_feed(config.get("feeds", []))
            elif task == "consolidation":
                result = await self._run_consolidation()
            elif task == "phi_measurement":
                result = await self._run_phi_measurement()
            else:
                result = {"ok": False, "err": f"unknown task: {task}"}

            run.items_fetched = result.get("fetched", 0)
            run.items_stored = result.get("stored", 0)
            run.domains_touched = result.get("domains", [])
            run.status = "completed"

        except Exception as e:
            run.errors.append(str(e)[:200])
            run.status = "failed"
            result = {"ok": False, "err": str(e)[:200]}
            logger.error(f"Learning task {task} failed: {e}")

        # Measure Φ after
        run.phi_after = await self._measure_global_phi()
        run.ended_at = time.time()

        # Persist run
        await self._persist_run(run)

        # Update schedule
        await self._conn.execute(
            """UPDATE learning_schedules
               SET last_run_at = ?, next_run_at = ?, run_count = run_count + 1,
                   total_items_learned = total_items_learned + ?
               WHERE task = ?""",
            (run.ended_at, run.ended_at + (await self._get_interval(task)) * 3600,
             run.items_stored, task),
        )
        await self._conn.commit()

        return {
            "task": task,
            "status": run.status,
            "fetched": run.items_fetched,
            "stored": run.items_stored,
            "phi": f"{run.phi_before:.3f}→{run.phi_after:.3f}",
            "domains": run.domains_touched[:5],
            "duration_s": round(run.ended_at - run.started_at, 1),
        }

    async def _get_interval(self, task: str) -> float:
        cursor = await self._conn.execute(
            "SELECT interval_hours FROM learning_schedules WHERE task = ?", (task,)
        )
        row = await cursor.fetchone()
        return row[0] if row else 24.0

    # ── Task Implementations ─────────────────────────────────────────────────

    async def _run_arxiv_ingestion(self, topic_key: str) -> Dict[str, Any]:
        """Fetch and ingest arXiv papers for a specific topic."""
        import urllib.request
        import xml.etree.ElementTree as ET

        topic = ARXIV_TOPICS.get(topic_key)
        if not topic:
            return {"ok": False, "err": f"unknown topic: {topic_key}", "fetched": 0, "stored": 0, "domains": []}

        # Query arXiv API directly — queries are pre-formatted with + for spaces
        # and field prefixes (cat:, ti:) that must not be double-encoded.
        raw_query = topic["query"]
        url = f"http://export.arxiv.org/api/query?search_query={raw_query}&sortBy=submittedDate&sortOrder=descending&max_results=5"

        paper_ids = []
        # Retry with backoff for 429 rate limits
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "PLTM/1.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    xml_data = resp.read().decode("utf-8")
                root = ET.fromstring(xml_data)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns):
                    id_el = entry.find("atom:id", ns)
                    if id_el is not None and id_el.text:
                        arxiv_id = id_el.text.strip().split("/")[-1]
                        if arxiv_id:
                            paper_ids.append(arxiv_id)
                break  # Success
            except urllib.request.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    wait = 3 * (attempt + 1)
                    logger.info(f"arXiv rate-limited, waiting {wait}s (attempt {attempt+1})")
                    import asyncio
                    await asyncio.sleep(wait)
                    continue
                logger.warning(f"arXiv search failed for {topic_key}: {e}")
                return {"ok": False, "err": str(e)[:200], "fetched": 0, "stored": 0, "domains": []}
            except Exception as e:
                logger.warning(f"arXiv search failed for {topic_key}: {e}")
                return {"ok": False, "err": str(e)[:200], "fetched": 0, "stored": 0, "domains": []}

        arxiv = self._get_arxiv()
        stored = 0
        domains = set()
        for paper_id in paper_ids:
            try:
                result = await arxiv.ingest_paper(paper_id, user_id="pltm_knowledge")
                if result.get("ok"):
                    stored += result.get("claims_stored", 0)
                    for cat in result.get("categories", []):
                        domains.add(cat)
            except Exception as e:
                logger.warning(f"Failed to ingest {paper_id}: {e}")

        return {
            "ok": True,
            "topic": topic_key,
            "fetched": len(paper_ids),
            "stored": stored,
            "domains": list(domains)[:10],
        }

    async def _run_github_trending(self, queries: List[str]) -> Dict[str, Any]:
        """Fetch trending GitHub repos."""
        import urllib.request
        import json as _json

        stored = 0
        fetched = 0
        domains = set()

        for query in queries[:4]:
            try:
                encoded = urllib.parse.quote(query)
                url = f"https://api.github.com/search/repositories?q={encoded}+stars:>50&sort=updated&order=desc&per_page=3"
                req = urllib.request.Request(url, headers={
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "PLTM/1.0",
                })
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = _json.loads(resp.read().decode("utf-8"))

                for item in data.get("items", [])[:3]:
                    fetched += 1
                    # Store as knowledge atom
                    if self._store:
                        from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
                        atom = MemoryAtom(
                            atom_type=AtomType.ENTITY,
                            subject=f"repo:{item.get('full_name', '')}",
                            predicate="trending_for",
                            object=f"{item.get('description', '')[:200]} (stars: {item.get('stargazers_count', 0)})",
                            confidence=0.7,
                            strength=0.7,
                            provenance=Provenance.INFERRED,
                            source_user="autonomous_learning",
                            contexts=["github_trending", query.replace(" ", "_")],
                            graph=GraphType.SUBSTANTIATED,
                        )
                        await self._store.add_atom(atom)
                        stored += 1
                        domains.add(query.replace(" ", "_"))
            except Exception as e:
                logger.warning(f"GitHub trending failed for '{query}': {e}")

        return {"ok": True, "fetched": fetched, "stored": stored, "domains": list(domains)}

    async def _run_news_feed(self, feeds: List[str]) -> Dict[str, Any]:
        """Fetch and store news from RSS feeds."""
        import urllib.request
        import xml.etree.ElementTree as ET

        stored = 0
        fetched = 0
        domains = set()

        for feed_url in feeds[:5]:
            try:
                with urllib.request.urlopen(feed_url, timeout=10) as resp:
                    xml_data = resp.read().decode("utf-8")
                root = ET.fromstring(xml_data)

                for item in root.findall(".//item")[:5]:
                    title_el = item.find("title")
                    desc_el = item.find("description")
                    if title_el is None:
                        continue
                    fetched += 1
                    title = title_el.text or ""
                    desc = (desc_el.text or "")[:500] if desc_el is not None else ""

                    if self._store:
                        from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
                        atom = MemoryAtom(
                            atom_type=AtomType.EVENT,
                            subject=f"news:{title[:80]}",
                            predicate="reported",
                            object=desc[:300],
                            confidence=0.6,
                            strength=0.6,
                            provenance=Provenance.INFERRED,
                            source_user="autonomous_learning",
                            contexts=["news_feed", feed_url[:50]],
                            graph=GraphType.SUBSTANTIATED,
                        )
                        await self._store.add_atom(atom)
                        stored += 1
                        domains.add("news")
            except Exception as e:
                logger.warning(f"News feed failed for {feed_url}: {e}")

        return {"ok": True, "fetched": fetched, "stored": stored, "domains": list(domains)}

    async def _run_consolidation(self) -> Dict[str, Any]:
        """
        Run episodic→semantic consolidation + decay.
        Uses existing ConsolidationEngine and DecayEngine if available.
        """
        result = {"ok": True, "fetched": 0, "stored": 0, "domains": ["consolidation"],
                  "consolidated": 0, "decayed": 0, "archived": 0}

        if not self._store:
            return result

        try:
            # Try to use PhiConsolidator (Tier 1)
            from src.memory.phi_rms import PhiConsolidator
            consolidator = PhiConsolidator(self._store)
            cons_result = await consolidator.consolidate("pltm_knowledge")
            result["consolidated"] = cons_result.get("promoted", 0) + cons_result.get("reinforced", 0)
            result["stored"] = result["consolidated"]
        except Exception as e:
            logger.warning(f"PhiConsolidator unavailable, trying basic: {e}")
            try:
                from src.memory.memory_intelligence import ConsolidationEngine
                engine = ConsolidationEngine(self._store, None)
                cons_result = await engine.consolidate("pltm_knowledge")
                result["consolidated"] = cons_result.get("promoted", 0)
                result["stored"] = result["consolidated"]
            except Exception as e2:
                logger.warning(f"Basic consolidation also failed: {e2}")

        try:
            from src.memory.memory_intelligence import DecayEngine
            decay_engine = DecayEngine(self._store)
            decay_result = await decay_engine.apply_decay("pltm_knowledge")
            result["decayed"] = decay_result.get("updated", 0)
            result["archived"] = decay_result.get("archived", 0)
        except Exception as e:
            logger.warning(f"Decay engine failed: {e}")

        return result

    async def _run_phi_measurement(self) -> Dict[str, Any]:
        """Measure Φ across all domains and persist snapshot."""
        phi_calc = self._get_phi_calc()
        if not self._store:
            return {"ok": False, "err": "no store", "fetched": 0, "stored": 0, "domains": []}

        # Get all atoms grouped by domain
        all_atoms = await self._store.get_all_atoms()
        domain_atoms = defaultdict(list)
        for atom in all_atoms:
            for ctx in getattr(atom, "contexts", []):
                domain_atoms[ctx].append(atom)

        phi_by_domain = {}
        total_nodes = 0
        total_edges = 0
        for domain, atoms in domain_atoms.items():
            if len(atoms) < 3:
                continue
            result = phi_calc.calculate(domain, atoms)
            phi_by_domain[domain] = result.phi
            total_nodes += result.n_nodes
            total_edges += result.n_edges

        global_phi = sum(phi_by_domain.values()) / max(len(phi_by_domain), 1)

        # Persist accumulation snapshot
        await self._conn.execute(
            """INSERT INTO learning_accumulation
               (measured_at, total_atoms, total_domains, phi_global, phi_by_domain,
                graph_nodes, graph_edges)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), len(all_atoms), len(phi_by_domain),
             round(global_phi, 4), json.dumps({k: round(v, 3) for k, v in
                                                sorted(phi_by_domain.items(), key=lambda x: -x[1])[:20]}),
             total_nodes, total_edges),
        )
        await self._conn.commit()

        return {
            "ok": True,
            "fetched": len(all_atoms),
            "stored": 0,
            "domains": list(phi_by_domain.keys())[:10],
            "phi_global": round(global_phi, 4),
            "top_domains": sorted(phi_by_domain.items(), key=lambda x: -x[1])[:5],
            "nodes": total_nodes,
            "edges": total_edges,
        }

    # ── Cross-Session Accumulation ───────────────────────────────────────────

    async def get_learning_digest(self, since_hours: float = 24.0) -> Dict[str, Any]:
        """
        Generate a compact digest of what was learned since last session.
        Designed to be injected into session_handoff.
        """
        since = time.time() - since_hours * 3600

        # Recent runs
        cursor = await self._conn.execute(
            """SELECT task, items_stored, phi_before, phi_after, domains, started_at
               FROM learning_runs WHERE started_at > ? AND status = 'completed'
               ORDER BY started_at""",
            (since,),
        )
        runs = await cursor.fetchall()

        total_learned = sum(r[1] for r in runs)
        all_domains = set()
        for r in runs:
            try:
                for d in json.loads(r[4] or "[]"):
                    all_domains.add(d)
            except Exception:
                pass

        # Φ trajectory
        cursor = await self._conn.execute(
            """SELECT phi_global, total_atoms, total_domains, graph_nodes, graph_edges, measured_at
               FROM learning_accumulation WHERE measured_at > ?
               ORDER BY measured_at""",
            (since,),
        )
        phi_rows = await cursor.fetchall()
        phi_trajectory = [
            {"phi": r[0], "atoms": r[1], "domains": r[2], "nodes": r[3], "edges": r[4],
             "at": datetime.fromtimestamp(r[5]).isoformat()[:16]}
            for r in phi_rows
        ]

        # Graph growth
        graph_growth = {}
        if len(phi_rows) >= 2:
            first, last = phi_rows[0], phi_rows[-1]
            graph_growth = {
                "atoms": (last[1] or 0) - (first[1] or 0),
                "domains": (last[2] or 0) - (first[2] or 0),
                "nodes": (last[3] or 0) - (first[3] or 0),
                "edges": (last[4] or 0) - (first[4] or 0),
            }

        # Learning velocity
        duration_hours = max(since_hours, 0.1)
        velocity = {
            "facts_per_hour": round(total_learned / duration_hours, 1),
            "runs_per_day": round(len(runs) / (duration_hours / 24), 1),
        }

        # Domain breakdown
        domain_counts = defaultdict(int)
        for r in runs:
            try:
                for d in json.loads(r[4] or "[]"):
                    domain_counts[d] += r[1]
            except Exception:
                pass
        top_domains = sorted(domain_counts.items(), key=lambda x: -x[1])[:8]

        return {
            "since_hours": since_hours,
            "runs": len(runs),
            "items_learned": total_learned,
            "domains_touched": len(all_domains),
            "phi_trajectory": phi_trajectory[-10:],
            "graph_growth": graph_growth,
            "velocity": velocity,
            "top_domains": [{"domain": d, "items": c} for d, c in top_domains],
        }

    async def get_phi_history(self, limit: int = 20) -> Dict[str, Any]:
        """Get Φ measurement history for trend analysis."""
        cursor = await self._conn.execute(
            """SELECT phi_global, total_atoms, total_domains, graph_nodes, graph_edges,
                      phi_by_domain, measured_at
               FROM learning_accumulation ORDER BY measured_at DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        history = []
        for r in rows:
            history.append({
                "phi": r[0],
                "atoms": r[1],
                "domains": r[2],
                "nodes": r[3],
                "edges": r[4],
                "by_domain": json.loads(r[5] or "{}"),
                "at": datetime.fromtimestamp(r[6]).isoformat()[:16],
            })
        history.reverse()

        trend = "stable"
        if len(history) >= 3:
            recent = [h["phi"] for h in history[-3:]]
            if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                trend = "growing"
            elif all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
                trend = "declining"

        return {"n": len(history), "trend": trend, "history": history}

    async def get_run_history(self, limit: int = 20) -> Dict[str, Any]:
        """Get recent learning run history."""
        cursor = await self._conn.execute(
            """SELECT run_id, task, started_at, ended_at, items_fetched, items_stored,
                      phi_before, phi_after, domains, status
               FROM learning_runs ORDER BY started_at DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        runs = []
        for r in rows:
            runs.append({
                "id": r[0],
                "task": r[1],
                "at": datetime.fromtimestamp(r[2]).isoformat()[:16],
                "dur": round(r[3] - r[2], 1) if r[3] else 0,
                "fetched": r[4],
                "stored": r[5],
                "phi": f"{r[6]:.3f}→{r[7]:.3f}",
                "status": r[9],
            })
        return {"n": len(runs), "runs": runs}

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _measure_global_phi(self) -> float:
        """Quick global Φ measurement."""
        if not self._store:
            return 0.0
        try:
            all_atoms = await self._store.get_all_atoms()
            if len(all_atoms) < 3:
                return 0.0
            phi_calc = self._get_phi_calc()
            result = phi_calc.calculate("global", all_atoms)
            return result.phi
        except Exception:
            return 0.0

    async def _persist_run(self, run: LearningRun):
        await self._conn.execute(
            """INSERT INTO learning_runs
               (run_id, task, started_at, ended_at, items_fetched, items_stored,
                phi_before, phi_after, domains, errors, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run.run_id, run.task, run.started_at, run.ended_at,
             run.items_fetched, run.items_stored, run.phi_before, run.phi_after,
             json.dumps(run.domains_touched), json.dumps(run.errors), run.status),
        )
        await self._conn.commit()

    def _get_phi_calc(self):
        if self._phi_calc is None:
            from src.analysis.phi_integration import PhiIntegrationCalculator
            self._phi_calc = PhiIntegrationCalculator(self.db_path)
        return self._phi_calc

    def _get_arxiv(self):
        if self._arxiv is None:
            from src.learning.arxiv_ingestion import ArxivIngestion
            self._arxiv = ArxivIngestion(self._store)
        return self._arxiv

    async def close(self):
        if self._conn:
            await self._conn.close()
            self._conn = None
