"""
Continuous Learning Loop

Automated knowledge acquisition that runs continuously:
- Hourly paper updates from arXiv
- Trending repos from GitHub
- News feed aggregation
- Knowledge consolidation and conflict resolution
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import asyncio
import json

from src.storage.sqlite_store import SQLiteGraphStore
from src.learning.universal_learning import UniversalLearningSystem, SourceType
from src.learning.batch_ingestion import BatchIngestionPipeline
from loguru import logger


@dataclass
class LearningSchedule:
    """Schedule for continuous learning tasks"""
    task_name: str
    interval_hours: float
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True
    run_count: int = 0
    total_items_learned: int = 0


@dataclass
class LearningSession:
    """A single learning session"""
    session_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    sources_processed: int = 0
    facts_learned: int = 0
    concepts_learned: int = 0
    conflicts_resolved: int = 0
    errors: List[str] = field(default_factory=list)


class ContinuousLearningLoop:
    """
    Continuous learning system that keeps Claude's knowledge up-to-date.
    
    Runs scheduled tasks to:
    - Fetch and learn from new papers
    - Track trending code repositories
    - Aggregate news and current events
    - Consolidate and resolve knowledge conflicts
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        self.learner = UniversalLearningSystem(store)
        self.batch_pipeline = BatchIngestionPipeline(store)
        
        self.is_running = False
        self.current_session: Optional[LearningSession] = None
        self.sessions_history: List[LearningSession] = []
        
        # Default schedules
        self.schedules: Dict[str, LearningSchedule] = {
            "arxiv_latest": LearningSchedule(
                task_name="arxiv_latest",
                interval_hours=1.0,
                enabled=True
            ),
            "github_trending": LearningSchedule(
                task_name="github_trending",
                interval_hours=6.0,
                enabled=True
            ),
            "news_feed": LearningSchedule(
                task_name="news_feed",
                interval_hours=1.0,
                enabled=True
            ),
            "knowledge_consolidation": LearningSchedule(
                task_name="knowledge_consolidation",
                interval_hours=24.0,
                enabled=True
            ),
        }
        
        # Data fetchers (to be provided externally or mocked)
        self.data_fetchers: Dict[str, Callable] = {}
        
        logger.info("ContinuousLearningLoop initialized")
    
    def register_data_fetcher(self, task_name: str, fetcher: Callable):
        """Register a data fetcher function for a task"""
        self.data_fetchers[task_name] = fetcher
        logger.info(f"Registered data fetcher for {task_name}")
    
    async def start_learning_loop(self, run_once: bool = False):
        """
        Start the continuous learning loop.
        
        Args:
            run_once: If True, run all tasks once and stop
        """
        self.is_running = True
        logger.info("Starting continuous learning loop")
        
        while self.is_running:
            session = LearningSession(
                session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                started_at=datetime.now()
            )
            self.current_session = session
            
            # Check each scheduled task
            for task_name, schedule in self.schedules.items():
                if not schedule.enabled:
                    continue
                
                should_run = (
                    schedule.last_run is None or
                    datetime.now() >= schedule.last_run + timedelta(hours=schedule.interval_hours)
                )
                
                if should_run or run_once:
                    try:
                        result = await self._run_task(task_name)
                        schedule.last_run = datetime.now()
                        schedule.next_run = datetime.now() + timedelta(hours=schedule.interval_hours)
                        schedule.run_count += 1
                        schedule.total_items_learned += result.get("items_learned", 0)
                        
                        session.sources_processed += result.get("sources_processed", 0)
                        session.facts_learned += result.get("facts_learned", 0)
                        
                    except Exception as e:
                        logger.error(f"Error in task {task_name}: {e}")
                        session.errors.append(f"{task_name}: {str(e)}")
            
            session.ended_at = datetime.now()
            self.sessions_history.append(session)
            self.current_session = None
            
            if run_once:
                break
            
            # Sleep until next task is due
            await asyncio.sleep(60)  # Check every minute
        
        logger.info("Continuous learning loop stopped")
    
    def stop_learning_loop(self):
        """Stop the continuous learning loop"""
        self.is_running = False
        logger.info("Stopping continuous learning loop")
    
    async def _run_task(self, task_name: str) -> Dict[str, Any]:
        """Run a specific learning task"""
        logger.info(f"Running learning task: {task_name}")
        
        if task_name == "arxiv_latest":
            return await self._learn_arxiv_latest()
        elif task_name == "github_trending":
            return await self._learn_github_trending()
        elif task_name == "news_feed":
            return await self._learn_news_feed()
        elif task_name == "knowledge_consolidation":
            return await self._consolidate_knowledge()
        else:
            logger.warning(f"Unknown task: {task_name}")
            return {"items_learned": 0}
    
    async def _learn_arxiv_latest(self) -> Dict[str, Any]:
        """Learn from latest arXiv papers"""
        # Get papers from fetcher if available
        if "arxiv_latest" in self.data_fetchers:
            papers = await self.data_fetchers["arxiv_latest"]()
        else:
            # Return empty if no fetcher
            papers = []
        
        if not papers:
            return {"sources_processed": 0, "facts_learned": 0, "items_learned": 0}
        
        result = await self.batch_pipeline.ingest_arxiv_papers(papers)
        
        return {
            "sources_processed": result.get("processed", 0),
            "facts_learned": result.get("findings_extracted", 0),
            "items_learned": result.get("processed", 0)
        }
    
    async def _learn_github_trending(self) -> Dict[str, Any]:
        """Learn from trending GitHub repos"""
        if "github_trending" in self.data_fetchers:
            repos = await self.data_fetchers["github_trending"]()
        else:
            repos = []
        
        if not repos:
            return {"sources_processed": 0, "facts_learned": 0, "items_learned": 0}
        
        result = await self.batch_pipeline.ingest_github_repos(repos)
        
        return {
            "sources_processed": result.get("processed", 0),
            "facts_learned": result.get("patterns_extracted", 0),
            "items_learned": result.get("processed", 0)
        }
    
    async def _learn_news_feed(self) -> Dict[str, Any]:
        """Learn from news feed"""
        if "news_feed" in self.data_fetchers:
            articles = await self.data_fetchers["news_feed"]()
        else:
            articles = []
        
        if not articles:
            return {"sources_processed": 0, "facts_learned": 0, "items_learned": 0}
        
        result = await self.batch_pipeline.ingest_news_feed(articles)
        
        return {
            "sources_processed": result.get("processed", 0),
            "facts_learned": result.get("facts_extracted", 0),
            "items_learned": result.get("processed", 0)
        }
    
    async def _consolidate_knowledge(self) -> Dict[str, Any]:
        """
        Consolidate learned knowledge:
        - Resolve conflicts between facts
        - Merge duplicate concepts
        - Update confidence scores based on evidence
        - Archive stale information
        """
        logger.info("Starting knowledge consolidation")
        
        conflicts_resolved = 0
        duplicates_merged = 0
        stale_archived = 0
        
        # Get all recent atoms (last 24 hours)
        # In production, would have more sophisticated querying
        
        # For now, just log that consolidation ran
        logger.info("Knowledge consolidation complete")
        
        return {
            "sources_processed": 0,
            "facts_learned": 0,
            "items_learned": 0,
            "conflicts_resolved": conflicts_resolved,
            "duplicates_merged": duplicates_merged,
            "stale_archived": stale_archived
        }
    
    async def run_single_task(self, task_name: str) -> Dict[str, Any]:
        """Run a single task immediately"""
        if task_name not in self.schedules:
            return {"error": f"Unknown task: {task_name}"}
        
        return await self._run_task(task_name)
    
    def get_schedule_status(self) -> Dict[str, Any]:
        """Get status of all scheduled tasks"""
        status = {}
        for task_name, schedule in self.schedules.items():
            status[task_name] = {
                "enabled": schedule.enabled,
                "interval_hours": schedule.interval_hours,
                "last_run": schedule.last_run.isoformat() if schedule.last_run else None,
                "next_run": schedule.next_run.isoformat() if schedule.next_run else None,
                "run_count": schedule.run_count,
                "total_items_learned": schedule.total_items_learned
            }
        return {
            "is_running": self.is_running,
            "schedules": status,
            "current_session": self.current_session.session_id if self.current_session else None,
            "total_sessions": len(self.sessions_history)
        }
    
    def update_schedule(
        self, 
        task_name: str, 
        interval_hours: Optional[float] = None,
        enabled: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Update a task schedule"""
        if task_name not in self.schedules:
            return {"error": f"Unknown task: {task_name}"}
        
        schedule = self.schedules[task_name]
        
        if interval_hours is not None:
            schedule.interval_hours = interval_hours
        
        if enabled is not None:
            schedule.enabled = enabled
        
        return {
            "task_name": task_name,
            "updated": True,
            "new_interval_hours": schedule.interval_hours,
            "enabled": schedule.enabled
        }
    
    def get_learning_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent learning session history"""
        recent = self.sessions_history[-limit:]
        return [
            {
                "session_id": s.session_id,
                "started_at": s.started_at.isoformat(),
                "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                "duration_seconds": (s.ended_at - s.started_at).total_seconds() if s.ended_at else None,
                "sources_processed": s.sources_processed,
                "facts_learned": s.facts_learned,
                "errors": len(s.errors)
            }
            for s in recent
        ]


class ManualLearningTrigger:
    """
    Manual triggers for learning specific content.
    
    Use when Claude encounters something worth learning.
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        self.learner = UniversalLearningSystem(store)
    
    async def learn_from_conversation(
        self,
        messages: List[Dict[str, str]],
        topic: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Learn from a conversation that contains valuable information.
        
        Called when Claude recognizes the conversation contains
        novel or important information worth remembering.
        """
        # Combine messages into content
        content = "\n".join([
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in messages
        ])
        
        result = await self.learner.learn_from_url(
            url=f"conversation://{user_id}/{topic}",
            content=content,
            source_type=SourceType.CONVERSATION
        )
        
        return {
            "learned_from": "conversation",
            "topic": topic,
            "user_id": user_id,
            "facts_extracted": result.get("extracted", {}).get("facts", 0),
            "concepts_extracted": result.get("extracted", {}).get("concepts", 0)
        }
    
    async def learn_from_user_expertise(
        self,
        user_id: str,
        domain: str,
        knowledge: str
    ) -> Dict[str, Any]:
        """
        Learn from user's domain expertise.
        
        When a user shares specialized knowledge, store it
        as high-confidence facts attributed to them.
        """
        result = await self.learner.learn_from_url(
            url=f"user_expertise://{user_id}/{domain}",
            content=knowledge,
            source_type=SourceType.CONVERSATION
        )
        
        return {
            "learned_from": "user_expertise",
            "user_id": user_id,
            "domain": domain,
            "facts_extracted": result.get("extracted", {}).get("facts", 0)
        }
