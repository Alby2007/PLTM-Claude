"""
Batch Ingestion Pipelines

Bulk knowledge ingestion from:
- Wikipedia dumps (6M+ articles)
- arXiv bulk downloads (2M+ papers)
- GitHub trending repos
- News feeds
- Documentation sites
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import asyncio
import json
import re

from src.storage.sqlite_store import SQLiteGraphStore
from src.learning.universal_learning import UniversalLearningSystem, SourceType
from loguru import logger


@dataclass
class IngestionJob:
    """A batch ingestion job"""
    job_id: str
    source_type: str
    total_items: int
    processed_items: int
    facts_extracted: int
    concepts_extracted: int
    started_at: datetime
    status: str  # pending, running, completed, failed
    errors: List[str]


class BatchIngestionPipeline:
    """
    Bulk knowledge ingestion from major sources.
    
    Handles:
    - Wikipedia article dumps
    - arXiv paper bulk downloads
    - GitHub repository scanning
    - News feed aggregation
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        self.learner = UniversalLearningSystem(store)
        self.active_jobs: Dict[str, IngestionJob] = {}
        
        logger.info("BatchIngestionPipeline initialized")
    
    async def ingest_wikipedia_articles(
        self,
        articles: List[Dict[str, str]],
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Ingest Wikipedia articles in batch.
        
        Args:
            articles: List of {"title": str, "content": str, "url": str}
            batch_size: How many to process concurrently
        
        Returns:
            Ingestion statistics
        """
        job_id = f"wiki_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = IngestionJob(
            job_id=job_id,
            source_type="wikipedia",
            total_items=len(articles),
            processed_items=0,
            facts_extracted=0,
            concepts_extracted=0,
            started_at=datetime.now(),
            status="running",
            errors=[]
        )
        self.active_jobs[job_id] = job
        
        logger.info(f"Starting Wikipedia ingestion job {job_id} with {len(articles)} articles")
        
        total_facts = 0
        total_concepts = 0
        
        # Process in batches
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            
            # Process batch concurrently
            tasks = []
            for article in batch:
                tasks.append(self._process_wikipedia_article(article))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    job.errors.append(str(result))
                else:
                    total_facts += result.get("extracted", {}).get("facts", 0)
                    total_concepts += result.get("extracted", {}).get("concepts", 0)
                job.processed_items += 1
            
            job.facts_extracted = total_facts
            job.concepts_extracted = total_concepts
            
            # Progress logging
            if job.processed_items % 100 == 0:
                logger.info(f"Job {job_id}: {job.processed_items}/{job.total_items} processed")
        
        job.status = "completed"
        
        return {
            "ok": True,
            "src": "wiki",
            "n": job.processed_items,
            "facts": total_facts,
            "err": len(job.errors)
        }
    
    async def _process_wikipedia_article(self, article: Dict[str, str]) -> Dict[str, Any]:
        """Process a single Wikipedia article"""
        return await self.learner.learn_from_url(
            url=article.get("url", f"https://en.wikipedia.org/wiki/{article['title']}"),
            content=article["content"],
            source_type=SourceType.WEB_PAGE
        )
    
    async def ingest_arxiv_papers(
        self,
        papers: List[Dict[str, Any]],
        batch_size: int = 5
    ) -> Dict[str, Any]:
        """
        Ingest arXiv papers in batch.
        
        Args:
            papers: List of paper metadata with content
            batch_size: How many to process concurrently
        
        Returns:
            Ingestion statistics
        """
        job_id = f"arxiv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = IngestionJob(
            job_id=job_id,
            source_type="arxiv",
            total_items=len(papers),
            processed_items=0,
            facts_extracted=0,
            concepts_extracted=0,
            started_at=datetime.now(),
            status="running",
            errors=[]
        )
        self.active_jobs[job_id] = job
        
        logger.info(f"Starting arXiv ingestion job {job_id} with {len(papers)} papers")
        
        total_findings = 0
        total_methods = 0
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            
            tasks = []
            for paper in batch:
                tasks.append(self._process_arxiv_paper(paper))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    job.errors.append(str(result))
                else:
                    total_findings += result.get("extracted", {}).get("findings", 0)
                    total_methods += result.get("extracted", {}).get("methods", 0)
                job.processed_items += 1
            
            job.facts_extracted = total_findings
        
        job.status = "completed"
        
        return {
            "ok": True,
            "src": "arxiv",
            "n": job.processed_items,
            "findings": total_findings,
            "err": len(job.errors)
        }
    
    async def _process_arxiv_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single arXiv paper"""
        return await self.learner.learn_from_paper(
            paper_id=paper.get("id", "unknown"),
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
            content=paper.get("content", paper.get("abstract", "")),
            authors=paper.get("authors", []),
            publication_date=paper.get("published")
        )
    
    async def ingest_github_repos(
        self,
        repos: List[Dict[str, Any]],
        batch_size: int = 3
    ) -> Dict[str, Any]:
        """
        Ingest GitHub repositories in batch.
        
        Args:
            repos: List of repo metadata with code samples
            batch_size: How many to process concurrently
        
        Returns:
            Ingestion statistics
        """
        job_id = f"github_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = IngestionJob(
            job_id=job_id,
            source_type="github",
            total_items=len(repos),
            processed_items=0,
            facts_extracted=0,
            concepts_extracted=0,
            started_at=datetime.now(),
            status="running",
            errors=[]
        )
        self.active_jobs[job_id] = job
        
        logger.info(f"Starting GitHub ingestion job {job_id} with {len(repos)} repos")
        
        total_patterns = 0
        total_techniques = 0
        
        for i in range(0, len(repos), batch_size):
            batch = repos[i:i + batch_size]
            
            tasks = []
            for repo in batch:
                tasks.append(self._process_github_repo(repo))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    job.errors.append(str(result))
                else:
                    total_patterns += result.get("extracted", {}).get("patterns", 0)
                    total_techniques += result.get("extracted", {}).get("techniques", 0)
                job.processed_items += 1
            
            job.facts_extracted = total_patterns
        
        job.status = "completed"
        
        return {
            "ok": True,
            "src": "github",
            "n": job.processed_items,
            "patterns": total_patterns,
            "err": len(job.errors)
        }
    
    async def _process_github_repo(self, repo: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single GitHub repo"""
        return await self.learner.learn_from_code(
            repo_url=repo.get("url", ""),
            repo_name=repo.get("name", "unknown"),
            description=repo.get("description", ""),
            languages=repo.get("languages", []),
            code_samples=repo.get("code_samples", [])
        )
    
    async def ingest_news_feed(
        self,
        articles: List[Dict[str, str]],
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Ingest news articles in batch.
        
        Args:
            articles: List of {"title": str, "content": str, "url": str, "published": str}
            batch_size: How many to process concurrently
        
        Returns:
            Ingestion statistics
        """
        job_id = f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        total_facts = 0
        processed = 0
        errors = []
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            
            tasks = []
            for article in batch:
                tasks.append(self.learner.learn_from_url(
                    url=article.get("url", ""),
                    content=article.get("content", ""),
                    source_type=SourceType.NEWS
                ))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    errors.append(str(result))
                else:
                    total_facts += result.get("extracted", {}).get("facts", 0)
                processed += 1
        
        return {
            "ok": True,
            "src": "news",
            "n": processed,
            "facts": total_facts,
            "err": len(errors)
        }
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an ingestion job"""
        job = self.active_jobs.get(job_id)
        if not job:
            return None
        
        return {
            "job_id": job.job_id,
            "source_type": job.source_type,
            "status": job.status,
            "progress": f"{job.processed_items}/{job.total_items}",
            "progress_percent": round(job.processed_items / job.total_items * 100, 1) if job.total_items > 0 else 0,
            "facts_extracted": job.facts_extracted,
            "concepts_extracted": job.concepts_extracted,
            "errors": len(job.errors),
            "started_at": job.started_at.isoformat(),
            "duration_seconds": (datetime.now() - job.started_at).total_seconds()
        }
    
    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get status of all jobs"""
        return [self.get_job_status(job_id) for job_id in self.active_jobs]
