"""FastAPI application for Procedural LTM MVP"""

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.core.config import settings
from src.pipeline.memory_pipeline import MemoryPipeline
from src.storage.sqlite_store import SQLiteGraphStore


# Global state
store: SQLiteGraphStore = None
pipeline: MemoryPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup/shutdown)"""
    global store, pipeline
    
    # Startup
    store = SQLiteGraphStore(settings.DB_PATH)
    await store.connect()
    
    pipeline = MemoryPipeline(store)
    
    yield
    
    # Shutdown
    await store.close()


app = FastAPI(
    title="Procedural Long-Term Memory - MVP",
    description="Jury-based memory system with conflict resolution",
    version="0.1.0",
    lifespan=lifespan,
)


# Request/Response models
class ProcessMessageRequest(BaseModel):
    """Request to process a message"""
    user_id: str
    message: str
    session_id: str = ""


class ProcessMessageResponse(BaseModel):
    """Response from processing a message"""
    atoms_extracted: int
    atoms_approved: int
    atoms_promoted: int
    conflicts_resolved: int


class MemoryAtomResponse(BaseModel):
    """Memory atom for API response"""
    subject: str
    predicate: str
    object: str
    confidence: float
    provenance: str
    graph: str
    promotion_tier: str


class GetMemoryResponse(BaseModel):
    """Response with user's memory"""
    user_id: str
    substantiated_count: int
    unsubstantiated_count: int
    facts: List[MemoryAtomResponse]


class StatsResponse(BaseModel):
    """System statistics"""
    messages_processed: int
    total_atoms: int
    substantiated_atoms: int
    unsubstantiated_atoms: int


# Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Procedural LTM MVP",
        "version": "0.1.0",
        "status": "running",
    }


@app.post("/process", response_model=ProcessMessageResponse)
async def process_message(request: ProcessMessageRequest):
    """
    Process a user message through the memory pipeline.
    
    Extracts facts, validates with jury, resolves conflicts, and stores.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Collect all updates
    atoms_extracted = 0
    atoms_approved = 0
    atoms_promoted = 0
    conflicts_resolved = 0
    
    async for update in pipeline.process_message(
        request.message,
        request.user_id,
        request.session_id,
    ):
        if update.stage == "extraction":
            atoms_extracted = update.atoms_extracted
        elif update.stage == "deliberation":
            atoms_approved = update.atoms_approved
        elif update.stage == "persistence":
            atoms_promoted = update.atoms_promoted
            conflicts_resolved = update.conflicts_resolved
    
    return ProcessMessageResponse(
        atoms_extracted=atoms_extracted,
        atoms_approved=atoms_approved,
        atoms_promoted=atoms_promoted,
        conflicts_resolved=conflicts_resolved,
    )


@app.get("/memory/{user_id}", response_model=GetMemoryResponse)
async def get_memory(user_id: str, include_unsubstantiated: bool = False):
    """
    Retrieve user's memory.
    
    By default returns only substantiated facts.
    Set include_unsubstantiated=true to also get shadow buffer.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Get substantiated memory
    substantiated = await pipeline.get_memory(user_id)
    
    facts = [
        MemoryAtomResponse(
            subject=atom.subject,
            predicate=atom.predicate,
            object=atom.object,
            confidence=atom.confidence,
            provenance=atom.provenance.value,
            graph=atom.graph.value,
            promotion_tier=atom.promotion_tier,
        )
        for atom in substantiated
    ]
    
    unsubstantiated_count = 0
    if include_unsubstantiated:
        unsubstantiated = await pipeline.get_unsubstantiated_memory(user_id)
        unsubstantiated_count = len(unsubstantiated)
        
        # Add unsubstantiated facts
        facts.extend([
            MemoryAtomResponse(
                subject=atom.subject,
                predicate=atom.predicate,
                object=atom.object,
                confidence=atom.confidence,
                provenance=atom.provenance.value,
                graph=atom.graph.value,
                promotion_tier=atom.promotion_tier,
            )
            for atom in unsubstantiated
        ])
    
    return GetMemoryResponse(
        user_id=user_id,
        substantiated_count=len(substantiated),
        unsubstantiated_count=unsubstantiated_count,
        facts=facts,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    if not pipeline or not store:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    pipeline_stats = pipeline.get_stats()
    storage_stats = await store.get_stats()
    
    return StatsResponse(
        messages_processed=pipeline_stats["messages_processed"],
        total_atoms=storage_stats.get("total_atoms", 0),
        substantiated_atoms=storage_stats.get("substantiated_count", 0),
        unsubstantiated_atoms=storage_stats.get("unsubstantiated_count", 0),
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not pipeline or not store:
        raise HTTPException(status_code=503, detail="System not ready")
    
    return {"status": "healthy"}
