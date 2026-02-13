"""
Typed Memory System handlers — store, recall, search, update, prune, etc.
Extracted from pltm_server.py for maintainability.
"""

import time as _time
from typing import Any, Dict, List

from mcp.types import TextContent

from mcp_server.handlers.registry import registry as R, compact_json


async def handle_store_typed(args: Dict[str, Any], mem_type: str) -> List[TextContent]:
    """Store a typed memory (episodic, semantic, belief, or procedural)."""
    from src.memory.memory_types import TypedMemory, MemoryType

    mt = MemoryType(mem_type)
    now = _time.time()

    mem = TypedMemory(
        id="",
        memory_type=mt,
        user_id=args["user_id"],
        content=args.get("content", args.get("trigger", "") + " → " + args.get("action", "")),
        context=args.get("context", ""),
        source=args.get("source", "observed"),
        strength=1.0,
        created_at=now,
        last_accessed=now,
        confidence=float(args.get("confidence", 0.5 if mt == MemoryType.BELIEF else 0.8)),
        episode_timestamp=now if mt == MemoryType.EPISODIC else 0,
        emotional_valence=float(args.get("emotional_valence", 0.0)),
        trigger=args.get("trigger", ""),
        action=args.get("action", ""),
        tags=args.get("tags", []),
    )

    mem_id = await R.typed_memory_store.store(mem)

    return [TextContent(type="text", text=compact_json({
        "stored": True,
        "id": mem_id,
        "type": mem_type,
        "content": mem.content[:100],
        "strength": mem.strength,
        "confidence": mem.confidence,
    }))]


async def handle_recall_memories(args: Dict[str, Any]) -> List[TextContent]:
    """Recall typed memories with type-aware retrieval."""
    from src.memory.memory_types import MemoryType

    mt = None
    if args.get("memory_type"):
        mt = MemoryType(args["memory_type"])

    memories = await R.typed_memory_store.query(
        user_id=args["user_id"],
        memory_type=mt,
        min_strength=float(args.get("min_strength", 0.1)),
        limit=int(args.get("limit", 20)),
        tags=args.get("tags"),
    )

    results = []
    for mem in memories:
        entry = {
            "id": mem.id,
            "type": mem.memory_type.value,
            "content": mem.content,
            "strength": round(mem.current_strength(), 3),
            "confidence": round(mem.confidence, 3),
            "tags": mem.tags,
        }
        if mem.memory_type.value == "episodic":
            entry["emotional_valence"] = mem.emotional_valence
            entry["context"] = mem.context
        elif mem.memory_type.value == "belief":
            entry["evidence_for"] = len(mem.evidence_for)
            entry["evidence_against"] = len(mem.evidence_against)
        elif mem.memory_type.value == "procedural":
            entry["trigger"] = mem.trigger
            entry["action"] = mem.action
            entry["success_rate"] = (
                round(mem.success_count / max(1, mem.success_count + mem.failure_count), 2)
            )
        results.append(entry)

    return [TextContent(type="text", text=compact_json({
        "count": len(results),
        "memories": results,
    }))]


async def handle_search_memories(args: Dict[str, Any]) -> List[TextContent]:
    """Full-text search across typed memories."""
    memories = await R.typed_memory_store.search(
        user_id=args["user_id"],
        query=args["query"],
        limit=int(args.get("limit", 20)),
    )

    results = [{
        "id": m.id,
        "type": m.memory_type.value,
        "content": m.content,
        "strength": round(m.current_strength(), 3),
        "confidence": round(m.confidence, 3),
    } for m in memories]

    return [TextContent(type="text", text=compact_json({
        "query": args["query"],
        "count": len(results),
        "results": results,
    }))]


async def handle_update_belief_mem(args: Dict[str, Any]) -> List[TextContent]:
    """Update a belief with new evidence."""
    # Accept 'delta'/'amount'/'confidence_change' as aliases for 'confidence_delta'
    confidence_delta = args.get("confidence_delta") or args.get("delta") or args.get("amount") or args.get("confidence_change")
    if confidence_delta is None:
        return [TextContent(type="text", text=compact_json({"error": "'confidence_delta' (number) is required"}))]
    # Accept 'direction'/'type' as alias for 'evidence_type'
    evidence_type = args.get("evidence_type") or args.get("direction") or args.get("type", "for")
    belief = await R.typed_memory_store.update_belief(
        belief_id=args.get("belief_id") or args.get("id", ""),
        evidence_type=evidence_type,
        evidence_id=args.get("evidence_id", ""),
        confidence_delta=float(confidence_delta),
    )

    if not belief:
        return [TextContent(type="text", text=compact_json({"error": "Belief not found or not a belief type"}))]

    return [TextContent(type="text", text=compact_json({
        "updated": True,
        "id": belief.id,
        "content": belief.content,
        "new_confidence": round(belief.confidence, 3),
        "evidence_for": len(belief.evidence_for),
        "evidence_against": len(belief.evidence_against),
    }))]


async def handle_record_procedure(args: Dict[str, Any]) -> List[TextContent]:
    """Record procedure outcome."""
    # Accept 'worked'/'result'/'outcome' as aliases for 'success'
    success = args.get("success")
    if success is None:
        success = args.get("worked") or args.get("result") or args.get("outcome")
    if success is None:
        return [TextContent(type="text", text=compact_json({"error": "'success' (boolean) is required"}))]
    proc = await R.typed_memory_store.record_procedure_outcome(
        procedure_id=args.get("procedure_id") or args.get("id", ""),
        success=bool(success),
    )

    if not proc:
        return [TextContent(type="text", text=compact_json({"error": "Procedure not found"}))]

    total = proc.success_count + proc.failure_count
    return [TextContent(type="text", text=compact_json({
        "updated": True,
        "id": proc.id,
        "trigger": proc.trigger,
        "action": proc.action,
        "success_count": proc.success_count,
        "failure_count": proc.failure_count,
        "success_rate": round(proc.success_count / max(1, total), 2),
        "strength": round(proc.strength, 3),
    }))]


async def handle_consolidate(args: Dict[str, Any]) -> List[TextContent]:
    """Run episodic → semantic consolidation."""
    new_semantics = await R.typed_memory_store.consolidate_episodes(
        user_id=args["user_id"],
        min_episodes=int(args.get("min_episodes", 3)),
    )

    results = [{
        "id": s.id,
        "content": s.content,
        "confidence": round(s.confidence, 3),
        "consolidated_from": len(s.consolidated_from),
    } for s in new_semantics]

    return [TextContent(type="text", text=compact_json({
        "consolidated": len(results),
        "new_semantic_memories": results,
    }))]


async def handle_memory_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Get typed memory statistics."""
    stats = await R.typed_memory_store.get_stats(args["user_id"])
    return [TextContent(type="text", text=compact_json(stats))]


async def handle_detect_contradictions(args: Dict[str, Any]) -> List[TextContent]:
    """Find contradicting memories."""
    contradictions = await R.typed_memory_store.detect_contradictions(args["user_id"])
    return [TextContent(type="text", text=compact_json({
        "contradictions": contradictions,
        "count": len(contradictions),
        "action": "Review each pair and correct_memory or forget_memory the wrong one." if contradictions else "No contradictions found."
    }))]


async def handle_what_do_i_know(args: Dict[str, Any]) -> List[TextContent]:
    """Synthesized cross-type retrieval for a topic."""
    result = await R.typed_memory_store.what_do_i_know_about(
        user_id=args["user_id"], topic=args["topic"],
        limit=int(args.get("limit", 30)),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_auto_tag(args: Dict[str, Any]) -> List[TextContent]:
    """Auto-tag all memories for a user."""
    result = await R.typed_memory_store.auto_tag_all(args["user_id"])
    return [TextContent(type="text", text=compact_json(result))]


async def handle_correct_memory(args: Dict[str, Any]) -> List[TextContent]:
    """Correct a memory's content."""
    # Accept 'correction' as alias for 'new_content' (Claude sometimes uses this)
    new_content = args.get("new_content") or args.get("correction", "")
    if not new_content:
        return [TextContent(type="text", text=compact_json({"error": "'new_content' is required"}))]
    mem = await R.typed_memory_store.correct_memory(
        memory_id=args["memory_id"], new_content=new_content,
        reason=args.get("reason", ""), new_confidence=args.get("new_confidence"),
    )
    if not mem:
        return [TextContent(type="text", text=compact_json({"error": "Memory not found"}))]
    return [TextContent(type="text", text=compact_json({
        "corrected": True, "id": mem.id, "new_content": mem.content,
        "confidence": round(mem.confidence, 3), "correction_history": mem.context[:200],
    }))]


async def handle_forget_memory(args: Dict[str, Any]) -> List[TextContent]:
    """Explicitly delete a memory."""
    deleted = await R.typed_memory_store.forget_memory(
        memory_id=args["memory_id"], reason=args.get("reason", ""),
    )
    return [TextContent(type="text", text=compact_json({
        "forgotten": deleted, "id": args["memory_id"],
    }))]


async def handle_auto_prune(args: Dict[str, Any]) -> List[TextContent]:
    """Auto-prune decayed memories."""
    result = await R.typed_memory_store.auto_prune(
        user_id=args["user_id"],
        strength_threshold=float(args.get("strength_threshold", 0.05)),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_get_relevant_context(args: Dict[str, Any]) -> List[TextContent]:
    """Pre-fetch conversation-relevant memories."""
    result = await R.typed_memory_store.get_relevant_context(
        user_id=args["user_id"],
        conversation_topic=args["conversation_topic"],
        limit=int(args.get("limit", 15)),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_user_timeline(args: Dict[str, Any]) -> List[TextContent]:
    """Chronological memory timeline."""
    result = await R.typed_memory_store.user_timeline(
        user_id=args["user_id"],
        limit=int(args.get("limit", 20)),
        offset=int(args.get("offset", 0)),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_semantic_search(args: Dict[str, Any]) -> List[TextContent]:
    """Semantic similarity search using embeddings."""
    if not R.embedding_store:
        return [TextContent(type="text", text=compact_json({"error": "Embedding store not initialized"}))]

    hits = await R.embedding_store.search(
        query=args["query"],
        limit=int(args.get("limit", 10)),
        min_similarity=float(args.get("min_similarity", 0.3)),
    )

    results = []
    for hit in hits:
        mem = await R.typed_memory_store.get(hit["memory_id"])
        if mem:
            results.append({
                "id": mem.id,
                "type": mem.memory_type.value,
                "content": mem.content,
                "similarity": hit["similarity"],
                "strength": round(mem.current_strength(), 3),
                "confidence": round(mem.confidence, 3),
                "tags": mem.tags,
            })

    return [TextContent(type="text", text=compact_json({
        "query": args["query"], "results": results, "count": len(results),
    }))]


async def handle_index_embeddings(args: Dict[str, Any]) -> List[TextContent]:
    """Batch-index all typed memories for a user."""
    if not R.embedding_store:
        return [TextContent(type="text", text=compact_json({"error": "Embedding store not initialized"}))]

    all_mems = await R.typed_memory_store.query(args["user_id"], limit=10000)

    batch = []
    for mem in all_mems:
        text = mem.content
        if mem.trigger:
            text += f" | trigger: {mem.trigger}"
        if mem.action:
            text += f" | action: {mem.action}"
        batch.append((mem.id, text))

    indexed = await R.embedding_store.index_batch(batch)
    stats = await R.embedding_store.get_stats()

    return [TextContent(type="text", text=compact_json({
        "newly_indexed": indexed,
        "total_memories": len(all_mems),
        "total_indexed": stats["indexed_count"],
        "model": stats["model"],
    }))]


async def handle_find_similar(args: Dict[str, Any]) -> List[TextContent]:
    """Find memories similar to a given memory."""
    if not R.embedding_store:
        return [TextContent(type="text", text=compact_json({"error": "Embedding store not initialized"}))]

    hits = await R.embedding_store.find_similar(
        memory_id=args["memory_id"],
        limit=int(args.get("limit", 5)),
        min_similarity=float(args.get("min_similarity", 0.5)),
    )

    results = []
    for hit in hits:
        mem = await R.typed_memory_store.get(hit["memory_id"])
        if mem:
            results.append({
                "id": mem.id,
                "type": mem.memory_type.value,
                "content": mem.content,
                "similarity": hit["similarity"],
                "tags": mem.tags,
            })

    return [TextContent(type="text", text=compact_json({
        "source_id": args["memory_id"], "similar": results, "count": len(results),
    }))]
