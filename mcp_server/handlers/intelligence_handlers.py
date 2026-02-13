"""
Memory Intelligence handlers — decay, consolidation, clustering, conflicts, etc.
Extracted from pltm_server.py for maintainability.
"""

from typing import Any, Dict, List

from mcp.types import TextContent

from mcp_server.handlers.registry import registry as R, compact_json


async def handle_process_conversation(args: Dict[str, Any]) -> List[TextContent]:
    """Process conversation through 3-lane pipeline."""
    if not R.typed_memory_pipeline:
        return [TextContent(type="text", text=compact_json({"error": "Pipeline not initialized"}))]

    result = await R.typed_memory_pipeline.process_conversation(
        user_id=args["user_id"],
        messages=args["messages"],
        context=args.get("context", ""),
    )

    return [TextContent(type="text", text=compact_json({
        "messages_processed": len(args["messages"]),
        "extracted": result.memories_extracted,
        "approved": result.memories_approved,
        "quarantined": result.memories_quarantined,
        "rejected": result.memories_rejected,
        "stored": result.memories_stored,
        "superseded": result.memories_superseded,
        "merged": result.memories_merged,
        "details": result.details,
    }))]


async def handle_pipeline_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Get 3-lane pipeline statistics."""
    if not R.typed_memory_pipeline:
        return [TextContent(type="text", text=compact_json({"error": "Pipeline not initialized"}))]
    return [TextContent(type="text", text=compact_json(R.typed_memory_pipeline.get_stats()))]


async def handle_apply_memory_decay(args: Dict[str, Any]) -> List[TextContent]:
    if not R.decay_engine:
        return [TextContent(type="text", text=compact_json({"error": "Not initialized"}))]
    result = await R.decay_engine.apply_decay(args["user_id"])
    return [TextContent(type="text", text=compact_json(result))]


async def handle_decay_forecast(args: Dict[str, Any]) -> List[TextContent]:
    if not R.decay_engine:
        return [TextContent(type="text", text=compact_json({"error": "Not initialized"}))]
    result = await R.decay_engine.get_decay_forecast(
        args["user_id"], hours_ahead=int(args.get("hours_ahead", 168)))
    return [TextContent(type="text", text=compact_json({"forecasts": result, "count": len(result)}))]


async def handle_consolidate_memories(args: Dict[str, Any]) -> List[TextContent]:
    if not R.consolidation_engine:
        return [TextContent(type="text", text=compact_json({"error": "Not initialized"}))]
    result = await R.consolidation_engine.consolidate(
        args["user_id"],
        min_cluster_size=int(args.get("min_cluster_size", 3)),
        similarity_threshold=float(args.get("similarity_threshold", 0.55)))
    return [TextContent(type="text", text=compact_json(result))]


async def handle_contextual_retrieve(args: Dict[str, Any]) -> List[TextContent]:
    if not R.contextual_retriever:
        return [TextContent(type="text", text=compact_json({"error": "Not initialized"}))]
    result = await R.contextual_retriever.retrieve_for_conversation(
        args["user_id"], args["messages"],
        max_memories=int(args.get("max_memories", 12)))
    return [TextContent(type="text", text=compact_json(result))]


async def handle_rank_by_importance(args: Dict[str, Any]) -> List[TextContent]:
    from src.memory.memory_intelligence import ImportanceScorer
    if not R.typed_memory_store:
        return [TextContent(type="text", text=compact_json({"error": "Not initialized"}))]
    result = await ImportanceScorer.rank_memories(
        R.typed_memory_store, args["user_id"], limit=int(args.get("limit", 50)))
    return [TextContent(type="text", text=compact_json({"ranked": result, "count": len(result)}))]


async def handle_surface_conflicts(args: Dict[str, Any]) -> List[TextContent]:
    if not R.conflict_surfacer:
        return [TextContent(type="text", text=compact_json({"error": "Not initialized"}))]
    result = await R.conflict_surfacer.detect_and_surface(args["user_id"])
    return [TextContent(type="text", text=compact_json({"conflicts": result, "count": len(result)}))]


async def handle_resolve_memory_conflict(args: Dict[str, Any]) -> List[TextContent]:
    if not R.conflict_surfacer:
        return [TextContent(type="text", text=compact_json({"error": "Not initialized"}))]
    result = await R.conflict_surfacer.resolve_conflict(
        args["conflict_id"], args["action"], args["user_id"])
    return [TextContent(type="text", text=compact_json(result))]


async def handle_memory_clusters(args: Dict[str, Any]) -> List[TextContent]:
    if not R.memory_clusterer:
        return [TextContent(type="text", text=compact_json({"error": "Not initialized"}))]
    result = await R.memory_clusterer.build_clusters(
        args["user_id"],
        similarity_threshold=float(args.get("similarity_threshold", 0.5)),
        min_cluster_size=int(args.get("min_cluster_size", 2)))
    return [TextContent(type="text", text=compact_json({"clusters": result, "count": len(result)}))]


async def handle_share_memory(args: Dict[str, Any]) -> List[TextContent]:
    if not R.shared_memory_mgr:
        return [TextContent(type="text", text=compact_json({"error": "Not initialized"}))]
    result = await R.shared_memory_mgr.share_memory(
        args["memory_id"], args["owner_id"], args["target_user_id"],
        permission=args.get("permission", "read"))
    return [TextContent(type="text", text=compact_json(result))]


async def handle_shared_with_me(args: Dict[str, Any]) -> List[TextContent]:
    if not R.shared_memory_mgr:
        return [TextContent(type="text", text=compact_json({"error": "Not initialized"}))]
    result = await R.shared_memory_mgr.get_shared_with_me(args["user_id"])
    return [TextContent(type="text", text=compact_json({"shared_memories": result, "count": len(result)}))]


async def handle_export_memory_profile(args: Dict[str, Any]) -> List[TextContent]:
    if not R.memory_portability:
        return [TextContent(type="text", text=compact_json({"error": "Not initialized"}))]
    result = await R.memory_portability.export_profile(args["user_id"])
    return [TextContent(type="text", text=compact_json(result))]


async def handle_import_memory_profile(args: Dict[str, Any]) -> List[TextContent]:
    if not R.memory_portability:
        return [TextContent(type="text", text=compact_json({"error": "Not initialized"}))]
    import json as _json
    try:
        profile = _json.loads(args["profile_json"])
    except Exception as e:
        return [TextContent(type="text", text=compact_json({"error": f"Invalid JSON: {e}"}))]
    result = await R.memory_portability.import_profile(
        profile, target_user_id=args.get("target_user_id"),
        merge=args.get("merge", True))
    return [TextContent(type="text", text=compact_json(result))]


async def handle_memory_provenance(args: Dict[str, Any]) -> List[TextContent]:
    if not R.provenance_tracker:
        return [TextContent(type="text", text=compact_json({"error": "Not initialized"}))]
    result = await R.provenance_tracker.get_provenance(args["memory_id"])
    if not result:
        return [TextContent(type="text", text=compact_json({"error": "No provenance found", "memory_id": args["memory_id"]}))]
    return [TextContent(type="text", text=compact_json(result))]


async def handle_apply_confidence_decay(args: Dict[str, Any]) -> List[TextContent]:
    if not R.confidence_decay_engine:
        return [TextContent(type="text", text=compact_json({"error": "Not initialized"}))]
    result = await R.confidence_decay_engine.apply_evidence_decay(args["user_id"])
    return [TextContent(type="text", text=compact_json({"adjustments": result, "count": len(result)}))]


async def handle_memory_audit(args: Dict[str, Any]) -> List[TextContent]:
    if not R.memory_auditor:
        return [TextContent(type="text", text=compact_json({"error": "Not initialized"}))]
    result = await R.memory_auditor.full_audit(args["user_id"])
    return [TextContent(type="text", text=compact_json(result))]
