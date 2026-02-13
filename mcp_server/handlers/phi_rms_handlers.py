"""
ΦRMS handlers — Phi Resource Management System tools.
Extracted handler functions using the shared registry pattern.
"""

from typing import Any, Dict, List

from mcp.types import TextContent

from mcp_server.handlers.registry import registry as R, compact_json


async def handle_phi_score_memories(args: Dict[str, Any]) -> List[TextContent]:
    if not R.phi_scorer:
        return [TextContent(type="text", text=compact_json({"error": "ΦRMS not initialized"}))]
    result = await R.phi_scorer.score_all(args["user_id"])
    return [TextContent(type="text", text=compact_json(result))]


async def handle_phi_prune(args: Dict[str, Any]) -> List[TextContent]:
    if not R.criticality_pruner:
        return [TextContent(type="text", text=compact_json({"error": "ΦRMS not initialized"}))]
    result = await R.criticality_pruner.prune(
        args["user_id"],
        target_token_savings=int(args.get("target_token_savings", 5000)),
        max_removals=int(args.get("max_removals", 20)))
    return [TextContent(type="text", text=compact_json(result))]


async def handle_phi_consolidate(args: Dict[str, Any]) -> List[TextContent]:
    if not R.phi_consolidator:
        return [TextContent(type="text", text=compact_json({"error": "ΦRMS not initialized"}))]
    result = await R.phi_consolidator.consolidate(
        args["user_id"],
        min_cluster_size=int(args.get("min_cluster_size", 3)),
        similarity_threshold=float(args.get("similarity_threshold", 0.55)))
    return [TextContent(type="text", text=compact_json(result))]


async def handle_phi_build_context(args: Dict[str, Any]) -> List[TextContent]:
    if not R.phi_context_builder:
        return [TextContent(type="text", text=compact_json({"error": "ΦRMS not initialized"}))]
    result = await R.phi_context_builder.build_context(
        args["user_id"],
        messages=args["messages"],
        token_budget=int(args.get("token_budget", 2000)))
    return [TextContent(type="text", text=compact_json(result))]
