"""
Shared component registry for handler modules.
Populated by initialize_pltm() in pltm_server.py.
Avoids circular imports between handlers and the main server.
"""

from typing import Any, Optional


class ComponentRegistry:
    """Holds references to all initialized PLTM components."""

    def __init__(self):
        self.store = None
        self.pipeline = None
        self.personality_agent = None
        self.personality_synth = None
        self.mood_tracker = None
        self.mood_patterns = None
        self.conflict_resolver = None
        self.contextual_personality = None
        self.typed_memory_store = None
        self.embedding_store = None
        self.typed_memory_pipeline = None
        self.decay_engine = None
        self.consolidation_engine = None
        self.contextual_retriever = None
        self.conflict_surfacer = None
        self.memory_clusterer = None
        self.shared_memory_mgr = None
        self.memory_portability = None
        self.provenance_tracker = None
        self.confidence_decay_engine = None
        self.memory_auditor = None
        self.phi_scorer = None
        self.criticality_pruner = None
        self.phi_consolidator = None
        self.phi_context_builder = None
        self.tool_analytics = None
        self.arch_snapshotter = None
        self.working_memory_compressor = None
        self.trajectory_encoder = None
        self.handoff_protocol = None


# Singleton registry — populated during server init
registry = ComponentRegistry()


def compact_json(obj) -> str:
    """Token-efficient JSON serialization - no whitespace"""
    import json
    return json.dumps(obj, separators=(',', ':'), default=str)
