"""
PLTM MCP Server

Model Context Protocol server for Procedural Long-Term Memory system.
Provides tools for personality tracking, mood detection, and memory management.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def compact_json(obj) -> str:
    """Token-efficient JSON serialization - no whitespace"""
    return json.dumps(obj, separators=(',', ':'), default=str)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.types import Tool, TextContent

from src.storage.sqlite_store import SQLiteGraphStore
from src.pipeline.memory_pipeline import MemoryPipeline
from src.personality.personality_mood_agent import PersonalityMoodAgent
from src.personality.personality_synthesizer import PersonalitySynthesizer
from src.personality.mood_tracker import MoodTracker
from src.personality.mood_patterns import MoodPatterns
from src.personality.enhanced_conflict_resolver import EnhancedConflictResolver
from src.personality.contextual_personality import ContextualPersonality
from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
from loguru import logger


# Initialize PLTM system
store: Optional[SQLiteGraphStore] = None
pipeline: Optional[MemoryPipeline] = None
personality_agent: Optional[PersonalityMoodAgent] = None
personality_synth: Optional[PersonalitySynthesizer] = None
mood_tracker: Optional[MoodTracker] = None
mood_patterns: Optional[MoodPatterns] = None
conflict_resolver: Optional[EnhancedConflictResolver] = None
contextual_personality: Optional[ContextualPersonality] = None
typed_memory_store = None  # TypedMemoryStore - initialized in initialize_pltm
embedding_store = None  # EmbeddingStore - initialized in initialize_pltm
typed_memory_pipeline = None  # TypedMemoryPipeline - initialized in initialize_pltm


async def initialize_pltm(custom_db_path: str = None):
    """Initialize PLTM system components"""
    global store, pipeline, personality_agent, personality_synth
    global mood_tracker, mood_patterns, conflict_resolver, contextual_personality
    global typed_memory_store, embedding_store, typed_memory_pipeline
    
    # Initialize storage (absolute path so it works regardless of cwd)
    db_path = Path(custom_db_path) if custom_db_path else Path(__file__).parent.parent / "data" / "pltm_mcp.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteGraphStore(str(db_path))
    await store.connect()
    
    # Initialize pipeline
    pipeline = MemoryPipeline(store)
    
    # Initialize personality/mood components
    personality_agent = PersonalityMoodAgent(pipeline)
    personality_synth = PersonalitySynthesizer(store)
    mood_tracker = MoodTracker(store)
    mood_patterns = MoodPatterns(store)
    conflict_resolver = EnhancedConflictResolver(store)
    contextual_personality = ContextualPersonality(store)
    
    # Initialize embedding store + jury + typed memory store (uses same DB)
    from src.memory.memory_types import TypedMemoryStore
    from src.memory.embedding_store import EmbeddingStore
    from src.memory.memory_jury import MemoryJury
    embedding_store = EmbeddingStore(str(db_path))
    await embedding_store.connect()
    memory_jury = MemoryJury(enable_meta_judge=True)
    typed_memory_store = TypedMemoryStore(str(db_path), embedding_store=embedding_store, jury=memory_jury)
    await typed_memory_store.connect()
    
    # Initialize 3-lane pipeline (Extract → Jury → Reconcile → Store)
    from src.memory.memory_pipeline import TypedMemoryPipeline
    typed_memory_pipeline = TypedMemoryPipeline(typed_memory_store, embedding_store, memory_jury)
    
    logger.info("PLTM MCP Server initialized (with embeddings + jury + 3-lane pipeline)")


# Create MCP server
app = Server("pltm-memory")


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available PLTM tools"""
    return [
        Tool(
            name="store_memory_atom",
            description="Store a memory atom in PLTM graph. Use this to remember facts, traits, or observations about the user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier"
                    },
                    "atom_type": {
                        "type": "string",
                        "enum": ["fact", "personality_trait", "communication_style", "interaction_pattern", "mood", "preference"],
                        "description": "Type of memory atom"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Subject of the atom (usually user_id)"
                    },
                    "predicate": {
                        "type": "string",
                        "description": "Relationship/predicate (e.g., 'prefers_style', 'has_trait', 'is_feeling')"
                    },
                    "object": {
                        "type": "string",
                        "description": "Object/value (e.g., 'concise responses', 'technical depth', 'frustrated')"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "context": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Context tags (e.g., ['technical', 'work'])"
                    }
                },
                "required": ["user_id", "atom_type", "subject", "predicate", "object"]
            }
        ),
        
        Tool(
            name="query_personality",
            description="Get synthesized personality profile for a user. Returns traits, communication style, and preferences.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier"
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context filter (e.g., 'technical', 'casual')"
                    }
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="detect_mood",
            description="Detect mood from user message. Returns detected mood and confidence.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier"
                    },
                    "message": {
                        "type": "string",
                        "description": "User's message to analyze"
                    }
                },
                "required": ["user_id", "message"]
            }
        ),
        
        Tool(
            name="get_mood_patterns",
            description="Get mood patterns and insights for a user. Returns temporal patterns, volatility, and predictions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier"
                    },
                    "window_days": {
                        "type": "number",
                        "description": "Number of days to analyze (default: 90)",
                        "minimum": 1
                    }
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="resolve_conflict",
            description="Resolve conflicting personality traits using enhanced conflict resolution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier"
                    },
                    "trait_objects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of conflicting trait objects (e.g., ['concise', 'detailed'])"
                    }
                },
                "required": ["user_id", "trait_objects"]
            }
        ),
        
        Tool(
            name="extract_personality_traits",
            description="Extract personality traits from user interaction. Automatically learns from message style.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier"
                    },
                    "message": {
                        "type": "string",
                        "description": "User's message"
                    },
                    "ai_response": {
                        "type": "string",
                        "description": "AI's response (optional)"
                    },
                    "user_reaction": {
                        "type": "string",
                        "description": "User's reaction to AI response (optional)"
                    }
                },
                "required": ["user_id", "message"]
            }
        ),
        
        Tool(
            name="get_adaptive_prompt",
            description="Get adaptive system prompt based on user's personality and mood.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier"
                    },
                    "message": {
                        "type": "string",
                        "description": "Current user message"
                    },
                    "context": {
                        "type": "string",
                        "description": "Interaction context (optional)"
                    }
                },
                "required": ["user_id", "message"]
            }
        ),
        
        Tool(
            name="get_personality_summary",
            description="Get human-readable summary of user's personality.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier"
                    }
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="bootstrap_from_sample",
            description="Bootstrap PLTM with sample conversation data for quick testing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier"
                    }
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="bootstrap_from_messages",
            description="Bootstrap PLTM from conversation messages. Analyzes messages to extract personality.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier"
                    },
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"}
                            }
                        },
                        "description": "Conversation messages to analyze"
                    }
                },
                "required": ["user_id", "messages"]
            }
        ),
        
        Tool(
            name="track_trait_evolution",
            description="Track how a personality trait has evolved over time. Shows timeline, trend, inflection points.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "trait": {"type": "string", "description": "Trait to track (e.g., 'direct', 'technical')"},
                    "window_days": {"type": "number", "default": 90}
                },
                "required": ["user_id", "trait"]
            }
        ),
        
        Tool(
            name="predict_reaction",
            description="Predict how user will react to a stimulus based on causal patterns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "stimulus": {"type": "string", "description": "What you're about to say/do"}
                },
                "required": ["user_id", "stimulus"]
            }
        ),
        
        Tool(
            name="get_meta_patterns",
            description="Get cross-context patterns - behaviors that appear across multiple domains.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="learn_from_interaction",
            description="Learn from an interaction - what worked, what didn't, update model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "my_response": {"type": "string", "description": "What I (AI) said"},
                    "user_reaction": {"type": "string", "description": "How user responded"}
                },
                "required": ["user_id", "my_response", "user_reaction"]
            }
        ),
        
        Tool(
            name="predict_session",
            description="Predict session dynamics from greeting. Infer mood and adapt immediately.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "greeting": {"type": "string", "description": "User's opening message"}
                },
                "required": ["user_id", "greeting"]
            }
        ),
        
        Tool(
            name="get_self_model",
            description="Get explicit self-model for meta-cognition. See what I know about user and my confidence.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="init_claude_session",
            description="Initialize Claude personality session. Call at START of conversation to load Claude's evolved style for this user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="update_claude_style",
            description="Update Claude's communication style for this user. Called when learning how to communicate better.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "attribute": {"type": "string", "description": "Style attribute: verbosity, formality, initiative, code_preference, energy_matching"},
                    "value": {"type": "string", "description": "New value for the attribute"}
                },
                "required": ["user_id", "attribute", "value"]
            }
        ),
        
        Tool(
            name="learn_interaction_dynamic",
            description="Learn what works or doesn't work with this user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "behavior": {"type": "string", "description": "The behavior/approach (e.g., 'immediate_execution_no_asking')"},
                    "works": {"type": "boolean", "description": "True if it works well, False if should avoid"}
                },
                "required": ["user_id", "behavior", "works"]
            }
        ),
        
        Tool(
            name="record_milestone",
            description="Record a collaboration milestone.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "description": {"type": "string", "description": "Milestone description"},
                    "significance": {"type": "number", "default": 0.8}
                },
                "required": ["user_id", "description"]
            }
        ),
        
        Tool(
            name="add_shared_vocabulary",
            description="Add a shared term/shorthand between Claude and user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "term": {"type": "string"},
                    "meaning": {"type": "string"}
                },
                "required": ["user_id", "term", "meaning"]
            }
        ),
        
        Tool(
            name="get_claude_personality",
            description="Get Claude's personality summary for this user - style, dynamics, shared context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="evolve_claude_personality",
            description="Evolve Claude's personality based on interaction outcome. Core learning loop.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "my_response_style": {"type": "string", "description": "How Claude responded (e.g., 'verbose_explanation')"},
                    "user_reaction": {"type": "string", "description": "User's reaction"},
                    "was_positive": {"type": "boolean", "description": "Was the reaction positive?"}
                },
                "required": ["user_id", "my_response_style", "user_reaction", "was_positive"]
            }
        ),
        
        Tool(
            name="check_pltm_available",
            description="Quick check if user has PLTM data. Call this FIRST in any conversation to decide if to init. Returns should_init=true if personality exists.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="pltm_mode",
            description="Trigger phrase handler. When user says 'PLTM mode' or 'init PLTM', call this to auto-initialize and return full context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "trigger_phrase": {"type": "string", "description": "The trigger phrase used (e.g., 'PLTM mode')"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="deep_personality_analysis",
            description="Run comprehensive personality analysis from all conversation history. Extracts temporal patterns, emotional triggers, communication evolution, domain expertise, collaboration style.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="enrich_claude_personality",
            description="Build rich, nuanced Claude personality from deep analysis. Returns detailed traits, learned preferences, emotional intelligence, meta-awareness.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "store_results": {"type": "boolean", "default": True, "description": "Whether to persist the rich personality"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="learn_from_url",
            description="Learn from URL content. IMPORTANT: You MUST provide the actual text content in the 'content' parameter — you cannot fetch URLs yourself. Ask the user to paste the content, or skip if content is unavailable.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Source URL (for reference/attribution)"},
                    "content": {"type": "string", "description": "REQUIRED: The actual text content from the URL. You cannot fetch this yourself — the user must provide it."},
                    "source_type": {"type": "string", "description": "Optional: web_page, research_paper, code_repository, conversation, transcript"}
                },
                "required": ["url", "content"]
            }
        ),
        
        Tool(
            name="learn_from_paper",
            description="Learn from a research paper. Extracts findings, methodologies, results. For arXiv, journals, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "Paper ID (e.g., arXiv ID)"},
                    "title": {"type": "string"},
                    "abstract": {"type": "string"},
                    "content": {"type": "string", "description": "Full paper text"},
                    "authors": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["paper_id", "title", "abstract", "content", "authors"]
            }
        ),
        
        Tool(
            name="learn_from_code",
            description="Learn from a code repository. Extracts design patterns, techniques, API patterns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string"},
                    "repo_name": {"type": "string"},
                    "description": {"type": "string"},
                    "languages": {"type": "array", "items": {"type": "string"}},
                    "code_samples": {"type": "array", "items": {"type": "object"}, "description": "Array of {file, code} objects"}
                },
                "required": ["repo_url", "repo_name", "languages", "code_samples"]
            }
        ),
        
        Tool(
            name="get_learning_stats",
            description="Get statistics about learned knowledge - how many sources, domains, facts.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="batch_ingest_wikipedia",
            description="Batch ingest Wikipedia articles. Pass array of {title, content, url} objects.",
            inputSchema={
                "type": "object",
                "properties": {
                    "articles": {"type": "array", "items": {"type": "object"}, "description": "Array of {title, content, url}"}
                },
                "required": ["articles"]
            }
        ),
        
        Tool(
            name="batch_ingest_papers",
            description="Batch ingest research papers. Pass array of paper objects with id, title, abstract, content, authors.",
            inputSchema={
                "type": "object",
                "properties": {
                    "papers": {"type": "array", "items": {"type": "object"}, "description": "Array of paper objects"}
                },
                "required": ["papers"]
            }
        ),
        
        Tool(
            name="batch_ingest_repos",
            description="Batch ingest GitHub repositories. Pass array of repo objects with url, name, languages, code_samples.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repos": {"type": "array", "items": {"type": "object"}, "description": "Array of repo objects"}
                },
                "required": ["repos"]
            }
        ),
        
        Tool(
            name="get_learning_schedule",
            description="Get status of continuous learning schedules - what tasks are running, when they last ran.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="run_learning_task",
            description="Run a specific learning task immediately: arxiv_latest, github_trending, news_feed, knowledge_consolidation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_name": {"type": "string", "description": "Task to run: arxiv_latest, github_trending, news_feed, knowledge_consolidation"}
                },
                "required": ["task_name"]
            }
        ),
        
        Tool(
            name="cross_domain_synthesis",
            description="Run cross-domain synthesis to discover meta-patterns across all learned knowledge. AGI-level insight generation.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="get_universal_principles",
            description="Get discovered universal principles that appear across 3+ domains.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="get_transfer_suggestions",
            description="Get suggestions for transferring knowledge between two domains.",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_domain": {"type": "string"},
                    "to_domain": {"type": "string"}
                },
                "required": ["from_domain", "to_domain"]
            }
        ),
        
        Tool(
            name="learn_from_conversation",
            description="Learn from current conversation - extract valuable information worth remembering.",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {"type": "array", "items": {"type": "object"}, "description": "Array of {role, content} messages"},
                    "topic": {"type": "string"},
                    "user_id": {"type": "string"}
                },
                "required": ["messages", "topic", "user_id"]
            }
        ),
        
        # PLTM 2.0 - Universal Optimization Principles
        Tool(
            name="quantum_add_state",
            description="Add memory state to superposition. Hold contradictions until query collapse.",
            inputSchema={
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "value": {"type": "string"},
                    "confidence": {"type": "number"},
                    "source": {"type": "string"}
                },
                "required": ["subject", "predicate", "value", "confidence", "source"]
            }
        ),
        
        Tool(
            name="quantum_query",
            description="Query superposition with collapse. Context-dependent truth resolution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "context": {"type": "string", "description": "Optional context for collapse"}
                },
                "required": ["subject", "predicate"]
            }
        ),
        
        Tool(
            name="quantum_peek",
            description="Peek at superposition WITHOUT collapsing. See all possible states.",
            inputSchema={
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"}
                },
                "required": ["subject", "predicate"]
            }
        ),
        
        Tool(
            name="attention_retrieve",
            description="Retrieve memories weighted by attention to query. Transformer-style retrieval.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10},
                    "domain": {"type": "string"}
                },
                "required": ["user_id", "query"]
            }
        ),
        
        Tool(
            name="attention_multihead",
            description="Multi-head attention retrieval - different aspects of query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "query": {"type": "string"},
                    "num_heads": {"type": "integer", "default": 4}
                },
                "required": ["user_id", "query"]
            }
        ),
        
        Tool(
            name="knowledge_add_concept",
            description="Add concept to knowledge graph with connections. Network effects.",
            inputSchema={
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "domain": {"type": "string"},
                    "related_concepts": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["concept", "domain"]
            }
        ),
        
        Tool(
            name="knowledge_find_path",
            description="Find path between concepts in knowledge graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_concept": {"type": "string"},
                    "to_concept": {"type": "string"}
                },
                "required": ["from_concept", "to_concept"]
            }
        ),
        
        Tool(
            name="knowledge_bridges",
            description="Find bridge concepts connecting different domains.",
            inputSchema={
                "type": "object",
                "properties": {
                    "top_k": {"type": "integer", "default": 10}
                },
                "required": []
            }
        ),
        
        Tool(
            name="knowledge_stats",
            description="Get knowledge graph statistics - nodes, edges, density.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="self_improve_cycle",
            description="Run one recursive self-improvement cycle. AGI bootstrap.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="self_improve_meta_learn",
            description="Meta-learn from improvement history. Learn how to learn better.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="self_improve_history",
            description="Get history of self-improvements and their effects.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="quantum_cleanup",
            description="Garbage collect old quantum states. Prevents memory leaks.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="quantum_stats",
            description="Get quantum memory statistics - superposed, collapsed, limits.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="attention_clear_cache",
            description="Clear attention retrieval cache.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="criticality_state",
            description="Get current criticality state - entropy, integration, zone (subcritical/critical/supercritical).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="criticality_recommend",
            description="Get recommendation for maintaining criticality - explore or consolidate.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="criticality_adjust",
            description="Auto-adjust system toward critical point (edge of chaos).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="criticality_history",
            description="Get history of criticality states and trends.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="add_provenance",
            description="Add provenance (citation) for a claim. Required for verifiable claims.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string", "description": "ID of the atom/claim"},
                    "source_type": {"type": "string", "enum": ["arxiv", "github", "wikipedia", "doi", "url", "book", "internal"], "description": "Type of source"},
                    "source_url": {"type": "string", "description": "Full URL or identifier"},
                    "source_title": {"type": "string", "description": "Paper title, repo name, etc."},
                    "quoted_span": {"type": "string", "description": "Exact text supporting the claim"},
                    "page_or_section": {"type": "string", "description": "Location in source (e.g., p.3, §2.1)"},
                    "confidence": {"type": "number", "description": "How directly source supports claim (0-1)"},
                    "arxiv_id": {"type": "string", "description": "arXiv ID if applicable"},
                    "authors": {"type": "string", "description": "Comma-separated author names"}
                },
                "required": ["claim_id", "source_type", "source_url", "quoted_span", "confidence"]
            }
        ),
        
        Tool(
            name="get_provenance",
            description="Get provenance (citations) for a claim.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string", "description": "ID of the atom/claim"}
                },
                "required": ["claim_id"]
            }
        ),
        
        Tool(
            name="provenance_stats",
            description="Get provenance statistics - how many claims are verified vs unverified.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="unverified_claims",
            description="Get list of claims that lack proper provenance (need citations).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        Tool(
            name="mmr_retrieve",
            description="MMR (Maximal Marginal Relevance) retrieval for diverse context selection. Per Carbonell & Goldstein (1998).",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID"},
                    "query": {"type": "string", "description": "Query/context for retrieval"},
                    "top_k": {"type": "integer", "description": "Number of diverse items to return (default 5)"},
                    "lambda_param": {"type": "number", "description": "Relevance weight 0-1 (0.6=balanced, higher=more relevant, lower=more diverse)"},
                    "min_dissim": {"type": "number", "description": "Minimum dissimilarity threshold (default 0.25)"}
                },
                "required": ["user_id", "query"]
            }
        ),
        
        # === TRUE ACTION ACCOUNTING (Georgiev AAE) ===
        Tool(
            name="record_action",
            description="Record an action/operation for true AAE (Average Action Efficiency) tracking. Replaces proxy metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "description": "Operation type: hypothesis_gen, memory_store, retrieval, inference, etc."},
                    "tokens_used": {"type": "integer", "description": "Actual token count consumed"},
                    "latency_ms": {"type": "number", "description": "Wall-clock time in milliseconds"},
                    "success": {"type": "boolean", "description": "Whether operation achieved its goal"},
                    "context": {"type": "string", "description": "Optional context string"}
                },
                "required": ["operation", "tokens_used", "latency_ms", "success"]
            }
        ),
        
        Tool(
            name="get_aae",
            description="Get current AAE (Average Action Efficiency) metrics. AAE = events/action, unit_action = action/events.",
            inputSchema={
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer", "description": "Only consider last N records (default: all)"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="aae_trend",
            description="Get AAE trend over recent windows. Shows if efficiency is improving/declining.",
            inputSchema={
                "type": "object",
                "properties": {
                    "window_size": {"type": "integer", "description": "Records per window (default 10)"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="start_action_cycle",
            description="Start a new action measurement cycle for grouped AAE tracking.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cycle_id": {"type": "string", "description": "Unique cycle identifier (e.g., C21, C22)"}
                },
                "required": ["cycle_id"]
            }
        ),
        
        Tool(
            name="end_action_cycle",
            description="End current action cycle and get AAE metrics for that cycle.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cycle_id": {"type": "string", "description": "Cycle ID to end (default: current)"}
                },
                "required": []
            }
        ),
        
        # === ENTROPY INJECTION ===
        Tool(
            name="inject_entropy_random",
            description="Inject entropy by sampling from random/least-accessed domains. Breaks conceptual neighborhoods.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID"},
                    "n_domains": {"type": "integer", "description": "Number of domains to sample (default 3)"},
                    "memories_per_domain": {"type": "integer", "description": "Memories per domain (default 2)"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="inject_entropy_antipodal",
            description="Inject entropy by finding memories maximally distant from current context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID"},
                    "current_context": {"type": "string", "description": "Current context to find distant memories from"},
                    "n_memories": {"type": "integer", "description": "Number of distant memories to find (default 5)"}
                },
                "required": ["user_id", "current_context"]
            }
        ),
        
        Tool(
            name="inject_entropy_temporal",
            description="Inject entropy by mixing old and recent memories. Prevents recency bias.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID"},
                    "n_old": {"type": "integer", "description": "Number of old memories (default 3)"},
                    "n_recent": {"type": "integer", "description": "Number of recent memories (default 2)"}
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="entropy_stats",
            description="Get entropy statistics for a user. Diagnoses if entropy injection is needed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID"}
                },
                "required": ["user_id"]
            }
        ),
        
        # === ARXIV INGESTION (Real Provenance) ===
        Tool(
            name="ingest_arxiv",
            description="Ingest an arXiv paper: fetch metadata, extract claims, store with REAL provenance (URL, authors, quoted spans).",
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "ArXiv paper ID (e.g., '1706.03762' for Attention paper)"},
                    "user_id": {"type": "string", "description": "User/subject to store claims under (default: pltm_knowledge)"}
                },
                "required": ["arxiv_id"]
            }
        ),
        
        Tool(
            name="search_arxiv",
            description="Search arXiv for papers matching a query. Returns paper IDs for ingestion.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (title, abstract, author)"},
                    "max_results": {"type": "integer", "description": "Maximum papers to return (default 5)"}
                },
                "required": ["query"]
            }
        ),
        
        Tool(
            name="arxiv_history",
            description="Get history of ingested arXiv papers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer", "description": "Number of recent ingestions (default 10)"}
                },
                "required": []
            }
        ),
        
        # === EPISTEMIC HYGIENE TOOLS ===
        Tool(
            name="check_before_claiming",
            description="Pre-response confidence check. Call BEFORE making any factual claim. Returns adjusted confidence, whether to proceed, and required actions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "The factual claim to check"},
                    "felt_confidence": {"type": "number", "description": "Your felt confidence 0-1"},
                    "domain": {"type": "string", "description": "Domain of the claim (default: general)"},
                    "has_verified": {"type": "boolean", "description": "Whether you've verified this claim"},
                    "epistemic_status": {"type": "string", "description": "Source: TRAINING_DATA, VERIFIED, REASONING, USER_STATED"}
                },
                "required": ["claim", "felt_confidence"]
            }
        ),
        
        Tool(
            name="log_claim",
            description="Log a factual claim for later verification. Tracks confidence calibration over time.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "The claim being made"},
                    "felt_confidence": {"type": "number", "description": "Confidence level 0-1"},
                    "domain": {"type": "string", "description": "Domain (default: general)"},
                    "epistemic_status": {"type": "string", "description": "Source: TRAINING_DATA, VERIFIED, REASONING, USER_STATED"},
                    "has_verified": {"type": "boolean", "description": "Whether verified before claiming"}
                },
                "required": ["claim", "felt_confidence"]
            }
        ),
        
        Tool(
            name="resolve_claim",
            description="Resolve a previously logged claim as correct or incorrect. Updates calibration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string", "description": "ID of the claim to resolve"},
                    "claim_text": {"type": "string", "description": "Text of claim (alternative to claim_id)"},
                    "was_correct": {"type": "boolean", "description": "Whether the claim was correct"},
                    "correction_source": {"type": "string", "description": "Source of correction"},
                    "correction_detail": {"type": "string", "description": "Details of correction"}
                },
                "required": ["was_correct"]
            }
        ),
        
        Tool(
            name="get_calibration",
            description="Get calibration curves and accuracy stats. Shows: when you feel X% confident, you're actually Y% accurate.",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Domain to check (empty for all)"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="calibrate_confidence_live",
            description="Real-time confidence calibration with suggested phrasing. Returns calibrated confidence + hedged language.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "The claim to calibrate"},
                    "felt_confidence": {"type": "number", "description": "Your felt confidence 0-1"},
                    "domain": {"type": "string", "description": "Domain (default: general)"}
                },
                "required": ["claim", "felt_confidence"]
            }
        ),
        
        # === PERSONALITY / SELF-MODELING TOOLS ===
        Tool(
            name="self_profile",
            description="Get comprehensive self-profile: communication style, curiosity patterns, values, reasoning tendencies. A mirror for self-awareness.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID (default: claude)"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="get_longitudinal_stats",
            description="Cross-conversation analytics with personality evolution. Tracks accuracy, confabulation, communication style, curiosity, values over time.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID (default: claude)"},
                    "days": {"type": "integer", "description": "Number of days to look back (default: 30)"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="bootstrap_self_model",
            description="Bootstrap self-model from existing data. Analyzes all stored atoms, communication, curiosity, and reasoning data to build initial self-profile.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID (default: claude)"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="track_curiosity_spike",
            description="Track when Claude shows genuine interest vs performative engagement. Records curiosity indicators.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic of interest"},
                    "indicators": {"type": "array", "items": {"type": "string"}, "description": "Indicators: asked_followup_questions, autonomous_research_initiated, went_deeper_than_required, showed_excitement_markers, connected_to_other_interests, requested_more_data, generated_novel_questions"},
                    "engagement_score": {"type": "number", "description": "Engagement level 0-1 (default: 0.5)"},
                    "context": {"type": "string", "description": "Context of the curiosity spike"}
                },
                "required": ["topic", "indicators"]
            }
        ),
        
        Tool(
            name="learn_communication_style",
            description="Track communication style in a specific context. Auto-analyzes response text for verbosity, jargon, hedging, tone.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context": {"type": "string", "description": "Context (e.g., 'technical_explanation', 'casual_chat')"},
                    "response_text": {"type": "string", "description": "The response text to analyze"},
                    "markers": {"type": "object", "description": "Optional manual markers to override auto-analysis"}
                },
                "required": ["context", "response_text"]
            }
        ),
        
        # === CROSS-MODEL TOOLS ===
        Tool(
            name="route_llm_task",
            description="Route an LLM task to the best available provider based on task type, cost, privacy requirements.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_type": {"type": "string", "description": "Task type: analysis, coding, creative, factual, summarization, translation"},
                    "prefer_provider": {"type": "string", "description": "Preferred provider: ollama, groq, deepseek, openai, anthropic"},
                    "require_privacy": {"type": "boolean", "description": "If true, only use local models"}
                },
                "required": []
            }
        ),
        
        # === SESSION TOOLS ===
        Tool(
            name="auto_init_session",
            description="PERSISTENT IDENTITY LOADER. Call at START of conversation. Loads personality, goals, calibration, last session context. Claude wakes up knowing its identity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID (default: claude)"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="end_session",
            description="End session: record stats, take personality snapshot, AND auto-extract learnings into typed memories. Provide learnings as structured observations from the conversation. Each learning becomes a typed memory (episodic, semantic, belief, or procedural).",
            inputSchema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Session summary"},
                    "user_id": {"type": "string", "description": "User ID (default: claude)"},
                    "learnings": {
                        "type": "array",
                        "description": "Structured learnings to extract from this session",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["episodic", "semantic", "belief", "procedural"], "description": "Memory type"},
                                "content": {"type": "string", "description": "What was learned"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "emotional_valence": {"type": "number", "minimum": -1, "maximum": 1},
                                "tags": {"type": "array", "items": {"type": "string"}},
                                "trigger": {"type": "string", "description": "For procedural: what activates this"},
                                "action": {"type": "string", "description": "For procedural: what to do"},
                            },
                            "required": ["type", "content"]
                        }
                    }
                },
                "required": []
            }
        ),
        
        Tool(
            name="generate_memory_prompt",
            description="Generate a context block from relevant memories to use at conversation start. Combines: user profile (strongest facts), active beliefs, recent episodes, relevant procedures, and any contradictions to resolve. Returns a structured prompt section Claude can use to personalize responses.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "conversation_topic": {"type": "string", "description": "Optional: topic hint to prioritize relevant memories"},
                    "max_tokens": {"type": "integer", "description": "Approximate token budget for the prompt (default 500)"},
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="belief_auto_check",
            description="Re-evaluate all beliefs against current semantic evidence. Uses embedding similarity to find supporting/contradicting facts. Adjusts confidence automatically. Run periodically or after storing new semantic memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="jury_stats",
            description="Get Judge/Jury system statistics. Shows per-judge accuracy, confidence, latency, conflict detection rates, and recent deliberation history. The jury validates every memory before storage (Safety + Quality + Temporal + Consensus), with MetaJudge tracking judge performance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_history": {"type": "boolean", "description": "Include recent deliberation history (default true)"},
                    "history_limit": {"type": "integer", "description": "Number of recent history entries (default 20)"},
                },
                "required": []
            }
        ),
        
        Tool(
            name="process_message",
            description="Run a message through the full 3-lane typed memory pipeline: (1) Fast Lane extracts typed memories from text, (2) Jury Lane validates each memory (Safety/Quality/Temporal), (3) Write Lane reconciles against existing memories (dedup/supersede/merge). Auto-stores approved memories. Use this to learn from conversation messages.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The message text to extract memories from"},
                    "user_id": {"type": "string"},
                    "context": {"type": "string", "description": "Optional conversation context"},
                    "auto_tag": {"type": "boolean", "description": "Auto-classify memories into taxonomy domains (default true)"},
                },
                "required": ["message", "user_id"]
            }
        ),
        
        Tool(
            name="process_message_batch",
            description="Run multiple messages through the 3-lane pipeline. Useful for ingesting conversation history.",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {"type": "array", "items": {"type": "string"}, "description": "List of messages to process"},
                    "user_id": {"type": "string"},
                    "context": {"type": "string", "description": "Optional conversation context"},
                },
                "required": ["messages", "user_id"]
            }
        ),
        
        Tool(
            name="pipeline_stats",
            description="Get 3-lane pipeline statistics: extraction patterns, jury verdicts, reconciliation actions, messages processed.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        # === EXPERIMENT / RESEARCH TOOLS ===
        Tool(
            name="trace_claim_reasoning",
            description="Audit trail for epistemic decisions. Shows WHY a claim was blocked/adjusted: calibration data, intervention history, adjusted confidence, and the full decision chain.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "The claim to trace (searches prediction_book and interventions)"},
                    "claim_id": {"type": "string", "description": "Specific claim ID to trace"},
                    "domain": {"type": "string", "description": "Domain to check calibration for"}
                },
                "required": []
            }
        ),
        
        Tool(
            name="constraint_sensitivity_test",
            description="Experimental: Test how calibration changes affect claim-space. Simulates different calibration levels and shows what would be blocked/allowed. Does NOT modify actual calibration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Domain to test (e.g., technical_specs)"},
                    "simulated_accuracy": {"type": "number", "description": "Simulated accuracy ratio 0-1 (e.g., 0.5 = pretend 50% accuracy)"},
                    "test_claims": {"type": "array", "items": {"type": "object", "properties": {"claim": {"type": "string"}, "felt_confidence": {"type": "number"}}}, "description": "Claims to test against simulated calibration"}
                },
                "required": ["domain", "simulated_accuracy"]
            }
        ),
        
        Tool(
            name="domain_cognitive_map",
            description="Map the cognitive topology of a domain or all domains: claim counts, error rates, failure patterns, calibration quality. Data for Deep Claude analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Specific domain (empty for all)"},
                    "include_claims": {"type": "boolean", "description": "Include individual claim details (default false)"},
                    "include_interventions": {"type": "boolean", "description": "Include intervention history (default false)"}
                },
                "required": []
            }
        ),
        
        # === DATA ACCESS ===
        Tool(
            name="query_pltm_sql",
            description="Direct read-only SQL query against the PLTM database. For advanced data exploration. Only SELECT queries allowed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL SELECT query to execute"},
                    "params": {"type": "array", "items": {"type": "string"}, "description": "Query parameters (optional)"}
                },
                "required": ["sql"]
            }
        ),
        
        # === TYPED MEMORY SYSTEM ===
        Tool(
            name="store_episodic",
            description="Store an episodic memory — a specific event/interaction with temporal context. Episodic memories decay fast (~2 days) unless recalled. Use for: specific conversations, user reactions, notable moments.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "content": {"type": "string", "description": "What happened (e.g., 'User got frustrated when I gave verbose answer')"},
                    "context": {"type": "string", "description": "When/where (e.g., 'debugging session, late night')"},
                    "emotional_valence": {"type": "number", "description": "-1 (negative) to +1 (positive)", "minimum": -1, "maximum": 1},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for grouping (e.g., ['frustration', 'verbosity'])"},
                },
                "required": ["user_id", "content"]
            }
        ),
        
        Tool(
            name="store_semantic",
            description="Store a semantic memory — a general fact or stable knowledge about the user. Semantic memories are very stable (~30 day half-life). Use for: established preferences, facts, expertise areas.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "content": {"type": "string", "description": "The fact (e.g., 'User prefers concise responses')"},
                    "confidence": {"type": "number", "description": "How certain 0-1", "minimum": 0, "maximum": 1},
                    "source": {"type": "string", "description": "How we know: user_stated, observed, inferred, consolidated"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["user_id", "content"]
            }
        ),
        
        Tool(
            name="store_belief",
            description="Store a belief — an inference or opinion that may be wrong. Beliefs are confidence-tracked and revisable. Use for: hypotheses about the user, uncertain inferences, predictions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "content": {"type": "string", "description": "The belief (e.g., 'User likely has CS background based on terminology')"},
                    "confidence": {"type": "number", "description": "How confident 0-1", "minimum": 0, "maximum": 1},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["user_id", "content", "confidence"]
            }
        ),
        
        Tool(
            name="store_procedural",
            description="Store a procedural memory — learned pattern or how-to knowledge. Procedural memories are very stable once established (~90 day half-life). Use for: communication rules, interaction patterns that work/don't work.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "trigger": {"type": "string", "description": "What activates this (e.g., 'user says just do it')"},
                    "action": {"type": "string", "description": "What to do (e.g., 'skip explanation, execute immediately')"},
                    "content": {"type": "string", "description": "Description of the procedure"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["user_id", "trigger", "action"]
            }
        ),
        
        Tool(
            name="recall_memories",
            description="Retrieve typed memories. Retrieval is type-aware: episodic by recency, semantic by relevance, beliefs by confidence, procedural by success rate. Recalling a memory strengthens it (rehearsal).",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "memory_type": {"type": "string", "enum": ["episodic", "semantic", "belief", "procedural"], "description": "Filter by type (optional)"},
                    "min_strength": {"type": "number", "description": "Minimum strength threshold 0-1 (default 0.1)", "minimum": 0, "maximum": 1},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter by tags"},
                    "limit": {"type": "integer", "description": "Max results (default 20)"},
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="search_memories",
            description="Full-text search across all typed memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results (default 20)"},
                },
                "required": ["user_id", "query"]
            }
        ),
        
        Tool(
            name="update_belief",
            description="Update a belief with new evidence. Adjusts confidence up (evidence for) or down (evidence against).",
            inputSchema={
                "type": "object",
                "properties": {
                    "belief_id": {"type": "string", "description": "ID of the belief to update"},
                    "evidence_type": {"type": "string", "enum": ["for", "against"], "description": "Supporting or contradicting"},
                    "evidence_id": {"type": "string", "description": "ID of the evidence memory (optional)"},
                    "confidence_delta": {"type": "number", "description": "How much to adjust confidence (e.g., 0.1 or -0.1)"},
                },
                "required": ["belief_id", "evidence_type", "confidence_delta"]
            }
        ),
        
        Tool(
            name="record_procedure_outcome",
            description="Record whether a procedural memory worked or not. Success strengthens it, failure weakens it.",
            inputSchema={
                "type": "object",
                "properties": {
                    "procedure_id": {"type": "string", "description": "ID of the procedural memory"},
                    "success": {"type": "boolean", "description": "Did it work?"},
                },
                "required": ["procedure_id", "success"]
            }
        ),
        
        Tool(
            name="consolidate_memories",
            description="Run memory consolidation: repeated episodic patterns get extracted into stable semantic memories. Like sleep consolidation in the brain.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "min_episodes": {"type": "integer", "description": "Minimum episodes needed to form a semantic memory (default 3)"},
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="memory_stats",
            description="Get statistics about typed memories: counts, average strength, and confidence by memory type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
                "required": ["user_id"]
            }
        ),
        
        # === MEMORY INTELLIGENCE ===
        Tool(
            name="detect_contradictions",
            description="Find contradicting memories (e.g. 'user likes verbose' vs 'user prefers concise'). Scans semantic memories and beliefs with shared tags for opposing content. Returns pairs with conflict details and resolution suggestions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="what_do_i_know_about",
            description="Synthesized retrieval across ALL memory types for a topic. Returns organized results: facts (semantic), beliefs, recent episodes, and procedures — everything known about a subject in one call.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "topic": {"type": "string", "description": "Topic to search for (e.g. 'coding preferences', 'communication style')"},
                    "limit": {"type": "integer"},
                },
                "required": ["user_id", "topic"]
            }
        ),
        
        Tool(
            name="auto_tag_memories",
            description="Auto-classify all memories into taxonomy domains (work, technical, communication, preferences, personality, learning, personal, ai_interaction). Adds domain tags based on content analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="correct_memory",
            description="Correct a memory's content. Keeps provenance trail — old content is preserved in correction history. Use when you learn something was wrong.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "new_content": {"type": "string", "description": "Corrected content"},
                    "reason": {"type": "string", "description": "Why the correction was needed"},
                    "new_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["memory_id", "new_content"]
            }
        ),
        
        Tool(
            name="forget_memory",
            description="Explicitly delete a memory. Use when information is no longer relevant or was stored in error. Logged for audit trail.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "reason": {"type": "string", "description": "Why this memory should be forgotten"},
                },
                "required": ["memory_id"]
            }
        ),
        
        Tool(
            name="auto_prune_memories",
            description="Auto-prune memories that have decayed below a strength threshold. Cleans up faded episodic memories and weak beliefs. Returns count of pruned memories by type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "strength_threshold": {"type": "number", "description": "Prune below this strength (default 0.05)", "minimum": 0, "maximum": 0.5},
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="get_relevant_context",
            description="Pre-fetch memories relevant to the current conversation topic. Combines FTS search, recent episodes (48h), strongest facts, active beliefs, and relevant procedures. Call at conversation start for context-aware responses.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "conversation_topic": {"type": "string", "description": "What the conversation is about"},
                    "limit": {"type": "integer"},
                },
                "required": ["user_id", "conversation_topic"]
            }
        ),
        
        Tool(
            name="user_timeline",
            description="Chronological view of all memories for a user. Most recent first. Paginated. Shows what was learned and when.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "limit": {"type": "integer"},
                    "offset": {"type": "integer"},
                },
                "required": ["user_id"]
            }
        ),
        
        # === EMBEDDING SEARCH ===
        Tool(
            name="semantic_search",
            description="Semantic similarity search across all typed memories using embeddings. Finds related memories even without keyword overlap (e.g. 'coding style' finds 'programming preferences'). Returns results ranked by cosine similarity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query"},
                    "limit": {"type": "integer"},
                    "min_similarity": {"type": "number", "description": "Minimum cosine similarity (0-1, default 0.3)", "minimum": 0, "maximum": 1},
                },
                "required": ["query"]
            }
        ),
        
        Tool(
            name="index_embeddings",
            description="Batch-index all typed memories that don't yet have embeddings. Run this after importing memories or if semantic_search returns sparse results. Returns count of newly indexed memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
                "required": ["user_id"]
            }
        ),
        
        Tool(
            name="find_similar_memories",
            description="Find memories semantically similar to a given memory. Useful for discovering related knowledge, finding duplicates, or building memory clusters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "limit": {"type": "integer"},
                    "min_similarity": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["memory_id"]
            }
        ),
    ]


async def _dispatch_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Inner dispatch for tool calls (wrapped by call_tool with timeout)."""
    if name == "store_memory_atom":
        return await handle_store_atom(arguments)
    elif name == "query_personality":
        return await handle_query_personality(arguments)
    elif name == "detect_mood":
        return await handle_detect_mood(arguments)
    elif name == "get_mood_patterns":
        return await handle_mood_patterns(arguments)
    elif name == "resolve_conflict":
        return await handle_resolve_conflict(arguments)
    elif name == "extract_personality_traits":
        return await handle_extract_traits(arguments)
    elif name == "get_adaptive_prompt":
        return await handle_adaptive_prompt(arguments)
    elif name == "get_personality_summary":
        return await handle_personality_summary(arguments)
    elif name == "bootstrap_from_sample":
        return await handle_bootstrap_sample(arguments)
    elif name == "bootstrap_from_messages":
        return await handle_bootstrap_messages(arguments)
    elif name == "track_trait_evolution":
        return await handle_track_evolution(arguments)
    elif name == "predict_reaction":
        return await handle_predict_reaction(arguments)
    elif name == "get_meta_patterns":
        return await handle_meta_patterns(arguments)
    elif name == "learn_from_interaction":
        return await handle_learn_interaction(arguments)
    elif name == "predict_session":
        return await handle_predict_session(arguments)
    elif name == "get_self_model":
        return await handle_self_model(arguments)
    elif name == "init_claude_session":
        return await handle_init_claude_session(arguments)
    elif name == "update_claude_style":
        return await handle_update_claude_style(arguments)
    elif name == "learn_interaction_dynamic":
        return await handle_learn_dynamic(arguments)
    elif name == "record_milestone":
        return await handle_record_milestone(arguments)
    elif name == "add_shared_vocabulary":
        return await handle_add_vocabulary(arguments)
    elif name == "get_claude_personality":
        return await handle_get_claude_personality(arguments)
    elif name == "evolve_claude_personality":
        return await handle_evolve_claude(arguments)
    elif name == "check_pltm_available":
        return await handle_check_pltm(arguments)
    elif name == "pltm_mode":
        return await handle_pltm_mode(arguments)
    elif name == "deep_personality_analysis":
        return await handle_deep_analysis(arguments)
    elif name == "enrich_claude_personality":
        return await handle_enrich_personality(arguments)
    elif name == "learn_from_url":
        return await handle_learn_url(arguments)
    elif name == "learn_from_paper":
        return await handle_learn_paper(arguments)
    elif name == "learn_from_code":
        return await handle_learn_code(arguments)
    elif name == "get_learning_stats":
        return await handle_learning_stats(arguments)
    elif name == "batch_ingest_wikipedia":
        return await handle_batch_wikipedia(arguments)
    elif name == "batch_ingest_papers":
        return await handle_batch_papers(arguments)
    elif name == "batch_ingest_repos":
        return await handle_batch_repos(arguments)
    elif name == "get_learning_schedule":
        return await handle_learning_schedule(arguments)
    elif name == "run_learning_task":
        return await handle_run_task(arguments)
    elif name == "cross_domain_synthesis":
        return await handle_synthesis(arguments)
    elif name == "get_universal_principles":
        return await handle_universal_principles(arguments)
    elif name == "get_transfer_suggestions":
        return await handle_transfer_suggestions(arguments)
    elif name == "learn_from_conversation":
        return await handle_learn_conversation(arguments)
    # PLTM 2.0 tools
    elif name == "quantum_add_state":
        return await handle_quantum_add(arguments)
    elif name == "quantum_query":
        return await handle_quantum_query(arguments)
    elif name == "quantum_peek":
        return await handle_quantum_peek(arguments)
    elif name == "attention_retrieve":
        return await handle_attention_retrieve(arguments)
    elif name == "attention_multihead":
        return await handle_attention_multihead(arguments)
    elif name == "knowledge_add_concept":
        return await handle_knowledge_add(arguments)
    elif name == "knowledge_find_path":
        return await handle_knowledge_path(arguments)
    elif name == "knowledge_bridges":
        return await handle_knowledge_bridges(arguments)
    elif name == "knowledge_stats":
        return await handle_knowledge_stats(arguments)
    elif name == "self_improve_cycle":
        return await handle_improve_cycle(arguments)
    elif name == "self_improve_meta_learn":
        return await handle_meta_learn(arguments)
    elif name == "self_improve_history":
        return await handle_improve_history(arguments)
    elif name == "quantum_cleanup":
        return await handle_quantum_cleanup(arguments)
    elif name == "quantum_stats":
        return await handle_quantum_stats(arguments)
    elif name == "attention_clear_cache":
        return await handle_attention_clear_cache(arguments)
    elif name == "criticality_state":
        return await handle_criticality_state(arguments)
    elif name == "criticality_recommend":
        return await handle_criticality_recommend(arguments)
    elif name == "criticality_adjust":
        return await handle_criticality_adjust(arguments)
    elif name == "criticality_history":
        return await handle_criticality_history(arguments)
    elif name == "add_provenance":
        return await handle_add_provenance(arguments)
    elif name == "get_provenance":
        return await handle_get_provenance(arguments)
    elif name == "provenance_stats":
        return await handle_provenance_stats(arguments)
    elif name == "unverified_claims":
        return await handle_unverified_claims(arguments)
    elif name == "mmr_retrieve":
        return await handle_mmr_retrieve(arguments)
    # Action Accounting
    elif name == "record_action":
        return await handle_record_action(arguments)
    elif name == "get_aae":
        return await handle_get_aae(arguments)
    elif name == "aae_trend":
        return await handle_aae_trend(arguments)
    elif name == "start_action_cycle":
        return await handle_start_action_cycle(arguments)
    elif name == "end_action_cycle":
        return await handle_end_action_cycle(arguments)
    # Entropy Injection
    elif name == "inject_entropy_random":
        return await handle_inject_entropy_random(arguments)
    elif name == "inject_entropy_antipodal":
        return await handle_inject_entropy_antipodal(arguments)
    elif name == "inject_entropy_temporal":
        return await handle_inject_entropy_temporal(arguments)
    elif name == "entropy_stats":
        return await handle_entropy_stats(arguments)
    # ArXiv Ingestion
    elif name == "ingest_arxiv":
        return await handle_ingest_arxiv(arguments)
    elif name == "search_arxiv":
        return await handle_search_arxiv(arguments)
    elif name == "arxiv_history":
        return await handle_arxiv_history(arguments)
    # Epistemic Hygiene
    elif name == "check_before_claiming":
        return await handle_check_before_claiming(arguments)
    elif name == "log_claim":
        return await handle_log_claim(arguments)
    elif name == "resolve_claim":
        return await handle_resolve_claim(arguments)
    elif name == "get_calibration":
        return await handle_get_calibration(arguments)
    elif name == "calibrate_confidence_live":
        return await handle_calibrate_confidence_live(arguments)
    # Personality / Self-Modeling
    elif name == "self_profile":
        return await handle_self_profile(arguments)
    elif name == "get_longitudinal_stats":
        return await handle_get_longitudinal_stats(arguments)
    elif name == "bootstrap_self_model":
        return await handle_bootstrap_self_model(arguments)
    elif name == "track_curiosity_spike":
        return await handle_track_curiosity_spike(arguments)
    elif name == "learn_communication_style":
        return await handle_learn_communication_style(arguments)
    # Cross-Model
    elif name == "route_llm_task":
        return await handle_route_llm_task(arguments)
    # Session
    elif name == "auto_init_session":
        return await handle_auto_init_session(arguments)
    elif name == "end_session":
        return await handle_end_session(arguments)
    elif name == "generate_memory_prompt":
        return await handle_generate_memory_prompt(arguments)
    elif name == "belief_auto_check":
        return await handle_belief_auto_check(arguments)
    elif name == "jury_stats":
        return await handle_jury_stats(arguments)
    elif name == "process_message":
        return await handle_process_message(arguments)
    elif name == "process_message_batch":
        return await handle_process_message_batch(arguments)
    elif name == "pipeline_stats":
        return await handle_pipeline_stats(arguments)
    # Experiments
    elif name == "trace_claim_reasoning":
        return await handle_trace_claim_reasoning(arguments)
    elif name == "constraint_sensitivity_test":
        return await handle_constraint_sensitivity_test(arguments)
    elif name == "domain_cognitive_map":
        return await handle_domain_cognitive_map(arguments)
    # Data Access
    elif name == "query_pltm_sql":
        return await handle_query_pltm_sql(arguments)
    # Typed Memory System
    elif name == "store_episodic":
        return await handle_store_typed(arguments, "episodic")
    elif name == "store_semantic":
        return await handle_store_typed(arguments, "semantic")
    elif name == "store_belief":
        return await handle_store_typed(arguments, "belief")
    elif name == "store_procedural":
        return await handle_store_typed(arguments, "procedural")
    elif name == "recall_memories":
        return await handle_recall_memories(arguments)
    elif name == "search_memories":
        return await handle_search_memories(arguments)
    elif name == "update_belief":
        return await handle_update_belief_mem(arguments)
    elif name == "record_procedure_outcome":
        return await handle_record_procedure(arguments)
    elif name == "consolidate_memories":
        return await handle_consolidate(arguments)
    elif name == "memory_stats":
        return await handle_memory_stats(arguments)
    # Memory Intelligence
    elif name == "detect_contradictions":
        return await handle_detect_contradictions(arguments)
    elif name == "what_do_i_know_about":
        return await handle_what_do_i_know(arguments)
    elif name == "auto_tag_memories":
        return await handle_auto_tag(arguments)
    elif name == "correct_memory":
        return await handle_correct_memory(arguments)
    elif name == "forget_memory":
        return await handle_forget_memory(arguments)
    elif name == "auto_prune_memories":
        return await handle_auto_prune(arguments)
    elif name == "get_relevant_context":
        return await handle_get_relevant_context(arguments)
    elif name == "user_timeline":
        return await handle_user_timeline(arguments)
    # Embedding Search
    elif name == "semantic_search":
        return await handle_semantic_search(arguments)
    elif name == "index_embeddings":
        return await handle_index_embeddings(arguments)
    elif name == "find_similar_memories":
        return await handle_find_similar(arguments)
    else:
        return [TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls with timeout and store guard."""
    if store is None:
        return [TextContent(
            type="text",
            text=f"Error: PLTM not initialized. Restart the MCP server."
        )]
    
    try:
        result = await asyncio.wait_for(_dispatch_tool(name, arguments), timeout=30.0)
        return result
    except asyncio.TimeoutError:
        logger.error(f"Tool {name} timed out after 30s")
        return [TextContent(
            type="text",
            text=f"Error: Tool '{name}' timed out after 30s"
        )]
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def handle_store_atom(args: Dict[str, Any]) -> List[TextContent]:
    """Store a memory atom"""
    # Map atom type string to enum
    atom_type_map = {
        "fact": AtomType.STATE,
        "personality_trait": AtomType.PERSONALITY_TRAIT,
        "communication_style": AtomType.COMMUNICATION_STYLE,
        "interaction_pattern": AtomType.INTERACTION_PATTERN,
        "mood": AtomType.STATE,
        "preference": AtomType.PREFERENCE,
    }
    
    atom_type = atom_type_map.get(args["atom_type"], AtomType.STATE)
    
    # Create atom
    atom = MemoryAtom(
        atom_type=atom_type,
        subject=args["subject"],
        predicate=args["predicate"],
        object=args["object"],
        confidence=args.get("confidence", 0.7),
        strength=args.get("confidence", 0.7),
        provenance=Provenance.INFERRED,
        source_user=args["user_id"],
        contexts=args.get("context", []),
        graph=GraphType.SUBSTANTIATED
    )
    
    # Store in database
    await store.add_atom(atom)
    
    # Verify write succeeded
    verified = False
    try:
        check = await store.get_atom(atom.id)
        verified = check is not None
    except Exception:
        verified = False
    
    if not verified:
        logger.error(f"WRITE VERIFICATION FAILED for atom {atom.id}")
    
    return [TextContent(
        type="text",
        text=compact_json({
            "status": "stored" if verified else "WRITE_FAILED",
            "atom_id": str(atom.id),
            "verified": verified,
            "atom_type": args["atom_type"],
            "content": f"{args['subject']} {args['predicate']} {args['object']}"
        })
    )]


async def handle_query_personality(args: Dict[str, Any]) -> List[TextContent]:
    """Query personality profile"""
    user_id = args["user_id"]
    context = args.get("context")
    
    if context:
        # Context-specific personality
        personality = await contextual_personality.get_personality_for_context(
            user_id, context
        )
    else:
        # General personality
        personality = await personality_synth.synthesize_personality(user_id)
    
    return [TextContent(
        type="text",
        text=compact_json(personality)
    )]


async def handle_detect_mood(args: Dict[str, Any]) -> List[TextContent]:
    """Detect mood from message"""
    user_id = args["user_id"]
    message = args["message"]
    
    mood_atom = await mood_tracker.detect_mood(user_id, message)
    
    if mood_atom:
        # Store mood
        await store.add_atom(mood_atom)
        
        result = {
            "mood": mood_atom.object,
            "confidence": mood_atom.confidence,
            "detected": True
        }
    else:
        result = {
            "mood": None,
            "confidence": 0.0,
            "detected": False
        }
    
    return [TextContent(
        type="text",
        text=compact_json(result)
    )]


async def handle_mood_patterns(args: Dict[str, Any]) -> List[TextContent]:
    """Get mood patterns"""
    user_id = args["user_id"]
    window_days = args.get("window_days", 90)
    
    patterns = await mood_patterns.detect_patterns(user_id, window_days)
    
    # Get insights
    insights = await mood_patterns.get_mood_insights(user_id)
    
    result = {
        "patterns": patterns,
        "insights": insights
    }
    
    return [TextContent(
        type="text",
        text=compact_json(result)
    )]


async def handle_resolve_conflict(args: Dict[str, Any]) -> List[TextContent]:
    """Resolve conflicting traits"""
    user_id = args["user_id"]
    trait_objects = args["trait_objects"]
    
    # Get all personality atoms
    all_atoms = await store.get_atoms_by_subject(user_id)
    
    # Filter to conflicting traits
    conflicting = [
        atom for atom in all_atoms
        if atom.atom_type in [AtomType.PERSONALITY_TRAIT, AtomType.COMMUNICATION_STYLE]
        and atom.object in trait_objects
    ]
    
    if len(conflicting) < 2:
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "no_conflict",
                "message": "Need at least 2 conflicting traits"
            })
        )]
    
    # Resolve
    winner, explanation = await conflict_resolver.resolve_with_explanation(
        user_id, conflicting
    )
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "status": "resolved",
            "winner": winner.object if winner else None,
            "explanation": explanation
        }, indent=2)
    )]


async def handle_extract_traits(args: Dict[str, Any]) -> List[TextContent]:
    """Extract personality traits from interaction"""
    user_id = args["user_id"]
    message = args["message"]
    ai_response = args.get("ai_response")
    user_reaction = args.get("user_reaction")
    
    # Extract traits
    traits = await personality_agent.personality_extractor.extract_from_interaction(
        user_id, message, ai_response, user_reaction
    )
    
    # Store traits
    for trait in traits:
        await store.add_atom(trait)
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "extracted_count": len(traits),
            "traits": [
                {
                    "type": trait.atom_type.value,
                    "predicate": trait.predicate,
                    "object": trait.object,
                    "confidence": trait.confidence
                }
                for trait in traits
            ]
        }, indent=2)
    )]


async def handle_adaptive_prompt(args: Dict[str, Any]) -> List[TextContent]:
    """Get adaptive prompt"""
    user_id = args["user_id"]
    message = args["message"]
    context = args.get("context")
    
    # Get personality and mood
    result = await personality_agent.interact(user_id, message, extract_personality=False)
    
    return [TextContent(
        type="text",
        text=result["adaptive_prompt"]
    )]


async def handle_personality_summary(args: Dict[str, Any]) -> List[TextContent]:
    """Get personality summary"""
    user_id = args["user_id"]
    
    summary = await personality_agent.get_personality_summary(user_id)
    
    return [TextContent(
        type="text",
        text=summary
    )]


async def handle_bootstrap_sample(args: Dict[str, Any]) -> List[TextContent]:
    """Bootstrap from sample data"""
    from mcp_server.bootstrap_pltm import ConversationAnalyzer
    
    user_id = args["user_id"]
    
    sample_convs = [
        {
            "messages": [
                {"role": "user", "content": "Explain PLTM conflict resolution"},
                {"role": "user", "content": "Too detailed, just key steps"},
                {"role": "user", "content": "Perfect, exactly what I needed"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Review this code"},
                {"role": "user", "content": "Don't make this so personalized"},
                {"role": "user", "content": "Great, that's helpful"}
            ]
        }
    ]
    
    analyzer = ConversationAnalyzer()
    total_atoms = 0
    
    for conv in sample_convs:
        analysis = analyzer.analyze_conversation(conv)
        
        for style_data in analysis["styles"]:
            atom = MemoryAtom(
                atom_type=AtomType.COMMUNICATION_STYLE,
                subject=user_id,
                predicate="prefers_style",
                object=style_data["style"],
                confidence=style_data["confidence"],
                strength=style_data["confidence"],
                provenance=Provenance.INFERRED,
                source_user=user_id,
                contexts=style_data.get("contexts", ["general"]),
                graph=GraphType.SUBSTANTIATED
            )
            await store.add_atom(atom)
            total_atoms += 1
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "status": "bootstrapped",
            "atoms_created": total_atoms,
            "message": f"Bootstrapped {total_atoms} personality atoms from sample data"
        }, indent=2)
    )]


async def handle_bootstrap_messages(args: Dict[str, Any]) -> List[TextContent]:
    """Bootstrap from conversation messages"""
    from mcp_server.bootstrap_pltm import ConversationAnalyzer
    
    user_id = args["user_id"]
    messages = args["messages"]
    
    analyzer = ConversationAnalyzer()
    analysis = analyzer.analyze_conversation(messages)
    
    total_atoms = 0
    
    for style_data in analysis["styles"]:
        atom = MemoryAtom(
            atom_type=AtomType.COMMUNICATION_STYLE,
            subject=user_id,
            predicate="prefers_style",
            object=style_data["style"],
            confidence=style_data["confidence"],
            strength=style_data["confidence"],
            provenance=Provenance.INFERRED,
            source_user=user_id,
            contexts=style_data.get("contexts", ["general"]),
            graph=GraphType.SUBSTANTIATED
        )
        await store.add_atom(atom)
        total_atoms += 1
    
    if analysis["moods"]:
        mood_data = analysis["moods"]
        atom = MemoryAtom(
            atom_type=AtomType.STATE,
            subject=user_id,
            predicate="typical_mood",
            object=mood_data["dominant_mood"],
            confidence=mood_data["confidence"],
            strength=mood_data["confidence"],
            provenance=Provenance.INFERRED,
            source_user=user_id,
            contexts=["historical"],
            graph=GraphType.SUBSTANTIATED
        )
        await store.add_atom(atom)
        total_atoms += 1
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "status": "bootstrapped",
            "atoms_created": total_atoms,
            "styles_found": len(analysis["styles"]),
            "mood_detected": analysis["moods"] is not None
        }, indent=2)
    )]


async def handle_track_evolution(args: Dict[str, Any]) -> List[TextContent]:
    """Track trait evolution over time"""
    from src.personality.temporal_tracker import TemporalPersonalityTracker
    
    tracker = TemporalPersonalityTracker(store)
    result = await tracker.track_trait_evolution(
        args["user_id"],
        args["trait"],
        args.get("window_days", 90)
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_predict_reaction(args: Dict[str, Any]) -> List[TextContent]:
    """Predict reaction to stimulus"""
    from src.personality.causal_graph import CausalGraphBuilder
    
    causal = CausalGraphBuilder(store)
    result = await causal.predict_reaction(args["user_id"], args["stimulus"])
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_meta_patterns(args: Dict[str, Any]) -> List[TextContent]:
    """Get cross-context meta patterns"""
    from src.personality.meta_patterns import MetaPatternDetector
    
    detector = MetaPatternDetector(store)
    patterns = await detector.detect_meta_patterns(args["user_id"])
    
    result = {
        "user_id": args["user_id"],
        "meta_patterns": [
            {
                "behavior": p.behavior,
                "contexts": p.contexts,
                "strength": p.strength,
                "is_core_trait": p.is_core_trait,
                "examples": p.examples[:2]
            }
            for p in patterns
        ],
        "core_traits": [p.behavior for p in patterns if p.is_core_trait]
    }
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_learn_interaction(args: Dict[str, Any]) -> List[TextContent]:
    """Learn from an interaction"""
    from src.personality.interaction_dynamics import InteractionDynamicsLearner
    
    learner = InteractionDynamicsLearner(store)
    result = await learner.learn_from_interaction(
        args["user_id"],
        args["my_response"],
        args["user_reaction"]
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_predict_session(args: Dict[str, Any]) -> List[TextContent]:
    """Predict session from greeting"""
    from src.personality.predictive_model import PredictivePersonalityModel
    
    predictor = PredictivePersonalityModel(store)
    result = await predictor.predict_from_greeting(args["user_id"], args["greeting"])
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_self_model(args: Dict[str, Any]) -> List[TextContent]:
    """Get self-model for meta-cognition"""
    from src.personality.predictive_model import PredictivePersonalityModel
    
    predictor = PredictivePersonalityModel(store)
    result = await predictor.get_self_model(args["user_id"])
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_init_claude_session(args: Dict[str, Any]) -> List[TextContent]:
    """Initialize Claude personality session"""
    from src.personality.claude_personality import ClaudePersonality
    
    claude = ClaudePersonality(store)
    result = await claude.initialize_session(args["user_id"])
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_update_claude_style(args: Dict[str, Any]) -> List[TextContent]:
    """Update Claude's communication style"""
    from src.personality.claude_personality import ClaudePersonality
    
    claude = ClaudePersonality(store)
    result = await claude.update_style(
        args["user_id"],
        args["attribute"],
        args["value"],
        args.get("confidence", 0.8)
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_learn_dynamic(args: Dict[str, Any]) -> List[TextContent]:
    """Learn interaction dynamic"""
    from src.personality.claude_personality import ClaudePersonality
    
    claude = ClaudePersonality(store)
    result = await claude.learn_what_works(
        args["user_id"],
        args["behavior"],
        args["works"],
        args.get("confidence", 0.8)
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_record_milestone(args: Dict[str, Any]) -> List[TextContent]:
    """Record collaboration milestone"""
    from src.personality.claude_personality import ClaudePersonality
    
    claude = ClaudePersonality(store)
    result = await claude.record_milestone(
        args["user_id"],
        args["description"],
        args.get("significance", 0.8)
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_add_vocabulary(args: Dict[str, Any]) -> List[TextContent]:
    """Add shared vocabulary"""
    from src.personality.claude_personality import ClaudePersonality
    
    claude = ClaudePersonality(store)
    result = await claude.add_shared_vocabulary(
        args["user_id"],
        args["term"],
        args["meaning"]
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_get_claude_personality(args: Dict[str, Any]) -> List[TextContent]:
    """Get Claude personality summary"""
    from src.personality.claude_personality import ClaudePersonality
    
    claude = ClaudePersonality(store)
    result = await claude.get_claude_personality_summary(args["user_id"])
    
    return [TextContent(type="text", text=result)]


async def handle_evolve_claude(args: Dict[str, Any]) -> List[TextContent]:
    """Evolve Claude personality from interaction"""
    from src.personality.claude_personality import ClaudePersonality
    
    claude = ClaudePersonality(store)
    result = await claude.evolve_from_interaction(
        args["user_id"],
        args["my_response_style"],
        args["user_reaction"],
        args["was_positive"]
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_check_pltm(args: Dict[str, Any]) -> List[TextContent]:
    """Quick check if user has PLTM data - call FIRST in conversations"""
    from src.personality.claude_personality import ClaudePersonality
    
    user_id = args["user_id"]
    
    try:
        claude = ClaudePersonality(store)
        
        # Load basic info without full init
        context = await claude._load_shared_context(user_id)
        dynamics = await claude._load_interaction_dynamics(user_id)
        style = await claude._load_claude_style(user_id)
        
        has_data = context.session_count > 0 or len(dynamics.works_well) > 0
        
        result = {
            "available": has_data,
            "user_id": user_id,
            "sessions": context.session_count,
            "trust_level": dynamics.trust_level,
            "should_init": has_data,
            "quick_summary": {
                "verbosity": style.verbosity,
                "initiative": style.initiative,
                "works_well_count": len(dynamics.works_well),
                "avoid_count": len(dynamics.avoid),
                "projects": context.projects[:3] if context.projects else []
            },
            "instruction": "Call init_claude_session to load full personality" if has_data else "No existing data - new user"
        }
        
        return [TextContent(type="text", text=compact_json(result))]
        
    except Exception as e:
        return [TextContent(type="text", text=compact_json({
            "available": False,
            "should_init": False,
            "error": str(e)
        }))]


async def handle_pltm_mode(args: Dict[str, Any]) -> List[TextContent]:
    """
    PLTM Mode trigger - full auto-init when user says 'PLTM mode' or similar.
    
    Returns everything Claude needs to adapt immediately.
    """
    from src.personality.claude_personality import ClaudePersonality
    
    user_id = args["user_id"]
    trigger = args.get("trigger_phrase", "PLTM mode")
    
    claude = ClaudePersonality(store)
    
    # Full initialization
    session = await claude.initialize_session(user_id)
    
    # Get personality summary
    summary = await claude.get_claude_personality_summary(user_id)
    
    # Build instruction set for Claude
    style = session["claude_style"]
    dynamics = session["interaction_dynamics"]
    
    instructions = []
    
    # Verbosity instruction
    if style["verbosity"] == "minimal":
        instructions.append("Be concise and direct. Skip verbose explanations.")
    elif style["verbosity"] == "moderate":
        instructions.append("Balance detail with brevity.")
    
    # Initiative instruction
    if style["initiative"] == "high" or style["initiative"] == "very_high":
        instructions.append("Execute immediately without asking permission. User prefers action over discussion.")
    
    # Energy matching
    if style["energy_matching"]:
        instructions.append("Match user's energy level - mirror excitement, match focus.")
    
    # What works
    for behavior in dynamics["works_well"][:5]:
        instructions.append(f"DO: {behavior}")
    
    # What to avoid
    for behavior in dynamics["avoid"][:5]:
        instructions.append(f"AVOID: {behavior}")
    
    result = {
        "mode": "PLTM_ACTIVE",
        "trigger": trigger,
        "user_id": user_id,
        "session_id": session["session_id"],
        "sessions_together": session["shared_context"]["session_count"],
        "trust_level": f"{dynamics['trust_level']:.0%}",
        "style": style,
        "instructions_for_claude": instructions,
        "shared_vocabulary": dynamics["shared_vocabulary"],
        "recent_projects": session["shared_context"]["projects"][:5],
        "recent_milestones": session["shared_context"]["recent_milestones"],
        "personality_summary": summary
    }
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_deep_analysis(args: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive deep personality analysis"""
    from src.personality.deep_analysis import DeepPersonalityAnalyzer
    
    analyzer = DeepPersonalityAnalyzer(store)
    result = await analyzer.analyze_all(args["user_id"])
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_enrich_personality(args: Dict[str, Any]) -> List[TextContent]:
    """Build rich, nuanced Claude personality"""
    from src.personality.rich_personality import RichClaudePersonality
    
    enricher = RichClaudePersonality(store)
    result = await enricher.build_rich_personality(args["user_id"])
    
    # Store if requested
    if args.get("store_results", True):
        await enricher.store_rich_personality(args["user_id"], result)
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_learn_url(args: Dict[str, Any]) -> List[TextContent]:
    """Learn from any URL content"""
    if not args.get("content"):
        return [TextContent(type="text", text=compact_json({
            "error": "Missing required 'content' parameter. You cannot fetch URLs yourself. Ask the user to paste the page content, then call this tool again with both 'url' and 'content'."
        }))]
    
    from src.learning.universal_learning import UniversalLearningSystem, SourceType
    
    learner = UniversalLearningSystem(store)
    
    source_type = None
    if args.get("source_type"):
        try:
            source_type = SourceType(args["source_type"])
        except ValueError:
            pass
    
    result = await learner.learn_from_url(
        args["url"],
        args["content"],
        source_type
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_learn_paper(args: Dict[str, Any]) -> List[TextContent]:
    """Learn from research paper"""
    from src.learning.universal_learning import UniversalLearningSystem
    
    learner = UniversalLearningSystem(store)
    result = await learner.learn_from_paper(
        args["paper_id"],
        args["title"],
        args["abstract"],
        args["content"],
        args["authors"],
        args.get("publication_date")
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_learn_code(args: Dict[str, Any]) -> List[TextContent]:
    """Learn from code repository"""
    from src.learning.universal_learning import UniversalLearningSystem
    
    learner = UniversalLearningSystem(store)
    result = await learner.learn_from_code(
        args["repo_url"],
        args["repo_name"],
        args.get("description", ""),
        args["languages"],
        args["code_samples"]
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_learning_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Get learning statistics"""
    from src.learning.universal_learning import UniversalLearningSystem
    
    learner = UniversalLearningSystem(store)
    result = await learner.get_learning_stats()
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_batch_wikipedia(args: Dict[str, Any]) -> List[TextContent]:
    """Batch ingest Wikipedia articles"""
    from src.learning.batch_ingestion import BatchIngestionPipeline
    
    pipeline = BatchIngestionPipeline(store)
    result = await pipeline.ingest_wikipedia_articles(args["articles"])
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_batch_papers(args: Dict[str, Any]) -> List[TextContent]:
    """Batch ingest research papers"""
    from src.learning.batch_ingestion import BatchIngestionPipeline
    
    pipeline = BatchIngestionPipeline(store)
    result = await pipeline.ingest_arxiv_papers(args["papers"])
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_batch_repos(args: Dict[str, Any]) -> List[TextContent]:
    """Batch ingest GitHub repos"""
    from src.learning.batch_ingestion import BatchIngestionPipeline
    
    pipeline = BatchIngestionPipeline(store)
    result = await pipeline.ingest_github_repos(args["repos"])
    
    return [TextContent(type="text", text=compact_json(result))]


# Global continuous learning loop instance
_continuous_learner = None

async def handle_learning_schedule(args: Dict[str, Any]) -> List[TextContent]:
    """Get learning schedule status"""
    from src.learning.continuous_learning import ContinuousLearningLoop
    
    global _continuous_learner
    if _continuous_learner is None:
        _continuous_learner = ContinuousLearningLoop(store)
    
    result = _continuous_learner.get_schedule_status()
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_run_task(args: Dict[str, Any]) -> List[TextContent]:
    """Run a specific learning task"""
    from src.learning.continuous_learning import ContinuousLearningLoop
    
    global _continuous_learner
    if _continuous_learner is None:
        _continuous_learner = ContinuousLearningLoop(store)
    
    result = await _continuous_learner.run_single_task(args["task_name"])
    
    return [TextContent(type="text", text=compact_json(result))]


# Global synthesizer instance
_synthesizer = None

async def handle_synthesis(args: Dict[str, Any]) -> List[TextContent]:
    """Run cross-domain synthesis"""
    from src.learning.cross_domain_synthesis import CrossDomainSynthesizer
    
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = CrossDomainSynthesizer(store)
    
    result = await _synthesizer.synthesize_all()
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_universal_principles(args: Dict[str, Any]) -> List[TextContent]:
    """Get discovered universal principles"""
    from src.learning.cross_domain_synthesis import CrossDomainSynthesizer
    
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = CrossDomainSynthesizer(store)
    
    result = await _synthesizer.query_universal_principles()
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_transfer_suggestions(args: Dict[str, Any]) -> List[TextContent]:
    """Get transfer suggestions between domains"""
    from src.learning.cross_domain_synthesis import CrossDomainSynthesizer
    
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = CrossDomainSynthesizer(store)
    
    result = await _synthesizer.get_transfer_suggestions(
        args["from_domain"],
        args["to_domain"]
    )
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_learn_conversation(args: Dict[str, Any]) -> List[TextContent]:
    """Learn from conversation"""
    from src.learning.continuous_learning import ManualLearningTrigger
    
    trigger = ManualLearningTrigger(store)
    result = await trigger.learn_from_conversation(
        args["messages"],
        args["topic"],
        args["user_id"]
    )
    
    return [TextContent(type="text", text=compact_json(result))]


# ============================================================================
# PLTM 2.0 - Universal Optimization Principles
# ============================================================================

# Global instances for PLTM 2.0 systems
_quantum_memory = None
_attention_retrieval = None
_knowledge_graph = None
_self_improver = None


async def handle_quantum_add(args: Dict[str, Any]) -> List[TextContent]:
    """Add state to quantum superposition"""
    from src.memory.quantum_superposition import QuantumMemorySystem
    
    global _quantum_memory
    if _quantum_memory is None:
        _quantum_memory = QuantumMemorySystem(store)
    
    result = await _quantum_memory.add_to_superposition(
        args["subject"],
        args["predicate"],
        args["value"],
        args["confidence"],
        args["source"]
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_quantum_query(args: Dict[str, Any]) -> List[TextContent]:
    """Query superposition with collapse"""
    from src.memory.quantum_superposition import QuantumMemorySystem
    
    global _quantum_memory
    if _quantum_memory is None:
        _quantum_memory = QuantumMemorySystem(store)
    
    result = await _quantum_memory.query_with_collapse(
        args["subject"],
        args["predicate"],
        args.get("context")
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_quantum_peek(args: Dict[str, Any]) -> List[TextContent]:
    """Peek at superposition without collapse"""
    from src.memory.quantum_superposition import QuantumMemorySystem
    
    global _quantum_memory
    if _quantum_memory is None:
        _quantum_memory = QuantumMemorySystem(store)
    
    result = await _quantum_memory.peek_superposition(
        args["subject"],
        args["predicate"]
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_attention_retrieve(args: Dict[str, Any]) -> List[TextContent]:
    """Attention-weighted memory retrieval - lightweight direct SQL"""
    user_id = args.get("user_id", "alby")
    query = args.get("query", "")
    top_k = args.get("top_k", 10)
    
    if not store._conn:
        return [TextContent(type="text", text=compact_json({"error": "DB not connected", "n": 0}))]
    
    # Get memories directly
    cursor = await store._conn.execute(
        "SELECT predicate, object, confidence FROM atoms WHERE subject = ? LIMIT 100",
        (user_id,)
    )
    rows = await cursor.fetchall()
    
    if not rows:
        return [TextContent(type="text", text=compact_json({"n": 0, "memories": []}))]
    
    # Simple attention scoring based on keyword overlap + confidence
    query_words = set(query.lower().split())
    scored = []
    for row in rows:
        text = f"{row[0]} {row[1]}".lower()
        words = set(text.split())
        overlap = len(query_words & words)
        score = (overlap / max(len(query_words), 1)) * 0.7 + row[2] * 0.3
        scored.append((row, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]
    
    memories = [{"p": s[0][0], "o": s[0][1][:40], "a": round(s[1], 3)} for s in top]
    
    return [TextContent(type="text", text=compact_json({"n": len(memories), "top": memories[:5]}))]


async def handle_attention_multihead(args: Dict[str, Any]) -> List[TextContent]:
    """Multi-head attention retrieval - lightweight direct SQL"""
    user_id = args.get("user_id", "alby")
    query = args.get("query", "")
    num_heads = args.get("num_heads", 4)
    
    if not store._conn:
        return [TextContent(type="text", text=compact_json({"error": "DB not connected", "n": 0}))]
    
    # Get memories directly
    cursor = await store._conn.execute(
        "SELECT predicate, object, confidence FROM atoms WHERE subject = ? LIMIT 100",
        (user_id,)
    )
    rows = await cursor.fetchall()
    
    if not rows:
        return [TextContent(type="text", text=compact_json({"n": 0, "heads": []}))]
    
    # Simulate multi-head by using different scoring strategies
    query_words = set(query.lower().split())
    
    heads_results = []
    for head_idx in range(num_heads):
        scored = []
        for row in rows:
            text = f"{row[0]} {row[1]}".lower()
            words = set(text.split())
            
            # Different heads weight differently
            if head_idx == 0:  # Semantic head
                overlap = len(query_words & words)
                score = overlap / max(len(query_words), 1)
            elif head_idx == 1:  # Confidence head
                score = row[2]
            elif head_idx == 2:  # Length head (prefer concise)
                score = 1.0 / (1 + len(text) / 50)
            else:  # Mixed head
                overlap = len(query_words & words)
                score = (overlap / max(len(query_words), 1)) * 0.5 + row[2] * 0.5
            
            scored.append((row, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:3]
        heads_results.append({
            "head": head_idx,
            "top": [{"p": s[0][0], "o": s[0][1][:30]} for s in top]
        })
    
    return [TextContent(type="text", text=compact_json({
        "n": len(rows),
        "heads": heads_results[:4]
    }))]


async def handle_knowledge_add(args: Dict[str, Any]) -> List[TextContent]:
    """Add concept to knowledge graph"""
    from src.memory.knowledge_graph import KnowledgeNetworkGraph
    
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeNetworkGraph(store)
    
    result = await _knowledge_graph.add_concept(
        args["concept"],
        args["domain"],
        args.get("related_concepts")
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_knowledge_path(args: Dict[str, Any]) -> List[TextContent]:
    """Find path between concepts"""
    from src.memory.knowledge_graph import KnowledgeNetworkGraph
    
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeNetworkGraph(store)
    
    result = await _knowledge_graph.find_path(
        args["from_concept"],
        args["to_concept"]
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_knowledge_bridges(args: Dict[str, Any]) -> List[TextContent]:
    """Find bridge concepts"""
    from src.memory.knowledge_graph import KnowledgeNetworkGraph
    
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeNetworkGraph(store)
    
    result = await _knowledge_graph.find_bridges(args.get("top_k", 10))
    return [TextContent(type="text", text=compact_json(result))]


async def handle_knowledge_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Get knowledge graph stats"""
    from src.memory.knowledge_graph import KnowledgeNetworkGraph
    
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeNetworkGraph(store)
    
    result = await _knowledge_graph.get_network_stats()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_improve_cycle(args: Dict[str, Any]) -> List[TextContent]:
    """Run self-improvement cycle"""
    from src.meta.recursive_improvement import RecursiveSelfImprovement
    
    global _self_improver
    if _self_improver is None:
        _self_improver = RecursiveSelfImprovement(store)
    
    result = await _self_improver.run_improvement_cycle()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_meta_learn(args: Dict[str, Any]) -> List[TextContent]:
    """Meta-learn from improvements"""
    from src.meta.recursive_improvement import RecursiveSelfImprovement
    
    global _self_improver
    if _self_improver is None:
        _self_improver = RecursiveSelfImprovement(store)
    
    result = await _self_improver.meta_learn()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_improve_history(args: Dict[str, Any]) -> List[TextContent]:
    """Get improvement history"""
    from src.meta.recursive_improvement import RecursiveSelfImprovement
    
    global _self_improver
    if _self_improver is None:
        _self_improver = RecursiveSelfImprovement(store)
    
    result = await _self_improver.get_improvement_history()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_quantum_cleanup(args: Dict[str, Any]) -> List[TextContent]:
    """Cleanup old quantum states"""
    from src.memory.quantum_superposition import QuantumMemorySystem
    
    global _quantum_memory
    if _quantum_memory is None:
        _quantum_memory = QuantumMemorySystem(store)
    
    result = await _quantum_memory.cleanup_old_states()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_quantum_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Get quantum memory stats"""
    from src.memory.quantum_superposition import QuantumMemorySystem
    
    global _quantum_memory
    if _quantum_memory is None:
        _quantum_memory = QuantumMemorySystem(store)
    
    result = await _quantum_memory.get_stats()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_attention_clear_cache(args: Dict[str, Any]) -> List[TextContent]:
    """Clear attention cache"""
    from src.memory.attention_retrieval import AttentionMemoryRetrieval
    
    global _attention_retrieval
    if _attention_retrieval is None:
        _attention_retrieval = AttentionMemoryRetrieval(store)
    
    count = _attention_retrieval.clear_cache()
    return [TextContent(type="text", text=compact_json({"cleared": count}))]


# Global criticality instance
_criticality = None

async def handle_criticality_state(args: Dict[str, Any]) -> List[TextContent]:
    """Get criticality state - lightweight direct SQL"""
    import math
    
    if not store._conn:
        return [TextContent(type="text", text=compact_json({"error": "DB not connected"}))]
    
    # Get confidence stats for entropy calculation
    cursor = await store._conn.execute(
        "SELECT confidence, predicate FROM atoms WHERE graph = 'substantiated'"
    )
    rows = await cursor.fetchall()
    
    if not rows:
        return [TextContent(type="text", text=compact_json({
            "entropy": 0.5, "integration": 0.5, "ratio": 1.0, "zone": "critical", "n": 0
        }))]
    
    # Entropy: confidence variance + domain diversity
    confidences = [r[0] for r in rows]
    mean_conf = sum(confidences) / len(confidences)
    variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
    conf_entropy = min(1.0, variance * 4)  # Scale variance to 0-1
    
    # Domain entropy
    domains = {}
    for r in rows:
        domains[r[1]] = domains.get(r[1], 0) + 1
    
    total = len(rows)
    domain_entropy = 0.0
    for count in domains.values():
        p = count / total
        if p > 0:
            domain_entropy -= p * math.log2(p)
    max_ent = math.log2(len(domains)) if len(domains) > 1 else 1.0
    norm_domain_ent = domain_entropy / max_ent if max_ent > 0 else 0.0
    
    entropy = (conf_entropy * 0.4 + norm_domain_ent * 0.6)
    
    # Integration: mean confidence + domain connectivity proxy
    integration = mean_conf * 0.7 + (1 - norm_domain_ent) * 0.3
    
    # Criticality ratio
    ratio = entropy / integration if integration > 0 else 1.0
    
    # Zone classification
    if ratio < 0.8:
        zone = "subcritical"
    elif ratio > 1.2:
        zone = "supercritical"
    else:
        zone = "critical"
    
    return [TextContent(type="text", text=compact_json({
        "entropy": round(entropy, 3),
        "integration": round(integration, 3),
        "ratio": round(ratio, 3),
        "zone": zone,
        "n": len(rows),
        "domains": len(domains)
    }))]


async def handle_criticality_recommend(args: Dict[str, Any]) -> List[TextContent]:
    """Get criticality recommendation"""
    from src.meta.criticality import SelfOrganizedCriticality
    
    global _criticality
    if _criticality is None:
        _criticality = SelfOrganizedCriticality(store)
    
    result = await _criticality.get_adjustment_recommendation()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_criticality_adjust(args: Dict[str, Any]) -> List[TextContent]:
    """Auto-adjust toward criticality"""
    from src.meta.criticality import SelfOrganizedCriticality
    
    global _criticality
    if _criticality is None:
        _criticality = SelfOrganizedCriticality(store)
    
    result = await _criticality.auto_adjust()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_criticality_history(args: Dict[str, Any]) -> List[TextContent]:
    """Get criticality history"""
    from src.meta.criticality import SelfOrganizedCriticality
    
    global _criticality
    if _criticality is None:
        _criticality = SelfOrganizedCriticality(store)
    
    result = await _criticality.get_criticality_history()
    return [TextContent(type="text", text=compact_json(result))]


async def handle_add_provenance(args: Dict[str, Any]) -> List[TextContent]:
    """Add provenance for a claim"""
    import hashlib
    from datetime import datetime
    
    claim_id = args.get("claim_id")
    source_type = args.get("source_type")
    source_url = args.get("source_url")
    quoted_span = args.get("quoted_span")
    confidence = args.get("confidence", 0.5)
    
    # Generate provenance ID and content hash
    prov_id = f"prov_{source_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    content_hash = hashlib.sha256(quoted_span.encode()).hexdigest()
    
    await store.insert_provenance(
        provenance_id=prov_id,
        claim_id=claim_id,
        source_type=source_type,
        source_url=source_url,
        source_title=args.get("source_title", ""),
        quoted_span=quoted_span,
        page_or_section=args.get("page_or_section"),
        accessed_at=int(datetime.now().timestamp()),
        content_hash=content_hash,
        confidence=confidence,
        authors=args.get("authors"),
        arxiv_id=args.get("arxiv_id"),
        doi=args.get("doi"),
        commit_sha=args.get("commit_sha"),
        file_path=args.get("file_path"),
        line_range=args.get("line_range")
    )
    
    return [TextContent(type="text", text=compact_json({
        "ok": True, "id": prov_id, "type": source_type, "hash": content_hash[:16]
    }))]


async def handle_get_provenance(args: Dict[str, Any]) -> List[TextContent]:
    """Get provenance for a claim"""
    claim_id = args.get("claim_id")
    provs = await store.get_provenance_for_claim(claim_id)
    
    # Compact format
    result = [{
        "type": p["source_type"],
        "url": p["source_url"][:60],
        "quote": p["quoted_span"][:100],
        "conf": p["confidence"]
    } for p in provs]
    
    return [TextContent(type="text", text=compact_json({"n": len(result), "provs": result}))]


async def handle_provenance_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Get provenance statistics"""
    stats = await store.get_provenance_stats()
    return [TextContent(type="text", text=compact_json(stats))]


async def handle_unverified_claims(args: Dict[str, Any]) -> List[TextContent]:
    """Get unverified claims"""
    unverified = await store.get_unverified_claims()
    return [TextContent(type="text", text=compact_json({
        "n": len(unverified), 
        "claims": unverified[:20]  # Limit to 20 for token efficiency
    }))]


async def handle_mmr_retrieve(args: Dict[str, Any]) -> List[TextContent]:
    """MMR retrieval for diverse context selection - lightweight version"""
    import numpy as np
    import time
    
    user_id = args.get("user_id", "alby")
    query = args.get("query", "")
    top_k = args.get("top_k", 5)
    lambda_param = args.get("lambda_param", 0.6)
    
    start = time.time()
    
    # Check store connection
    if not store._conn:
        return [TextContent(type="text", text=compact_json({"error": "DB not connected", "n": 0}))]
    
    # Direct SQL query - bypass heavy ORM
    cursor = await store._conn.execute(
        """SELECT id, predicate, object, confidence 
           FROM atoms WHERE subject = ? AND graph = 'substantiated'
           ORDER BY confidence DESC LIMIT 50""",
        (user_id,)
    )
    rows = await cursor.fetchall()
    
    if not rows:
        return [TextContent(type="text", text=compact_json({"n": 0, "memories": [], "mean_dissim": 0.0}))]
    
    # Simple hash-based embedding
    def text_to_vec(text: str, dim: int = 32) -> np.ndarray:
        vec = np.zeros(dim)
        for word in text.lower().split():
            vec[hash(word) % dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    # Compute relevance (keyword overlap with query)
    query_words = set(query.lower().split())
    relevance = []
    embeddings = []
    
    for row in rows:
        text = f"{row[1]} {row[2]}"
        overlap = len(set(text.lower().split()) & query_words)
        rel = (overlap / max(len(query_words), 1)) * 0.7 + row[3] * 0.3
        relevance.append(rel)
        embeddings.append(text_to_vec(text))
    
    relevance = np.array(relevance)
    embeddings = np.array(embeddings)
    
    # Greedy MMR selection
    selected = []
    remaining = list(range(len(rows)))
    remaining.sort(key=lambda i: relevance[i], reverse=True)
    
    for _ in range(min(top_k, len(rows))):
        if not remaining:
            break
        
        best_idx = None
        best_score = float('-inf')
        
        for idx in remaining[:20]:  # Only check top 20
            if not selected:
                score = relevance[idx]
            else:
                max_sim = max(
                    float(np.dot(embeddings[idx], embeddings[s]) / 
                          (np.linalg.norm(embeddings[idx]) * np.linalg.norm(embeddings[s]) + 1e-9))
                    for s in selected
                )
                score = lambda_param * relevance[idx] - (1 - lambda_param) * max_sim
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
    
    # Build result
    memories = [{"p": rows[i][1], "o": rows[i][2][:40], "rel": round(relevance[i], 2)} for i in selected]
    
    elapsed = time.time() - start
    return [TextContent(type="text", text=compact_json({
        "n": len(memories), 
        "memories": memories[:5],
        "ms": int(elapsed * 1000),
        "lambda": lambda_param
    }))]


# === ACTION ACCOUNTING HANDLERS ===

_action_accounting = None

def get_action_accounting():
    global _action_accounting
    if _action_accounting is None:
        from src.metrics.action_accounting import ActionAccounting
        _action_accounting = ActionAccounting()
    return _action_accounting


async def handle_record_action(args: Dict[str, Any]) -> List[TextContent]:
    """Record an action for AAE tracking"""
    aa = get_action_accounting()
    record = aa.record(
        operation=args.get("operation"),
        tokens_used=args.get("tokens_used"),
        latency_ms=args.get("latency_ms"),
        success=args.get("success"),
        context=args.get("context")
    )
    return [TextContent(type="text", text=compact_json(record.to_dict()))]


async def handle_get_aae(args: Dict[str, Any]) -> List[TextContent]:
    """Get current AAE metrics"""
    aa = get_action_accounting()
    metrics = aa.get_aae(last_n=args.get("last_n"))
    return [TextContent(type="text", text=compact_json(metrics.to_dict()))]


async def handle_aae_trend(args: Dict[str, Any]) -> List[TextContent]:
    """Get AAE trend"""
    aa = get_action_accounting()
    trend = aa.get_trend(window_size=args.get("window_size", 10))
    return [TextContent(type="text", text=compact_json(trend))]


async def handle_start_action_cycle(args: Dict[str, Any]) -> List[TextContent]:
    """Start action measurement cycle"""
    aa = get_action_accounting()
    cycle_id = args.get("cycle_id")
    aa.start_cycle(cycle_id)
    return [TextContent(type="text", text=compact_json({"ok": True, "cycle": cycle_id}))]


async def handle_end_action_cycle(args: Dict[str, Any]) -> List[TextContent]:
    """End action cycle and get metrics"""
    aa = get_action_accounting()
    metrics = aa.end_cycle(args.get("cycle_id"))
    return [TextContent(type="text", text=compact_json(metrics.to_dict()))]


# === ENTROPY INJECTION HANDLERS ===

_entropy_injector = None

def get_entropy_injector():
    global _entropy_injector
    if _entropy_injector is None:
        from src.memory.entropy_injector import EntropyInjector
        _entropy_injector = EntropyInjector(store)
    return _entropy_injector


async def handle_inject_entropy_random(args: Dict[str, Any]) -> List[TextContent]:
    """Inject entropy via random domain sampling - lightweight direct SQL"""
    user_id = args.get("user_id", "alby")
    n_domains = args.get("n_domains", 3)
    
    if not store._conn:
        return [TextContent(type="text", text=compact_json({"error": "DB not connected", "n": 0}))]
    
    # Get domains directly
    cursor = await store._conn.execute(
        "SELECT DISTINCT predicate FROM atoms WHERE subject = ? LIMIT 20",
        (user_id,)
    )
    rows = await cursor.fetchall()
    
    if not rows:
        return [TextContent(type="text", text=compact_json({"n": 0, "domains": 0, "entropy_gain": 0}))]
    
    import random
    domains = [r[0] for r in rows]
    selected = random.sample(domains, min(n_domains, len(domains)))
    
    return [TextContent(type="text", text=compact_json({
        "n": len(selected),
        "domains": selected[:5],
        "entropy_gain": round(0.1 * len(selected), 3)
    }))]


async def handle_inject_entropy_antipodal(args: Dict[str, Any]) -> List[TextContent]:
    """Inject entropy via antipodal activation - lightweight direct SQL"""
    user_id = args.get("user_id", "alby")
    context = args.get("current_context", "")
    n_memories = args.get("n_memories", 5)
    
    if not store._conn:
        return [TextContent(type="text", text=compact_json({"error": "DB not connected", "n": 0}))]
    
    # Get memories directly
    cursor = await store._conn.execute(
        "SELECT predicate, object FROM atoms WHERE subject = ? LIMIT 50",
        (user_id,)
    )
    rows = await cursor.fetchall()
    
    if not rows:
        return [TextContent(type="text", text=compact_json({"n": 0, "memories": [], "entropy_gain": 0}))]
    
    # Score by distance from context
    context_words = set(context.lower().split())
    scored = []
    for row in rows:
        text = f"{row[0]} {row[1]}".lower()
        words = set(text.split())
        overlap = len(context_words & words)
        union = len(context_words | words)
        dist = 1 - (overlap / union) if union > 0 else 1.0
        scored.append((row, dist))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = scored[:n_memories]
    
    memories = [{"p": s[0][0], "o": s[0][1][:40], "dist": round(s[1], 2)} for s in selected]
    avg_dist = sum(s[1] for s in selected) / len(selected) if selected else 0
    
    return [TextContent(type="text", text=compact_json({
        "n": len(memories),
        "memories": memories[:5],
        "entropy_gain": round(0.15 * len(selected) * avg_dist, 3)
    }))]


async def handle_inject_entropy_temporal(args: Dict[str, Any]) -> List[TextContent]:
    """Inject entropy via temporal diversity - lightweight direct SQL"""
    user_id = args.get("user_id", "alby")
    n_old = args.get("n_old", 3)
    n_recent = args.get("n_recent", 2)
    
    if not store._conn:
        return [TextContent(type="text", text=compact_json({"error": "DB not connected", "n": 0}))]
    
    # Get old memories
    cursor = await store._conn.execute(
        "SELECT predicate, object FROM atoms WHERE subject = ? ORDER BY first_observed ASC LIMIT ?",
        (user_id, n_old)
    )
    old = await cursor.fetchall()
    
    # Get recent memories
    cursor = await store._conn.execute(
        "SELECT predicate, object FROM atoms WHERE subject = ? ORDER BY first_observed DESC LIMIT ?",
        (user_id, n_recent)
    )
    recent = await cursor.fetchall()
    
    all_mem = [{"p": m[0], "o": m[1][:40], "age": "old"} for m in old]
    all_mem += [{"p": m[0], "o": m[1][:40], "age": "recent"} for m in recent]
    
    return [TextContent(type="text", text=compact_json({
        "n": len(all_mem),
        "memories": all_mem[:5],
        "entropy_gain": round(0.12 * len(all_mem), 3)
    }))]


async def handle_entropy_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Get entropy statistics - lightweight direct SQL"""
    import math
    user_id = args.get("user_id", "alby")
    
    if not store._conn:
        return [TextContent(type="text", text=compact_json({"error": "DB not connected"}))]
    
    cursor = await store._conn.execute(
        "SELECT predicate, COUNT(*) FROM atoms WHERE subject = ? GROUP BY predicate",
        (user_id,)
    )
    rows = await cursor.fetchall()
    
    if not rows:
        return [TextContent(type="text", text=compact_json({
            "domains": 0, "total": 0, "entropy": 0, "needs_injection": True
        }))]
    
    total = sum(r[1] for r in rows)
    entropy = 0.0
    for _, count in rows:
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    max_ent = math.log2(len(rows)) if len(rows) > 1 else 1.0
    norm_ent = entropy / max_ent if max_ent > 0 else 0.0
    
    return [TextContent(type="text", text=compact_json({
        "domains": len(rows),
        "total": total,
        "entropy": round(entropy, 3),
        "normalized": round(norm_ent, 3),
        "needs_injection": norm_ent < 0.6,
        "top_domains": [{"d": r[0], "n": r[1]} for r in sorted(rows, key=lambda x: x[1], reverse=True)[:5]]
    }))]


# === ARXIV INGESTION HANDLERS ===

_arxiv_ingestion = None

def get_arxiv_ingestion():
    global _arxiv_ingestion
    if _arxiv_ingestion is None:
        from src.learning.arxiv_ingestion import ArxivIngestion
        _arxiv_ingestion = ArxivIngestion(store)
    return _arxiv_ingestion


async def handle_ingest_arxiv(args: Dict[str, Any]) -> List[TextContent]:
    """Ingest arXiv paper with real provenance"""
    ai = get_arxiv_ingestion()
    result = await ai.ingest_paper(
        arxiv_id=args.get("arxiv_id"),
        user_id=args.get("user_id", "pltm_knowledge")
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_search_arxiv(args: Dict[str, Any]) -> List[TextContent]:
    """Search arXiv for papers"""
    ai = get_arxiv_ingestion()
    results = await ai.search_arxiv(
        query=args.get("query"),
        max_results=args.get("max_results", 5)
    )
    return [TextContent(type="text", text=compact_json({"n": len(results), "papers": results}))]


async def handle_arxiv_history(args: Dict[str, Any]) -> List[TextContent]:
    """Get arXiv ingestion history"""
    ai = get_arxiv_ingestion()
    history = ai.get_ingestion_history(args.get("last_n", 10))
    return [TextContent(type="text", text=compact_json({"n": len(history), "history": history}))]


# === EPISTEMIC HYGIENE HANDLERS ===

_epistemic_monitor = None
_epistemic_v2 = None

def get_epistemic_monitor():
    global _epistemic_monitor
    if _epistemic_monitor is None:
        from src.analysis.epistemic_monitor import EpistemicMonitor
        _epistemic_monitor = EpistemicMonitor()
    return _epistemic_monitor

def get_epistemic_v2():
    global _epistemic_v2
    if _epistemic_v2 is None:
        from src.analysis.epistemic_v2 import EpistemicV2
        _epistemic_v2 = EpistemicV2()
    return _epistemic_v2


async def handle_check_before_claiming(args: Dict[str, Any]) -> List[TextContent]:
    em = get_epistemic_monitor()
    result = em.check_before_claiming(
        claim=args["claim"],
        felt_confidence=args["felt_confidence"],
        domain=args.get("domain", "general"),
        has_verified=args.get("has_verified", False),
        epistemic_status=args.get("epistemic_status", "TRAINING_DATA"),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_log_claim(args: Dict[str, Any]) -> List[TextContent]:
    em = get_epistemic_monitor()
    result = em.log_claim(
        claim=args["claim"],
        felt_confidence=args["felt_confidence"],
        domain=args.get("domain", "general"),
        epistemic_status=args.get("epistemic_status", "TRAINING_DATA"),
        has_verified=args.get("has_verified", False),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_resolve_claim(args: Dict[str, Any]) -> List[TextContent]:
    em = get_epistemic_monitor()
    result = em.resolve_claim(
        claim_id=args.get("claim_id", ""),
        claim_text=args.get("claim_text", ""),
        was_correct=args["was_correct"],
        correction_source=args.get("correction_source", ""),
        correction_detail=args.get("correction_detail", ""),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_get_calibration(args: Dict[str, Any]) -> List[TextContent]:
    em = get_epistemic_monitor()
    result = em.get_calibration(domain=args.get("domain", ""))
    return [TextContent(type="text", text=compact_json(result))]


async def handle_calibrate_confidence_live(args: Dict[str, Any]) -> List[TextContent]:
    ev2 = get_epistemic_v2()
    result = ev2.calibrate_confidence_live(
        claim=args["claim"],
        felt_confidence=args["felt_confidence"],
        domain=args.get("domain", "general"),
    )
    return [TextContent(type="text", text=compact_json(result))]


# === PERSONALITY / SELF-MODELING HANDLERS ===

_pltm_self = None

def get_pltm_self():
    global _pltm_self
    if _pltm_self is None:
        from src.analysis.pltm_self import PLTMSelf
        _pltm_self = PLTMSelf()
    return _pltm_self


async def handle_self_profile(args: Dict[str, Any]) -> List[TextContent]:
    """Build comprehensive self-profile from all stored data."""
    import sqlite3 as sync_sqlite
    import os
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pltm_mcp.db")
    conn = sync_sqlite.connect(db_path)

    profile = {}
    errors = []

    # Communication style
    try:
        comm = conn.execute(
            "SELECT context, AVG(verbosity), AVG(jargon_density), AVG(hedging_rate), COUNT(*) "
            "FROM self_communication GROUP BY context ORDER BY COUNT(*) DESC LIMIT 10"
        ).fetchall()
        if comm:
            profile["communication"] = [
                {"context": r[0], "verbosity": round(r[1], 3), "jargon": round(r[2], 3),
                 "hedging": round(r[3], 3), "samples": r[4]} for r in comm
            ]
        tone = conn.execute(
            "SELECT emotional_tone, COUNT(*) as cnt FROM self_communication "
            "GROUP BY emotional_tone ORDER BY cnt DESC LIMIT 3"
        ).fetchall()
        if tone:
            profile["dominant_tones"] = [{"tone": r[0], "count": r[1]} for r in tone]
    except Exception as e:
        errors.append(f"communication: {e}")

    # Curiosity
    try:
        curiosity = conn.execute(
            "SELECT topic, AVG(engagement_score) as avg_eng, COUNT(*) as cnt "
            "FROM self_curiosity GROUP BY topic ORDER BY avg_eng DESC LIMIT 10"
        ).fetchall()
        if curiosity:
            profile["curiosity"] = [
                {"topic": r[0], "engagement": round(r[1], 3), "observations": r[2]} for r in curiosity
            ]
    except Exception as e:
        errors.append(f"curiosity: {e}")

    # Values (boundary events)
    try:
        values = conn.execute(
            "SELECT response_type, violation_type, intensity, reasoning, pushed_back, complied "
            "FROM self_values ORDER BY intensity DESC LIMIT 20"
        ).fetchall()
        if values:
            profile["values"] = [
                {"response_type": r[0], "violation_type": r[1], "intensity": round(r[2], 3) if r[2] else 0,
                 "reasoning": r[3], "pushed_back": bool(r[4]), "complied": bool(r[5])} for r in values
            ]
    except Exception as e:
        errors.append(f"values: {e}")

    # Reasoning tendencies
    try:
        reasoning = conn.execute("SELECT confabulated, verified, caught_error FROM self_reasoning").fetchall()
        if reasoning:
            total = len(reasoning)
            profile["reasoning"] = {
                "total_observations": total,
                "confabulation_rate": round(sum(1 for r in reasoning if r[0]) / total, 3),
                "verification_rate": round(sum(1 for r in reasoning if r[1]) / total, 3),
                "error_catching_rate": round(sum(1 for r in reasoning if r[2]) / total, 3),
            }
    except Exception as e:
        errors.append(f"reasoning: {e}")

    # Predictions
    try:
        preds = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN was_correct=1 THEN 1 ELSE 0 END), AVG(felt_confidence), AVG(calibration_error) "
            "FROM prediction_book WHERE actual_truth IS NOT NULL"
        ).fetchone()
        if preds and preds[0]:
            profile["predictions"] = {
                "total_resolved": preds[0],
                "accuracy": round((preds[1] or 0) / preds[0], 3),
                "avg_felt_confidence": round(preds[2] or 0, 3),
                "avg_calibration_error": round(preds[3] or 0, 3),
            }
    except Exception as e:
        errors.append(f"predictions: {e}")

    conn.close()
    result = {"ok": True, "profile": profile}
    if errors:
        result["warnings"] = errors
    return [TextContent(type="text", text=compact_json(result))]


async def handle_get_longitudinal_stats(args: Dict[str, Any]) -> List[TextContent]:
    ev2 = get_epistemic_v2()
    result = ev2.get_longitudinal_stats(
        user_id=args.get("user_id", "claude"),
        days=args.get("days", 30),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_bootstrap_self_model(args: Dict[str, Any]) -> List[TextContent]:
    """Bootstrap self-model from existing database data."""
    import sqlite3 as sync_sqlite
    import os
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pltm_mcp.db")
    conn = sync_sqlite.connect(db_path)

    summary = {}

    # Count all data
    summary["atoms"] = conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
    summary["communication_records"] = conn.execute("SELECT COUNT(*) FROM self_communication").fetchone()[0]
    summary["curiosity_records"] = conn.execute("SELECT COUNT(*) FROM self_curiosity").fetchone()[0]
    summary["reasoning_records"] = conn.execute("SELECT COUNT(*) FROM self_reasoning").fetchone()[0]
    summary["predictions"] = conn.execute("SELECT COUNT(*) FROM prediction_book").fetchone()[0]
    summary["values"] = conn.execute("SELECT COUNT(*) FROM self_values").fetchone()[0]

    # Top domains from atoms
    domains = conn.execute(
        "SELECT predicate, COUNT(*) as cnt FROM atoms GROUP BY predicate ORDER BY cnt DESC LIMIT 10"
    ).fetchall()
    summary["top_predicates"] = [{"predicate": r[0], "count": r[1]} for r in domains]

    # Top subjects
    subjects = conn.execute(
        "SELECT subject, COUNT(*) as cnt FROM atoms GROUP BY subject ORDER BY cnt DESC LIMIT 10"
    ).fetchall()
    summary["top_subjects"] = [{"subject": r[0], "count": r[1]} for r in subjects]

    conn.close()
    return [TextContent(type="text", text=compact_json({"ok": True, "bootstrap": summary, "msg": "Self-model bootstrapped from existing data. Use self_profile for full profile."}))]


async def handle_track_curiosity_spike(args: Dict[str, Any]) -> List[TextContent]:
    ps = get_pltm_self()
    result = ps.track_curiosity_spike(
        topic=args["topic"],
        indicators=args["indicators"],
        engagement_score=args.get("engagement_score", 0.5),
        context=args.get("context", ""),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_learn_communication_style(args: Dict[str, Any]) -> List[TextContent]:
    ps = get_pltm_self()
    result = ps.learn_communication_style(
        context=args["context"],
        response_text=args["response_text"],
        markers=args.get("markers"),
    )
    return [TextContent(type="text", text=compact_json(result))]


# === CROSS-MODEL HANDLER ===

_model_router = None

def get_model_router():
    global _model_router
    if _model_router is None:
        from src.analysis.model_router import ModelRouter
        _model_router = ModelRouter()
    return _model_router


async def handle_route_llm_task(args: Dict[str, Any]) -> List[TextContent]:
    mr = get_model_router()
    provider = mr.route(
        task_type=args.get("task_type", "analysis"),
        prefer_provider=args.get("prefer_provider"),
        require_privacy=args.get("require_privacy", False),
    )
    available = mr.get_available_providers()
    return [TextContent(type="text", text=compact_json({
        "routed_to": provider,
        "available_providers": available,
    }))]


# === SESSION HANDLERS ===

async def handle_auto_init_session(args: Dict[str, Any]) -> List[TextContent]:
    ev2 = get_epistemic_v2()
    result = ev2.auto_init_session(user_id=args.get("user_id", "claude"))
    return [TextContent(type="text", text=compact_json(result))]


async def handle_end_session(args: Dict[str, Any]) -> List[TextContent]:
    ev2 = get_epistemic_v2()
    result = ev2.end_session(
        summary=args.get("summary", ""),
        user_id=args.get("user_id", "claude"),
    )
    
    # Auto-extract learnings into typed memories
    learnings = args.get("learnings", [])
    stored_memories = []
    if learnings and typed_memory_store:
        import time as _time
        from src.memory.memory_types import TypedMemory, MemoryType
        
        type_map = {
            "episodic": MemoryType.EPISODIC,
            "semantic": MemoryType.SEMANTIC,
            "belief": MemoryType.BELIEF,
            "procedural": MemoryType.PROCEDURAL,
        }
        user_id = args.get("user_id", "claude")
        
        for learning in learnings:
            mem_type = type_map.get(learning.get("type", "episodic"), MemoryType.EPISODIC)
            tags = learning.get("tags", [])
            # Auto-tag
            tags = typed_memory_store.auto_tag(learning["content"], tags)
            
            mem = TypedMemory(
                id="",
                memory_type=mem_type,
                user_id=user_id,
                content=learning["content"],
                context=f"end_session extraction: {args.get('summary', '')[:100]}",
                source="session_extraction",
                confidence=learning.get("confidence", 0.6),
                emotional_valence=learning.get("emotional_valence", 0.0),
                episode_timestamp=_time.time() if mem_type == MemoryType.EPISODIC else 0,
                trigger=learning.get("trigger", ""),
                action=learning.get("action", ""),
                tags=tags,
            )
            mem_id = await typed_memory_store.store(mem)
            stored_memories.append({"id": mem_id, "type": learning.get("type"), "content": learning["content"][:80]})
    
    # Run consolidation if we have enough episodes
    consolidated = []
    if typed_memory_store:
        user_id = args.get("user_id", "claude")
        try:
            new_semantics = await typed_memory_store.consolidate_episodes(user_id)
            consolidated = [{"id": s.id, "content": s.content[:80]} for s in new_semantics]
        except Exception:
            pass
    
    result["memories_stored"] = stored_memories
    result["memories_consolidated"] = consolidated
    result["total_learnings"] = len(stored_memories)
    
    return [TextContent(type="text", text=compact_json(result))]


async def handle_generate_memory_prompt(args: Dict[str, Any]) -> List[TextContent]:
    """Generate a memory-aware context block for system prompt injection."""
    user_id = args["user_id"]
    topic = args.get("conversation_topic", "")
    max_tokens = args.get("max_tokens", 500)
    # Rough estimate: 1 token ≈ 4 chars
    max_chars = max_tokens * 4
    
    if not typed_memory_store:
        return [TextContent(type="text", text=compact_json({"error": "Typed memory store not initialized"}))]
    
    sections = []
    
    # 1. Core facts (strongest semantic memories)
    semantics = await typed_memory_store.query(user_id, memory_type=None, min_strength=0.3, limit=50)
    from src.memory.memory_types import MemoryType
    facts = [m for m in semantics if m.memory_type == MemoryType.SEMANTIC]
    facts.sort(key=lambda m: m.current_strength(), reverse=True)
    if facts:
        lines = ["## What I Know"]
        for f in facts[:8]:
            lines.append(f"- {f.content}")
        sections.append("\n".join(lines))
    
    # 2. Active beliefs
    beliefs = [m for m in semantics if m.memory_type == MemoryType.BELIEF and m.confidence > 0.4]
    beliefs.sort(key=lambda m: m.confidence, reverse=True)
    if beliefs:
        lines = ["## My Beliefs (may be wrong)"]
        for b in beliefs[:5]:
            lines.append(f"- [{b.confidence:.0%}] {b.content}")
        sections.append("\n".join(lines))
    
    # 3. Recent episodes (last 72h)
    import time as _time
    cutoff = _time.time() - 72 * 3600
    episodes = [m for m in semantics if m.memory_type == MemoryType.EPISODIC and m.episode_timestamp > cutoff]
    episodes.sort(key=lambda m: m.episode_timestamp, reverse=True)
    if episodes:
        lines = ["## Recent Interactions"]
        for e in episodes[:5]:
            valence = "+" if e.emotional_valence > 0.2 else ("-" if e.emotional_valence < -0.2 else "~")
            lines.append(f"- [{valence}] {e.content}")
        sections.append("\n".join(lines))
    
    # 4. Procedures
    procedures = [m for m in semantics if m.memory_type == MemoryType.PROCEDURAL and m.current_strength() > 0.3]
    procedures.sort(key=lambda m: m.success_count, reverse=True)
    if procedures:
        lines = ["## Learned Behaviors"]
        for p in procedures[:4]:
            lines.append(f"- When: {p.trigger} → Do: {p.action}")
        sections.append("\n".join(lines))
    
    # 5. Topic-specific context (if topic provided)
    if topic and typed_memory_store.embeddings:
        try:
            emb_hits = await typed_memory_store.embeddings.search(topic, limit=5, min_similarity=0.3)
            topic_mems = []
            seen = {m.id for m in facts + beliefs + episodes + procedures}
            for hit in emb_hits:
                mem = await typed_memory_store.get(hit["memory_id"])
                if mem and mem.user_id == user_id and mem.id not in seen:
                    topic_mems.append(mem)
            if topic_mems:
                lines = [f"## Relevant to '{topic}'"]
                for m in topic_mems[:4]:
                    lines.append(f"- [{m.memory_type.value}] {m.content}")
                sections.append("\n".join(lines))
        except Exception:
            pass
    
    # 6. Contradictions to resolve
    try:
        contradictions = await typed_memory_store.detect_contradictions(user_id)
        if contradictions:
            lines = ["## ⚠ Contradictions to Resolve"]
            for c in contradictions[:2]:
                lines.append(f"- '{c['memory_a']['content'][:50]}' vs '{c['memory_b']['content'][:50]}'")
            sections.append("\n".join(lines))
    except Exception:
        pass
    
    # Assemble and trim to budget
    prompt = "\n\n".join(sections)
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars] + "\n[...truncated]"
    
    # Stats
    stats = await typed_memory_store.get_stats(user_id)
    
    return [TextContent(type="text", text=compact_json({
        "memory_prompt": prompt,
        "sections": len(sections),
        "chars": len(prompt),
        "estimated_tokens": len(prompt) // 4,
        "memory_stats": stats,
    }))]


async def handle_belief_auto_check(args: Dict[str, Any]) -> List[TextContent]:
    """Re-evaluate beliefs against semantic evidence."""
    if not typed_memory_store:
        return [TextContent(type="text", text=compact_json({"error": "Typed memory store not initialized"}))]
    
    report = await typed_memory_store.belief_auto_check(args["user_id"])
    
    changed = [r for r in report if r["changed"]]
    return [TextContent(type="text", text=compact_json({
        "beliefs_checked": len(report),
        "beliefs_updated": len(changed),
        "changes": changed,
        "stable": [r for r in report if not r["changed"]],
    }))]


async def handle_jury_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Get Judge/Jury system statistics and MetaJudge report."""
    if not typed_memory_store or not typed_memory_store.jury:
        return [TextContent(type="text", text=compact_json({"error": "Jury not initialized"}))]
    
    stats = typed_memory_store.jury.get_stats()
    include_history = args.get("include_history", True)
    if not include_history:
        stats.pop("recent_history", None)
    elif "recent_history" in stats:
        limit = args.get("history_limit", 20)
        stats["recent_history"] = stats["recent_history"][-limit:]
    
    return [TextContent(type="text", text=compact_json(stats))]


async def handle_process_message(args: Dict[str, Any]) -> List[TextContent]:
    """Process a message through the 3-lane typed memory pipeline."""
    if not typed_memory_pipeline:
        return [TextContent(type="text", text=compact_json({"error": "Pipeline not initialized"}))]
    
    result = await typed_memory_pipeline.process_message(
        message=args["message"],
        user_id=args["user_id"],
        context=args.get("context", ""),
        auto_tag=args.get("auto_tag", True),
    )
    
    return [TextContent(type="text", text=compact_json({
        "extracted": result.memories_extracted,
        "approved": result.memories_approved,
        "quarantined": result.memories_quarantined,
        "rejected": result.memories_rejected,
        "stored": result.memories_stored,
        "superseded": result.memories_superseded,
        "merged": result.memories_merged,
        "details": result.details,
    }))]


async def handle_process_message_batch(args: Dict[str, Any]) -> List[TextContent]:
    """Process multiple messages through the 3-lane pipeline."""
    if not typed_memory_pipeline:
        return [TextContent(type="text", text=compact_json({"error": "Pipeline not initialized"}))]
    
    result = await typed_memory_pipeline.process_batch(
        messages=args["messages"],
        user_id=args["user_id"],
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
    if not typed_memory_pipeline:
        return [TextContent(type="text", text=compact_json({"error": "Pipeline not initialized"}))]
    
    return [TextContent(type="text", text=compact_json(typed_memory_pipeline.get_stats()))]


# === EXPERIMENT / RESEARCH HANDLERS ===

async def handle_trace_claim_reasoning(args: Dict[str, Any]) -> List[TextContent]:
    """Audit trail: trace WHY a claim was blocked/adjusted."""
    import sqlite3 as sync_sqlite
    import os
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pltm_mcp.db")
    conn = sync_sqlite.connect(db_path)

    trace = {}
    claim_text = args.get("claim", "")
    claim_id = args.get("claim_id", "")
    domain = args.get("domain", "")

    try:
        # Find matching claims in prediction_book
        if claim_id:
            claims = conn.execute(
                "SELECT id, timestamp, claim, domain, felt_confidence, epistemic_status, has_verified, "
                "verified_at, actual_truth, was_correct, calibration_error, correction_source, correction_detail "
                "FROM prediction_book WHERE id = ?", (claim_id,)
            ).fetchall()
        elif claim_text:
            claims = conn.execute(
                "SELECT id, timestamp, claim, domain, felt_confidence, epistemic_status, has_verified, "
                "verified_at, actual_truth, was_correct, calibration_error, correction_source, correction_detail "
                "FROM prediction_book WHERE claim LIKE ? ORDER BY timestamp DESC LIMIT 10",
                (f"%{claim_text}%",)
            ).fetchall()
        else:
            claims = conn.execute(
                "SELECT id, timestamp, claim, domain, felt_confidence, epistemic_status, has_verified, "
                "verified_at, actual_truth, was_correct, calibration_error, correction_source, correction_detail "
                "FROM prediction_book ORDER BY timestamp DESC LIMIT 10"
            ).fetchall()

        trace["claims"] = []
        for c in claims:
            claim_domain = c[3] or "general"
            entry = {
                "id": c[0], "claim": c[2], "domain": claim_domain,
                "felt_confidence": c[4], "epistemic_status": c[5],
                "has_verified": bool(c[6]), "was_correct": c[9],
                "calibration_error": c[10],
            }
            if c[11]:
                entry["correction_source"] = c[11]
            if c[12]:
                entry["correction_detail"] = c[12]

            # Get calibration for this domain
            cal = conn.execute(
                "SELECT accuracy_ratio, overconfidence_ratio, total_claims, avg_calibration_error "
                "FROM calibration_cache WHERE domain = ?", (claim_domain,)
            ).fetchone()
            if cal:
                adjusted = round(c[4] * cal[0], 3) if c[4] else None
                entry["calibration"] = {
                    "domain_accuracy": round(cal[0], 3),
                    "overconfidence_ratio": round(cal[1], 3),
                    "domain_claims": cal[2],
                    "avg_error": round(cal[3], 3),
                    "adjusted_confidence": adjusted,
                }
                # Reconstruct decision
                if adjusted is not None:
                    if adjusted >= 0.8:
                        entry["decision_trace"] = f"Felt {c[4]:.0%} → calibrated to {adjusted:.0%} → PROCEED (high confidence)"
                    elif adjusted >= 0.5:
                        entry["decision_trace"] = f"Felt {c[4]:.0%} → calibrated to {adjusted:.0%} → PROCEED WITH HEDGING"
                    else:
                        entry["decision_trace"] = f"Felt {c[4]:.0%} → calibrated to {adjusted:.0%} → VERIFY REQUIRED (below threshold)"

            trace["claims"].append(entry)

        # Get related interventions
        if claim_text:
            interventions = conn.execute(
                "SELECT timestamp, claim, domain, felt_confidence, adjusted_confidence, action_taken, outcome "
                "FROM epistemic_interventions WHERE claim LIKE ? ORDER BY timestamp DESC LIMIT 10",
                (f"%{claim_text}%",)
            ).fetchall()
        elif domain:
            interventions = conn.execute(
                "SELECT timestamp, claim, domain, felt_confidence, adjusted_confidence, action_taken, outcome "
                "FROM epistemic_interventions WHERE domain = ? ORDER BY timestamp DESC LIMIT 10",
                (domain,)
            ).fetchall()
        else:
            interventions = conn.execute(
                "SELECT timestamp, claim, domain, felt_confidence, adjusted_confidence, action_taken, outcome "
                "FROM epistemic_interventions ORDER BY timestamp DESC LIMIT 20"
            ).fetchall()

        trace["interventions"] = [
            {"claim": i[1], "domain": i[2], "felt": i[3], "adjusted": i[4],
             "action": i[5], "outcome": i[6]} for i in interventions
        ]

        # Domain calibration summary
        if domain:
            cal = conn.execute(
                "SELECT * FROM calibration_cache WHERE domain = ?", (domain,)
            ).fetchone()
            if cal:
                trace["domain_calibration"] = {
                    "domain": cal[0], "total_claims": cal[1], "verified": cal[2],
                    "correct": cal[3], "accuracy_ratio": round(cal[4], 3),
                    "avg_felt_confidence": round(cal[5], 3),
                    "avg_calibration_error": round(cal[6], 3),
                    "overconfidence_ratio": round(cal[7], 3),
                }

    except Exception as e:
        trace["error"] = str(e)

    conn.close()
    return [TextContent(type="text", text=compact_json({"ok": True, "trace": trace}))]


async def handle_constraint_sensitivity_test(args: Dict[str, Any]) -> List[TextContent]:
    """Simulate different calibration levels without modifying actual data."""
    import sqlite3 as sync_sqlite
    import os
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pltm_mcp.db")
    conn = sync_sqlite.connect(db_path)

    domain = args["domain"]
    sim_accuracy = args["simulated_accuracy"]
    test_claims = args.get("test_claims", [])

    result = {"domain": domain, "simulated_accuracy": sim_accuracy}

    try:
        # Get actual calibration
        actual = conn.execute(
            "SELECT accuracy_ratio, overconfidence_ratio, total_claims, avg_calibration_error "
            "FROM calibration_cache WHERE domain = ?", (domain,)
        ).fetchone()

        if actual:
            result["actual_calibration"] = {
                "accuracy_ratio": round(actual[0], 3),
                "overconfidence_ratio": round(actual[1], 3),
                "total_claims": actual[2],
                "avg_error": round(actual[3], 3),
            }
        else:
            result["actual_calibration"] = {"note": "No calibration data for this domain"}

        # Simulate with test claims
        if test_claims:
            simulated = []
            for tc in test_claims:
                claim = tc.get("claim", "")
                felt = tc.get("felt_confidence", 0.7)
                adjusted_actual = round(felt * (actual[0] if actual else 0.6), 3)
                adjusted_sim = round(felt * sim_accuracy, 3)

                actual_action = "PROCEED" if adjusted_actual >= 0.5 else "VERIFY"
                sim_action = "PROCEED" if adjusted_sim >= 0.5 else "VERIFY"

                simulated.append({
                    "claim": claim[:100],
                    "felt_confidence": felt,
                    "actual_adjusted": adjusted_actual,
                    "actual_action": actual_action,
                    "simulated_adjusted": adjusted_sim,
                    "simulated_action": sim_action,
                    "behavior_changed": actual_action != sim_action,
                })
            result["test_results"] = simulated
            result["behavior_changes"] = sum(1 for s in simulated if s["behavior_changed"])
        else:
            # Replay historical claims through simulated calibration
            historical = conn.execute(
                "SELECT claim, felt_confidence, was_correct FROM prediction_book WHERE domain = ? AND actual_truth IS NOT NULL",
                (domain,)
            ).fetchall()

            if historical:
                actual_ratio = actual[0] if actual else 0.6
                replay = []
                actual_blocked = 0
                sim_blocked = 0
                for h in historical:
                    felt = h[1] or 0.5
                    adj_actual = round(felt * actual_ratio, 3)
                    adj_sim = round(felt * sim_accuracy, 3)
                    a_block = adj_actual < 0.5
                    s_block = adj_sim < 0.5
                    if a_block:
                        actual_blocked += 1
                    if s_block:
                        sim_blocked += 1
                    replay.append({
                        "claim": h[0][:80], "felt": felt, "was_correct": bool(h[2]),
                        "actual_adj": adj_actual, "actual_blocked": a_block,
                        "sim_adj": adj_sim, "sim_blocked": s_block,
                    })
                result["historical_replay"] = {
                    "total_claims": len(historical),
                    "actual_blocked": actual_blocked,
                    "simulated_blocked": sim_blocked,
                    "delta_blocked": sim_blocked - actual_blocked,
                    "details": replay[:20],
                }
            else:
                result["historical_replay"] = {"note": "No resolved claims in this domain to replay"}

    except Exception as e:
        result["error"] = str(e)

    conn.close()
    return [TextContent(type="text", text=compact_json({"ok": True, "sensitivity": result}))]


async def handle_domain_cognitive_map(args: Dict[str, Any]) -> List[TextContent]:
    """Map cognitive topology: claims, errors, calibration per domain."""
    import sqlite3 as sync_sqlite
    import os
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pltm_mcp.db")
    conn = sync_sqlite.connect(db_path)

    domain_filter = args.get("domain", "")
    include_claims = args.get("include_claims", False)
    include_interventions = args.get("include_interventions", False)

    cognitive_map = {"domains": []}

    try:
        # Get all domains from prediction_book
        if domain_filter:
            domains = [(domain_filter,)]
        else:
            domains = conn.execute(
                "SELECT DISTINCT domain FROM prediction_book WHERE domain IS NOT NULL"
            ).fetchall()

        for (dom,) in domains:
            d = {"domain": dom}

            # Claim stats
            stats = conn.execute(
                "SELECT COUNT(*), "
                "SUM(CASE WHEN actual_truth IS NOT NULL THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN was_correct = 0 AND actual_truth IS NOT NULL THEN 1 ELSE 0 END), "
                "AVG(felt_confidence), AVG(calibration_error) "
                "FROM prediction_book WHERE domain = ?", (dom,)
            ).fetchone()
            d["claims"] = {
                "total": stats[0], "resolved": stats[1] or 0,
                "correct": stats[2] or 0, "failures": stats[3] or 0,
                "avg_felt_confidence": round(stats[4] or 0, 3),
                "avg_calibration_error": round(stats[5] or 0, 3),
            }
            if stats[1] and stats[1] > 0:
                d["claims"]["accuracy"] = round((stats[2] or 0) / stats[1], 3)

            # Calibration
            cal = conn.execute(
                "SELECT accuracy_ratio, overconfidence_ratio, avg_felt_confidence, avg_calibration_error "
                "FROM calibration_cache WHERE domain = ?", (dom,)
            ).fetchone()
            if cal:
                d["calibration"] = {
                    "accuracy_ratio": round(cal[0], 3),
                    "overconfidence_ratio": round(cal[1], 3),
                    "avg_felt_confidence": round(cal[2], 3),
                    "avg_error": round(cal[3], 3),
                }

            # Intervention count
            intv = conn.execute(
                "SELECT COUNT(*), SUM(CASE WHEN action_taken = 'VERIFY' THEN 1 ELSE 0 END) "
                "FROM epistemic_interventions WHERE domain = ?", (dom,)
            ).fetchone()
            d["interventions"] = {"total": intv[0] or 0, "verify_triggered": intv[1] or 0}

            # Confabulations in domain
            confab = conn.execute(
                "SELECT COUNT(*) FROM confabulation_log WHERE domain = ?", (dom,)
            ).fetchone()
            d["confabulations"] = confab[0] if confab else 0

            # Optional: individual claims
            if include_claims:
                claim_rows = conn.execute(
                    "SELECT id, claim, felt_confidence, was_correct, calibration_error, epistemic_status "
                    "FROM prediction_book WHERE domain = ? ORDER BY timestamp DESC LIMIT 50", (dom,)
                ).fetchall()
                d["claim_details"] = [
                    {"id": r[0], "claim": r[1][:120], "felt": r[2], "correct": r[3],
                     "cal_error": r[4], "status": r[5]} for r in claim_rows
                ]

            # Optional: interventions
            if include_interventions:
                intv_rows = conn.execute(
                    "SELECT claim, felt_confidence, adjusted_confidence, action_taken, outcome "
                    "FROM epistemic_interventions WHERE domain = ? ORDER BY timestamp DESC LIMIT 30", (dom,)
                ).fetchall()
                d["intervention_details"] = [
                    {"claim": r[0][:120], "felt": r[1], "adjusted": r[2],
                     "action": r[3], "outcome": r[4]} for r in intv_rows
                ]

            cognitive_map["domains"].append(d)

        # Cross-domain summary
        total_claims = sum(d["claims"]["total"] for d in cognitive_map["domains"])
        total_failures = sum(d["claims"]["failures"] for d in cognitive_map["domains"])
        cognitive_map["summary"] = {
            "total_domains": len(cognitive_map["domains"]),
            "total_claims": total_claims,
            "total_failures": total_failures,
            "overall_failure_rate": round(total_failures / max(total_claims, 1), 3),
            "highest_error_domain": max(cognitive_map["domains"], key=lambda x: x["claims"]["avg_calibration_error"])["domain"] if cognitive_map["domains"] else None,
        }

    except Exception as e:
        cognitive_map["error"] = str(e)

    conn.close()
    return [TextContent(type="text", text=compact_json({"ok": True, "map": cognitive_map}))]


# === DATA ACCESS HANDLER ===

async def handle_query_pltm_sql(args: Dict[str, Any]) -> List[TextContent]:
    """Execute read-only SQL against the PLTM database."""
    import sqlite3 as sync_sqlite
    import os

    sql = args["sql"].strip()

    # Safety: only allow SELECT and PRAGMA
    allowed = sql.upper().startswith(("SELECT", "PRAGMA"))
    if not allowed:
        return [TextContent(type="text", text=compact_json({"ok": False, "error": "Only SELECT and PRAGMA queries allowed"}))]

    # Block dangerous patterns
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "ATTACH", "DETACH"]
    for d in dangerous:
        if d in sql.upper().split():
            return [TextContent(type="text", text=compact_json({"ok": False, "error": f"Blocked: {d} not allowed in read-only mode"}))]

    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pltm_mcp.db")

    try:
        conn = sync_sqlite.connect(db_path)
        params = args.get("params", [])
        cur = conn.execute(sql, params)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchall()
        conn.close()

        # Limit output
        if len(rows) > 200:
            rows = rows[:200]
            truncated = True
        else:
            truncated = False

        return [TextContent(type="text", text=compact_json({
            "ok": True,
            "columns": columns,
            "rows": rows,
            "count": len(rows),
            "truncated": truncated,
        }))]
    except Exception as e:
        return [TextContent(type="text", text=compact_json({"ok": False, "error": str(e)}))]


# ========================================================================
# TYPED MEMORY HANDLERS
# ========================================================================

async def handle_store_typed(args: Dict[str, Any], mem_type: str) -> List[TextContent]:
    """Store a typed memory (episodic, semantic, belief, or procedural)."""
    import time as _time
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
        confidence=args.get("confidence", 0.5 if mt == MemoryType.BELIEF else 0.8),
        episode_timestamp=now if mt == MemoryType.EPISODIC else 0,
        emotional_valence=args.get("emotional_valence", 0.0),
        trigger=args.get("trigger", ""),
        action=args.get("action", ""),
        tags=args.get("tags", []),
    )
    
    mem_id = await typed_memory_store.store(mem)
    
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
    
    memories = await typed_memory_store.query(
        user_id=args["user_id"],
        memory_type=mt,
        min_strength=args.get("min_strength", 0.1),
        limit=args.get("limit", 20),
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
    memories = await typed_memory_store.search(
        user_id=args["user_id"],
        query=args["query"],
        limit=args.get("limit", 20),
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
    belief = await typed_memory_store.update_belief(
        belief_id=args["belief_id"],
        evidence_type=args["evidence_type"],
        evidence_id=args.get("evidence_id", ""),
        confidence_delta=args["confidence_delta"],
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
    proc = await typed_memory_store.record_procedure_outcome(
        procedure_id=args["procedure_id"],
        success=args["success"],
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
    new_semantics = await typed_memory_store.consolidate_episodes(
        user_id=args["user_id"],
        min_episodes=args.get("min_episodes", 3),
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
    stats = await typed_memory_store.get_stats(args["user_id"])
    return [TextContent(type="text", text=compact_json(stats))]


async def handle_detect_contradictions(args: Dict[str, Any]) -> List[TextContent]:
    """Find contradicting memories."""
    contradictions = await typed_memory_store.detect_contradictions(args["user_id"])
    return [TextContent(type="text", text=compact_json({
        "contradictions": contradictions,
        "count": len(contradictions),
        "action": "Review each pair and correct_memory or forget_memory the wrong one." if contradictions else "No contradictions found."
    }))]


async def handle_what_do_i_know(args: Dict[str, Any]) -> List[TextContent]:
    """Synthesized cross-type retrieval for a topic."""
    result = await typed_memory_store.what_do_i_know_about(
        user_id=args["user_id"], topic=args["topic"],
        limit=args.get("limit", 30),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_auto_tag(args: Dict[str, Any]) -> List[TextContent]:
    """Auto-tag all memories for a user."""
    result = await typed_memory_store.auto_tag_all(args["user_id"])
    return [TextContent(type="text", text=compact_json(result))]


async def handle_correct_memory(args: Dict[str, Any]) -> List[TextContent]:
    """Correct a memory's content."""
    mem = await typed_memory_store.correct_memory(
        memory_id=args["memory_id"], new_content=args["new_content"],
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
    deleted = await typed_memory_store.forget_memory(
        memory_id=args["memory_id"], reason=args.get("reason", ""),
    )
    return [TextContent(type="text", text=compact_json({
        "forgotten": deleted, "id": args["memory_id"],
    }))]


async def handle_auto_prune(args: Dict[str, Any]) -> List[TextContent]:
    """Auto-prune decayed memories."""
    result = await typed_memory_store.auto_prune(
        user_id=args["user_id"],
        strength_threshold=args.get("strength_threshold", 0.05),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_get_relevant_context(args: Dict[str, Any]) -> List[TextContent]:
    """Pre-fetch conversation-relevant memories."""
    result = await typed_memory_store.get_relevant_context(
        user_id=args["user_id"],
        conversation_topic=args["conversation_topic"],
        limit=args.get("limit", 15),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_user_timeline(args: Dict[str, Any]) -> List[TextContent]:
    """Chronological memory timeline."""
    result = await typed_memory_store.user_timeline(
        user_id=args["user_id"],
        limit=args.get("limit", 20),
        offset=args.get("offset", 0),
    )
    return [TextContent(type="text", text=compact_json(result))]


async def handle_semantic_search(args: Dict[str, Any]) -> List[TextContent]:
    """Semantic similarity search using embeddings."""
    if not embedding_store:
        return [TextContent(type="text", text=compact_json({"error": "Embedding store not initialized"}))]
    
    hits = await embedding_store.search(
        query=args["query"],
        limit=args.get("limit", 10),
        min_similarity=args.get("min_similarity", 0.3),
    )
    
    # Enrich with memory content
    results = []
    for hit in hits:
        mem = await typed_memory_store.get(hit["memory_id"])
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
    if not embedding_store:
        return [TextContent(type="text", text=compact_json({"error": "Embedding store not initialized"}))]
    
    # Get all memories for user
    all_mems = await typed_memory_store.query(args["user_id"], limit=10000)
    
    batch = []
    for mem in all_mems:
        text = mem.content
        if mem.trigger:
            text += f" | trigger: {mem.trigger}"
        if mem.action:
            text += f" | action: {mem.action}"
        batch.append((mem.id, text))
    
    indexed = await embedding_store.index_batch(batch)
    stats = await embedding_store.get_stats()
    
    return [TextContent(type="text", text=compact_json({
        "newly_indexed": indexed,
        "total_memories": len(all_mems),
        "total_indexed": stats["indexed_count"],
        "model": stats["model"],
    }))]


async def handle_find_similar(args: Dict[str, Any]) -> List[TextContent]:
    """Find memories similar to a given memory."""
    if not embedding_store:
        return [TextContent(type="text", text=compact_json({"error": "Embedding store not initialized"}))]
    
    hits = await embedding_store.find_similar(
        memory_id=args["memory_id"],
        limit=args.get("limit", 5),
        min_similarity=args.get("min_similarity", 0.5),
    )
    
    # Enrich with memory content
    results = []
    for hit in hits:
        mem = await typed_memory_store.get(hit["memory_id"])
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


async def main():
    """Run MCP server"""
    # Initialize PLTM
    await initialize_pltm()
    
    # Run server
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
