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


async def initialize_pltm():
    """Initialize PLTM system components"""
    global store, pipeline, personality_agent, personality_synth
    global mood_tracker, mood_patterns, conflict_resolver, contextual_personality
    
    # Initialize storage
    store = SQLiteGraphStore("pltm_mcp.db")
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
    
    logger.info("PLTM MCP Server initialized")


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
            description="Learn from any URL content. Extracts facts, concepts, relationships. AGI path - continuous learning from web.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Source URL"},
                    "content": {"type": "string", "description": "Text content from the URL"},
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
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    
    try:
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
        
        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
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
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "status": "stored",
            "atom_id": str(atom.id),
            "atom_type": args["atom_type"],
            "content": f"{args['subject']} {args['predicate']} {args['object']}"
        }, indent=2)
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
