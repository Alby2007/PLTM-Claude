"""
Lifelong Learning Agent

An AI agent that improves over time by learning from every interaction.
This is an OPTIONAL layer on top of the existing memory system - it does not
modify or break any existing functionality.

Key features:
- Learns user preferences, expertise, projects over time
- Retrieves relevant context for personalized responses
- Extracts learnings from each interaction
- Improves accuracy and relevance with accumulated knowledge

This is designed for research and experimentation in lifelong learning.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime

from loguru import logger

from src.core.models import MemoryAtom, AtomType
from src.storage.sqlite_store import SQLiteGraphStore
from src.extraction.rule_based import RuleBasedExtractor
from src.pipeline.memory_pipeline import MemoryPipeline


class LifelongLearningAgent:
    """
    Agent that improves through experience by accumulating personalized knowledge.
    
    This agent:
    1. Retrieves relevant memories before responding
    2. Uses memories to personalize responses
    3. Extracts learnings from interactions
    4. Stores learnings for future use
    
    Over time, the agent becomes increasingly personalized and useful.
    """
    
    def __init__(
        self,
        memory_pipeline: MemoryPipeline,
        user_id: str,
        learning_enabled: bool = True
    ):
        """
        Initialize lifelong learning agent.
        
        Args:
            memory_pipeline: Existing memory pipeline (no modifications needed)
            user_id: User identifier for personalization
            learning_enabled: Whether to extract and store learnings
        """
        self.pipeline = memory_pipeline
        self.user_id = user_id
        self.learning_enabled = learning_enabled
        
        # Stats for tracking improvement
        self.interaction_count = 0
        self.learnings_extracted = 0
        self.context_retrievals = 0
        
        logger.info(
            f"LifelongLearningAgent initialized for user {user_id} "
            f"(learning_enabled={learning_enabled})"
        )
    
    async def get_context(
        self,
        query: str,
        limit: int = 20,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from accumulated memories.
        
        Args:
            query: Current user message/query
            limit: Maximum number of memories to retrieve
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of relevant memory atoms with metadata
        """
        self.context_retrievals += 1
        
        # Get all non-historical atoms for this user
        # In a production system, you'd use semantic search here
        atoms = await self.pipeline.store.find_by_triple(
            subject=self.user_id,
            exclude_historical=True
        )
        
        # Filter by confidence
        relevant_atoms = [
            atom for atom in atoms
            if atom.confidence >= min_confidence
        ]
        
        # Sort by confidence and recency
        relevant_atoms.sort(
            key=lambda a: (a.confidence, a.created_at),
            reverse=True
        )
        
        # Limit results
        relevant_atoms = relevant_atoms[:limit]
        
        # Format for context
        context = []
        for atom in relevant_atoms:
            context.append({
                'statement': f"{atom.subject} {atom.predicate} {atom.object}",
                'confidence': atom.confidence,
                'type': atom.atom_type.value,
                'created_at': atom.created_at.isoformat() if atom.created_at else None,
                'context': atom.context,
            })
        
        logger.debug(
            f"Retrieved {len(context)} relevant memories for user {self.user_id}"
        )
        
        return context
    
    def format_context_for_prompt(self, context: List[Dict[str, Any]]) -> str:
        """
        Format retrieved context for LLM prompt.
        
        Args:
            context: List of memory dicts from get_context()
        
        Returns:
            Formatted string for prompt injection
        """
        if not context:
            return "No prior knowledge about this user yet."
        
        lines = ["What you know about this user:"]
        
        # Group by type
        by_type = {}
        for item in context:
            atom_type = item['type']
            if atom_type not in by_type:
                by_type[atom_type] = []
            by_type[atom_type].append(item)
        
        # Format each type
        for atom_type, items in by_type.items():
            lines.append(f"\n{atom_type.upper()}:")
            for item in items:
                confidence_str = f" (confidence: {item['confidence']:.0%})"
                context_str = f" [in context: {item['context']}]" if item['context'] else ""
                lines.append(f"  - {item['statement']}{confidence_str}{context_str}")
        
        return "\n".join(lines)
    
    async def extract_learnings(
        self,
        user_message: str,
        assistant_response: str,
        interaction_metadata: Optional[Dict] = None
    ) -> List[str]:
        """
        Extract learnings from an interaction.
        
        This is a simple rule-based extraction. In production, you'd use an LLM
        to extract more nuanced learnings.
        
        Args:
            user_message: What the user said
            assistant_response: What the assistant responded
            interaction_metadata: Optional metadata (task success, user feedback, etc.)
        
        Returns:
            List of learning statements to store
        """
        if not self.learning_enabled:
            return []
        
        learnings = []
        
        # Extract explicit preferences from user message
        # Example: "I prefer Python" -> learning
        preference_keywords = [
            'prefer', 'like', 'love', 'enjoy', 'want',
            'dislike', 'hate', 'avoid', 'don\'t like'
        ]
        
        for keyword in preference_keywords:
            if keyword in user_message.lower():
                # This is a simplified extraction - in production use the existing
                # RuleBasedExtractor or an LLM
                learnings.append(user_message)
                break
        
        # Extract expertise signals
        expertise_keywords = [
            'I work on', 'I\'m working on', 'my project',
            'I\'m building', 'I specialize in', 'I\'m experienced in'
        ]
        
        for keyword in expertise_keywords:
            if keyword in user_message.lower():
                learnings.append(user_message)
                break
        
        # Extract from metadata if provided
        if interaction_metadata:
            if interaction_metadata.get('task_completed'):
                learnings.append(f"User successfully completed task: {interaction_metadata.get('task_name')}")
            
            if interaction_metadata.get('user_feedback'):
                learnings.append(f"User feedback: {interaction_metadata['user_feedback']}")
        
        self.learnings_extracted += len(learnings)
        
        logger.debug(f"Extracted {len(learnings)} learnings from interaction")
        
        return learnings
    
    async def process_learnings(self, learnings: List[str]) -> int:
        """
        Process and store extracted learnings.
        
        Args:
            learnings: List of learning statements
        
        Returns:
            Number of learnings successfully stored
        """
        stored_count = 0
        
        for learning in learnings:
            try:
                # Process through existing pipeline (no modifications needed!)
                await self.pipeline.process(self.user_id, learning)
                stored_count += 1
            except Exception as e:
                logger.warning(f"Failed to store learning: {learning} - {e}")
        
        logger.info(f"Stored {stored_count}/{len(learnings)} learnings for user {self.user_id}")
        
        return stored_count
    
    async def respond(
        self,
        message: str,
        llm_callable: Optional[Any] = None,
        extract_learnings: bool = True,
        interaction_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate a response with memory-augmented context.
        
        This is the main entry point for the lifelong learning agent.
        
        Args:
            message: User's message
            llm_callable: Optional LLM function to call (if None, returns context only)
            extract_learnings: Whether to extract learnings from this interaction
            interaction_metadata: Optional metadata about the interaction
        
        Returns:
            Dict with response, context, learnings, and stats
        """
        self.interaction_count += 1
        
        # Step 1: Retrieve relevant context
        context = await self.get_context(message, limit=20)
        
        # Step 2: Format context for prompt
        context_str = self.format_context_for_prompt(context)
        
        # Step 3: Generate response (if LLM provided)
        response = None
        if llm_callable:
            prompt = f"""You are an AI assistant with long-term memory.

{context_str}

User message: {message}

Respond based on your accumulated knowledge about this user.
Be personalized, relevant, and reference prior knowledge when appropriate.
"""
            response = await llm_callable(prompt)
        
        # Step 4: Extract learnings
        learnings = []
        if extract_learnings and response:
            learnings = await self.extract_learnings(
                message,
                response,
                interaction_metadata
            )
        
        # Step 5: Store learnings
        stored_count = 0
        if learnings:
            stored_count = await self.process_learnings(learnings)
        
        return {
            'response': response,
            'context': context,
            'context_count': len(context),
            'learnings': learnings,
            'learnings_stored': stored_count,
            'interaction_number': self.interaction_count,
            'total_learnings_extracted': self.learnings_extracted,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics for tracking improvement over time.
        
        Returns:
            Dict with interaction count, learnings, context retrievals
        """
        return {
            'user_id': self.user_id,
            'interaction_count': self.interaction_count,
            'learnings_extracted': self.learnings_extracted,
            'context_retrievals': self.context_retrievals,
            'learning_enabled': self.learning_enabled,
            'avg_learnings_per_interaction': (
                self.learnings_extracted / self.interaction_count
                if self.interaction_count > 0 else 0.0
            ),
        }
    
    async def get_user_profile(self) -> Dict[str, Any]:
        """
        Generate a comprehensive user profile from accumulated memories.
        
        Returns:
            Dict with user preferences, expertise, projects, etc.
        """
        # Get all memories
        atoms = await self.pipeline.store.find_by_triple(
            subject=self.user_id,
            exclude_historical=True
        )
        
        # Group by type
        profile = {
            'user_id': self.user_id,
            'total_memories': len(atoms),
            'preferences': [],
            'expertise': [],
            'affiliations': [],
            'attributes': [],
            'facts': [],
        }
        
        for atom in atoms:
            statement = f"{atom.predicate} {atom.object}"
            
            if atom.atom_type == AtomType.PREFERENCE:
                profile['preferences'].append(statement)
            elif atom.atom_type == AtomType.EXPERTISE:
                profile['expertise'].append(statement)
            elif atom.atom_type == AtomType.AFFILIATION:
                profile['affiliations'].append(statement)
            elif atom.atom_type == AtomType.ATTRIBUTE:
                profile['attributes'].append(statement)
            elif atom.atom_type == AtomType.FACT:
                profile['facts'].append(statement)
        
        return profile
    
    async def reset_user_memories(self) -> int:
        """
        Reset all memories for this user (for testing/experimentation).
        
        Returns:
            Number of memories deleted
        """
        atoms = await self.pipeline.store.find_by_triple(
            subject=self.user_id,
            exclude_historical=False  # Get all, including historical
        )
        
        count = len(atoms)
        
        # Delete all atoms for this user
        for atom in atoms:
            await self.pipeline.store.delete_atom(atom.id)
        
        logger.warning(f"Reset {count} memories for user {self.user_id}")
        
        return count


class LifelongLearningExperiment:
    """
    Experimental framework for measuring lifelong learning improvement.
    
    This class helps run controlled experiments to measure how agent
    performance improves over time as memories accumulate.
    """
    
    def __init__(self, agent: LifelongLearningAgent):
        self.agent = agent
        self.experiment_data = []
    
    async def run_interaction(
        self,
        message: str,
        llm_callable: Any,
        ground_truth_response: Optional[str] = None,
        task_success: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Run a single interaction and record metrics.
        
        Args:
            message: User message
            llm_callable: LLM function
            ground_truth_response: Optional expected response for evaluation
            task_success: Optional task completion indicator
        
        Returns:
            Interaction results with metrics
        """
        # Run interaction
        result = await self.agent.respond(
            message,
            llm_callable,
            extract_learnings=True,
            interaction_metadata={'task_completed': task_success}
        )
        
        # Record metrics
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'interaction_number': result['interaction_number'],
            'message': message,
            'response': result['response'],
            'context_count': result['context_count'],
            'learnings_stored': result['learnings_stored'],
            'task_success': task_success,
        }
        
        # Evaluate if ground truth provided
        if ground_truth_response and result['response']:
            # Simple similarity check (in production use semantic similarity)
            similarity = self._calculate_similarity(
                result['response'],
                ground_truth_response
            )
            metrics['response_quality'] = similarity
        
        self.experiment_data.append(metrics)
        
        return metrics
    
    def _calculate_similarity(self, response: str, ground_truth: str) -> float:
        """Simple similarity metric (placeholder for semantic similarity)"""
        # This is a very naive implementation
        # In production, use sentence embeddings or BLEU score
        response_words = set(response.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        if not truth_words:
            return 0.0
        
        overlap = len(response_words & truth_words)
        return overlap / len(truth_words)
    
    def get_improvement_metrics(self) -> Dict[str, Any]:
        """
        Calculate improvement metrics over time.
        
        Returns:
            Dict with baseline vs current performance
        """
        if len(self.experiment_data) < 2:
            return {'error': 'Need at least 2 interactions for comparison'}
        
        # Split into baseline (first 10%) and current (last 10%)
        n = len(self.experiment_data)
        baseline_size = max(1, n // 10)
        
        baseline_data = self.experiment_data[:baseline_size]
        current_data = self.experiment_data[-baseline_size:]
        
        # Calculate metrics
        def avg_metric(data, key):
            values = [d[key] for d in data if key in d and d[key] is not None]
            return sum(values) / len(values) if values else 0.0
        
        baseline_quality = avg_metric(baseline_data, 'response_quality')
        current_quality = avg_metric(current_data, 'response_quality')
        
        baseline_context = avg_metric(baseline_data, 'context_count')
        current_context = avg_metric(current_data, 'context_count')
        
        return {
            'total_interactions': n,
            'baseline_response_quality': baseline_quality,
            'current_response_quality': current_quality,
            'quality_improvement': current_quality - baseline_quality,
            'quality_improvement_pct': (
                ((current_quality - baseline_quality) / baseline_quality * 100)
                if baseline_quality > 0 else 0.0
            ),
            'baseline_context_count': baseline_context,
            'current_context_count': current_context,
            'context_growth': current_context - baseline_context,
        }
    
    def generate_report(self) -> str:
        """Generate experiment report"""
        metrics = self.get_improvement_metrics()
        
        if 'error' in metrics:
            return f"Experiment Report: {metrics['error']}"
        
        report = f"""
================================================================================
LIFELONG LEARNING EXPERIMENT REPORT
================================================================================

Total Interactions: {metrics['total_interactions']}

RESPONSE QUALITY
  Baseline (first 10%):  {metrics['baseline_response_quality']:.1%}
  Current (last 10%):    {metrics['current_response_quality']:.1%}
  Improvement:           {metrics['quality_improvement']:+.1%}
  Improvement %:         {metrics['quality_improvement_pct']:+.1f}%

CONTEXT ACCUMULATION
  Baseline context:      {metrics['baseline_context_count']:.1f} memories
  Current context:       {metrics['current_context_count']:.1f} memories
  Growth:                {metrics['context_growth']:+.1f} memories

AGENT STATS
{self._format_agent_stats()}

================================================================================
"""
        return report
    
    def _format_agent_stats(self) -> str:
        """Format agent statistics"""
        stats = self.agent.get_stats()
        return f"""  User ID:               {stats['user_id']}
  Interactions:          {stats['interaction_count']}
  Learnings Extracted:   {stats['learnings_extracted']}
  Avg Learnings/Int:     {stats['avg_learnings_per_interaction']:.2f}
  Learning Enabled:      {stats['learning_enabled']}"""
