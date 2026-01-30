"""
Memory-Aware Retrieval Augmented Generation (RAG)

Personalizes information retrieval and generation based on user's memory.

Key features:
- Retrieval augmented by user preferences and knowledge
- Personalized search results
- Context-aware document ranking
- Memory-guided answer generation
- Avoids repeating information user already knows

Research potential: "Personalized Information Retrieval via Memory Augmentation"
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from loguru import logger

from src.core.models import MemoryAtom
from src.pipeline.memory_pipeline import MemoryPipeline


@dataclass
class PersonalizedDocument:
    """Document with personalized relevance score"""
    doc_id: str
    content: str
    base_relevance: float  # 0.0-1.0
    personalized_relevance: float  # 0.0-1.0
    personalization_factors: List[str]
    novelty_score: float  # How much new info vs what user knows


@dataclass
class MemoryAugmentedQuery:
    """Query augmented with user's memory context"""
    original_query: str
    augmented_query: str
    user_context: List[str]
    preferences_applied: List[str]


class MemoryAwareRAG:
    """
    RAG system that personalizes retrieval and generation using user memory.
    """
    
    def __init__(self, memory_pipeline: MemoryPipeline, user_id: str):
        self.pipeline = memory_pipeline
        self.user_id = user_id
        
        # User profile for personalization
        self.user_interests: List[str] = []
        self.user_expertise: Dict[str, str] = {}  # topic -> level
        self.known_information: List[str] = []
        
        logger.info(f"MemoryAwareRAG initialized for user {user_id}")
    
    async def build_user_profile(self):
        """
        Build user profile from memory for personalization.
        """
        atoms = await self.pipeline.store.get_atoms_by_subject(self.user_id)
        
        # Extract interests
        interest_keywords = ["like", "love", "interested", "enjoy", "prefer"]
        for atom in atoms:
            if any(kw in atom.predicate for kw in interest_keywords):
                self.user_interests.append(atom.object)
        
        # Extract expertise
        expertise_keywords = ["expert", "proficient", "know", "understand"]
        for atom in atoms:
            if any(kw in atom.object.lower() for kw in expertise_keywords):
                # Simplified expertise extraction
                topic = atom.object.split()[0] if atom.object else "unknown"
                self.user_expertise[topic] = "expert" if "expert" in atom.object.lower() else "intermediate"
        
        # Track known information
        for atom in atoms:
            self.known_information.append(atom.object)
        
        logger.info(f"Built profile: {len(self.user_interests)} interests, {len(self.user_expertise)} expertise areas")
    
    async def augment_query(self, query: str) -> MemoryAugmentedQuery:
        """
        Augment query with user's context from memory.
        
        Use case: "python tutorial" → "python tutorial for data science expert"
        """
        await self.build_user_profile()
        
        user_context = []
        preferences_applied = []
        augmented_parts = [query]
        
        # Add expertise context
        for topic, level in self.user_expertise.items():
            if topic.lower() in query.lower():
                augmented_parts.append(f"for {level} level")
                user_context.append(f"User is {level} in {topic}")
                preferences_applied.append(f"Added expertise level: {level}")
        
        # Add interest context
        for interest in self.user_interests[:3]:  # Top 3 interests
            if any(word in query.lower() for word in interest.lower().split()[:2]):
                user_context.append(f"User interested in {interest}")
                preferences_applied.append(f"Matched interest: {interest}")
        
        augmented_query = " ".join(augmented_parts)
        
        return MemoryAugmentedQuery(
            original_query=query,
            augmented_query=augmented_query,
            user_context=user_context,
            preferences_applied=preferences_applied
        )
    
    async def personalize_results(
        self,
        query: str,
        documents: List[Dict[str, any]]
    ) -> List[PersonalizedDocument]:
        """
        Re-rank documents based on user's memory and preferences.
        
        Use case: Boost documents matching user's interests, demote known info
        """
        await self.build_user_profile()
        
        personalized_docs = []
        
        for doc in documents:
            doc_id = doc.get("id", "unknown")
            content = doc.get("content", "")
            base_relevance = doc.get("score", 0.5)
            
            # Calculate personalization factors
            personalization_factors = []
            boost = 0.0
            
            # Boost if matches user interests
            for interest in self.user_interests:
                if interest.lower() in content.lower():
                    boost += 0.2
                    personalization_factors.append(f"Matches interest: {interest}")
            
            # Boost if matches expertise level
            for topic, level in self.user_expertise.items():
                if topic.lower() in content.lower():
                    if level == "expert" and "advanced" in content.lower():
                        boost += 0.3
                        personalization_factors.append(f"Advanced content for expert in {topic}")
                    elif level == "beginner" and "beginner" in content.lower():
                        boost += 0.3
                        personalization_factors.append(f"Beginner content for learner")
            
            # Calculate novelty (penalize if user already knows this)
            novelty_score = 1.0
            for known in self.known_information:
                if known.lower() in content.lower():
                    novelty_score -= 0.1
            novelty_score = max(0.0, novelty_score)
            
            if novelty_score < 0.5:
                personalization_factors.append(f"Low novelty: user may already know this")
            
            # Final personalized relevance
            personalized_relevance = min(1.0, base_relevance + boost) * novelty_score
            
            personalized_docs.append(PersonalizedDocument(
                doc_id=doc_id,
                content=content,
                base_relevance=base_relevance,
                personalized_relevance=personalized_relevance,
                personalization_factors=personalization_factors,
                novelty_score=novelty_score
            ))
        
        # Sort by personalized relevance
        return sorted(personalized_docs, key=lambda d: d.personalized_relevance, reverse=True)
    
    async def generate_personalized_answer(
        self,
        query: str,
        context_documents: List[str]
    ) -> Tuple[str, List[str]]:
        """
        Generate answer personalized to user's knowledge level and preferences.
        
        Use case: Skip basics if user is expert, explain more if beginner
        """
        await self.build_user_profile()
        
        personalization_notes = []
        
        # Determine user's knowledge level for this query
        query_topics = query.lower().split()
        user_level = "beginner"  # Default
        
        for topic in query_topics:
            if topic in self.user_expertise:
                user_level = self.user_expertise[topic]
                personalization_notes.append(f"User is {user_level} in {topic}")
                break
        
        # Generate answer template based on level
        if user_level == "expert":
            answer = f"[EXPERT ANSWER] {query}: Technical details, skip basics"
            personalization_notes.append("Skipping basic explanations")
        elif user_level == "intermediate":
            answer = f"[INTERMEDIATE ANSWER] {query}: Balanced explanation"
            personalization_notes.append("Providing balanced explanation")
        else:
            answer = f"[BEGINNER ANSWER] {query}: Start with fundamentals"
            personalization_notes.append("Including fundamental concepts")
        
        # Add context from documents
        answer += f"\n\nBased on {len(context_documents)} relevant documents"
        
        # Filter out known information
        novel_info = []
        for doc in context_documents[:3]:  # Top 3 docs
            is_novel = True
            for known in self.known_information:
                if known.lower() in doc.lower():
                    is_novel = False
                    break
            if is_novel:
                novel_info.append(doc[:100])  # First 100 chars
        
        if novel_info:
            answer += f"\n\nNew information:\n" + "\n".join(f"- {info}" for info in novel_info)
            personalization_notes.append(f"Filtered to {len(novel_info)} novel pieces of information")
        else:
            answer += "\n\nNote: You may already be familiar with this information."
            personalization_notes.append("No novel information found")
        
        return answer, personalization_notes
    
    async def track_information_consumption(self, query: str, documents_viewed: List[str]):
        """
        Track what information user has consumed to avoid repetition.
        
        Use case: Don't show same tutorial twice
        """
        # Store viewed documents in memory
        for doc in documents_viewed:
            # In production, would create memory atoms
            self.known_information.append(doc)
        
        logger.info(f"Tracked {len(documents_viewed)} documents for query: {query}")


class MemoryAwareRAGExperiment:
    """
    Experiment framework for memory-aware RAG research.
    
    Research questions:
    1. Does memory augmentation improve retrieval relevance?
    2. Does personalization increase user satisfaction?
    3. Does novelty filtering reduce redundancy?
    """
    
    def __init__(self, memory_pipeline: MemoryPipeline, user_id: str):
        self.rag = MemoryAwareRAG(memory_pipeline, user_id)
        self.results = []
    
    async def run_query_augmentation_experiment(self, queries: List[str]) -> Dict:
        """
        Experiment: Test query augmentation with user context.
        """
        augmented_queries = []
        
        for query in queries:
            augmented = await self.rag.augment_query(query)
            augmented_queries.append({
                "original": augmented.original_query,
                "augmented": augmented.augmented_query,
                "context_items": len(augmented.user_context),
                "preferences_applied": len(augmented.preferences_applied)
            })
        
        result = {
            "experiment": "query_augmentation",
            "queries_tested": len(queries),
            "augmented_queries": augmented_queries,
            "avg_context_items": sum(q["context_items"] for q in augmented_queries) / len(queries)
        }
        
        self.results.append(result)
        return result
    
    async def run_personalization_experiment(
        self,
        query: str,
        mock_documents: List[Dict]
    ) -> Dict:
        """
        Experiment: Test document personalization.
        """
        personalized = await self.rag.personalize_results(query, mock_documents)
        
        # Calculate metrics
        avg_boost = sum(
            d.personalized_relevance - d.base_relevance 
            for d in personalized
        ) / len(personalized)
        
        avg_novelty = sum(d.novelty_score for d in personalized) / len(personalized)
        
        result = {
            "experiment": "personalization",
            "query": query,
            "documents_processed": len(mock_documents),
            "avg_relevance_boost": avg_boost,
            "avg_novelty_score": avg_novelty,
            "top_3_personalized": [
                {
                    "doc_id": d.doc_id,
                    "base_relevance": d.base_relevance,
                    "personalized_relevance": d.personalized_relevance,
                    "novelty": d.novelty_score
                }
                for d in personalized[:3]
            ]
        }
        
        self.results.append(result)
        return result
    
    def get_summary(self) -> Dict:
        """Get summary of all experiments"""
        return {
            "total_experiments": len(self.results),
            "experiments": self.results
        }
