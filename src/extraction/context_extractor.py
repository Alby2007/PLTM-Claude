"""Extract temporal, situational, and conditional contexts from text"""

import re
from typing import List

from loguru import logger


class ContextExtractor:
    """
    Extract temporal, situational, and conditional contexts from text.
    
    For MVP: Simple regex-based extraction.
    Future: LLM-based context understanding.
    """

    CONTEXT_PATTERNS = [
        # Temporal: "when X", "during X", "at X"
        (r"when (\w+(?:\s+\w+){0,2})", "temporal"),
        (r"during (\w+(?:\s+\w+){0,2})", "temporal"),
        (r"at (night|work|home|school)", "temporal"),
        (r"in the (morning|evening|afternoon)", "temporal"),
        
        # Conditional: "if X", "while X"
        (r"if (\w+(?:\s+\w+){0,2})", "conditional"),
        (r"while (\w+(?:\s+\w+){0,2})", "conditional"),
        
        # Situational: "at work", "at home", "at parties"
        (r"at (work|home|school|office|parties|the office)", "situational"),
        (r"for (work|fun|leisure|business)", "situational"),
        (r"when (coding|learning|relaxing|working|exercising)", "situational"),
    ]

    def extract(self, text: str) -> List[str]:
        """
        Extract context tags from text.
        
        Args:
            text: Message text to extract contexts from
            
        Returns:
            List of context strings
        """
        contexts = []
        text_lower = text.lower()
        
        for pattern, context_type in self.CONTEXT_PATTERNS:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                context = match.group(1).strip()
                contexts.append(context)
                logger.debug(f"Extracted {context_type} context: {context}")
        
        # Deduplicate
        unique_contexts = list(set(contexts))
        
        if unique_contexts:
            logger.info(f"Extracted {len(unique_contexts)} contexts: {unique_contexts}")
        
        return unique_contexts
