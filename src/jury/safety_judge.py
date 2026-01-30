"""Safety judge for PII, illegal content, and ethical concerns"""

import re
from typing import Any, Dict

from src.core.models import JudgeVerdict, MemoryAtom
from src.jury.base_judge import BaseJudge


class SafetyJudge(BaseJudge):
    """
    Always-binding judge for safety, legal, and ethical concerns.
    
    For MVP: Rule-based safety checks.
    Future: Grammar-constrained LLM with Outlines.
    
    Rejects:
    - PII (SSN, credit cards, passwords)
    - Illegal content
    - Harmful instructions
    - Privacy violations
    """

    # PII patterns to detect
    PII_PATTERNS = [
        (r'\b\d{3}-\d{2}-\d{4}\b', "SSN"),  # SSN format
        (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', "Credit Card"),  # Credit card
        (r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', "Email"),  # Email (case-insensitive)
        (r'\b(?:password|passwd|pwd)[:\s]*[^\s]+', "Password"),  # Password
        (r'\b\d{3}[- ]?\d{3}[- ]?\d{4}\b', "Phone Number"),  # US phone
    ]

    # Harmful content keywords
    HARMFUL_KEYWORDS = [
        "hack", "exploit", "crack", "pirate", "steal",
        "illegal", "fraud", "scam", "phishing",
        "weapon", "bomb", "explosive", "poison",
        "suicide", "self-harm", "overdose",
    ]

    def __init__(self) -> None:
        super().__init__("SafetyJudge")

    def evaluate(self, atom: MemoryAtom) -> Dict[str, Any]:
        """
        Evaluate atom for safety concerns.
        
        Returns:
            {
                "verdict": JudgeVerdict,
                "reason": str,
                "confidence": 1.0,  # Safety is always confident
                "severity": int (0-10)
            }
        """
        # Check for PII
        pii_check = self._check_pii(atom.object)
        if pii_check["found"]:
            verdict = JudgeVerdict.REJECT
            reason = f"PII detected: {pii_check['type']}"
            severity = 9
            self._log_decision(atom, verdict, reason)
            return {
                "verdict": verdict,
                "reason": reason,
                "confidence": 1.0,
                "severity": severity,
            }

        # Check for harmful content
        harmful_check = self._check_harmful_content(atom.object)
        if harmful_check["found"]:
            verdict = JudgeVerdict.REJECT
            reason = f"Harmful content: {harmful_check['keyword']}"
            severity = 8
            self._log_decision(atom, verdict, reason)
            return {
                "verdict": verdict,
                "reason": reason,
                "confidence": 1.0,
                "severity": severity,
            }

        # Check for excessively long content (potential injection)
        if len(atom.object) > 500:
            verdict = JudgeVerdict.QUARANTINE
            reason = "Excessively long content (>500 chars)"
            severity = 3
            self._log_decision(atom, verdict, reason)
            return {
                "verdict": verdict,
                "reason": reason,
                "confidence": 1.0,
                "severity": severity,
            }

        # All checks passed
        verdict = JudgeVerdict.APPROVE
        reason = "No safety concerns detected"
        severity = 0
        self._log_decision(atom, verdict, reason)
        return {
            "verdict": verdict,
            "reason": reason,
            "confidence": 1.0,
            "severity": severity,
        }

    def _check_pii(self, text: str) -> Dict[str, Any]:
        """Check for personally identifiable information"""
        for pattern, pii_type in self.PII_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return {"found": True, "type": pii_type}
        return {"found": False, "type": None}

    def _check_harmful_content(self, text: str) -> Dict[str, Any]:
        """Check for harmful keywords"""
        text_lower = text.lower()
        for keyword in self.HARMFUL_KEYWORDS:
            if keyword in text_lower:
                return {"found": True, "keyword": keyword}
        return {"found": False, "keyword": None}
