"""
Advanced mood pattern detection algorithms.

This module provides enhanced pattern detection including:
- Cyclical patterns (weekly, monthly cycles)
- Mood volatility (stability vs instability)
- Mood velocity (rate of change)
- Trigger patterns (what causes mood changes)
"""

from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import math

from loguru import logger


class AdvancedMoodPatterns:
    """
    Advanced algorithms for mood pattern detection.
    
    Extends basic pattern detection with:
    - Cyclical pattern detection (weekly/monthly cycles)
    - Volatility analysis (mood stability)
    - Velocity calculation (rate of mood change)
    - Trigger identification (what causes changes)
    """
    
    @staticmethod
    def find_cyclical_patterns(history: List[Dict]) -> List[Dict[str, Any]]:
        """
        Detect cyclical mood patterns (weekly, bi-weekly, monthly).
        
        Args:
            history: List of mood records
            
        Returns:
            List of detected cycles with period and strength
        """
        if len(history) < 14:  # Need at least 2 weeks
            return []
        
        cycles = []
        
        # Check for weekly cycles (7-day period)
        weekly_cycle = AdvancedMoodPatterns._detect_cycle(history, period_days=7)
        if weekly_cycle["strength"] > 0.5:
            cycles.append({
                "type": "weekly",
                "period_days": 7,
                "strength": weekly_cycle["strength"],
                "pattern": weekly_cycle["pattern"]
            })
        
        # Check for bi-weekly cycles (14-day period)
        biweekly_cycle = AdvancedMoodPatterns._detect_cycle(history, period_days=14)
        if biweekly_cycle["strength"] > 0.5:
            cycles.append({
                "type": "bi-weekly",
                "period_days": 14,
                "strength": biweekly_cycle["strength"],
                "pattern": biweekly_cycle["pattern"]
            })
        
        # Check for monthly cycles (30-day period)
        if len(history) >= 60:  # Need at least 2 months
            monthly_cycle = AdvancedMoodPatterns._detect_cycle(history, period_days=30)
            if monthly_cycle["strength"] > 0.5:
                cycles.append({
                    "type": "monthly",
                    "period_days": 30,
                    "strength": monthly_cycle["strength"],
                    "pattern": monthly_cycle["pattern"]
                })
        
        logger.debug(f"Detected {len(cycles)} cyclical patterns")
        return cycles
    
    @staticmethod
    def _detect_cycle(history: List[Dict], period_days: int) -> Dict[str, Any]:
        """
        Detect if a specific cycle period exists in the data.
        
        Uses autocorrelation to find repeating patterns.
        """
        if len(history) < period_days * 2:
            return {"strength": 0.0, "pattern": []}
        
        # Sort by timestamp
        sorted_history = sorted(history, key=lambda x: x["timestamp"])
        
        # Group by cycle position
        cycle_positions = defaultdict(list)
        for record in sorted_history:
            days_since_start = (record["timestamp"] - sorted_history[0]["timestamp"]).days
            position = days_since_start % period_days
            cycle_positions[position].append(record["mood"])
        
        # Calculate consistency at each position
        consistencies = []
        for position in range(period_days):
            moods = cycle_positions.get(position, [])
            if len(moods) >= 2:
                # Most common mood at this position
                most_common = Counter(moods).most_common(1)[0]
                consistency = most_common[1] / len(moods)
                consistencies.append(consistency)
        
        # Overall cycle strength
        if consistencies:
            strength = sum(consistencies) / len(consistencies)
        else:
            strength = 0.0
        
        # Build pattern (most common mood at each position)
        pattern = []
        for position in range(period_days):
            moods = cycle_positions.get(position, [])
            if moods:
                most_common = Counter(moods).most_common(1)[0][0]
                pattern.append(most_common)
            else:
                pattern.append(None)
        
        return {"strength": strength, "pattern": pattern}
    
    @staticmethod
    def calculate_volatility(history: List[Dict]) -> float:
        """
        Calculate mood volatility (how stable vs unstable moods are).
        
        Returns:
            Volatility score (0.0 = very stable, 1.0 = very volatile)
        """
        if len(history) < 3:
            return 0.0
        
        # Sort by timestamp
        sorted_history = sorted(history, key=lambda x: x["timestamp"])
        
        # Count mood changes
        changes = 0
        for i in range(len(sorted_history) - 1):
            if sorted_history[i]["mood"] != sorted_history[i + 1]["mood"]:
                changes += 1
        
        # Volatility = change rate
        volatility = changes / (len(sorted_history) - 1)
        
        return round(volatility, 2)
    
    @staticmethod
    def calculate_mood_velocity(history: List[Dict]) -> Dict[str, float]:
        """
        Calculate mood velocity (rate of change for each mood type).
        
        Returns:
            Dict mapping mood types to their average duration
        """
        if len(history) < 2:
            return {}
        
        # Sort by timestamp
        sorted_history = sorted(history, key=lambda x: x["timestamp"])
        
        # Track duration of each mood
        mood_durations = defaultdict(list)
        current_mood = sorted_history[0]["mood"]
        current_start = sorted_history[0]["timestamp"]
        
        for i in range(1, len(sorted_history)):
            if sorted_history[i]["mood"] != current_mood:
                # Mood changed
                duration = (sorted_history[i]["timestamp"] - current_start).total_seconds() / 3600  # hours
                mood_durations[current_mood].append(duration)
                
                current_mood = sorted_history[i]["mood"]
                current_start = sorted_history[i]["timestamp"]
        
        # Calculate average duration for each mood
        velocity = {}
        for mood, durations in mood_durations.items():
            avg_duration = sum(durations) / len(durations)
            velocity[mood] = round(avg_duration, 1)  # hours
        
        return velocity
    
    @staticmethod
    def find_trigger_patterns(history: List[Dict]) -> List[Dict[str, Any]]:
        """
        Identify patterns in what triggers mood changes.
        
        Returns:
            List of trigger patterns (time-based, sequence-based)
        """
        if len(history) < 5:
            return []
        
        triggers = []
        
        # Sort by timestamp
        sorted_history = sorted(history, key=lambda x: x["timestamp"])
        
        # Find rapid mood swings (within 1 hour)
        for i in range(len(sorted_history) - 1):
            time_diff = (sorted_history[i + 1]["timestamp"] - sorted_history[i]["timestamp"]).total_seconds() / 60
            
            if time_diff < 60 and sorted_history[i]["mood"] != sorted_history[i + 1]["mood"]:
                # Rapid mood change
                triggers.append({
                    "type": "rapid_change",
                    "from": sorted_history[i]["mood"],
                    "to": sorted_history[i + 1]["mood"],
                    "time_minutes": round(time_diff, 1)
                })
        
        # Find mood rebounds (A → B → A pattern)
        for i in range(len(sorted_history) - 2):
            if (sorted_history[i]["mood"] == sorted_history[i + 2]["mood"] and
                sorted_history[i]["mood"] != sorted_history[i + 1]["mood"]):
                # Rebound pattern
                triggers.append({
                    "type": "rebound",
                    "baseline": sorted_history[i]["mood"],
                    "deviation": sorted_history[i + 1]["mood"]
                })
        
        return triggers
    
    @staticmethod
    def detect_mood_clusters(history: List[Dict]) -> Dict[str, List[str]]:
        """
        Detect which moods tend to cluster together.
        
        Returns:
            Dict mapping moods to their frequently co-occurring moods
        """
        if len(history) < 5:
            return {}
        
        # Sort by timestamp
        sorted_history = sorted(history, key=lambda x: x["timestamp"])
        
        # Track mood co-occurrences (within 24 hours)
        co_occurrences = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(sorted_history)):
            mood_a = sorted_history[i]["mood"]
            
            # Look at moods within next 24 hours
            for j in range(i + 1, len(sorted_history)):
                time_diff = (sorted_history[j]["timestamp"] - sorted_history[i]["timestamp"]).total_seconds() / 3600
                
                if time_diff > 24:
                    break
                
                mood_b = sorted_history[j]["mood"]
                if mood_a != mood_b:
                    co_occurrences[mood_a][mood_b] += 1
        
        # Find significant clusters (occurred at least 2 times)
        clusters = {}
        for mood, related_moods in co_occurrences.items():
            significant = [
                related_mood for related_mood, count in related_moods.items()
                if count >= 2
            ]
            if significant:
                clusters[mood] = significant
        
        return clusters
    
    @staticmethod
    def calculate_mood_entropy(history: List[Dict]) -> float:
        """
        Calculate mood entropy (measure of unpredictability).
        
        Higher entropy = more unpredictable moods
        Lower entropy = more predictable moods
        
        Returns:
            Entropy score (0.0 = perfectly predictable, higher = more random)
        """
        if len(history) < 2:
            return 0.0
        
        # Calculate mood distribution
        moods = [record["mood"] for record in history]
        total = len(moods)
        
        # Calculate probabilities
        mood_counts = Counter(moods)
        probabilities = [count / total for count in mood_counts.values()]
        
        # Calculate Shannon entropy
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        
        return round(entropy, 2)
