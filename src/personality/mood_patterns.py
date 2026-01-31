"""
Detect and predict mood patterns over time.

This module analyzes mood history to find temporal, sequential, and cyclical patterns,
enabling mood prediction and better understanding of user emotional states.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from loguru import logger

from src.storage.sqlite_store import SQLiteGraphStore
from src.personality.mood_tracker import MoodTracker
from src.personality.advanced_mood_patterns import AdvancedMoodPatterns


class MoodPatterns:
    """
    Detect recurring mood patterns and predict future moods.
    
    Finds patterns like:
    - Temporal: Stressed on certain days/times
    - Sequential: Frustration before breakthroughs
    - Cyclical: Mood cycles over weeks
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        self.mood_tracker = MoodTracker(store)
        logger.info("MoodPatterns initialized")
    
    async def detect_patterns(
        self,
        user_id: str,
        window_days: int = 90
    ) -> Dict[str, Any]:
        """
        Detect all mood patterns for a user.
        
        Args:
            user_id: User identifier
            window_days: Number of days to analyze
            
        Returns:
            Dictionary with detected patterns
        """
        # Get mood history
        history = await self.mood_tracker.get_mood_history(user_id, days=window_days)
        
        if len(history) < 5:
            logger.debug(f"Insufficient mood data for pattern detection ({len(history)} records)")
            return {
                "temporal_patterns": {},
                "sequential_patterns": [],
                "typical_trajectory": None,
                "mood_distribution": {},
                "cyclical_patterns": [],
                "volatility": 0.0,
                "mood_velocity": {},
                "trigger_patterns": [],
                "data_points": len(history)
            }
        
        patterns = {
            "temporal_patterns": self._find_time_patterns(history),
            "sequential_patterns": self._find_sequences(history),
            "typical_trajectory": self._find_baseline(history),
            "mood_distribution": self._calculate_distribution(history),
            "cyclical_patterns": self._find_cyclical_patterns(history),
            "volatility": self._calculate_volatility(history),
            "mood_velocity": self._calculate_mood_velocity(history),
            "trigger_patterns": self._find_trigger_patterns(history),
            "data_points": len(history)
        }
        
        logger.info(f"Detected mood patterns from {len(history)} data points")
        return patterns
    
    def _find_time_patterns(self, history: List[Dict]) -> Dict[str, Any]:
        """
        Find temporal patterns (time-of-day, day-of-week).
        
        Returns patterns like:
        - "Stressed on Mondays"
        - "Happy in mornings"
        - "Frustrated in evenings"
        """
        # Group by day of week
        by_weekday = defaultdict(list)
        for record in history:
            weekday = record["timestamp"].strftime("%A")
            by_weekday[weekday].append(record["mood"])
        
        # Group by hour of day
        by_hour = defaultdict(list)
        for record in history:
            hour = record["timestamp"].hour
            by_hour[hour].append(record["mood"])
        
        # Find dominant moods for each time period
        weekday_patterns = {}
        for day, moods in by_weekday.items():
            if len(moods) >= 2:  # Need at least 2 observations
                most_common = Counter(moods).most_common(1)[0]
                if most_common[1] / len(moods) > 0.5:  # >50% of the time
                    weekday_patterns[day] = most_common[0]
        
        hour_patterns = {}
        for hour, moods in by_hour.items():
            if len(moods) >= 2:
                most_common = Counter(moods).most_common(1)[0]
                if most_common[1] / len(moods) > 0.5:
                    hour_patterns[f"{hour:02d}:00"] = most_common[0]
        
        return {
            "by_weekday": weekday_patterns,
            "by_hour": hour_patterns
        }
    
    def _find_sequences(self, history: List[Dict]) -> List[Dict[str, Any]]:
        """
        Find sequential patterns (mood A often followed by mood B).
        
        Returns patterns like:
        - "Frustration → Happy" (breakthrough pattern)
        - "Stressed → Calm" (resolution pattern)
        """
        if len(history) < 3:
            return []
        
        # Build transition counts
        transitions = defaultdict(int)
        for i in range(len(history) - 1):
            current_mood = history[i]["mood"]
            next_mood = history[i + 1]["mood"]
            transitions[(current_mood, next_mood)] += 1
        
        # Find significant transitions (occurred at least 2 times)
        patterns = []
        for (from_mood, to_mood), count in transitions.items():
            if count >= 2 and from_mood != to_mood:
                patterns.append({
                    "from": from_mood,
                    "to": to_mood,
                    "count": count,
                    "pattern_type": self._classify_transition(from_mood, to_mood)
                })
        
        # Sort by count
        patterns.sort(key=lambda x: x["count"], reverse=True)
        return patterns
    
    def _classify_transition(self, from_mood: str, to_mood: str) -> str:
        """Classify the type of mood transition"""
        negative_moods = ["frustrated", "sad", "stressed", "confused"]
        positive_moods = ["happy", "excited", "calm"]
        
        if from_mood in negative_moods and to_mood in positive_moods:
            return "recovery"
        elif from_mood in positive_moods and to_mood in negative_moods:
            return "decline"
        elif from_mood in negative_moods and to_mood in negative_moods:
            return "persistence"
        else:
            return "neutral"
    
    def _find_baseline(self, history: List[Dict]) -> Optional[str]:
        """
        Find the typical/baseline mood.
        
        Returns the most common mood over the time period.
        """
        if not history:
            return None
        
        moods = [record["mood"] for record in history]
        most_common = Counter(moods).most_common(1)[0]
        
        # Only return if it's significantly more common (>30%)
        if most_common[1] / len(moods) > 0.3:
            return most_common[0]
        
        return None
    
    def _calculate_distribution(self, history: List[Dict]) -> Dict[str, float]:
        """Calculate percentage distribution of moods"""
        if not history:
            return {}
        
        moods = [record["mood"] for record in history]
        total = len(moods)
        
        distribution = {}
        for mood, count in Counter(moods).items():
            distribution[mood] = round(count / total * 100, 1)
        
        return distribution
    
    def _find_cyclical_patterns(self, history: List[Dict]) -> List[Dict[str, Any]]:
        """Detect cyclical patterns using advanced algorithms"""
        return AdvancedMoodPatterns.find_cyclical_patterns(history)
    
    def _calculate_volatility(self, history: List[Dict]) -> float:
        """Calculate mood volatility"""
        return AdvancedMoodPatterns.calculate_volatility(history)
    
    def _calculate_mood_velocity(self, history: List[Dict]) -> Dict[str, float]:
        """Calculate mood velocity (duration of each mood)"""
        return AdvancedMoodPatterns.calculate_mood_velocity(history)
    
    def _find_trigger_patterns(self, history: List[Dict]) -> List[Dict[str, Any]]:
        """Find trigger patterns for mood changes"""
        return AdvancedMoodPatterns.find_trigger_patterns(history)
    
    async def predict_mood(
        self,
        user_id: str,
        for_time: Optional[datetime] = None
    ) -> Optional[Tuple[str, float]]:
        """
        Predict likely mood for a given time.
        
        Args:
            user_id: User identifier
            for_time: Time to predict for (default: now)
            
        Returns:
            Tuple of (predicted_mood, confidence) or None
        """
        if for_time is None:
            for_time = datetime.now()
        
        # Get patterns
        patterns = await self.detect_patterns(user_id, window_days=90)
        
        if patterns["data_points"] < 5:
            return None
        
        # Check temporal patterns
        weekday = for_time.strftime("%A")
        hour = f"{for_time.hour:02d}:00"
        
        temporal_patterns = patterns["temporal_patterns"]
        
        # Weekday pattern
        if weekday in temporal_patterns.get("by_weekday", {}):
            predicted_mood = temporal_patterns["by_weekday"][weekday]
            return (predicted_mood, 0.7)
        
        # Hour pattern
        if hour in temporal_patterns.get("by_hour", {}):
            predicted_mood = temporal_patterns["by_hour"][hour]
            return (predicted_mood, 0.6)
        
        # Fall back to baseline
        baseline = patterns["typical_trajectory"]
        if baseline:
            return (baseline, 0.5)
        
        return None
    
    async def get_mood_insights(self, user_id: str) -> str:
        """
        Generate human-readable insights about mood patterns.
        
        Args:
            user_id: User identifier
            
        Returns:
            Formatted string with insights
        """
        patterns = await self.detect_patterns(user_id, window_days=90)
        
        if patterns["data_points"] < 5:
            return "Insufficient mood data for insights (need at least 5 mood records)"
        
        insights = ["Mood Pattern Insights:"]
        insights.append("")
        
        # Distribution
        if patterns["mood_distribution"]:
            insights.append("Mood Distribution:")
            for mood, percentage in sorted(
                patterns["mood_distribution"].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                insights.append(f"  {mood}: {percentage}%")
            insights.append("")
        
        # Baseline
        if patterns["typical_trajectory"]:
            insights.append(f"Typical mood: {patterns['typical_trajectory']}")
            insights.append("")
        
        # Temporal patterns
        temporal = patterns["temporal_patterns"]
        if temporal.get("by_weekday"):
            insights.append("Day-of-week patterns:")
            for day, mood in temporal["by_weekday"].items():
                insights.append(f"  {day}: tends to be {mood}")
            insights.append("")
        
        if temporal.get("by_hour"):
            insights.append("Time-of-day patterns:")
            for hour, mood in sorted(temporal["by_hour"].items()):
                insights.append(f"  {hour}: tends to be {mood}")
            insights.append("")
        
        # Sequential patterns
        if patterns["sequential_patterns"]:
            insights.append("Common mood transitions:")
            for pattern in patterns["sequential_patterns"][:3]:  # Top 3
                insights.append(
                    f"  {pattern['from']} → {pattern['to']} "
                    f"({pattern['count']} times, {pattern['pattern_type']})"
                )
            insights.append("")
        
        insights.append(f"Based on {patterns['data_points']} mood records")
        
        return "\n".join(insights)
