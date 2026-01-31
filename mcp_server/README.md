# PLTM MCP Server

Model Context Protocol server for Procedural Long-Term Memory system.

---

## 🎯 What This Is

An MCP server that exposes PLTM's personality and mood tracking capabilities to Claude Desktop (or any MCP client).

**Capabilities:**
- Store memory atoms (facts, traits, moods)
- Query personality profiles
- Detect and track moods
- Resolve conflicting traits
- Extract personality from interactions
- Generate adaptive prompts

---

## 🚀 Quick Start

### 1. Install MCP SDK

```bash
pip install mcp
```

### 2. Configure Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on Mac):

```json
{
  "mcpServers": {
    "pltm-memory": {
      "command": "python",
      "args": [
        "C:/Users/alber/CascadeProjects/LLTM/mcp_server/pltm_server.py"
      ]
    }
  }
}
```

### 3. Restart Claude Desktop

The PLTM tools will now be available in Claude Desktop!

---

## 🛠️ Available Tools

### 1. `store_memory_atom`
Store a memory atom in PLTM graph.

**Parameters:**
- `user_id` (required): User identifier
- `atom_type` (required): Type of atom (fact, personality_trait, communication_style, interaction_pattern, mood, preference)
- `subject` (required): Subject of the atom
- `predicate` (required): Relationship/predicate
- `object` (required): Object/value
- `confidence` (optional): Confidence score (0.0-1.0)
- `context` (optional): Context tags

**Example:**
```json
{
  "user_id": "alice",
  "atom_type": "communication_style",
  "subject": "alice",
  "predicate": "prefers_style",
  "object": "concise responses",
  "confidence": 0.8,
  "context": ["technical"]
}
```

### 2. `query_personality`
Get synthesized personality profile.

**Parameters:**
- `user_id` (required): User identifier
- `context` (optional): Context filter

**Returns:**
```json
{
  "core_traits": ["direct", "analytical"],
  "communication_style": ["concise", "technical depth"],
  "formality_level": "professional",
  "humor_preference": false,
  "detail_level": "detailed"
}
```

### 3. `detect_mood`
Detect mood from user message.

**Parameters:**
- `user_id` (required): User identifier
- `message` (required): User's message

**Returns:**
```json
{
  "mood": "frustrated",
  "confidence": 0.8,
  "detected": true
}
```

### 4. `get_mood_patterns`
Get mood patterns and insights.

**Parameters:**
- `user_id` (required): User identifier
- `window_days` (optional): Number of days to analyze (default: 90)

**Returns:**
```json
{
  "patterns": {
    "temporal_patterns": {...},
    "cyclical_patterns": [...],
    "volatility": 0.3,
    "mood_distribution": {...}
  },
  "insights": "Mood Pattern Insights:\n..."
}
```

### 5. `resolve_conflict`
Resolve conflicting personality traits.

**Parameters:**
- `user_id` (required): User identifier
- `trait_objects` (required): List of conflicting trait objects

**Returns:**
```json
{
  "status": "resolved",
  "winner": "concise responses",
  "explanation": "Resolved to: 'concise responses' (score: 0.85)\n..."
}
```

### 6. `extract_personality_traits`
Extract personality traits from interaction.

**Parameters:**
- `user_id` (required): User identifier
- `message` (required): User's message
- `ai_response` (optional): AI's response
- `user_reaction` (optional): User's reaction

**Returns:**
```json
{
  "extracted_count": 3,
  "traits": [
    {
      "type": "communication_style",
      "predicate": "prefers_style",
      "object": "concise",
      "confidence": 0.7
    }
  ]
}
```

### 7. `get_adaptive_prompt`
Get adaptive system prompt based on personality and mood.

**Parameters:**
- `user_id` (required): User identifier
- `message` (required): Current user message
- `context` (optional): Interaction context

**Returns:**
```
You are an AI assistant with an adaptive personality.
Use casual, friendly language.
Be concise and to-the-point.

User: [message]
```

### 8. `get_personality_summary`
Get human-readable personality summary.

**Parameters:**
- `user_id` (required): User identifier

**Returns:**
```
Personality Profile:
  Traits: direct, analytical
  Style: concise, technical depth
  Formality: professional
  Humor: No
  Detail: detailed
```

---

## 🧪 Testing Protocol

### Phase 1: Baseline (Day 1)

**User:** "Explain quicksort"  
**Claude:** [vanilla response]  
**User:** "Too long, just give me the algorithm"  
**Claude:** [calls `extract_personality_traits`]
- Extracts: `prefers_style: concise`
- Stores with confidence 0.7

### Phase 2: Learning (Days 2-7)

**User:** "Explain merge sort"  
**Claude:** [calls `query_personality`]
- Sees: `prefers_concise: 0.8`
- Generates shorter response  
**User:** "Perfect"  
**Claude:** [calls `extract_personality_traits` with positive feedback]
- Reinforces trait, confidence → 0.9

### Phase 3: Validation (Day 30)

**User:** "Explain any algorithm"  
**Claude:** [calls `get_adaptive_prompt`]
- Automatically generates concise response
- Personality is stable and predictable

### Conflict Testing

**User:** "Actually, give me more detail this time"  
**Claude:** [calls `extract_personality_traits`]
- Creates conflicting atom: `prefers_detail`
- [calls `resolve_conflict`]
- Jury examines context
- Creates context-dependent personality

### Mood Testing

**User:** "This isn't working, I'm so frustrated"  
**Claude:** [calls `detect_mood`]
- Detects: `frustrated` (confidence 0.8)
- [calls `get_adaptive_prompt`]
- Adapts: more patient, solutions-focused
- Stores mood atom with timestamp

**User:** [next day] "Let's try again"  
**Claude:** [calls `get_mood_patterns`]
- Sees frustration yesterday
- Slightly more careful/supportive tone

---

## 📊 Data Storage

**Database:** `pltm_mcp.db` (SQLite)

**Location:** Same directory as server script

**Schema:** Standard PLTM dual-graph architecture
- Substantiated graph (verified facts)
- Historical graph (superseded facts)

---

## 🔧 Development

### Running Locally

```bash
# From LLTM directory
python mcp_server/pltm_server.py
```

### Testing Tools

```bash
# Use MCP inspector
npx @modelcontextprotocol/inspector python mcp_server/pltm_server.py
```

### Debugging

Enable debug logging in `pltm_server.py`:

```python
logger.add("pltm_mcp_debug.log", level="DEBUG")
```

---

## 🎯 Use Cases

### 1. Personalized Coding Assistant

Claude learns your coding style:
- Concise vs verbose explanations
- Functional vs OOP preferences
- Comment style preferences
- Error handling approaches

### 2. Adaptive Tutor

Claude adapts to your learning style:
- Detail level preference
- Example-based vs theory-based
- Pace of learning
- Areas of struggle

### 3. Mood-Aware Support

Claude responds to your emotional state:
- Patient when frustrated
- Encouraging when struggling
- Celebratory when succeeding
- Calm when stressed

### 4. Long-Term Memory

Claude remembers across sessions:
- Your preferences
- Your projects
- Your communication style
- Your mood patterns

---

## 🚀 Advanced Features

### Context-Aware Personality

```python
# Query personality for specific context
{
  "user_id": "alice",
  "context": "technical"
}

# Returns different personality than:
{
  "user_id": "alice",
  "context": "casual"
}
```

### Mood Pattern Analysis

```python
# Get comprehensive mood insights
{
  "user_id": "alice",
  "window_days": 90
}

# Returns:
# - Temporal patterns (stressed on Mondays)
# - Cyclical patterns (weekly cycles)
# - Volatility (mood stability)
# - Predictions (likely mood for given time)
```

### Conflict Resolution

```python
# Resolve conflicting traits
{
  "user_id": "alice",
  "trait_objects": ["concise", "detailed"]
}

# Returns:
# - Winning trait based on 6-factor scoring
# - Detailed explanation
# - Confidence in resolution
```

---

## 📚 Documentation

- **Main README**: `../README.md`
- **Experiment 8 Details**: `../EXPERIMENT_8_COMPLETE.md`
- **API Reference**: See tool schemas above
- **Testing Protocol**: See testing section above

---

## 🎉 Status

**Implementation:** Complete ✅  
**Testing:** Ready for integration ✅  
**Documentation:** Complete ✅

**Ready to use with Claude Desktop!**
