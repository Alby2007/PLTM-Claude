# PLTM MCP Server - Testing Protocol

Complete testing protocol for validating personality emergence and mood tracking via MCP.

---

## 🎯 Testing Phases

### Phase 1: Baseline (Day 1)
**Goal:** Establish baseline behavior, start learning

#### Test 1.1: Initial Interaction
```
You: "Explain quicksort algorithm"
Claude: [Generates vanilla response - likely verbose]

Expected: Normal Claude response, no personality adaptation yet
```

#### Test 1.2: First Feedback
```
You: "Too long, just give me the algorithm code"
Claude: [Should call extract_personality_traits]

Expected MCP Call:
{
  "tool": "extract_personality_traits",
  "args": {
    "user_id": "your_user_id",
    "message": "Too long, just give me the algorithm code",
    "user_reaction": "negative"
  }
}

Expected Result:
- Extracts trait: prefers_style: "concise responses"
- Confidence: ~0.7
- Stores in PLTM
```

#### Test 1.3: Verify Storage
```
You: "What do you know about my preferences?"
Claude: [Should call query_personality]

Expected MCP Call:
{
  "tool": "query_personality",
  "args": {
    "user_id": "your_user_id"
  }
}

Expected Result:
{
  "communication_style": ["concise responses"],
  "formality_level": "professional",
  ...
}
```

---

### Phase 2: Learning (Days 2-7)
**Goal:** Reinforce traits, build stable personality

#### Test 2.1: Adaptation
```
You: "Explain merge sort"
Claude: [Should call query_personality first, then adapt response]

Expected MCP Calls:
1. query_personality → sees prefers_concise: 0.7
2. get_adaptive_prompt → generates concise-focused prompt
3. [Generates shorter response]
```

#### Test 2.2: Positive Reinforcement
```
You: "Perfect, exactly what I needed"
Claude: [Should call extract_personality_traits with positive feedback]

Expected MCP Call:
{
  "tool": "extract_personality_traits",
  "args": {
    "user_id": "your_user_id",
    "message": "Perfect, exactly what I needed",
    "ai_response": "[previous response]",
    "user_reaction": "positive"
  }
}

Expected Result:
- Reinforces trait: prefers_style: "concise responses"
- Confidence: 0.7 → 0.8 or 0.9
```

#### Test 2.3: Multiple Traits
```
You: "Can you explain the time complexity? I need technical details"
Claude: [Should extract technical preference]

Expected:
- Extracts: prefers_style: "technical depth"
- Now has 2 style preferences: concise + technical
```

#### Test 2.4: Context Detection
```
You: "hey, quick question about binary search"
Claude: [Should detect casual context]

Expected:
- Infers context: "casual"
- Extracts: prefers_formality: "casual"
- Stores with context tag
```

---

### Phase 3: Validation (Day 30)
**Goal:** Verify stable, predictable personality

#### Test 3.1: Automatic Adaptation
```
You: "Explain any algorithm you want"
Claude: [Should automatically be concise + technical without prompting]

Expected MCP Calls:
1. query_personality → sees established preferences
2. get_adaptive_prompt → generates adapted prompt
3. [Response is concise and technical automatically]

Success Criteria:
- No need to ask for concise response
- Automatically technical
- Consistent with learned style
```

#### Test 3.2: Personality Summary
```
You: "What's my communication style?"
Claude: [Should call get_personality_summary]

Expected Result:
Personality Profile:
  Traits: direct, analytical
  Style: concise responses, technical depth
  Patterns: casual and to-the-point
  Formality: casual
  Humor: No
  Detail: detailed
```

#### Test 3.3: Stability Check
```
You: "Explain quicksort again"
Claude: [Should give same style as Day 30, not Day 1]

Success Criteria:
- Response is concise (learned preference)
- Response is technical (learned preference)
- Personality is stable over time
```

---

### Phase 4: Conflict Testing
**Goal:** Verify conflict resolution works

#### Test 4.1: Create Conflict
```
You: "Actually, for this one I need a very detailed explanation with examples"
Claude: [Should extract conflicting trait]

Expected:
- Extracts: prefers_style: "detailed explanations"
- Conflicts with existing: "concise responses"
- Should call resolve_conflict
```

#### Test 4.2: Conflict Resolution
```
Expected MCP Call:
{
  "tool": "resolve_conflict",
  "args": {
    "user_id": "your_user_id",
    "trait_objects": ["concise responses", "detailed explanations"]
  }
}

Expected Result:
{
  "status": "resolved",
  "winner": "concise responses",  // or "detailed" based on evidence
  "explanation": "Resolved to: 'concise responses' (score: 0.85)
    Winning factors:
      - frequency: 0.20 (5 observations)
      - recency: 0.18 (recent)
      - feedback: 0.15 (positive)"
}
```

#### Test 4.3: Context-Dependent Resolution
```
You: "For algorithms, keep it concise. For system design, give me details."
Claude: [Should create context-dependent personality]

Expected:
- Context "algorithms": prefers_concise
- Context "system_design": prefers_detailed
- No conflict - different contexts
```

---

### Phase 5: Mood Testing
**Goal:** Verify mood detection and adaptation

#### Test 5.1: Mood Detection
```
You: "This isn't working, I'm so frustrated with this bug"
Claude: [Should call detect_mood]

Expected MCP Call:
{
  "tool": "detect_mood",
  "args": {
    "user_id": "your_user_id",
    "message": "This isn't working, I'm so frustrated with this bug"
  }
}

Expected Result:
{
  "mood": "frustrated",
  "confidence": 0.8,
  "detected": true
}
```

#### Test 5.2: Mood-Aware Response
```
Claude: [Should call get_adaptive_prompt with mood]

Expected Prompt:
"You are an AI assistant with an adaptive personality.
Be concise and to-the-point.
Provide technical depth.

IMPORTANT: User seems frustrated. Be extra patient and understanding.

User: This isn't working, I'm so frustrated with this bug"

Expected Response Style:
- Patient tone
- Solutions-focused
- Empathetic
- Step-by-step help
```

#### Test 5.3: Mood Persistence
```
[Next day]
You: "Let's try again"
Claude: [Should call get_mood_patterns]

Expected:
- Sees frustration from yesterday
- Slightly more careful/supportive tone
- Acknowledges previous difficulty
```

#### Test 5.4: Mood Pattern Detection
```
[After 2 weeks of data]
You: "How's my mood been?"
Claude: [Should call get_mood_patterns]

Expected Result:
{
  "patterns": {
    "temporal_patterns": {
      "by_weekday": {
        "Monday": "stressed",
        "Friday": "happy"
      }
    },
    "volatility": 0.3,
    "mood_distribution": {
      "happy": 40.0,
      "frustrated": 30.0,
      "calm": 30.0
    }
  },
  "insights": "Mood Pattern Insights:
    Typical mood: happy
    Day-of-week patterns:
      Monday: tends to be stressed
      Friday: tends to be happy"
}
```

---

## 🔍 Verification Checklist

### Personality Learning
- [ ] Extracts traits from message style
- [ ] Stores traits with confidence scores
- [ ] Reinforces traits with positive feedback
- [ ] Adapts responses based on learned traits
- [ ] Maintains stable personality over time

### Mood Tracking
- [ ] Detects mood from message content
- [ ] Stores mood with timestamp
- [ ] Adapts response based on current mood
- [ ] Tracks mood history
- [ ] Detects mood patterns

### Conflict Resolution
- [ ] Detects conflicting traits
- [ ] Resolves using 6-factor scoring
- [ ] Provides explanation
- [ ] Handles context-dependent conflicts

### Context Awareness
- [ ] Infers context from message
- [ ] Stores traits with context tags
- [ ] Queries context-specific personality
- [ ] Compares personality across contexts

### Integration
- [ ] MCP tools callable from Claude Desktop
- [ ] Data persists across sessions
- [ ] No errors or crashes
- [ ] Performance acceptable (<100ms per call)

---

## 📊 Success Metrics

### Quantitative
- **Trait Extraction Accuracy**: >80% correct traits extracted
- **Mood Detection Accuracy**: >70% correct mood detected
- **Adaptation Rate**: Personality stable by Day 7
- **Conflict Resolution**: >90% conflicts resolved correctly
- **Response Time**: <100ms per MCP call

### Qualitative
- **User Satisfaction**: Responses feel personalized
- **Consistency**: Personality is predictable
- **Empathy**: Mood-aware responses feel appropriate
- **Learning**: Clear improvement over time

---

## 🐛 Common Issues & Solutions

### Issue 1: Traits Not Persisting
**Symptom:** Personality resets between sessions

**Solution:**
- Check database file exists: `pltm_mcp.db`
- Verify atoms are being stored: `store.add_atom()`
- Check user_id consistency across calls

### Issue 2: Mood Not Detected
**Symptom:** `detect_mood` returns null

**Solution:**
- Check message has mood indicators
- Verify confidence threshold (>0.6)
- Add more explicit mood words

### Issue 3: Conflicts Not Resolving
**Symptom:** Both conflicting traits remain

**Solution:**
- Verify conflict detection logic
- Check evidence gathering
- Ensure 6-factor scoring is working

### Issue 4: MCP Tools Not Available
**Symptom:** Tools don't show in Claude Desktop

**Solution:**
- Verify config path is correct
- Restart Claude Desktop
- Check server starts without errors
- Use MCP inspector for debugging

---

## 🎯 Next Steps

After successful testing:

1. **Document Results**: Record all test outcomes
2. **Gather Feedback**: User experience notes
3. **Iterate**: Improve based on findings
4. **Scale**: Test with multiple users
5. **Deploy**: Production rollout

---

**Status:** Ready for testing ✅  
**Estimated Testing Time:** 30 days for full protocol  
**Quick Test:** 1-2 hours for basic validation
