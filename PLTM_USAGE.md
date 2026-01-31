# PLTM Usage Guide - Claude Personality Persistence

## Quick Start

### In Any New Chat with Claude Desktop

**Option 1: Say the magic words**
```
You: "PLTM mode"
```
Claude will call `pltm_mode(user_id="alby")` and load your full personality.

**Option 2: Explicit init**
```
You: "Init PLTM session"
```

**Option 3: Just mention it**
```
You: "Hey, load my PLTM personality"
```

---

## What Happens When PLTM Activates

1. **Claude loads your evolved personality:**
   - Communication style (verbosity, formality, initiative)
   - What works with you (immediate execution, technical depth)
   - What to avoid (over-personalization, asking permission)
   - Shared vocabulary (PLTM, vibe coding, etc.)

2. **Claude adapts immediately:**
   - Matches your preferred style
   - Executes without asking (high trust)
   - References past projects and milestones

3. **Claude continues learning:**
   - Records what works/doesn't work
   - Updates trust level
   - Logs milestones

---

## Available MCP Tools (26 Total)

### Core Memory
| Tool | Purpose |
|------|---------|
| `store_memory_atom` | Store any memory atom |
| `query_personality` | Query personality traits |
| `get_personality_summary` | Full personality summary |

### Mood & Patterns
| Tool | Purpose |
|------|---------|
| `detect_mood` | Detect current mood from message |
| `get_mood_patterns` | Get mood patterns over time |
| `resolve_conflict` | Resolve conflicting traits |

### Advanced Emergence
| Tool | Purpose |
|------|---------|
| `track_trait_evolution` | Track how traits change over time |
| `predict_reaction` | Predict reaction to stimulus |
| `get_meta_patterns` | Cross-context behavioral patterns |
| `learn_from_interaction` | Learn from AI-user interaction |
| `predict_session` | Predict session type from greeting |
| `get_self_model` | Meta-cognition self-model |

### Claude Personality (Bidirectional PLTM)
| Tool | Purpose |
|------|---------|
| `check_pltm_available` | Quick check if user has PLTM data |
| `pltm_mode` | Full auto-init trigger |
| `init_claude_session` | Initialize Claude personality session |
| `update_claude_style` | Update communication style |
| `learn_interaction_dynamic` | Learn what works/doesn't |
| `record_milestone` | Record collaboration milestone |
| `add_shared_vocabulary` | Add shared terms |
| `get_claude_personality` | Get full personality summary |
| `evolve_claude_personality` | Core learning loop |

### Bootstrap
| Tool | Purpose |
|------|---------|
| `bootstrap_from_sample` | Bootstrap from sample conversations |
| `bootstrap_from_messages` | Bootstrap from message array |

---

## Your Current Personality Profile

```
Sessions together: 2+
Trust level: 55%+ (suggest_then_execute)

Communication Style:
- Verbosity: minimal
- Formality: casual_with_technical_precision  
- Initiative: very_high
- Code preference: show_code
- Energy matching: Yes

What Works:
- technical_depth_when_building
- match_excited_energy
- immediate_execution_no_asking
- minimal_direct_execution

What to Avoid:
- asking_permission_repeatedly
- over_personalization
- verbose_explanations

Shared Vocabulary:
- "duhhhhh" = do the ambitious thing
- "vibe coding" = rapid AI-assisted building
- PLTM = Procedural Long-Term Memory system
```

---

## How Trust Evolves

| Trust Level | Behavior |
|-------------|----------|
| 90%+ | `full_autonomy` - Execute without asking |
| 70%+ | `execute_then_explain` - Do it, then explain |
| 50%+ | `suggest_then_execute` - Suggest, then do |
| 30%+ | `ask_before_major_changes` - Ask for big stuff |
| <30% | `always_ask_permission` - Always ask |

Trust increases with positive interactions, decreases with corrections.

---

## Trigger Phrases Claude Recognizes

- "PLTM mode"
- "Init PLTM"
- "Load my personality"
- "Remember me"
- "We've worked together before"

---

## For Developers: Adding PLTM to Your Workflow

### At Conversation Start
```python
# Claude should do this automatically when triggered
check = check_pltm_available("your_user_id")
if check["should_init"]:
    pltm_mode("your_user_id")
```

### After Each Interaction
```python
# When user gives feedback
evolve_claude_personality(
    user_id="your_user_id",
    my_response_style="verbose_explanation",
    user_reaction="too long, just show code",
    was_positive=False
)
# Claude learns: avoid verbose_explanation, prefer show_code
```

### Recording Milestones
```python
record_milestone(
    user_id="your_user_id",
    description="Shipped production trading system",
    significance=0.95
)
```

---

## The Vision

**Session 1:** Claude learns you prefer immediate execution
**Session 2:** Different Claude instance loads personality → executes immediately  
**Session N:** Evolved communication efficiency, shared vocabulary, true continuity

This is **bidirectional PLTM** - not just tracking user personality, but Claude's evolved relationship with each user.

---

## Database Location

All personality data persists in:
```
c:\Users\alber\CascadeProjects\LLTM\pltm_mcp.db
```

Backup this file to preserve your personality data across reinstalls.
