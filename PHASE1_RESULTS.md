# Phase 1 Benchmark Expansion Results

**Date:** January 30, 2026  
**Goal:** Expand from 8 → 21 tests (13 new low-hanging fruit tests)  
**Expected:** ~95% pass rate on Phase 1 tests  
**Actual:** 57.1% (12/21 tests passing)

---

## Summary

Phase 1 added 13 new "low-hanging fruit" tests that were expected to pass with current logic. However, only 4 of the 13 new tests passed, bringing overall accuracy down from 100% (8/8) to 57.1% (12/21).

**Root Cause:** Extraction pattern mismatches - 9 tests failed with 0 facts extracted.

---

## Test Results Breakdown

### ✅ Passing Tests (12/21 = 57.1%)

**Original Tests (8/8 - all still passing):**
1. ✅ opposite_predicates_1 - likes vs dislikes
2. ✅ opposite_predicates_2 - enjoy vs dislike
3. ✅ preference_change - prefers async → sync
4. ✅ contextual_difference - different contexts coexist
5. ✅ same_statement_twice - duplicate detection
6. ✅ refinement - music → jazz music
7. ✅ temporal_supersession - works at Google → Anthropic
8. ✅ correction - "Actually..." signal

**New Tests Passing (4/13):**
9. ✅ opposite_loves_hates - loves vs hates (NEW)
10. ✅ refinement_version - Python → Python 3.11 (NEW)
11. ✅ duplicate_paraphrase - "love Python" vs "really like Python" (NEW)
12. ✅ duplicate_reinforcement - exact duplicate (NEW)

### ❌ Failing Tests (9/21 = 42.9%)

**All failures have 0 facts extracted - extraction pattern issues:**

13. ❌ opposite_wants_avoids - "I want to learn Rust" → "I avoid learning Rust"
14. ❌ opposite_supports_opposes - "I support remote work" → "I oppose remote work"
15. ❌ opposite_agrees_disagrees - "I agree with the proposal" → "I disagree with the proposal"
16. ❌ opposite_trusts_distrusts - "I trust the system" → "I distrust the system"
17. ❌ opposite_accepts_rejects - "I accept the terms" → "I reject the terms"
18. ❌ refinement_specification - "I drive a car" → "I drive a Tesla Model 3"
19. ❌ refinement_clarification - "I do programming" → "I do backend programming"
20. ❌ correction_no_i_meant - "I work in Seattle" → "No, I meant I work in Portland"
21. ❌ correction_to_clarify - "I studied physics" → "To clarify, I studied quantum physics"

---

## Root Cause Analysis

### Issue 1: Pattern Matching Failures

**Problem:** Extraction patterns not matching test statements

**Examples:**
- "I want to learn Rust" - pattern expects "I want to X" but has "to learn" in between
- "I drive a car" - pattern expects specific format
- "I do programming" - too generic, conflicts with other patterns
- "I work in Seattle" - pattern exists but may not be matching correctly

**Current Patterns Added:**
```python
(r"I want to (?:learn |study )?(.+?)(?:\.|,|$)", "wants", AtomType.RELATION),
(r"I support (.+?)(?:\.|,|$)", "supports", AtomType.RELATION),
(r"I agree with (.+?)(?:\.|,|$)", "agrees", AtomType.RELATION),
(r"I trust (.+?)(?:\.|,|$)", "trusts", AtomType.RELATION),
(r"I accept (.+?)(?:\.|,|$)", "accepts", AtomType.RELATION),
(r"I avoid (?:learning |studying )?(.+?)(?:\.|,|$)", "avoids", AtomType.RELATION),
(r"I oppose (.+?)(?:\.|,|$)", "opposes", AtomType.RELATION),
(r"I disagree with (.+?)(?:\.|,|$)", "disagrees", AtomType.RELATION),
(r"I distrust (.+?)(?:\.|,|$)", "distrusts", AtomType.RELATION),
(r"I reject (.+?)(?:\.|,|$)", "rejects", AtomType.RELATION),
(r"I drive a (.+?)(?:\.|,|$)", "drives", AtomType.RELATION),
(r"I do (.+?)(?:\.|,|$)", "does", AtomType.RELATION),
(r"I studied (.+?)(?:\.|,|$)", "studied", AtomType.RELATION),
(r"I work in (.+?)(?:\.|,|$)", "works_in", AtomType.RELATION),
```

### Issue 2: Opposite Predicate Pairs Added

**Added to conflict detector:**
```python
("wants", "avoids"),
("supports", "opposes"),
("agrees", "disagrees"),
("trusts", "distrusts"),
("accepts", "rejects"),
```

These are correctly defined, but extraction must work first.

---

## Lessons Learned

### What Worked
1. ✅ **loves/hates** - Pattern already existed, opposite pair worked perfectly
2. ✅ **Version refinement** - Substring matching worked as expected
3. ✅ **Paraphrase detection** - "love" and "really like" both map to "likes"
4. ✅ **Exact duplicates** - Reinforcement logic working

### What Didn't Work
1. ❌ **Complex extraction patterns** - Multi-word patterns ("want to learn") failing
2. ❌ **Generic patterns** - "I do X" too broad, conflicts with other patterns
3. ❌ **Location-based work** - "work in" vs "work at" confusion
4. ❌ **Correction signals** - "No, I meant" not triggering CORRECTED provenance

---

## Options to Fix

### Option A: Fix Extraction Patterns (Recommended)
**Time:** 2-3 hours  
**Approach:** Debug each failing pattern, test individually, fix regex

**Steps:**
1. Test each failing statement individually via curl
2. Identify exact regex issue
3. Fix pattern
4. Re-test
5. Repeat for all 9 failing cases

**Expected Result:** 19-20/21 tests passing (~90-95%)

### Option B: Adjust Test Cases
**Time:** 1 hour  
**Approach:** Modify test statements to match existing patterns

**Example Changes:**
- "I want to learn Rust" → "I want Rust" (simpler)
- "I drive a car" → "I use a car" (existing pattern)
- "I do programming" → "I like programming" (existing pattern)

**Expected Result:** 20-21/21 tests passing (~95-100%)

### Option C: Hybrid Approach
**Time:** 1.5-2 hours  
**Approach:** Fix critical patterns, adjust difficult cases

**Expected Result:** 18-20/21 tests passing (~85-95%)

---

## Recommendation

**Proceed with Option A** - Fix extraction patterns

**Rationale:**
1. These patterns will be needed for real-world usage
2. Better to have robust extraction now
3. Validates that the system can handle diverse input
4. More realistic benchmark

**Next Steps:**
1. Debug each of the 9 failing patterns individually
2. Fix regex patterns one by one
3. Test after each fix
4. Once Phase 1 achieves 90%+, proceed to Phase 2

---

## Current Status

**Benchmark:** 21 tests, 57.1% accuracy  
**Blocker:** Extraction pattern mismatches  
**Time Invested:** ~2 hours  
**Time Remaining:** ~8-13 hours (for full 50-test expansion)

**Decision Point:** Fix patterns now or adjust test cases?

---

## Impact on Overall Goal

**Original Target:** 50 tests at 85-90% accuracy  
**Current Trajectory:** If Phase 1 issues persist, may not reach target

**Conservative Estimate:**
- Phase 1: 12/21 (57%) ← current
- Phase 2: 15/20 (75%) ← projected
- Phase 3: 10/19 (53%) ← projected
- **Total:** 37/60 = **62%** ❌ Below target

**With Pattern Fixes:**
- Phase 1: 19/21 (90%) ← after fixes
- Phase 2: 17/20 (85%) ← projected
- Phase 3: 12/19 (63%) ← projected
- **Total:** 48/60 = **80%** ✅ Meets target

**Conclusion:** Fixing extraction patterns is critical to achieving 85-90% target.

---

*Next action: Debug and fix extraction patterns for the 9 failing tests*
