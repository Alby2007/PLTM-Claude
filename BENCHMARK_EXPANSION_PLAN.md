# Benchmark Expansion Plan: 8 → 50 Test Cases

**Current Status:** 8/8 tests passing (100%)  
**Target:** 50 test cases with 85-90% accuracy  
**Timeline:** 10-15 hours

---

## Current Test Coverage (8 tests)

### Category Breakdown
1. **Opposite Predicates:** 2 tests (likes vs dislikes, enjoy vs dislike)
2. **Exclusive Predicates:** 2 tests (preference change, temporal supersession)
3. **Contextual:** 1 test (different contexts allow coexistence)
4. **Duplicate Detection:** 1 test (same statement twice)
5. **Refinement:** 1 test (music → jazz music)
6. **Correction Signals:** 1 test (Actually...)

### Gaps in Current Coverage
- No multi-hop conflicts (A → B → C chains)
- No temporal reasoning ("used to" vs present)
- No negation handling ("don't like" vs "like")
- No quantifiers ("always" vs "sometimes")
- No conditional statements ("if X then Y")
- No partial updates ("Python" → "Python 3.11")
- No multi-predicate conflicts
- No ambiguous cases
- No edge cases (empty strings, special characters)
- No performance stress tests

---

## Expanded Test Suite Design (50 tests)

### Category 1: Opposite Predicates (8 tests)
**Current:** 2 tests | **New:** 6 additional tests

1. ✅ Direct contradiction (likes vs dislikes) - EXISTING
2. ✅ Opposite sentiment (enjoy vs dislike) - EXISTING
3. **NEW:** loves vs hates
4. **NEW:** wants vs avoids
5. **NEW:** supports vs opposes
6. **NEW:** agrees vs disagrees
7. **NEW:** trusts vs distrusts
8. **NEW:** accepts vs rejects

**Expected Pass Rate:** 100% (all should work with current logic)

---

### Category 2: Exclusive Predicates (10 tests)
**Current:** 2 tests | **New:** 8 additional tests

1. ✅ Preference change (prefers async → sync) - EXISTING
2. ✅ Temporal supersession (works at Google → Anthropic) - EXISTING
3. **NEW:** Location change (lives in Seattle → San Francisco)
4. **NEW:** Identity change (is engineer → is manager)
5. **NEW:** Marital status (married to Alice → married to Bob)
6. **NEW:** Reports to (reports to John → reports to Sarah)
7. **NEW:** Multiple exclusive predicates (works at X, lives in Y simultaneously)
8. **NEW:** Rapid changes (3+ location changes in sequence)
9. **NEW:** Conflicting exclusive predicates (is X and is Y)
10. **NEW:** Exclusive with low similarity (works at NASA → works at SpaceX)

**Expected Pass Rate:** 90% (may need tuning for edge cases)

---

### Category 3: Contextual Coexistence (8 tests)
**Current:** 1 test | **New:** 7 additional tests

1. ✅ Different contexts (likes jazz when relaxing vs hates when working) - EXISTING
2. **NEW:** Temporal contexts (likes mornings vs dislikes evenings)
3. **NEW:** Situational contexts (confident at work vs shy at parties)
4. **NEW:** Conditional contexts (if tired then dislikes, if energized then likes)
5. **NEW:** Multiple contexts (likes A when X, dislikes A when Y, neutral when Z)
6. **NEW:** Overlapping contexts (likes when relaxing OR working)
7. **NEW:** Nested contexts (likes jazz when relaxing at home)
8. **NEW:** Context without explicit markers (implicit context extraction)

**Expected Pass Rate:** 75% (context extraction may need improvement)

---

### Category 4: Temporal Reasoning (6 tests)
**Current:** 0 tests | **New:** 6 tests

1. **NEW:** Past vs present ("used to like" vs "like")
2. **NEW:** Future intentions ("will start" vs "started")
3. **NEW:** Temporal progression (liked → loved → obsessed)
4. **NEW:** Temporal reversal (liked → neutral → disliked)
5. **NEW:** Explicit time markers ("in 2020" vs "in 2025")
6. **NEW:** Duration ("always liked" vs "recently started liking")

**Expected Pass Rate:** 60% (temporal reasoning not fully implemented)

---

### Category 5: Negation Handling (5 tests)
**Current:** 0 tests | **New:** 5 tests

1. **NEW:** Simple negation ("like" vs "don't like")
2. **NEW:** Double negation ("don't dislike" = "like"?)
3. **NEW:** Partial negation ("not always like")
4. **NEW:** Negation with context ("don't like when tired")
5. **NEW:** Negation ambiguity ("not bad" = "good"?)

**Expected Pass Rate:** 50% (negation parsing needs implementation)

---

### Category 6: Quantifiers & Modifiers (5 tests)
**Current:** 0 tests | **New:** 5 tests

1. **NEW:** Frequency ("always" vs "sometimes" vs "never")
2. **NEW:** Intensity ("love" vs "kinda like" vs "slightly prefer")
3. **NEW:** Certainty ("definitely" vs "maybe" vs "probably")
4. **NEW:** Scope ("all Python" vs "some Python features")
5. **NEW:** Degree ("very much" vs "a little bit")

**Expected Pass Rate:** 40% (quantifier extraction not implemented)

---

### Category 7: Refinement & Partial Updates (4 tests)
**Current:** 1 test | **New:** 3 additional tests

1. ✅ Simple refinement (music → jazz music) - EXISTING
2. **NEW:** Version update (Python → Python 3.11)
3. **NEW:** Specification (car → Tesla Model 3)
4. **NEW:** Clarification (programming → backend programming)

**Expected Pass Rate:** 100% (refinement logic already works)

---

### Category 8: Multi-Hop Conflicts (3 tests)
**Current:** 0 tests | **New:** 3 tests

1. **NEW:** A → B → C chain (likes X, neutral on X, dislikes X)
2. **NEW:** Circular conflicts (A supersedes B, B supersedes C, C supersedes A)
3. **NEW:** Transitive conflicts (if A conflicts with B, and B with C, does A conflict with C?)

**Expected Pass Rate:** 70% (may need conflict chain resolution)

---

### Category 9: Duplicate & Reinforcement (3 tests)
**Current:** 1 test | **New:** 2 additional tests

1. ✅ Exact duplicate (same statement twice) - EXISTING
2. **NEW:** Paraphrase duplicate ("love Python" vs "really like Python")
3. **NEW:** Reinforcement with increased confidence

**Expected Pass Rate:** 80% (paraphrase detection needs work)

---

### Category 10: Correction Signals (3 tests)
**Current:** 1 test | **New:** 2 additional tests

1. ✅ "Actually..." correction - EXISTING
2. **NEW:** "No, I meant..." correction
3. **NEW:** "To clarify..." correction

**Expected Pass Rate:** 100% (correction detection already works)

---

### Category 11: Edge Cases (5 tests)
**Current:** 0 tests | **New:** 5 tests

1. **NEW:** Empty object handling
2. **NEW:** Special characters in objects
3. **NEW:** Very long objects (>200 chars)
4. **NEW:** Unicode and emoji handling
5. **NEW:** Case sensitivity ("Python" vs "python")

**Expected Pass Rate:** 80% (may expose parsing bugs)

---

## Implementation Strategy

### Phase 1: Low-Hanging Fruit (2-3 hours)
Implement tests expected to pass with current logic:
- Opposite predicates (6 new tests)
- Refinement (3 new tests)
- Correction signals (2 new tests)
- Duplicate detection (2 new tests)

**Expected:** 13 new tests, ~95% pass rate

### Phase 2: Moderate Difficulty (3-4 hours)
Implement tests requiring minor enhancements:
- Exclusive predicates (8 new tests)
- Contextual coexistence (7 new tests)
- Edge cases (5 new tests)

**Expected:** 20 new tests, ~75% pass rate

### Phase 3: Advanced Features (5-8 hours)
Implement tests requiring new capabilities:
- Temporal reasoning (6 new tests)
- Negation handling (5 new tests)
- Quantifiers (5 new tests)
- Multi-hop conflicts (3 new tests)

**Expected:** 19 new tests, ~55% pass rate

---

## Expected Overall Results

### Conservative Estimate
- **Phase 1:** 13 tests × 95% = 12.4 passing
- **Phase 2:** 20 tests × 75% = 15 passing
- **Phase 3:** 19 tests × 55% = 10.5 passing
- **Existing:** 8 tests × 100% = 8 passing

**Total:** 45.9 / 58 tests = **79% accuracy** ✅ (within 85-90% range if we optimize)

### Optimistic Estimate (with fixes)
- **Phase 1:** 13 tests × 100% = 13 passing
- **Phase 2:** 20 tests × 85% = 17 passing
- **Phase 3:** 19 tests × 65% = 12.4 passing
- **Existing:** 8 tests × 100% = 8 passing

**Total:** 50.4 / 58 tests = **87% accuracy** ✅

---

## Risk Areas

### High Risk (likely to fail initially)
1. **Temporal reasoning** - No temporal logic implemented
2. **Negation parsing** - Not in current extraction
3. **Quantifiers** - Not extracted or compared
4. **Multi-hop conflicts** - No chain resolution logic

### Medium Risk
1. **Context extraction** - Limited to simple patterns
2. **Paraphrase detection** - No semantic similarity
3. **Exclusive predicates edge cases** - May need tuning

### Low Risk
1. **Opposite predicates** - Already working perfectly
2. **Refinement** - Already working
3. **Correction signals** - Already working

---

## Success Criteria

### Must-Have (for 85% target)
- ✅ All Phase 1 tests passing (13 tests)
- ✅ 80%+ of Phase 2 tests passing (16+ tests)
- ✅ 50%+ of Phase 3 tests passing (10+ tests)
- ✅ All existing tests still passing (8 tests)

**Total:** 47+ / 58 tests = **81%+**

### Nice-to-Have (for 90% target)
- All Phase 1 tests passing (13 tests)
- 90%+ of Phase 2 tests passing (18+ tests)
- 65%+ of Phase 3 tests passing (12+ tests)
- All existing tests still passing (8 tests)

**Total:** 51+ / 58 tests = **88%+**

---

## Implementation Timeline

### Day 1 (4 hours)
- Design all 50 test cases
- Implement Phase 1 tests (low-hanging fruit)
- Run initial benchmark
- **Deliverable:** 21 total tests, ~90% accuracy

### Day 2 (4 hours)
- Implement Phase 2 tests (moderate difficulty)
- Fix failing Phase 2 tests
- Improve context extraction
- **Deliverable:** 41 total tests, ~80% accuracy

### Day 3 (4 hours)
- Implement Phase 3 tests (advanced features)
- Identify gaps in temporal/negation/quantifier handling
- Implement critical fixes
- **Deliverable:** 58 total tests, ~75% accuracy

### Day 4 (3 hours)
- Optimize failing cases
- Tune thresholds and rules
- Final benchmark run
- **Deliverable:** 58 total tests, **85-90% accuracy** ✅

---

## Next Steps

1. **Create test case definitions** (detailed specifications)
2. **Implement test infrastructure** (extend current benchmark suite)
3. **Run Phase 1** (low-hanging fruit)
4. **Analyze failures** and implement fixes
5. **Iterate** through Phases 2 and 3
6. **Document results** and learnings

---

**Status:** Ready to begin implementation  
**Estimated Time:** 10-15 hours  
**Expected Outcome:** 85-90% accuracy on 50 test cases
