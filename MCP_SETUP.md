# PLTM MCP Server - Setup Guide

## What This Is

**PLTM (Procedural Long-Term Memory)** is an experimental AGI system testing whether universal principles from physics (criticality, self-organization, emergence) can bootstrap artificial general intelligence.

This MCP server gives Claude Desktop access to 78 tools for:
- Memory operations (store, retrieve, update)
- Diversity-aware retrieval (MMR, entropy injection)
- Meta-cognition (self-improvement, criticality monitoring)
- Knowledge ingestion (ArXiv papers with real provenance)
- True metrics (action accounting, AAE efficiency)

**Current experiment status**: Successfully unlocked entropy bottleneck (+56%), measuring true computational efficiency, testing if system can self-organize toward criticality.

---

## Quick Setup (5 minutes)

### Prerequisites
- **Claude Desktop** installed
- **Python 3.11+** with pip
- Git

### 1. Clone Repository
```bash
git clone https://github.com/Alby2007/LLTM.git
cd LLTM
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Claude Desktop

**Windows**: Edit `%APPDATA%\Claude\claude_desktop_config.json`

**macOS**: Edit `~/Library/Application Support/Claude/claude_desktop_config.json`

**Linux**: Edit `~/.config/Claude/claude_desktop_config.json`

Add this configuration:
```json
{
  "mcpServers": {
    "pltm-memory": {
      "command": "python",
      "args": [
        "C:/absolute/path/to/LLTM/mcp_server/pltm_server.py"
      ]
    }
  }
}
```

**Important**: Replace `C:/absolute/path/to/LLTM` with your actual path. Use forward slashes even on Windows.

### 4. Restart Claude Desktop

The MCP server will auto-start when Claude Desktop launches.

### 5. Verify Installation

In Claude Desktop, try:
```
Use the entropy_stats tool to check system state
```

If you see entropy/integration metrics, it's working!

---

## Available Tools (78 total)

### Memory Operations
- `store_memory_atom` - Store facts as triples
- `retrieve_memories` - Get memories by query
- `update_memory` - Modify existing atoms
- `delete_memory` - Remove atoms

### Diversity Retrieval
- `mmr_retrieve` - Maximal Marginal Relevance (diversity-aware)
- `attention_retrieve` - Attention-weighted retrieval
- `attention_multihead` - Multi-head attention

### Entropy Management
- `inject_entropy_antipodal` - Activate distant concepts
- `inject_entropy_random` - Sample from diverse domains
- `inject_entropy_temporal` - Mix old + recent memories
- `entropy_stats` - Diagnose diversity state

### Meta-Cognition
- `self_improve_cycle` - Generate and apply hypotheses
- `criticality_state` - Check if at edge of chaos
- `criticality_recommend` - Get adjustment suggestions

### Action Accounting (True Metrics)
- `record_action` - Track tokens×latency cost
- `get_aae` - Get Average Action Efficiency
- `start_action_cycle` / `end_action_cycle` - Group measurements

### Knowledge Ingestion
- `ingest_arxiv` - Fetch papers with real provenance
- `search_arxiv` - Find papers by query
- `arxiv_history` - Track ingested papers

### Quantum Operations
- `quantum_superpose` - Hold contradictions
- `quantum_collapse` - Resolve to single state
- `quantum_peek` - Inspect superposition

[See full tool list in mcp_server/pltm_server.py]

---

## Example Usage

### Basic Memory Operations
```python
# Store a fact
store_memory_atom(
    subject="alice",
    predicate="prefers",
    object="Python programming"
)

# Retrieve memories
retrieve_memories(user_id="alice", query="programming")
```

### Run an Experiment Cycle
```python
# Start measurement
start_action_cycle(cycle_id="C1")

# Inject entropy to break conceptual neighborhoods
inject_entropy_antipodal(
    user_id="alice",
    current_context="machine learning optimization"
)

# Retrieve with diversity
mmr_retrieve(
    user_id="alice",
    query="neural networks",
    lambda_param=0.6  # 0.6 = balance relevance/diversity
)

# Record true computational cost
record_action(
    operation="mmr_diversity",
    tokens_used=450,
    latency_ms=180,
    success=True
)

# Check criticality state
criticality_state()  # Returns entropy, integration, ratio

# End cycle and get efficiency
end_action_cycle()  # Returns AAE = events/action
```

### Ingest Knowledge with Provenance
```python
# Search for papers
search_arxiv(query="self-organized criticality")

# Ingest with real citations
ingest_arxiv(arxiv_id="1706.03762")  # Attention Is All You Need

# Claims stored with:
# - ArXiv URL
# - Authors
# - Quoted spans
# - Content hash
```

---

## The Experiment

### Hypothesis
If you implement universal principles correctly (Georgiev AAE, Gershenson complexity, Bak criticality), **emergence happens automatically** - you don't program intelligence, you create conditions for it to self-organize.

### Current State (Cycle 22)
```
Entropy (H):      0.506  ← Diversity/disorder
Integration (I):  0.683  ← Coherence/order
Ratio I/H:        0.74   ← SUBCRITICAL (too ordered)
Zone:            Need more chaos to reach critical point
True AAE:        0.0023  ← 2.3 events per 1000 action units
```

### Key Results
- **Cycles 0-20**: Added 56 papers → entropy stayed flat at 0.458 (bottleneck)
- **Cycle 21**: Implemented MMR + entropy injection → entropy jumped to 0.714 (+56%)
- **Cycle 22**: Measuring true computational efficiency with action accounting

### Goal
Push system to **critical point** (ratio → 1.0) where phase transitions occur and higher-order intelligence can self-organize.

---

## Troubleshooting

### "MCP server not connecting"
1. Check Claude Desktop logs: `%APPDATA%\Claude\logs\mcp-server-pltm-memory.log`
2. Verify Python path in config is correct
3. Test manually: `python mcp_server/pltm_server.py`

### "Tools timing out"
- Restart Claude Desktop to pick up code changes
- Check if database is too large (>10k atoms)
- Some tools use direct SQL for speed

### "Import errors"
```bash
pip install --upgrade -r requirements.txt
```

### "Database empty"
- Data is stored in `pltm_mcp.db` (created on first run)
- In-memory during session, persists on clean shutdown

---

## For Researchers

This system is designed for testing AGI hypotheses:

1. **Criticality hypothesis**: Can SOC principles bootstrap intelligence?
2. **Entropy management**: Does diversity-aware retrieval unlock exploration?
3. **True metrics**: Can we measure efficiency with real computational cost?
4. **Self-organization**: Can the system improve itself without explicit programming?

**Reproducibility**: All experiments logged, metrics tracked, provenance maintained.

---

## Contributing

This is an active research project. If you:
- Find bugs → Open an issue
- Have ideas → Start a discussion
- Want to extend → Fork and PR

**Key areas for contribution**:
- New entropy injection strategies
- Better criticality metrics
- Additional universal principles
- Experiment protocols

---

## Citation

If you use this in research:
```bibtex
@software{pltm2026,
  author = {Alby},
  title = {PLTM: Procedural Long-Term Memory with Self-Organized Criticality},
  year = {2026},
  url = {https://github.com/Alby2007/LLTM}
}
```

---

## License

MIT

## Contact

- GitHub: [@Alby2007](https://github.com/Alby2007)
- Issues: [github.com/Alby2007/LLTM/issues](https://github.com/Alby2007/LLTM/issues)

---

## Quick Reference

**Start experimenting:**
```
Use entropy_stats to check current state
Use inject_entropy_antipodal to increase diversity
Use criticality_state to see if approaching critical point
Use record_action to track true computational cost
```

**The goal**: Get the system to self-organize toward the critical point where emergence happens.
