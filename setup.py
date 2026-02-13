"""
PLTM Build Configuration — Cython Compilation for IP Protection

Compiles Tier 1 proprietary algorithms into binary .so shared libraries.
Source .py files are gitignored; only compiled binaries ship in the repo.

Usage:
    python setup.py build_ext --inplace    # Compile all Tier 1 modules
    python build_shield.py                  # Convenience wrapper (recommended)
"""

from setuptools import setup, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# ── Tier 1: Core differentiating IP — MUST be compiled ──────────────────────
TIER1_MODULES = [
    "src/memory/phi_rms.py",
    "src/memory/memory_pipeline.py",
    "src/memory/memory_jury.py",
    "src/memory/session_continuity.py",
    "src/analysis/epistemic_v2.py",
    "src/analysis/pltm_self.py",
]

# ── Tier 2: Consider shielding — opt-in via SHIELD_TIER2=1 env var ──────────
import os
TIER2_MODULES = [
    "src/analysis/grounded_reasoning.py",
    "src/memory/memory_intelligence.py",
    "src/core/ontology.py",
    "src/memory/attention_retrieval.py",
    "src/learning/cross_domain_synthesis.py",
]

modules_to_compile = list(TIER1_MODULES)
if os.environ.get("SHIELD_TIER2", "0") == "1":
    modules_to_compile.extend(TIER2_MODULES)

setup(
    name="pltm",
    version="0.1.0",
    description="Persistent Long-Term Memory for AI",
    packages=find_packages(),
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(
        modules_to_compile,
        compiler_directives={
            "language_level": "3",       # Python 3 semantics
            "boundscheck": True,         # Keep bounds checking (safe with dynamic code)
            "wraparound": True,          # Keep negative indexing (safe with dynamic code)
        },
        # Don't generate .html annotation files
        annotate=False,
    ),
    python_requires=">=3.11",
)
