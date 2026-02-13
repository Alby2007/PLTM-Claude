#!/usr/bin/env python3
"""
PLTM Binary Shield Builder

Compiles Tier 1 proprietary algorithms into .so shared libraries,
then verifies imports still work. Run before git push.

Usage:
    python build_shield.py              # Compile Tier 1 only
    SHIELD_TIER2=1 python build_shield.py  # Compile Tier 1 + Tier 2
    python build_shield.py --verify     # Verify existing .so files work
    python build_shield.py --clean      # Remove .c and build artifacts
"""

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

TIER1_MODULES = [
    ("src/memory/phi_rms.py", "src.memory.phi_rms"),
    ("src/memory/memory_pipeline.py", "src.memory.memory_pipeline"),
    ("src/memory/memory_jury.py", "src.memory.memory_jury"),
    ("src/memory/session_continuity.py", "src.memory.session_continuity"),
    ("src/analysis/epistemic_v2.py", "src.analysis.epistemic_v2"),
    ("src/analysis/pltm_self.py", "src.analysis.pltm_self"),
]

TIER2_MODULES = [
    ("src/analysis/grounded_reasoning.py", "src.analysis.grounded_reasoning"),
    ("src/memory/memory_intelligence.py", "src.memory.memory_intelligence"),
    ("src/core/ontology.py", "src.core.ontology"),
    ("src/memory/attention_retrieval.py", "src.memory.attention_retrieval"),
    ("src/learning/cross_domain_synthesis.py", "src.learning.cross_domain_synthesis"),
]


def compile_modules():
    """Compile Tier 1 (and optionally Tier 2) modules via setup.py."""
    print("=" * 60)
    print("PLTM Binary Shield — Compiling proprietary modules")
    print("=" * 60)

    env = os.environ.copy()
    tier2 = env.get("SHIELD_TIER2", "0") == "1"

    modules = list(TIER1_MODULES)
    if tier2:
        modules.extend(TIER2_MODULES)
        print(f"\nCompiling {len(modules)} modules (Tier 1 + Tier 2)")
    else:
        print(f"\nCompiling {len(modules)} modules (Tier 1 only)")

    # Check all source files exist
    missing = [f for f, _ in modules if not (ROOT / f).exists()]
    if missing:
        print(f"\n❌ Missing source files: {missing}")
        return False

    # Run Cython compilation
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"\n❌ Compilation failed:\n{result.stderr[-2000:]}")
        return False

    print("\n✅ Compilation succeeded")
    return True


def verify_imports():
    """Verify all compiled modules can be imported."""
    print("\n" + "=" * 60)
    print("Verifying imports...")
    print("=" * 60)

    # Add project root to path
    sys.path.insert(0, str(ROOT))

    modules = list(TIER1_MODULES)
    if os.environ.get("SHIELD_TIER2", "0") == "1":
        modules.extend(TIER2_MODULES)

    ok = 0
    fail = 0
    for filepath, module_name in modules:
        so_pattern = filepath.replace(".py", ".cpython-*.so")
        so_files = list(ROOT.glob(so_pattern))

        try:
            # Force reimport
            if module_name in sys.modules:
                del sys.modules[module_name]
            mod = importlib.import_module(module_name)
            source = getattr(mod, "__file__", "?")
            is_so = source and source.endswith(".so")
            status = "✅ .so" if is_so else "⚠️  .py (not compiled)"
            print(f"  {status}  {module_name}")
            ok += 1
        except Exception as e:
            print(f"  ❌ FAIL  {module_name}: {e}")
            fail += 1

    print(f"\n{'✅' if fail == 0 else '❌'} {ok} passed, {fail} failed")
    return fail == 0


def list_so_files():
    """List all compiled .so files with sizes."""
    print("\n" + "=" * 60)
    print("Compiled .so files:")
    print("=" * 60)

    total_size = 0
    for filepath, _ in TIER1_MODULES + TIER2_MODULES:
        so_pattern = filepath.replace(".py", ".cpython-*.so")
        for so_file in sorted(ROOT.glob(so_pattern)):
            size = so_file.stat().st_size
            total_size += size
            print(f"  {size:>10,} bytes  {so_file.relative_to(ROOT)}")

    if total_size == 0:
        print("  (none found — run build_shield.py first)")
    else:
        print(f"\n  Total: {total_size:,} bytes ({total_size / 1024:.0f} KB)")


def clean():
    """Remove .c files and build/ directory."""
    print("Cleaning build artifacts...")
    removed = 0

    # Remove generated .c files
    for filepath, _ in TIER1_MODULES + TIER2_MODULES:
        c_file = ROOT / filepath.replace(".py", ".c")
        if c_file.exists():
            c_file.unlink()
            removed += 1

    # Remove build/ directory
    build_dir = ROOT / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
        removed += 1

    print(f"  Removed {removed} artifacts")


def print_gitignore_reminder():
    """Print reminder about .gitignore setup."""
    print("\n" + "=" * 60)
    print("IMPORTANT: .gitignore setup for git push")
    print("=" * 60)
    print("""
Before pushing to git, ensure your .gitignore has:

  # Tier 1 source files — NEVER push to public repo
  src/memory/phi_rms.py
  src/memory/memory_pipeline.py
  src/memory/memory_jury.py
  src/memory/session_continuity.py
  src/analysis/epistemic_v2.py
  src/analysis/pltm_self.py

  # Allow compiled .so files (override global *.so ignore)
  !src/memory/*.so
  !src/analysis/*.so
  !src/core/*.so
  !src/learning/*.so

Keep a PRIVATE backup of the .py source files outside the repo!
""")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify_imports()
        list_so_files()
    elif "--clean" in sys.argv:
        clean()
    else:
        success = compile_modules()
        if success:
            clean()  # Remove .c intermediates
            verify_imports()
            list_so_files()
            print_gitignore_reminder()
