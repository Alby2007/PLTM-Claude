#!/usr/bin/env python3
"""
PLTM — Full Automated Setup
============================
One command does everything:

    python setup_pltm.py

  1. Check Python version (3.10+)
  2. Create virtual environment
  3. Install all dependencies
  4. Create .env from template
  5. Initialize database (if needed)
  6. Pre-download embedding model
  7. Auto-configure Claude Desktop (backup existing config)
  8. Verify installation

Flags:
  --skip-claude    Skip Claude Desktop configuration
  --skip-model     Skip embedding model download (faster install)
  --reset          Reset: delete .venv and DB, start fresh
  --uninstall      Remove PLTM from Claude Desktop config
"""

import subprocess
import sys
import os
import json
import platform
import shutil
import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
VENV = ROOT / ".venv"
DATA = ROOT / "data"
DB_PATH = DATA / "pltm_mcp.db"
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"

IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
PYTHON = VENV / ("Scripts" if IS_WIN else "bin") / ("python.exe" if IS_WIN else "python3")
PIP = VENV / ("Scripts" if IS_WIN else "bin") / ("pip.exe" if IS_WIN else "pip")

PASS = "[OK]"
FAIL = "[!!]"
WARN = "[??]"
SKIP = "[--]"


# ── Helpers ────────────────────────────────────────────────
def header(step, total, msg):
    print(f"\n{'─'*60}")
    print(f"  [{step}/{total}] {msg}")
    print(f"{'─'*60}")


def run(cmd, capture=False, check=True, **kwargs):
    cmd_str = " ".join(str(c) for c in cmd)
    if len(cmd_str) > 100:
        cmd_str = cmd_str[:97] + "..."
    print(f"  $ {cmd_str}")
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    else:
        result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        print(f"  {FAIL} Command failed (exit code {result.returncode})")
        if capture and result.stderr:
            for line in result.stderr.strip().splitlines()[:5]:
                print(f"      {line}")
        return None
    return result


def get_claude_config_path() -> Path:
    """Detect Claude Desktop config path for any OS."""
    if IS_WIN:
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
        return Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
    elif IS_MAC:
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    else:
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def read_groq_key() -> str:
    """Try to find GROQ_API_KEY from .env or environment."""
    # Check environment first
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    # Check .env
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("GROQ_API_KEY=") and not line.endswith("="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val and "your" not in val.lower() and "sk-ant-" not in val:
                    return val
    return ""


# ── Steps ──────────────────────────────────────────────────
def step_python_check(step, total):
    header(step, total, "Checking Python version")
    v = sys.version_info
    print(f"  Python {v.major}.{v.minor}.{v.micro} ({sys.executable})")
    if v < (3, 10):
        print(f"  {FAIL} Python 3.10+ required. You have {v.major}.{v.minor}.")
        print()
        if IS_WIN:
            print("  Install Python 3.11+:")
            print("    winget install Python.Python.3.11")
            print("  or download from https://www.python.org/downloads/")
        elif IS_MAC:
            print("  Install Python 3.11+:")
            print("    brew install python@3.11")
        else:
            print("  Install Python 3.11+:")
            print("    sudo apt install python3.11  # Debian/Ubuntu")
            print("    sudo dnf install python3.11  # Fedora")
        sys.exit(1)
    print(f"  {PASS} Python {v.major}.{v.minor} OK")


def step_venv(step, total):
    header(step, total, "Creating virtual environment")
    if VENV.exists() and PYTHON.exists():
        print(f"  {SKIP} .venv already exists, skipping")
        return
    if VENV.exists():
        print("  Removing broken .venv...")
        shutil.rmtree(VENV, ignore_errors=True)
    result = run([sys.executable, "-m", "venv", str(VENV)])
    if result is None:
        print(f"  {FAIL} Failed to create venv. Try: python -m ensurepip")
        sys.exit(1)
    print(f"  {PASS} Virtual environment created")


def step_deps(step, total):
    header(step, total, "Installing dependencies")
    run([str(PIP), "install", "--upgrade", "pip", "-q"], check=False)

    req_file = ROOT / "requirements-lite.txt"
    if not req_file.exists():
        req_file = ROOT / "requirements.txt"

    result = run([str(PIP), "install", "-r", str(req_file)])
    if result is None:
        print(f"  {FAIL} Dependency install failed. Check errors above.")
        sys.exit(1)
    print(f"  {PASS} Dependencies installed from {req_file.name}")


def step_env(step, total):
    header(step, total, "Setting up environment")
    if ENV_FILE.exists():
        print(f"  {SKIP} .env already exists")
    elif ENV_EXAMPLE.exists():
        shutil.copy2(ENV_EXAMPLE, ENV_FILE)
        print(f"  {PASS} Created .env from template")
    else:
        ENV_FILE.write_text(
            "# PLTM Configuration\n"
            "# Get a free key at https://console.groq.com\n"
            "GROQ_API_KEY=\n"
            "\n"
            "# Optional\n"
            "DEEPSEEK_API_KEY=\n"
            "LOG_LEVEL=INFO\n"
        )
        print(f"  {PASS} Created .env template")

    key = read_groq_key()
    if key:
        print(f"  {PASS} GROQ_API_KEY found (...{key[-6:]})")
    else:
        print(f"  {WARN} GROQ_API_KEY not set — LLM tools (ingestion, fact-check) won't work")
        print("       Core memory tools work fine without it.")
        print("       Get a free key: https://console.groq.com")


def step_db(step, total):
    header(step, total, "Initializing database")
    DATA.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(DB_PATH))
            tables = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
            conn.close()
            print(f"  {PASS} Database exists ({tables} tables)")
        except Exception:
            print(f"  {PASS} Database exists")
        return

    print("  Creating fresh database...")
    init_script = (
        "import asyncio, sys\n"
        f"sys.path.insert(0, r'{ROOT}')\n"
        "from mcp_server.pltm_server import initialize_pltm\n"
        "asyncio.run(initialize_pltm())\n"
        "print('done')\n"
    )
    result = run([str(PYTHON), "-c", init_script], cwd=str(ROOT), check=False)
    if result and result.returncode == 0:
        print(f"  {PASS} Database created at data/pltm_mcp.db")
    else:
        # Fallback: just create the data dir, server will init on first run
        print(f"  {WARN} Could not pre-initialize DB (will be created on first run)")


def step_model(step, total):
    header(step, total, "Downloading embedding model")
    dl_script = (
        "try:\n"
        "    from sentence_transformers import SentenceTransformer\n"
        "    m = SentenceTransformer('all-MiniLM-L6-v2')\n"
        "    print('ready')\n"
        "except ImportError:\n"
        "    print('skip')\n"
        "except Exception as e:\n"
        "    print(f'error:{e}')\n"
    )
    result = run([str(PYTHON), "-c", dl_script], cwd=str(ROOT), capture=True, check=False)
    if result and "ready" in (result.stdout or ""):
        print(f"  {PASS} Embedding model cached (all-MiniLM-L6-v2)")
    elif result and "skip" in (result.stdout or ""):
        print(f"  {WARN} sentence-transformers not installed — embedding search unavailable")
        print("       Install later: .venv/bin/pip install sentence-transformers")
    else:
        print(f"  {WARN} Model download failed — will retry on first use")


def step_configure_claude(step, total):
    header(step, total, "Configuring Claude Desktop")

    config_path = get_claude_config_path()
    print(f"  Config: {config_path}")

    # Check venv python exists
    if not PYTHON.exists():
        print(f"  {FAIL} Venv Python not found at {PYTHON}")
        sys.exit(1)

    # Build PLTM entry
    groq_key = read_groq_key()
    env_block = {"PYTHONPATH": str(ROOT)}
    if groq_key:
        env_block["GROQ_API_KEY"] = groq_key

    pltm_entry = {
        "command": str(PYTHON),
        "args": ["-m", "mcp_server.pltm_server"],
        "env": env_block,
    }

    # Load existing config
    config = {}
    if config_path.exists():
        try:
            raw = config_path.read_text(encoding="utf-8")
            # Handle BOM (common Windows issue)
            if raw.startswith("\ufeff"):
                raw = raw[1:]
            config = json.loads(raw)
            server_count = len(config.get("mcpServers", {}))
            print(f"  Found existing config with {server_count} MCP server(s)")
        except json.JSONDecodeError as e:
            # Common MCP bug: malformed JSON
            print(f"  {WARN} Existing config has invalid JSON: {e}")
            # Backup the broken file
            backup = config_path.with_suffix(f".backup-{datetime.datetime.now():%Y%m%d-%H%M%S}.json")
            shutil.copy2(config_path, backup)
            print(f"  Backed up broken config to {backup.name}")
            config = {}
        except Exception as e:
            print(f"  {WARN} Could not read config: {e}")
            config = {}

    # Backup before modifying
    if config_path.exists() and config:
        backup = config_path.with_suffix(f".pre-pltm-{datetime.datetime.now():%Y%m%d-%H%M%S}.json")
        shutil.copy2(config_path, backup)
        print(f"  Backed up to {backup.name}")

    # Merge
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    was_update = "pltm" in config["mcpServers"]
    config["mcpServers"]["pltm"] = pltm_entry

    # Write with proper encoding (fixes Windows UTF-8 BOM issues)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    action = "Updated" if was_update else "Added"
    print(f"  {PASS} {action} 'pltm' server in Claude Desktop config")

    # Validate the written config is parseable (catch write corruption)
    try:
        json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        print(f"  {FAIL} Written config is corrupt — restoring backup")
        if backup.exists():
            shutil.copy2(backup, config_path)
        sys.exit(1)


def step_verify(step, total):
    header(step, total, "Verifying installation")
    errors = 0

    # Check venv python
    if PYTHON.exists():
        print(f"  {PASS} Virtual environment")
    else:
        print(f"  {FAIL} Virtual environment missing")
        errors += 1

    # Check core import
    result = run(
        [str(PYTHON), "-c", "from mcp_server import pltm_server; print('ok')"],
        capture=True, check=False, cwd=str(ROOT)
    )
    if result and "ok" in (result.stdout or ""):
        print(f"  {PASS} MCP server importable")
    else:
        print(f"  {FAIL} MCP server import failed")
        if result and result.stderr:
            for line in result.stderr.strip().splitlines()[:3]:
                print(f"       {line}")
        errors += 1

    # Check DB
    if DB_PATH.exists():
        print(f"  {PASS} Database")
    else:
        print(f"  {WARN} Database will be created on first run")

    # Check Claude config
    config_path = get_claude_config_path()
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            if "pltm" in cfg.get("mcpServers", {}):
                print(f"  {PASS} Claude Desktop configured")
            else:
                print(f"  {WARN} Claude Desktop config exists but no 'pltm' entry")
        except Exception:
            print(f"  {WARN} Claude Desktop config unreadable")
    else:
        print(f"  {WARN} Claude Desktop config not found")

    return errors


def handle_reset():
    """Delete .venv and DB for a fresh start."""
    print("\n  Resetting PLTM installation...\n")
    if VENV.exists():
        print(f"  Removing {VENV}...")
        shutil.rmtree(VENV, ignore_errors=True)
    if DB_PATH.exists():
        print(f"  Removing {DB_PATH}...")
        DB_PATH.unlink()
    if ENV_FILE.exists():
        print(f"  Keeping .env (delete manually if needed)")
    print("  Reset complete. Run setup_pltm.py again to reinstall.\n")
    sys.exit(0)


def handle_uninstall():
    """Remove PLTM from Claude Desktop config."""
    print("\n  Removing PLTM from Claude Desktop...\n")
    config_path = get_claude_config_path()
    if not config_path.exists():
        print("  No Claude Desktop config found.")
        sys.exit(0)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if "pltm" in config.get("mcpServers", {}):
            del config["mcpServers"]["pltm"]
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            print(f"  {PASS} Removed 'pltm' from {config_path}")
            print("  Restart Claude Desktop to apply.")
        else:
            print("  'pltm' not found in config.")
    except Exception as e:
        print(f"  {FAIL} Error: {e}")
    sys.exit(0)


# ── Main ───────────────────────────────────────────────────
def main():
    args = sys.argv[1:]

    if "--reset" in args:
        handle_reset()
    if "--uninstall" in args:
        handle_uninstall()

    skip_claude = "--skip-claude" in args
    skip_model = "--skip-model" in args

    total = 8 - (1 if skip_claude else 0) - (1 if skip_model else 0)

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║   PLTM — Persistent Long-Term Memory        ║")
    print("  ║   Automated Setup                            ║")
    print("  ╚══════════════════════════════════════════════╝")

    s = 1
    step_python_check(s, total); s += 1
    step_venv(s, total); s += 1
    step_deps(s, total); s += 1
    step_env(s, total); s += 1
    step_db(s, total); s += 1

    if not skip_model:
        step_model(s, total); s += 1

    if not skip_claude:
        step_configure_claude(s, total); s += 1

    errors = step_verify(s, total)

    # Final message
    print()
    print("  ╔══════════════════════════════════════════════╗")
    if errors == 0:
        print("  ║   Setup complete!                            ║")
        print("  ║                                              ║")
        print("  ║   Restart Claude Desktop to activate PLTM.   ║")
        print("  ║   136 tools will be available.               ║")
    else:
        print("  ║   Setup finished with warnings.              ║")
        print("  ║   Check errors above.                        ║")
    print("  ╚══════════════════════════════════════════════╝")

    if not skip_claude:
        print()
        print("  Just restart Claude Desktop — that's it!")
        print()
        print("  Verify by asking Claude:")
        print('    "Use auto_init_session to check system state"')

    groq_key = read_groq_key()
    if not groq_key:
        print()
        print("  Optional: Add GROQ_API_KEY to .env for LLM tools")
        print("  (free at https://console.groq.com)")
        print("  Then re-run: python setup_pltm.py")

    print()


if __name__ == "__main__":
    main()
