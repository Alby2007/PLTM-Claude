#!/bin/bash
# PLTM — One-line installer for macOS/Linux
# Usage: curl -fsSL https://raw.githubusercontent.com/Alby2007/PLTM-Claude/main/install.sh | bash
set -e

echo ""
echo "  PLTM — Persistent Long-Term Memory for Claude"
echo "  Installing..."
echo ""

# Check Python
if command -v python3.11 &>/dev/null; then
    PY=python3.11
elif command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "  ERROR: Python not found. Install Python 3.10+ first:"
    echo "    macOS:  brew install python@3.11"
    echo "    Ubuntu: sudo apt install python3.11"
    exit 1
fi

# Check version
VER=$($PY -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MAJOR=$($PY -c "import sys; print(sys.version_info.major)")
MINOR=$($PY -c "import sys; print(sys.version_info.minor)")
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
    echo "  ERROR: Python 3.10+ required (found $VER)"
    exit 1
fi
echo "  Using Python $VER ($PY)"

# Clone
INSTALL_DIR="$HOME/PLTM"
if [ -d "$INSTALL_DIR" ]; then
    echo "  Updating existing install at $INSTALL_DIR..."
    cd "$INSTALL_DIR"
    git pull --quiet
else
    echo "  Cloning PLTM..."
    git clone --quiet https://github.com/Alby2007/PLTM-Claude.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Run setup
$PY setup_pltm.py "$@"

echo ""
echo "  Done! Restart Claude Desktop to activate PLTM."
echo ""
