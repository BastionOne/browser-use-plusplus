#!/bin/bash
# Installation script for browser-use-plusplus (bupp) with local dependencies
# This script sets up bupp for local development with editable browser-use and llm-lib
#
# Usage:
#   ./install_local.sh              - Install in current directory (bupp)
#   ./install_local.sh <path>       - Install in specified directory

set -e

TARGET_DIR="${1:-.}"

# Resolve to absolute path
if [[ "$TARGET_DIR" != /* ]]; then
    TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"
fi

echo "Installing browser-use-plusplus in: $TARGET_DIR"

# Verify we're in the right place
if [[ ! -f "$TARGET_DIR/pyproject.toml" ]]; then
    echo "Error: pyproject.toml not found in $TARGET_DIR"
    exit 1
fi

cd "$TARGET_DIR"

# Check for parent repo dependencies
PARENT_DIR="$(dirname "$TARGET_DIR")"
HAS_BROWSER_USE=false
HAS_LLM_LIB=false

if [[ -d "$PARENT_DIR/browser-use" ]]; then
    HAS_BROWSER_USE=true
    echo "Found browser-use at: $PARENT_DIR/browser-use"
fi

if [[ -d "$PARENT_DIR/llm-lib" ]]; then
    HAS_LLM_LIB=true
    echo "Found llm-lib at: $PARENT_DIR/llm-lib"
fi

# Set up Python environment
echo ""
echo "=== Setting up Python environment ==="

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "Creating virtual environment with uv..."
    uv venv .venv
fi

if [[ -f ".venv/bin/activate" ]]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate

    echo ""
    echo "=== Installing packages ==="

    # Install local dependencies if available
    if [[ "$HAS_BROWSER_USE" == true ]]; then
        echo "Installing browser-use from local path (editable)..."
        uv pip install -e "$PARENT_DIR/browser-use"
    else
        echo "Warning: browser-use not found locally, will be installed from PyPI"
    fi

    if [[ "$HAS_LLM_LIB" == true ]]; then
        echo "Installing llm-lib from local path (editable)..."
        uv pip install -e "$PARENT_DIR/llm-lib"
    else
        echo "Warning: llm-lib not found locally, will be installed from PyPI"
    fi

    # Install bupp package
    echo "Installing browser-use-plusplus (editable)..."
    uv pip install -e .

    echo ""
    echo "=== Installation complete! ==="
    echo ""
    if [[ "$HAS_BROWSER_USE" == true || "$HAS_LLM_LIB" == true ]]; then
        echo "Local dependencies installed:"
        [[ "$HAS_BROWSER_USE" == true ]] && echo "  - browser-use (editable)"
        [[ "$HAS_LLM_LIB" == true ]] && echo "  - llm-lib (editable)"
    fi
    echo ""
    echo "To activate the virtual environment:"
    echo "  source .venv/bin/activate"
else
    echo "Error: Failed to create virtual environment"
    exit 1
fi
