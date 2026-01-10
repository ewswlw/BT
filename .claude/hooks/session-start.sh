#!/bin/bash
# Claude Session Start Hook
# This hook runs when a new Claude session starts
# Use it to set up your environment, check dependencies, and prepare the workspace

set -e

echo "🚀 Starting Claude session for BT (Backtesting Framework)..."

# Check if we're in the correct directory
if [[ ! -f "pyproject.toml" ]]; then
    echo "⚠️  Warning: pyproject.toml not found. You may not be in the project root."
fi

# Display current branch
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
echo "📍 Current branch: $CURRENT_BRANCH"

# Check if poetry is available
if command -v poetry &> /dev/null; then
    echo "✓ Poetry is available"

    # Check if virtual environment is activated
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo "✓ Virtual environment is active: $VIRTUAL_ENV"
    else
        echo "ℹ️  Virtual environment not active. Run 'poetry shell' if needed."
    fi
else
    echo "⚠️  Poetry not found. Install it for dependency management."
fi

# Display project structure overview
echo ""
echo "📁 Project structure:"
echo "  - cad_ig_er_index_backtesting/: Core backtesting framework"
echo "  - market data pipeline/: Data ingestion and processing"
echo "  - tests/: Test suites"
echo "  - .claude/: Claude configuration and tools"

# Check for uncommitted changes
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo ""
    echo "⚠️  You have uncommitted changes:"
    git status --short
fi

echo ""
echo "✓ Session initialized. Ready to work!"
echo ""
