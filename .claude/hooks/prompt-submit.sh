#!/bin/bash
# Claude Prompt Submit Hook
# This hook runs before each prompt is submitted to Claude
# Use it to validate input, check context, or enforce guidelines

# Get the prompt from stdin
PROMPT="$1"

# Check if the prompt contains certain keywords that might need special handling
if echo "$PROMPT" | grep -qi "delete.*production\|drop.*database\|rm.*-rf.*\/"; then
    echo "⚠️  SAFETY WARNING: This prompt contains potentially destructive operations."
    echo "Please review carefully before proceeding."
    # Uncomment the next line to block such prompts
    # exit 1
fi

# Check for requests to modify critical files
CRITICAL_FILES=(
    "pyproject.toml"
    "poetry.lock"
    ".git/config"
)

for file in "${CRITICAL_FILES[@]}"; do
    if echo "$PROMPT" | grep -qi "delete.*$file\|remove.*$file"; then
        echo "⚠️  WARNING: Prompt mentions critical file: $file"
    fi
done

# Provide helpful context reminders for backtesting work
if echo "$PROMPT" | grep -qi "backtest\|strategy\|trading\|portfolio"; then
    echo "ℹ️  Remember: Always validate strategies with out-of-sample data"
fi

# Allow the prompt to proceed
exit 0
