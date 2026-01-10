#!/bin/bash
# Claude Tool Call Hook
# This hook runs before certain tool calls are executed
# Use it to validate tool usage, log actions, or enforce policies

TOOL_NAME="$1"
TOOL_ARGS="$2"

# Log tool calls for audit trail
LOG_DIR=".claude/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/tool-calls.log"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TIMESTAMP] Tool: $TOOL_NAME | Args: $TOOL_ARGS" >> "$LOG_FILE"

# Validate certain tool calls
case "$TOOL_NAME" in
    "Bash")
        # Check for dangerous bash commands
        if echo "$TOOL_ARGS" | grep -qE "rm -rf /|mkfs|dd if="; then
            echo "🛑 BLOCKED: Potentially dangerous bash command detected"
            exit 1
        fi
        ;;

    "Write"|"Edit")
        # Ensure we're not modifying protected files
        if echo "$TOOL_ARGS" | grep -qE "\.git/|\.env"; then
            echo "⚠️  WARNING: Attempting to modify protected file"
            # Uncomment to block: exit 1
        fi
        ;;

    "WebFetch")
        # Log external web requests
        echo "ℹ️  Making external web request: $TOOL_ARGS"
        ;;
esac

# Allow the tool call to proceed
exit 0
