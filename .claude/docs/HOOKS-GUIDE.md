# Hooks Guide

Hooks are shell scripts that run automatically at specific points in your Claude workflow. They provide automation, safety checks, and workflow enhancements.

## Table of Contents

- [Overview](#overview)
- [Available Hooks](#available-hooks)
- [Hook Configuration](#hook-configuration)
- [Writing Custom Hooks](#writing-custom-hooks)
- [Best Practices](#best-practices)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

### What are Hooks?

Hooks are executable shell scripts that Claude Code runs automatically in response to specific events. They enable you to:

- **Automate Setup**: Initialize your environment when starting a session
- **Enforce Safety**: Validate operations before they execute
- **Log Activity**: Track actions for audit and debugging
- **Provide Context**: Display relevant information at the right time
- **Prevent Errors**: Catch issues before they cause problems

### Hook Types

1. **Session Start**: Runs once when Claude session begins
2. **Prompt Submit**: Runs before each user prompt is sent
3. **Tool Call**: Runs before Claude executes tools

### How Hooks Work

```
Event Occurs → Hook Triggered → Hook Executes → Result Evaluated → Proceed or Block
```

If a hook exits with code 0: Operation proceeds
If a hook exits with non-zero code: Operation may be blocked (depends on configuration)

## Available Hooks

### 1. Session Start Hook

**Location**: `.claude/hooks/session-start.sh`

**When it runs**: Once at the beginning of each Claude session

**Purpose**:
- Initialize the development environment
- Display project context
- Check prerequisites
- Show current status

**Current Implementation**:

```bash
#!/bin/bash
# Session Start Hook

set -e

echo "🚀 Starting Claude session for BT (Backtesting Framework)..."

# Check project root
if [[ ! -f "pyproject.toml" ]]; then
    echo "⚠️  Warning: pyproject.toml not found"
fi

# Display current branch
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
echo "📍 Current branch: $CURRENT_BRANCH"

# Check Poetry availability
if command -v poetry &> /dev/null; then
    echo "✓ Poetry is available"
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo "✓ Virtual environment is active"
    else
        echo "ℹ️  Run 'poetry shell' to activate virtual environment"
    fi
else
    echo "⚠️  Poetry not found"
fi

# Display project structure
echo ""
echo "📁 Project structure:"
echo "  - cad_ig_er_index_backtesting/: Core framework"
echo "  - market data pipeline/: Data processing"
echo "  - tests/: Test suites"
echo "  - .claude/: Claude configuration"

# Check for uncommitted changes
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo ""
    echo "⚠️  You have uncommitted changes:"
    git status --short
fi

echo ""
echo "✓ Session initialized. Ready to work!"
echo ""

exit 0
```

**Customization Ideas**:
- Load environment variables
- Check for dependency updates
- Display recent commits
- Show open pull requests
- Run quick health checks
- Set up temporary directories

### 2. Prompt Submit Hook

**Location**: `.claude/hooks/prompt-submit.sh`

**When it runs**: Before each user prompt is sent to Claude

**Purpose**:
- Validate user input
- Provide contextual warnings
- Block dangerous operations
- Add helpful reminders

**Current Implementation**:

```bash
#!/bin/bash
# Prompt Submit Hook

PROMPT="$1"

# Safety checks for destructive operations
if echo "$PROMPT" | grep -qi "delete.*production\|drop.*database\|rm.*-rf.*\/"; then
    echo "⚠️  SAFETY WARNING: Potentially destructive operation detected"
    echo "Please review carefully before proceeding"
    # Uncomment to block: exit 1
fi

# Check for critical file modifications
CRITICAL_FILES=("pyproject.toml" "poetry.lock" ".git/config")

for file in "${CRITICAL_FILES[@]}"; do
    if echo "$PROMPT" | grep -qi "delete.*$file\|remove.*$file"; then
        echo "⚠️  WARNING: Prompt mentions critical file: $file"
    fi
done

# Contextual reminders
if echo "$PROMPT" | grep -qi "backtest\|strategy\|trading"; then
    echo "ℹ️  Remember: Always validate with out-of-sample data"
fi

exit 0
```

**Use Cases**:
- Block commands that could delete data
- Warn about operations on production systems
- Remind about best practices
- Suggest related commands
- Validate syntax before sending

**Customization Ideas**:
- Check for required context
- Validate data references
- Suggest optimizations
- Enforce naming conventions
- Add project-specific warnings

### 3. Tool Call Hook

**Location**: `.claude/hooks/tool-call.sh`

**When it runs**: Before Claude executes certain tools (Bash, Write, Edit, etc.)

**Purpose**:
- Log tool usage for audit
- Validate tool parameters
- Block dangerous operations
- Enforce policies

**Current Implementation**:

```bash
#!/bin/bash
# Tool Call Hook

TOOL_NAME="$1"
TOOL_ARGS="$2"

# Log all tool calls
LOG_DIR=".claude/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/tool-calls.log"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TIMESTAMP] Tool: $TOOL_NAME | Args: $TOOL_ARGS" >> "$LOG_FILE"

# Validate specific tools
case "$TOOL_NAME" in
    "Bash")
        # Block dangerous bash commands
        if echo "$TOOL_ARGS" | grep -qE "rm -rf /|mkfs|dd if="; then
            echo "🛑 BLOCKED: Potentially dangerous bash command"
            exit 1
        fi
        ;;

    "Write"|"Edit")
        # Warn about protected files
        if echo "$TOOL_ARGS" | grep -qE "\.git/|\.env"; then
            echo "⚠️  WARNING: Modifying protected file"
            # Uncomment to block: exit 1
        fi
        ;;

    "WebFetch")
        # Log external requests
        echo "ℹ️  Making external web request"
        ;;
esac

exit 0
```

**Use Cases**:
- Audit all file modifications
- Prevent accidental deletions
- Enforce code review requirements
- Track external API calls
- Monitor resource usage

**Customization Ideas**:
- Require approval for certain operations
- Enforce file naming conventions
- Validate git operations
- Check resource availability
- Rate limit API calls
- Enforce security policies

## Hook Configuration

### Enabling Hooks

Add to your Claude Code settings (usually `~/.config/claude/settings.json`):

```json
{
  "hooks": {
    "session_start": {
      "enabled": true,
      "path": ".claude/hooks/session-start.sh",
      "timeout": 30000
    },
    "prompt_submit": {
      "enabled": true,
      "path": ".claude/hooks/prompt-submit.sh",
      "timeout": 5000,
      "blocking": false
    },
    "tool_call": {
      "enabled": true,
      "path": ".claude/hooks/tool-call.sh",
      "timeout": 5000,
      "blocking": true,
      "tools": ["Bash", "Write", "Edit", "WebFetch"]
    }
  }
}
```

### Configuration Options

**enabled**: `true|false` - Whether hook is active

**path**: `string` - Path to hook script (relative to project root)

**timeout**: `number` - Maximum execution time in milliseconds

**blocking**: `true|false` - Whether non-zero exit blocks operation

**tools**: `array` - (tool_call only) Which tools trigger the hook

### Disabling Hooks Temporarily

To disable all hooks temporarily:

```bash
# Rename hook directory
mv .claude/hooks .claude/hooks.disabled

# Work without hooks
...

# Re-enable
mv .claude/hooks.disabled .claude/hooks
```

To disable a specific hook:

```bash
# Rename specific hook
mv .claude/hooks/tool-call.sh .claude/hooks/tool-call.sh.disabled
```

## Writing Custom Hooks

### Hook Template

```bash
#!/bin/bash
# Custom Hook Name
# Description of what this hook does

# Exit on error
set -e

# Get hook arguments
ARG1="$1"
ARG2="$2"

# Your logic here
echo "Hook executing..."

# Perform checks or operations
if [[ condition ]]; then
    echo "✓ Check passed"
else
    echo "✗ Check failed"
    exit 1  # Non-zero exit blocks operation
fi

# Success
exit 0
```

### Hook Arguments

**Session Start**: No arguments

**Prompt Submit**:
- `$1`: The user's prompt text

**Tool Call**:
- `$1`: Tool name (e.g., "Bash", "Write", "Edit")
- `$2`: Tool arguments (as JSON string)

### Exit Codes

- `0`: Success, allow operation to proceed
- `1`: Failure, block operation (if blocking enabled)
- Other: Treated as failure

### Best Practices

#### 1. Keep Hooks Fast
Hooks should execute quickly (< 1 second) to avoid slowing down the workflow.

```bash
# Good: Quick check
if [[ -f "critical-file.txt" ]]; then
    echo "File exists"
fi

# Bad: Slow operation
poetry install  # Don't do expensive operations
```

#### 2. Provide Clear Output
Users should understand what the hook is doing.

```bash
# Good: Clear messages
echo "✓ Code formatting check passed"
echo "⚠️  Warning: Missing test coverage"
echo "🛑 BLOCKED: Cannot delete production data"

# Bad: Silent or cryptic
echo "Check 1 OK"  # What check?
```

#### 3. Use Appropriate Exit Codes
Only block operations when necessary.

```bash
# Critical safety issue - block
if dangerous_operation; then
    echo "🛑 BLOCKED: Dangerous operation"
    exit 1
fi

# Warning only - don't block
if suboptimal_operation; then
    echo "⚠️  Warning: Consider alternatives"
    exit 0  # Still allow
fi
```

#### 4. Log Important Actions
Keep an audit trail.

```bash
LOG_FILE=".claude/logs/hook-activity.log"
echo "[$(date)] Hook executed: $TOOL_NAME" >> "$LOG_FILE"
```

#### 5. Handle Errors Gracefully
Don't let hook failures break Claude.

```bash
# Use set -e carefully
set -e

# Or handle errors explicitly
if ! command_that_might_fail; then
    echo "Warning: Optional check failed"
    exit 0  # Don't block on optional checks
fi
```

#### 6. Make Hooks Portable
Work across different environments.

```bash
# Good: Check before using
if command -v poetry &> /dev/null; then
    poetry check
fi

# Bad: Assume tools exist
poetry check  # Fails if Poetry not installed
```

## Examples

### Example 1: Pre-Commit Check Hook

```bash
#!/bin/bash
# Run linting before commits

if echo "$PROMPT" | grep -qi "commit"; then
    echo "Running pre-commit checks..."

    # Check code formatting
    if ! poetry run black --check . ; then
        echo "🛑 BLOCKED: Code formatting issues"
        echo "Run: poetry run black ."
        exit 1
    fi

    # Check linting
    if ! poetry run ruff check .; then
        echo "🛑 BLOCKED: Linting issues found"
        echo "Run: poetry run ruff check --fix ."
        exit 1
    fi

    echo "✓ Pre-commit checks passed"
fi

exit 0
```

### Example 2: Data Quality Gate

```bash
#!/bin/bash
# Validate data quality before backtests

if echo "$PROMPT" | grep -qi "backtest\|run.*strategy"; then
    echo "Checking data quality..."

    # Check data freshness
    DATA_DIR="data/processed"
    if [[ -d "$DATA_DIR" ]]; then
        LATEST=$(find "$DATA_DIR" -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
        AGE=$(( $(date +%s) - $(stat -c %Y "$LATEST") ))

        if (( AGE > 86400 )); then  # 24 hours
            echo "⚠️  Warning: Data is $(( AGE / 3600 )) hours old"
            echo "Consider refreshing: poetry run python scripts/update_data.py"
        else
            echo "✓ Data is up to date"
        fi
    fi
fi

exit 0
```

### Example 3: Resource Monitoring

```bash
#!/bin/bash
# Check system resources before heavy operations

TOOL_NAME="$1"

if [[ "$TOOL_NAME" == "Bash" ]]; then
    # Check disk space
    DISK_USAGE=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//')

    if (( DISK_USAGE > 90 )); then
        echo "🛑 BLOCKED: Disk usage critical (${DISK_USAGE}%)"
        echo "Free up space before continuing"
        exit 1
    elif (( DISK_USAGE > 80 )); then
        echo "⚠️  Warning: Disk usage high (${DISK_USAGE}%)"
    fi

    # Check memory
    MEMORY_AVAILABLE=$(free -m | awk 'NR==2 {print $7}')

    if (( MEMORY_AVAILABLE < 1000 )); then
        echo "⚠️  Warning: Low memory (${MEMORY_AVAILABLE}MB available)"
    fi
fi

exit 0
```

### Example 4: Branch Protection

```bash
#!/bin/bash
# Prevent direct commits to main/master

TOOL_ARGS="$2"
CURRENT_BRANCH=$(git branch --show-current)

if [[ "$TOOL_ARGS" =~ "git commit" ]] || [[ "$TOOL_ARGS" =~ "git push" ]]; then
    if [[ "$CURRENT_BRANCH" == "main" ]] || [[ "$CURRENT_BRANCH" == "master" ]]; then
        echo "🛑 BLOCKED: Cannot commit directly to $CURRENT_BRANCH"
        echo "Create a feature branch: git checkout -b feature/your-feature"
        exit 1
    fi
fi

exit 0
```

## Troubleshooting

### Hook Not Running

**Problem**: Hook doesn't execute when expected

**Solutions**:
1. Check file permissions: `chmod +x .claude/hooks/your-hook.sh`
2. Verify hook path in configuration
3. Check if hook is enabled in settings
4. Test hook manually: `.claude/hooks/your-hook.sh "test"`

### Hook Fails Silently

**Problem**: Hook errors not visible

**Solutions**:
1. Add explicit logging:
   ```bash
   exec 2>> .claude/logs/hook-errors.log
   set -x  # Debug mode
   ```
2. Check error logs: `cat .claude/logs/hook-errors.log`
3. Test hook manually with sample inputs

### Hook is Too Slow

**Problem**: Hook takes too long to execute

**Solutions**:
1. Profile the hook: Add `time` commands
2. Remove expensive operations
3. Cache results when possible
4. Use background processes for non-critical checks
5. Increase timeout in configuration

### Hook Blocks Valid Operations

**Problem**: Hook incorrectly blocks legitimate operations

**Solutions**:
1. Review blocking conditions
2. Make checks more specific
3. Add whitelisting
4. Use warnings instead of blocking
5. Adjust regex patterns

## Summary

Hooks provide powerful automation and safety features:

- ✅ **Automate** repetitive setup tasks
- ✅ **Enforce** safety and quality standards
- ✅ **Monitor** operations and resources
- ✅ **Prevent** errors before they happen
- ✅ **Log** activity for audit trails

Start with the provided hooks and customize them for your workflow!

---

**Next**: Read [COMMANDS-GUIDE.md](./COMMANDS-GUIDE.md) to learn about slash commands.
