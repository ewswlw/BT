# Getting Started with Claude Configuration

This guide will help you set up and start using the Claude configuration for the BT backtesting framework.

## Prerequisites

- Claude Code CLI installed
- BT repository cloned
- Basic understanding of backtesting concepts
- Python environment set up with Poetry

## Installation

### 1. Verify Setup

The `.claude` directory should already be in your repository. Verify it exists:

```bash
ls -la .claude/
```

You should see:
```
hooks/
commands/
agents/
docs/
config/
templates/
```

### 2. Make Hooks Executable

Ensure all hook scripts have execute permissions:

```bash
chmod +x .claude/hooks/*.sh
```

Verify permissions:
```bash
ls -la .claude/hooks/
```

### 3. Configure Claude Settings

Edit your Claude Code settings to enable hooks:

```json
{
  "hooks": {
    "session_start": ".claude/hooks/session-start.sh",
    "prompt_submit": ".claude/hooks/prompt-submit.sh",
    "tool_call": ".claude/hooks/tool-call.sh"
  },
  "commands_path": ".claude/commands",
  "agents_path": ".claude/agents"
}
```

## First Steps

### 1. Start a Session

When you start Claude Code, the session-start hook will run automatically:

```bash
claude-code
```

You should see output like:
```
🚀 Starting Claude session for BT (Backtesting Framework)...
📍 Current branch: your-branch
✓ Poetry is available
✓ Virtual environment is active
...
```

### 2. Try a Slash Command

Test the backtest command:

```
/backtest

Could you help me understand the current backtesting framework structure?
```

Claude will switch into backtesting specialist mode and provide detailed analysis.

### 3. Work with an Agent

Request specific agent expertise:

```
I need help analyzing the risk profile of my momentum strategy.
Please use the risk manager agent.
```

## Common First Tasks

### Task 1: Explore the Codebase

```
/review-code

Please review the cad_ig_er_index_backtesting/ directory and give me an
overview of the code structure and any immediate concerns.
```

### Task 2: Run a Backtest

```
/backtest

I want to run a backtest on the momentum strategy with the following parameters:
- Lookback: 126 days
- Rebalance: Monthly
- Universe: S&P 500

Please validate the data and run the analysis.
```

### Task 3: Analyze Performance

```
/analyze-strategy

Analyze the performance of the last backtest run. Focus on:
1. Risk-adjusted returns
2. Drawdown analysis
3. Factor exposures
4. Suggestions for improvement
```

### Task 4: Optimize Parameters

```
/optimize-params

Help me optimize the lookback period for the momentum strategy.
- Test range: 60 to 252 days
- Objective: Sharpe ratio
- Use walk-forward validation
```

## Understanding the Workflow

### Standard Development Cycle

```
1. /review-code → Understand current state
2. Implement changes → Write code with Claude
3. /review-code → Validate implementation
4. /backtest → Test the strategy
5. /analyze-strategy → Evaluate results
6. /optimize-params → Tune parameters
7. /risk-analysis → Assess risks
8. Deploy → Push to production
```

### Data Pipeline Development

```
1. /data-pipeline → Get expert guidance
2. Design pipeline → Plan architecture
3. Implement → Write code
4. /review-code → Quality check
5. Test → Validate data quality
6. Deploy → Set up monitoring
```

## Using Hooks Effectively

### Session Start Hook

Automatically runs when you start Claude. It:
- Shows current branch
- Checks environment setup
- Displays project structure
- Warns about uncommitted changes

Customize it by editing `.claude/hooks/session-start.sh`.

### Prompt Submit Hook

Runs before each prompt is sent to Claude. It:
- Detects potentially dangerous operations
- Provides context-specific reminders
- Validates command syntax

Use it to enforce safety practices.

### Tool Call Hook

Runs before tool execution. It:
- Logs all tool calls for audit
- Blocks dangerous operations
- Validates file modifications
- Tracks external requests

Essential for production safety.

## Customizing for Your Workflow

### Add Custom Commands

Create a new file: `.claude/commands/my-command.md`

```markdown
# /my-command - Brief Description

You are a specialized assistant for [specific task].

## Your Role
[Define the role and responsibilities]

## Key Responsibilities
1. [Responsibility 1]
2. [Responsibility 2]

## Guidelines
[Specific guidelines for this command]
```

### Modify Existing Agents

Edit agent files in `.claude/agents/` to adjust:
- Personality and communication style
- Specific expertise areas
- Default behaviors
- Quality standards

### Configure Settings

Edit `.claude/config/settings.json`:

```json
{
  "default_model": "claude-sonnet-4-5",
  "agents": {
    "quantitative-analyst": {
      "model": "claude-opus-4-5",
      "temperature": 0.1
    },
    "risk-manager": {
      "model": "claude-sonnet-4-5",
      "temperature": 0.0
    }
  },
  "hooks": {
    "enable_logging": true,
    "log_directory": ".claude/logs"
  },
  "safety": {
    "block_dangerous_commands": true,
    "require_confirmation": ["git push --force", "rm -rf"]
  }
}
```

## Best Practices

### 1. Start Small
- Begin with one slash command
- Get comfortable with the workflow
- Gradually add more tools

### 2. Iterate Quickly
- Make small changes
- Test frequently
- Use code review command

### 3. Document Everything
- Add comments to your code
- Keep notes in documentation
- Track decisions and rationale

### 4. Use Version Control
- Commit frequently
- Write clear commit messages
- Review diffs before committing

### 5. Monitor Quality
- Run tests regularly
- Check data quality metrics
- Review risk indicators

## Troubleshooting

### Hooks Don't Run

**Problem**: Hooks not executing automatically

**Solutions**:
1. Check permissions: `chmod +x .claude/hooks/*.sh`
2. Verify hook paths in settings
3. Check for shell errors: Run hooks manually
4. Review logs: `cat .claude/logs/hook-errors.log`

### Commands Not Found

**Problem**: Slash commands not recognized

**Solutions**:
1. Verify file naming: `/command-name` → `command-name.md`
2. Check markdown formatting
3. Ensure commands_path is set
4. Restart Claude session

### Agent Not Responding Correctly

**Problem**: Agent behavior not as expected

**Solutions**:
1. Review agent markdown file
2. Be more specific in your request
3. Explicitly reference the agent
4. Check agent's expertise area matches your need

### Poor Performance

**Problem**: Claude responses are slow

**Solutions**:
1. Use faster model for simple tasks (haiku)
2. Reduce context size
3. Break complex tasks into smaller pieces
4. Cache frequently used data

## Next Steps

Once you're comfortable with the basics:

1. **Read the guides**:
   - [HOOKS-GUIDE.md](./HOOKS-GUIDE.md)
   - [COMMANDS-GUIDE.md](./COMMANDS-GUIDE.md)
   - [AGENTS-GUIDE.md](./AGENTS-GUIDE.md)

2. **Explore templates**:
   - Check `.claude/templates/` for code examples
   - Use as starting points for your own code

3. **Customize your setup**:
   - Add project-specific commands
   - Create custom agents for your team
   - Configure hooks for your workflow

4. **Share with team**:
   - Document your customizations
   - Share best practices
   - Contribute improvements

## Getting Help

### Resources

1. **Documentation**: Complete docs in `.claude/docs/`
2. **Examples**: Working examples in `.claude/templates/`
3. **Logs**: Debug info in `.claude/logs/`

### Common Questions

**Q: Which command should I use?**
A: Use `/backtest` for running tests, `/analyze-strategy` for deep analysis, `/review-code` for code quality.

**Q: When should I use agents vs commands?**
A: Commands provide focused workflows; agents provide specialized expertise. Use commands for specific tasks, agents for ongoing conversations.

**Q: How do I modify hook behavior?**
A: Edit the shell scripts in `.claude/hooks/`. Make them executable after changes.

**Q: Can I disable hooks temporarily?**
A: Yes, comment out hook paths in your settings or rename the hook files.

## Summary

You now know how to:
- ✅ Set up the Claude configuration
- ✅ Use slash commands
- ✅ Work with specialized agents
- ✅ Leverage hooks for safety
- ✅ Customize for your workflow
- ✅ Troubleshoot common issues

Start experimenting and happy backtesting! 🚀

---

**Next**: Read [COMMANDS-GUIDE.md](./COMMANDS-GUIDE.md) for detailed command documentation.
