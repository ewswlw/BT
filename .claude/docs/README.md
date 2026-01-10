# Claude Configuration for BT (Backtesting Framework)

Welcome to the Claude configuration directory for the BT backtesting framework. This directory contains hooks, commands, agents, and documentation to enhance your workflow with Claude Code.

## 📁 Directory Structure

```
.claude/
├── hooks/              # Shell scripts that run on events
│   ├── session-start.sh      # Runs when session starts
│   ├── prompt-submit.sh      # Runs before prompts
│   └── tool-call.sh          # Runs before tool execution
├── commands/           # Custom slash commands
│   ├── backtest.md           # /backtest command
│   ├── analyze-strategy.md   # /analyze-strategy command
│   ├── optimize-params.md    # /optimize-params command
│   ├── review-code.md        # /review-code command
│   ├── risk-analysis.md      # /risk-analysis command
│   └── data-pipeline.md      # /data-pipeline command
├── agents/             # Specialized sub-agents
│   ├── quantitative-analyst.md
│   ├── risk-manager.md
│   ├── code-reviewer.md
│   └── data-engineer.md
├── docs/               # Documentation
│   ├── README.md             # This file
│   ├── GETTING-STARTED.md
│   ├── HOOKS-GUIDE.md
│   ├── COMMANDS-GUIDE.md
│   └── AGENTS-GUIDE.md
├── config/             # Configuration files
│   └── settings.json
├── templates/          # Code templates
│   └── strategy-template.py
└── logs/               # Log files (created at runtime)
    └── tool-calls.log
```

## 🚀 Quick Start

### 1. Using Hooks

Hooks are automatically executed at specific points in your workflow:

- **session-start.sh**: Runs when you start a new Claude session
- **prompt-submit.sh**: Validates prompts before submission
- **tool-call.sh**: Checks tool calls for safety

To enable hooks, configure them in your Claude settings.

### 2. Using Slash Commands

Invoke specialized behaviors with slash commands:

```
/backtest - Run comprehensive backtesting analysis
/analyze-strategy - Deep dive into strategy performance
/optimize-params - Parameter optimization with validation
/review-code - Code review for quant systems
/risk-analysis - Comprehensive risk assessment
/data-pipeline - Data pipeline management and troubleshooting
```

Example:
```
User: /backtest
Claude: [Activates backtesting specialist mode with full context]
```

### 3. Working with Agents

Agents are specialized personas for specific tasks. Reference them in your prompts:

- **Quantitative Analyst**: Strategy development and analysis
- **Risk Manager**: Risk assessment and mitigation
- **Code Reviewer**: Code quality and correctness
- **Data Engineer**: Data pipeline and infrastructure

Example:
```
User: @quantitative-analyst Please review the momentum strategy implementation
Claude: [Responds with quantitative analyst expertise]
```

## 📚 Documentation

Detailed guides are available in the `docs/` directory:

1. **[GETTING-STARTED.md](./GETTING-STARTED.md)**: Setup and first steps
2. **[HOOKS-GUIDE.md](./HOOKS-GUIDE.md)**: Comprehensive hooks documentation
3. **[COMMANDS-GUIDE.md](./COMMANDS-GUIDE.md)**: All available slash commands
4. **[AGENTS-GUIDE.md](./AGENTS-GUIDE.md)**: Agent capabilities and usage

## 🎯 Common Workflows

### Developing a New Strategy

1. Use `/backtest` to understand current framework
2. Implement strategy with Claude's help
3. Use `/review-code` to validate implementation
4. Use `/analyze-strategy` to evaluate performance
5. Use `/optimize-params` to tune parameters
6. Use `/risk-analysis` to assess risk profile

### Improving Data Pipeline

1. Use `/data-pipeline` for expert guidance
2. Implement changes with data engineer agent
3. Use `/review-code` for quality check
4. Test thoroughly with provided validation scripts

### Debugging Issues

1. Describe the issue to Claude
2. Use appropriate specialist agent
3. Follow systematic troubleshooting
4. Validate fixes with tests

## ⚙️ Configuration

Settings are stored in `config/settings.json`. Customize:

- Default models for different agents
- Hook behavior and thresholds
- Alert preferences
- Logging levels

## 🛡️ Safety Features

### Hooks Provide Safety
- Detect dangerous commands
- Validate data operations
- Log all actions
- Enforce best practices

### Quality Gates
- Code review checklist
- Data validation framework
- Risk limits and monitoring
- Test coverage requirements

## 📊 Monitoring

### Logs
- Session activity: `logs/session.log`
- Tool calls: `logs/tool-calls.log`
- Pipeline runs: `logs/pipeline.log`

### Metrics
- Code quality scores
- Data quality metrics
- Risk indicators
- Performance benchmarks

## 🤝 Best Practices

### Working with Claude

1. **Be Specific**: Clearly state your goals
2. **Use Commands**: Leverage specialized slash commands
3. **Reference Agents**: Call out specific expertise needed
4. **Iterative Development**: Work in small, testable increments
5. **Review Everything**: Use code review before deployment

### Code Quality

1. **Write Tests First**: TDD for critical components
2. **Document Assumptions**: Be explicit about constraints
3. **Validate Data**: Never trust input data
4. **Handle Errors**: Graceful degradation
5. **Monitor Production**: Continuous observability

### Risk Management

1. **Test Thoroughly**: Out-of-sample validation
2. **Set Hard Limits**: Automated risk controls
3. **Monitor Continuously**: Real-time alerting
4. **Document Risks**: Clear risk documentation
5. **Review Regularly**: Periodic risk assessments

## 🔧 Troubleshooting

### Hooks Not Running
- Check file permissions: `chmod +x .claude/hooks/*.sh`
- Verify hook configuration in settings
- Check logs for error messages

### Commands Not Working
- Ensure markdown files are properly formatted
- Check command name matches filename
- Verify no syntax errors in command files

### Agents Not Responding Correctly
- Review agent markdown for clarity
- Ensure proper context is provided
- Check that request matches agent expertise

## 📈 Advanced Usage

### Custom Hooks
Create your own hooks by adding scripts to `.claude/hooks/`:

```bash
#!/bin/bash
# .claude/hooks/custom-hook.sh

# Your custom logic here
echo "Custom hook executed"
```

### Custom Commands
Add new slash commands by creating markdown files in `.claude/commands/`:

```markdown
# /my-command - Description

You are a specialized assistant for...

## Your Role
...
```

### Custom Agents
Define new agents in `.claude/agents/`:

```markdown
# My Custom Agent

## Agent Identity
**Name**: My Agent
**Specialization**: ...
```

## 🆘 Getting Help

1. **Documentation**: Check the docs/ directory
2. **Examples**: See templates/ for working examples
3. **Issues**: Review common issues in troubleshooting
4. **Community**: Share learnings with team

## 📝 License

This configuration is part of the BT backtesting framework project.

---

**Last Updated**: 2026-01-10
**Maintained By**: BT Development Team
**Claude Version**: Compatible with Claude Code CLI
