# Repository Setup + Agentic Coding Optimization

Date: 2026-04-26 (UTC)

## Current state
- Repository is already present at `/workspace/BT` and is a valid Git working tree.
- No remote URL is configured in this environment (`git remote -v` produced no entries).

## What was optimized for agentic coding
1. **Import-path compatibility shims**
   - Added a `core/` compatibility package that aliases `cad_ig_er_index_backtesting.core` under the legacy `core.*` namespace.
   - Added a `data_pipelines/` compatibility package that exposes `DataPipeline` from the legacy file path (`market data pipeline/fetch_data.py`).

2. **Stable pytest defaults for bots/agents**
   - Added `[tool.pytest.ini_options]` in `pyproject.toml` with `pythonpath = ["."]` so agents do not need to remember `PYTHONPATH=. ...`.

3. **Single-command smoke check for automation**
   - Added executable script: `scripts/agent_smoke_check.sh`
   - Script verifies:
     - git worktree is valid
     - Python runtime is available
     - critical imports succeed
     - focused fast tests run

## Recommended agent workflow
```bash
./scripts/agent_smoke_check.sh
PYTHONPATH=. python -m pytest -q
```

## Latest observed full-test status
- `PYTHONPATH=. python -m pytest -q`
- Result: collection now succeeds (previous `ModuleNotFoundError` issues removed), but there are functional test failures still to fix in core logic (config serialization, metrics expectations, validator behavior, and strategy/report edge cases).
