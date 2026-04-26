#!/usr/bin/env bash
set -euo pipefail

printf "[1/4] Git worktree...\n"
git rev-parse --is-inside-work-tree >/dev/null

printf "[2/4] Python...\n"
python -V

printf "[3/4] Import compatibility...\n"
PYTHONPATH=. python - <<'PY'
from data_pipelines.fetch_data import DataPipeline
from cad_ig_er_index_backtesting.core.config import DataConfig
from cad_ig_er_index_backtesting.strategies.base_strategy import BaseStrategy
print("compat_imports_ok")
PY

printf "[4/4] Fast test gate...\n"
PYTHONPATH=. python -m pytest --collect-only -q
