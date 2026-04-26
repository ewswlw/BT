"""Compatibility namespace for legacy absolute imports.

This package aliases ``cad_ig_er_index_backtesting.core`` to ``core`` so that
existing strategy modules and tests that import ``core.*`` continue to work.
"""

from importlib import import_module
import sys

_BASE = "cad_ig_er_index_backtesting.core"

# Expose commonly used submodules under the legacy `core.*` namespace.
for _submodule in [
    "config",
    "data_loader",
    "feature_engineering",
    "metrics",
    "portfolio",
    "reporting",
    "validation",
]:
    sys.modules[f"core.{_submodule}"] = import_module(f"{_BASE}.{_submodule}")

# Re-export package-level symbols.
from cad_ig_er_index_backtesting.core import *  # noqa: F401,F403
