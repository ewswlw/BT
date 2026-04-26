"""Compatibility wrapper for the historical `data_pipelines.fetch_data` path."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_LEGACY_FILE = (
    Path(__file__).resolve().parent.parent
    / "market data pipeline"
    / "fetch_data.py"
)

_spec = spec_from_file_location("_legacy_fetch_data", _LEGACY_FILE)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load legacy fetch_data module from {_LEGACY_FILE}")

_module = module_from_spec(_spec)

try:
    _spec.loader.exec_module(_module)
    DataPipeline = _module.DataPipeline
except BaseException as exc:  # pragma: no cover - compatibility fallback
    class DataPipeline:  # type: ignore[no-redef]
        """Fallback stub when optional Bloomberg deps are unavailable."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Legacy DataPipeline dependencies are unavailable in this environment"
            ) from exc

__all__ = ["DataPipeline"]
