"""
Lazy import manager for performance optimization.
Defers loading of heavy libraries until they're actually needed.
"""

import importlib
import sys
from typing import Any, Dict, Optional, Callable
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class LazyImporter:
    """Manages lazy loading of expensive imports."""
    
    def __init__(self):
        self._modules: Dict[str, Any] = {}
        self._load_times: Dict[str, float] = {}
        
    def register_lazy_import(self, 
                           module_name: str, 
                           install_name: Optional[str] = None,
                           fallback: Optional[Callable] = None):
        """Register a module for lazy importing."""
        if module_name not in self._modules:
            self._modules[module_name] = LazyModule(
                module_name, install_name, fallback
            )
        return self._modules[module_name]
    
    def get_module(self, module_name: str) -> Any:
        """Get a lazily imported module."""
        if module_name in self._modules:
            return self._modules[module_name].get()
        else:
            # Direct import as fallback
            return importlib.import_module(module_name)
    
    def preload_critical(self, modules: list):
        """Preload critical modules for better startup performance."""
        for module_name in modules:
            if module_name in self._modules:
                self._modules[module_name].get()

class LazyModule:
    """Wrapper for a lazily imported module."""
    
    def __init__(self, 
                 module_name: str, 
                 install_name: Optional[str] = None,
                 fallback: Optional[Callable] = None):
        self.module_name = module_name
        self.install_name = install_name or module_name
        self.fallback = fallback
        self._module = None
        self._attempted = False
        
    def get(self) -> Any:
        """Get the actual module, importing if necessary."""
        if self._module is not None:
            return self._module
            
        if self._attempted:
            if self.fallback:
                return self.fallback()
            else:
                raise ImportError(f"Module {self.module_name} not available")
        
        self._attempted = True
        
        try:
            import time
            start_time = time.time()
            self._module = importlib.import_module(self.module_name)
            load_time = time.time() - start_time
            logger.debug(f"Loaded {self.module_name} in {load_time:.3f}s")
            return self._module
            
        except ImportError as e:
            logger.warning(f"Failed to import {self.module_name}: {e}")
            if self.fallback:
                self._module = self.fallback()
                return self._module
            else:
                raise ImportError(
                    f"Module {self.module_name} not available. "
                    f"Install with: pip install {self.install_name}"
                )
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the actual module."""
        return getattr(self.get(), name)

# Global lazy importer instance
lazy_importer = LazyImporter()

# Register heavy modules for lazy loading
def setup_lazy_imports():
    """Setup lazy imports for heavy modules."""
    
    # ML Libraries - Heavy
    lazy_importer.register_lazy_import(
        'torch', 'torch', 
        fallback=lambda: None
    )
    
    lazy_importer.register_lazy_import(
        'lightgbm', 'lightgbm',
        fallback=lambda: None
    )
    
    lazy_importer.register_lazy_import(
        'xgboost', 'xgboost',
        fallback=lambda: None
    )
    
    lazy_importer.register_lazy_import(
        'catboost', 'catboost',
        fallback=lambda: None
    )
    
    # Visualization - Heavy
    lazy_importer.register_lazy_import(
        'plotly', 'plotly',
        fallback=lambda: _create_mock_plotly()
    )
    
    lazy_importer.register_lazy_import(
        'matplotlib.pyplot', 'matplotlib',
        fallback=lambda: _create_mock_matplotlib()
    )
    
    lazy_importer.register_lazy_import(
        'seaborn', 'seaborn',
        fallback=lambda: None
    )
    
    lazy_importer.register_lazy_import(
        'dash', 'dash',
        fallback=lambda: None
    )
    
    # Analysis libraries
    lazy_importer.register_lazy_import(
        'vectorbt', 'vectorbt',
        fallback=lambda: _create_mock_vectorbt()
    )
    
    lazy_importer.register_lazy_import(
        'quantstats', 'quantstats',
        fallback=lambda: _create_mock_quantstats()
    )
    
    # Optimization libraries
    lazy_importer.register_lazy_import(
        'optuna', 'optuna',
        fallback=lambda: None
    )
    
    # Feature engineering
    lazy_importer.register_lazy_import(
        'tsfresh', 'tsfresh',
        fallback=lambda: None
    )

def _create_mock_plotly():
    """Create mock plotly for fallback."""
    class MockPlotly:
        def __init__(self):
            self.graph_objects = self
            
        def Figure(self, *args, **kwargs):
            logger.warning("Plotly not available, skipping visualization")
            return self
            
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    return MockPlotly()

def _create_mock_matplotlib():
    """Create mock matplotlib for fallback."""
    class MockMatplotlib:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
            
        def show(self):
            logger.warning("Matplotlib not available, skipping plot")
    
    return MockMatplotlib()

def _create_mock_vectorbt():
    """Create mock vectorbt for fallback."""
    class MockVectorBT:
        Portfolio = None
        settings = type('Settings', (), {'array_wrapper': {}})()
        
    return MockVectorBT()

def _create_mock_quantstats():
    """Create mock quantstats for fallback."""
    class MockQuantStats:
        stats = type('Stats', (), {})()
        reports = type('Reports', (), {})()
        
    return MockQuantStats()

def require_module(module_name: str):
    """Decorator to ensure a module is available before function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                lazy_importer.get_module(module_name)
                return func(*args, **kwargs)
            except ImportError as e:
                logger.error(f"Function {func.__name__} requires {module_name}: {e}")
                raise
        return wrapper
    return decorator

# Convenience functions for common imports
def get_torch():
    """Get torch module if available."""
    return lazy_importer.get_module('torch')

def get_plotly():
    """Get plotly module if available."""
    return lazy_importer.get_module('plotly')

def get_vectorbt():
    """Get vectorbt module if available."""
    return lazy_importer.get_module('vectorbt')

def get_quantstats():
    """Get quantstats module if available."""
    return lazy_importer.get_module('quantstats')

# Initialize lazy imports
setup_lazy_imports()