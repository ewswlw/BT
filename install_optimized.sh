#!/bin/bash

# Optimized Installation Script for Financial Backtesting Framework
# Installs dependencies based on usage patterns to minimize startup time

set -e

echo "====================================="
echo "Optimized Installation Script"
echo "====================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    print_error "Poetry not found. Please install Poetry first:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

print_status "Poetry found. Proceeding with optimized installation..."

# Function to prompt for feature selection
select_features() {
    echo ""
    echo "Select features to install (reduces startup time by excluding unused dependencies):"
    echo "You can install additional features later with: poetry install --extras \"feature1 feature2\""
    echo ""

    # Core features (always installed)
    echo "Core dependencies (pandas, numpy, yaml): Always installed"
    
    # Optional feature selection
    read -p "Install Bloomberg data providers? (xbbg, pdblp) [y/N]: " bloomberg
    read -p "Install machine learning libraries? (lightgbm, xgboost, torch) [y/N]: " ml
    read -p "Install visualization libraries? (plotly, matplotlib, dash) [y/N]: " viz
    read -p "Install analysis extensions? (vectorbt, quantstats) [y/N]: " analysis
    read -p "Install Jupyter notebook support? [y/N]: " notebooks
    read -p "Install optimization libraries? (optuna, flaml) [y/N]: " optimization
    read -p "Install all features (maximum functionality)? [y/N]: " all_features

    # Build extras string
    extras=""
    
    if [[ $all_features =~ ^[Yy]$ ]]; then
        extras="all"
        print_warning "Installing all features. This will increase startup time but provide full functionality."
    else
        if [[ $bloomberg =~ ^[Yy]$ ]]; then
            extras="${extras} bloomberg"
        fi
        if [[ $ml =~ ^[Yy]$ ]]; then
            extras="${extras} ml"
        fi
        if [[ $viz =~ ^[Yy]$ ]]; then
            extras="${extras} visualization"
        fi
        if [[ $analysis =~ ^[Yy]$ ]]; then
            extras="${extras} analysis"
        fi
        if [[ $notebooks =~ ^[Yy]$ ]]; then
            extras="${extras} notebooks"
        fi
        if [[ $optimization =~ ^[Yy]$ ]]; then
            extras="${extras} optimization"
        fi
    fi

    echo $extras
}

# Performance mode selection
select_performance_mode() {
    echo ""
    echo "Select performance mode:"
    echo "1) Minimal (core only) - Fastest startup, basic functionality"
    echo "2) Essential (core + analysis) - Good balance of speed and features"
    echo "3) Full (all features) - Maximum features, slower startup"
    echo "4) Custom (interactive selection)"
    echo ""
    read -p "Choose mode [1-4]: " mode

    case $mode in
        1)
            echo ""
            ;;
        2)
            echo "analysis data"
            ;;
        3)
            echo "all"
            ;;
        4)
            select_features
            ;;
        *)
            print_warning "Invalid selection. Using minimal mode."
            echo ""
            ;;
    esac
}

# Backup original pyproject.toml
backup_pyproject() {
    if [ -f "pyproject.toml" ]; then
        print_status "Backing up original pyproject.toml..."
        cp pyproject.toml pyproject.toml.backup
    fi
}

# Install optimized dependencies
install_optimized() {
    local extras="$1"
    
    print_status "Installing optimized dependencies..."
    
    # Use optimized pyproject.toml if available
    if [ -f "pyproject_optimized.toml" ]; then
        print_status "Using optimized dependency configuration..."
        cp pyproject_optimized.toml pyproject.toml
    fi
    
    # Install based on selected features
    if [ -n "$extras" ]; then
        print_status "Installing with extras: $extras"
        poetry install --extras "$extras"
    else
        print_status "Installing minimal dependencies..."
        poetry install --no-dev
    fi
}

# Performance testing
run_performance_test() {
    print_status "Running startup performance test..."
    
    # Test import time
    python3 -c "
import time
start = time.time()
try:
    import pandas as pd
    import numpy as np
    import yaml
    core_time = time.time() - start
    print(f'Core imports: {core_time:.3f}s')
    
    # Test heavy imports if available
    try:
        import_start = time.time()
        import torch
        torch_time = time.time() - import_start
        print(f'Torch import: {torch_time:.3f}s')
    except ImportError:
        print('Torch not installed (good for startup performance)')
    
    try:
        import_start = time.time()
        import plotly
        plotly_time = time.time() - import_start
        print(f'Plotly import: {plotly_time:.3f}s')
    except ImportError:
        print('Plotly not installed (good for startup performance)')
        
    print(f'Total core startup time: {core_time:.3f}s')
    
except Exception as e:
    print(f'Import test failed: {e}')
"
}

# Setup lazy imports
setup_lazy_imports() {
    print_status "Setting up lazy import system..."
    
    # Create lazy import configuration
    if [ ! -f "data_pipelines/lazy_imports.py" ]; then
        print_warning "Lazy imports file not found. Creating basic version..."
        mkdir -p data_pipelines
        cat > data_pipelines/lazy_imports.py << 'EOF'
"""Basic lazy import system for performance optimization."""
import importlib
import logging

logger = logging.getLogger(__name__)

class LazyImporter:
    def __init__(self):
        self._modules = {}
    
    def get_module(self, module_name):
        if module_name not in self._modules:
            try:
                self._modules[module_name] = importlib.import_module(module_name)
            except ImportError:
                logger.warning(f"Module {module_name} not available")
                return None
        return self._modules[module_name]

lazy_importer = LazyImporter()
EOF
    fi
}

# Create performance monitoring
setup_performance_monitoring() {
    print_status "Setting up performance monitoring..."
    
    cat > performance_monitor.py << 'EOF'
#!/usr/bin/env python3
"""Simple performance monitoring for the framework."""

import time
import psutil
import logging

logger = logging.getLogger(__name__)

def monitor_startup():
    """Monitor application startup performance."""
    start_time = time.time()
    
    # Memory before imports
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Time core imports
    import_start = time.time()
    import pandas as pd
    import numpy as np
    core_import_time = time.time() - import_start
    
    # Memory after core imports
    core_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    total_time = time.time() - start_time
    memory_delta = core_memory - initial_memory
    
    print(f"Startup Performance Report:")
    print(f"Core imports: {core_import_time:.3f}s")
    print(f"Memory usage: {core_memory:.1f}MB (+{memory_delta:.1f}MB)")
    print(f"Total startup: {total_time:.3f}s")
    
    return {
        'core_import_time': core_import_time,
        'total_time': total_time,
        'memory_mb': core_memory,
        'memory_delta_mb': memory_delta
    }

if __name__ == "__main__":
    monitor_startup()
EOF

    chmod +x performance_monitor.py
}

# Main installation process
main() {
    print_status "Starting optimized installation process..."
    
    # Backup existing configuration
    backup_pyproject
    
    # Select performance mode
    extras=$(select_performance_mode)
    
    # Setup lazy imports
    setup_lazy_imports
    
    # Install dependencies
    install_optimized "$extras"
    
    # Setup performance monitoring
    setup_performance_monitoring
    
    # Run performance test
    run_performance_test
    
    print_status "Installation completed successfully!"
    echo ""
    echo "Performance Optimization Tips:"
    echo "================================"
    echo "1. Use 'poetry run python performance_monitor.py' to check startup performance"
    echo "2. Install additional features only when needed:"
    echo "   poetry install --extras \"ml visualization\""
    echo "3. Consider using the lazy import system for heavy dependencies"
    echo "4. Monitor memory usage with the built-in performance tools"
    echo ""
    echo "Next steps:"
    echo "- Run 'python performance_optimizer.py' to analyze your code"
    echo "- Use 'python performance_monitor.py' to benchmark improvements"
    echo "- Check 'performance_report.md' for detailed optimization recommendations"
}

# Run main function
main "$@"