#!/usr/bin/env python3
"""
Test runner script for cad_ig_er_index_backtesting test suite.
Can be run directly or via pytest.
"""

import sys
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if __name__ == '__main__':
    # Get the test directory
    test_dir = Path(__file__).parent
    
    # Run pytest with verbose output
    exit_code = pytest.main([
        str(test_dir),
        '-v',
        '--tb=short',
        '--strict-markers',
        '--disable-warnings'
    ])
    
    sys.exit(exit_code)

