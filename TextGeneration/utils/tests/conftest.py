"""Test configuration for utils tests."""

import sys
from pathlib import Path

# Add project root to Python path to ensure tests can import modules correctly
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
