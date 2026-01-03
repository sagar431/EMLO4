"""Pytest configuration and fixtures."""
import sys
from pathlib import Path

# Add the project root to the path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))
