"""
Pytest configuration ensuring project root is in sys.path.
"""

import sys
import os

# Insert project root into sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
