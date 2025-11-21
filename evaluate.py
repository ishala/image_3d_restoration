"""
Main evaluation script - wrapper for src/evaluation/evaluate.py
Easier access from project root
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run evaluation
from evaluation.evaluate import main

if __name__ == "__main__":
    main()
