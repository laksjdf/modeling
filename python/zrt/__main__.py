"""Allow running as: python -m python.zrt --model-id <MODEL> [options]"""
import sys
import os
# Ensure `python/` is on sys.path so that `from zrt.*` imports work
# (used throughout the training module).
_python_dir = os.path.join(os.path.dirname(__file__), "..")
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

from python.zrt.cli import main

main()
