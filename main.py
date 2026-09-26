"""Convenience entry point: ``python main.py run VIDEO`` == ``python -m mimir run VIDEO``."""
import sys

from mimir.cli import main

if __name__ == "__main__":
    sys.exit(main())
