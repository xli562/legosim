#!/usr/bin/env python3
"""Compatibility wrapper for the refactored chiplet estimator package."""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from chiplet_estimator.main import main


if __name__ == "__main__":
    raise SystemExit(main())
