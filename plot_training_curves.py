"""Backward-compatible root-level launcher for evaluation.plot_training_curves."""
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.plot_training_curves import *
try:
    from evaluation.plot_training_curves import main
except ImportError:
    main = None

if __name__ == "__main__":
    if main is not None:
        main()
