"""One-click reproduction script for Table I ($n=20, m=4$) of the paper.

Runs evaluation on the 100 fixed test instances across all dynamic rates
(\\phi \\in {0.10, 0.25, 0.50, 0.75}) for Greedy, MARDAM, MAAM, LiDRL, AMCVN,
and DVNDA using pretrained checkpoints from the ``checkpoints/`` directory.

Usage:
    python reproduce.py [--strict] [--device cuda|cpu]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction.evaluate_table1 import main

if __name__ == "__main__":
    # If no arguments provided, default to strict verification
    if len(sys.argv) == 1:
        sys.argv.append("--strict")
    raise SystemExit(main())
