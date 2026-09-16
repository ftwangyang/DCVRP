"""Fine-tune DVNDA on n=20/35/50 under the paper-aligned environment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCALES = (
    (20, 4, "checkpoints/DVNDA.pt"),
    (35, 7, "checkpoints/DVNDA_n35.pt"),
    (50, 10, "checkpoints/DVNDA_n50.pt"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain DVNDA at all Table I scales.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/paper_align"))
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--scales", nargs="+", type=int, default=[20, 35, 50])
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for n, m, init in SCALES:
        if n not in args.scales:
            continue
        cmd = [
            sys.executable,
            str(ROOT / "train.py"),
            "--method",
            "DVNDA",
            "-n",
            str(n),
            "-m",
            str(m),
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--output-dir",
            str(args.output_dir),
            "--rollouts",
            "3",
        ]
        init_path = ROOT / init
        if not args.from_scratch and init_path.exists():
            cmd.extend(["--init-from", str(init_path)])
        print(" ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()
