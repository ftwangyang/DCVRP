"""Train the five Table I neural methods one after another on one GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("DVNDA", "AMCVN", "LiDRL", "MAAM", "MARDAM")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--customer-count", type=int, default=20)
    parser.add_argument("-m", "--vehicle-count", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-epoch", type=int, default=20)
    parser.add_argument("--early-stop-mae", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("checkpoints/table1_5knob_seq"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n = args.customer_count
    m = args.vehicle_count if args.vehicle_count is not None else max(1, round(n / 5))
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / f"n{n}_s{args.seed}_summary.jsonl"

    for method in args.methods:
        out_dir = args.output_root / f"n{n}_s{args.seed}" / method
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "train.py"),
            "--method",
            method,
            "-n",
            str(n),
            "-m",
            str(m),
            "--epochs",
            str(args.epochs),
            "--steps-per-epoch",
            str(args.steps_per_epoch),
            "--seed",
            str(args.seed),
            "--rollouts",
            "3",
            "--device",
            args.device,
            "--output-dir",
            str(out_dir),
            "--early-stop-epoch",
            str(args.early_stop_epoch),
            "--early-stop-mae",
            str(args.early_stop_mae),
        ]
        print("=" * 76, flush=True)
        print(f"METHOD {method}  n={n}  output={out_dir}", flush=True)
        print(" ".join(cmd), flush=True)
        started = datetime.now(timezone.utc).isoformat()
        code = subprocess.call(cmd, cwd=str(ROOT))
        suffix = "" if n == 20 else f"_n{n}"
        table1 = out_dir / f"{method}{suffix}_table1_best.pt"
        record = {
            "method": method,
            "seed": args.seed,
            "customer_count": n,
            "exit_code": code,
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(out_dir),
        }
        if table1.exists():
            import torch

            chk = torch.load(table1, map_location="cpu", weights_only=False)
            record.update(
                {
                    "best_epoch": chk.get("epoch"),
                    "table1_mae": chk.get("table1_mae"),
                    "table1_max_gap": chk.get("table1_max_gap"),
                    "table1_gap_by_rate": chk.get("table1_gap_by_rate"),
                    "table1_all_within": chk.get("table1_all_within"),
                    "val_distance_by_rate": chk.get("val_distance_by_rate"),
                    "passed_gate": bool(chk.get("table1_all_within"))
                    if "table1_all_within" in chk
                    else (
                        chk.get("table1_max_gap") is not None
                        and float(chk["table1_max_gap"]) <= args.early_stop_mae
                    ),
                }
            )
        else:
            record["passed_gate"] = False
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if code != 0:
            print(f"METHOD {method} failed with exit code {code}", flush=True)
            sys.exit(code)


if __name__ == "__main__":
    main()
