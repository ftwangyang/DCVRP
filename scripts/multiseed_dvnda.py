"""Multi-seed DVNDA runs with a 20-epoch / per-cell +/-5% Table I gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (42, 7, 2024, 3407, 20260821)
INIT = {
    20: ROOT / "checkpoints" / "DVNDA.pt",
    35: ROOT / "checkpoints" / "DVNDA_n35.pt",
    50: ROOT / "checkpoints" / "DVNDA_n50.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-seed DVNDA Table I search.")
    parser.add_argument("-n", "--customer-count", type=int, default=20)
    parser.add_argument("-m", "--vehicle-count", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-epoch", type=int, default=20)
    parser.add_argument("--early-stop-mae", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        default=True,
        help="Train from random init (default). Fine-tune old checkpoints fail the 8% gate.",
    )
    parser.add_argument(
        "--init-from-checkpoint",
        action="store_true",
        help="Fine-tune official checkpoints instead of training from scratch.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("checkpoints/multiseed"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n = args.customer_count
    m = args.vehicle_count if args.vehicle_count is not None else max(1, round(n / 5))
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / f"n{n}_summary.jsonl"

    for seed in args.seeds:
        out_dir = args.output_root / f"n{n}_s{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
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
            "--seed",
            str(seed),
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
        init = INIT.get(n)
        use_init = args.init_from_checkpoint or (not args.from_scratch)
        if use_init and init is not None and init.exists():
            cmd.extend(["--init-from", str(init)])
        else:
            print("Training from scratch (no --init-from).", flush=True)
        print("=" * 76, flush=True)
        print(f"SEED {seed}  n={n}  output={out_dir}", flush=True)
        print(" ".join(cmd), flush=True)
        started = datetime.now(timezone.utc).isoformat()
        code = subprocess.call(cmd, cwd=str(ROOT))
        table1 = out_dir / "DVNDA_table1_best.pt"
        if n != 20:
            table1 = out_dir / f"DVNDA_n{n}_table1_best.pt"
        record = {
            "seed": seed,
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
                    "val_distance": chk.get("val_distance"),
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
            print(f"SEED {seed} failed with exit code {code}", flush=True)


if __name__ == "__main__":
    main()
