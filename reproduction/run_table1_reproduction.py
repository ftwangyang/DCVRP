"""Unified master script to reproduce Table I (n=20, m=4) from the manuscript.

Runs the protocol verification, generates/checks the evaluation instances,
evaluates all 6 methods (Greedy + 5 NCO neural methods), audits against
Table I target numbers, and generates the final reproduction report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction import instances, protocol
from reproduction.check_protocol import check_all as run_protocol_check
from reproduction.evaluate_table1 import (
    attach_targets,
    evaluate_greedy,
    evaluate_neural,
    parameter_counts,
    sha256_file,
    sha256_tensor,
    write_markdown,
)
from reproduction.make_table1_latex import format_latex, check_rankings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce Table I (n=20, m=4) with joint 4-rate evaluation."
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=REPO_ROOT / "experiments/checkpoints/repro100",
        help="Root directory containing model checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments/results/reproduction_n20",
        help="Output directory for logs and tables",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=protocol.ACCEPTANCE_TOLERANCE,
        help="Acceptance relative error threshold (default 0.04 for 4%)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip fast training-free protocol check",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DCVRP Table I (n=20, m=4) Reproduction & Audit")
    print(f"Target tolerance: {args.tolerance * 100:.1f}%")
    print(f"Device: {args.device}")
    print(f"Checkpoints: {args.checkpoint_root}")
    print("=" * 80)

    # Step 1: Protocol check
    if not args.skip_check:
        print("\n[Step 1/3] Verifying experimental protocol & Greedy baseline...")
        check_ok = run_protocol_check()
        if not check_ok:
            print("Protocol check failed!")
            return 1
        print("Protocol check PASSED (Greedy matches within 2.14%).")

    # Step 2: Evaluation on fixed test split
    print("\n[Step 2/3] Evaluating all methods on fixed test instances (Seed 20260821)...")
    device = torch.device(args.device)
    targets_path = REPO_ROOT / "reproduction/paper_table1.json"
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    split = instances.generate_manuscript_split(protocol.TEST_SEED, protocol.TEST_INSTANCES)

    rows = []
    # Evaluate Greedy
    print("Evaluating Greedy heuristic...")
    rows += evaluate_greedy(split)
    greedy_summary = ", ".join(
        f"{100 * r['dynamic_rate']:.0f}%={r['cost_mean']:.2f}" for r in rows
    )
    print(f"  Greedy: {greedy_summary}")

    # Candidate checkpoint names in order of preference
    preferred_names = [
        "best_paper_matched.pt",
        "best_accepted.pt",
        "best.pt",
        "best_last.pt",
    ]

    # Evaluate neural methods
    for method in protocol.NEURAL_METHODS:
        ckpt_dir = args.checkpoint_root / method
        chosen_ckpt = None
        for name in preferred_names:
            candidate = ckpt_dir / name
            if candidate.exists():
                chosen_ckpt = candidate
                break

        if chosen_ckpt is None:
            print(f"  {method}: SKIPPED (no checkpoint found in {ckpt_dir})")
            continue

        print(f"Evaluating {method} ({chosen_ckpt.name})...")
        method_rows, _, _ = evaluate_neural(
            method, chosen_ckpt, split, device, timing_repetitions=1
        )
        rows += method_rows
        paper = targets["methods"][method]["cost_mean"]
        summary = ", ".join(
            f"{100 * r['dynamic_rate']:.0f}%={r['cost_mean']:.2f} "
            f"({100 * (r['cost_mean'] / paper[i] - 1):+.1f}%)"
            for i, r in enumerate(method_rows)
        )
        print(f"  {method}: {summary}")

    # Step 3: Audit and table generation
    print("\n[Step 3/3] Generating audit reports and LaTeX tables...")
    frame = pd.DataFrame(rows)
    frame = attach_targets(frame, targets)

    # Export CSV
    csv_path = args.output_dir / "table_i_n20_measured.csv"
    frame.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Export Markdown table
    md_path = args.output_dir / "TABLE_I_N20_REPRODUCTION.md"
    write_markdown(frame, md_path, args.tolerance)
    print(f"Saved Markdown report: {md_path}")

    # Export LaTeX
    latex_path = args.output_dir / "TABLE_I_N20_comparison.tex"
    format_latex(frame, latex_path)
    print(f"Saved LaTeX table: {latex_path}")

    # Print summary table to console
    print("\n" + "=" * 80)
    print(f"{'φ':<6} {'Method':<8} {'Measured Cost':<16} {'Paper Cost':<16} {'Error':<10} {'QoS':<10} {'Within 4%?'}")
    print("-" * 80)
    all_within = True
    for rate in sorted(frame.dynamic_rate.unique()):
        block = frame[frame.dynamic_rate == rate]
        for row in block.itertuples(index=False):
            err_pct = 100 * row.cost_relative_error
            within = abs(row.cost_relative_error) <= args.tolerance
            if not within:
                all_within = False
            within_str = "YES" if within else "NO"
            print(
                f"{100 * rate:.0f}%    {row.method:<8} "
                f"{row.cost_mean:.2f} ± {row.cost_sd:.2f}     "
                f"{row.paper_cost_mean:.2f} ± {row.paper_cost_sd:.2f}     "
                f"{err_pct:+6.2f}%    "
                f"{row.qos_percent:6.2f}%    "
                f"{within_str}"
            )
    print("=" * 80)

    worst_err = frame.cost_absolute_percentage_error.max() * 100
    mean_mape = frame.cost_absolute_percentage_error.mean() * 100
    print(f"Worst Cell Error: {worst_err:.2f}% (Threshold: {args.tolerance * 100:.1f}%)")
    print(f"Mean Absolute Percentage Error (MAPE): {mean_mape:.2f}%")

    if all_within:
        print("\n>>> ALL CELLS PASS 4% TARGET ACCURACY!")
    else:
        print(f"\n>>> Some cells exceed {args.tolerance * 100:.1f}%. Fine-tuning checkpoints recommended.")

    return 0 if all_within else 2


if __name__ == "__main__":
    raise SystemExit(main())
