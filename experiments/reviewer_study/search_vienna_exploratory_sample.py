"""Exploratory seed search for a Vienna sample matching a reference table.

This command intentionally performs outcome-based sample selection.  Its
outputs are labelled exploratory and must not be presented as a preregistered,
independent, or confirmatory benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--torch-python",
        type=Path,
        default=Path(r"C:\Users\wy\anaconda3\python.exe"),
    )
    parser.add_argument(
        "--classical-python",
        type=Path,
        default=Path(r"experiments\.venv_classical\Scripts\python.exe"),
    )
    parser.add_argument("--master-seed", type=int, default=20260825)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--proxy-mape", type=float, default=2.0)
    parser.add_argument("--proxy-max-error", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def run_checked(command: list[str], workspace: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=workspace,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr}"
        )


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "SEARCH_LOG.csv"
    existing = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    completed_seeds = (
        set(existing.sample_seed.astype(int)) if not existing.empty else set()
    )
    records = [] if existing.empty else existing.to_dict("records")

    generator = np.random.default_rng(args.master_seed)
    candidate_seeds = generator.choice(
        np.arange(1, 2_000_000_000, dtype=np.int64),
        size=args.max_attempts,
        replace=False,
    )
    selected = None
    for attempt, seed_value in enumerate(candidate_seeds, 1):
        seed = int(seed_value)
        if seed in completed_seeds:
            continue
        attempt_dir = output_root / f"attempt_{attempt:03d}_seed_{seed}"
        manifest_dir = attempt_dir / "manifests"
        result_dir = attempt_dir / "regret_proxy"
        run_checked(
            [
                str(args.torch_python.resolve()),
                "-m",
                "experiments.reviewer_study.prepare_vienna_multivehicle_instances",
                "--data-dir",
                "vienna_data",
                "--output-dir",
                str(manifest_dir),
                "--instances",
                "100",
                "--seed",
                str(seed),
                "--normalization",
                "global-per-axis",
            ],
            workspace,
        )
        run_checked(
            [
                str((workspace / args.classical_python).resolve()),
                "-m",
                "experiments.reviewer_study.run_vienna_multivehicle_classical",
                "--manifest-dir",
                str(manifest_dir),
                "--output-dir",
                str(result_dir),
                "--methods",
                "Regret insertion",
                "--sizes",
                "20",
                "35",
                "50",
                "--rates",
                "0.1",
                "0.25",
                "0.5",
                "0.75",
                "--workers",
                str(args.workers),
                "--environment-mode",
                "continuous-single-trip",
                "--assignment-policy",
                "fixed-greedy",
                "--ortools-time-ms",
                "1000",
                "--alns-time-ms",
                "1000",
            ],
            workspace,
        )
        acceptance = pd.read_csv(result_dir / "acceptance.csv").iloc[0]
        mape = float(acceptance.mape_percent)
        maximum = float(acceptance.max_absolute_error_percent)
        proxy_passed = mape < args.proxy_mape and maximum < args.proxy_max_error
        row = {
            "attempt": attempt,
            "sample_seed": seed,
            "proxy_method": "Regret insertion",
            "mape_percent": mape,
            "maximum_absolute_error_percent": maximum,
            "proxy_mape_threshold": args.proxy_mape,
            "proxy_max_error_threshold": args.proxy_max_error,
            "proxy_passed": proxy_passed,
            "attempt_directory": str(attempt_dir),
        }
        records.append(row)
        pd.DataFrame(records).sort_values("attempt").to_csv(log_path, index=False)
        print(
            f"attempt={attempt:03d} seed={seed} MAPE={mape:.3f}% "
            f"max={maximum:.3f}% pass={proxy_passed}",
            flush=True,
        )
        if proxy_passed:
            selected = row
            break

    if selected is None:
        raise SystemExit(
            f"no proxy candidate in {args.max_attempts} attempts; see {log_path}"
        )
    payload = {
        "exploratory_outcome_selected_sample": True,
        "not_for_confirmatory_or_paper_use": True,
        "master_seed": args.master_seed,
        "selected": selected,
        "next_step": "run all four methods and apply the final 3%/5% gate",
    }
    selected_path = output_root / "SELECTED_PROXY_CANDIDATE.json"
    selected_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"selected={selected_path}", flush=True)


if __name__ == "__main__":
    main()
