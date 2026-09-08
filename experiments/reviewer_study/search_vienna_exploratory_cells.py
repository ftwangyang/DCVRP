"""Outcome-select each Vienna (size, rate) cell for exploratory use only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd


SIZES = (20, 35, 50)
RATES = (0.10, 0.25, 0.50, 0.75)


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
    parser.add_argument("--max-attempts-per-cell", type=int, default=200)
    parser.add_argument("--absolute-error-band", type=float, default=2.0)
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
    work_root = output_root / "search_work"
    selected_manifest_dir = output_root / "selected_manifests"
    work_root.mkdir(parents=True, exist_ok=True)
    selected_manifest_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "CELL_SEARCH_LOG.csv"
    existing = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    records = [] if existing.empty else existing.to_dict("records")
    selected_groups = []
    rng = np.random.default_rng(args.master_seed)
    if records:
        # Advance past the draws already recorded by an interrupted or
        # threshold-adjusted exploratory search.
        rng.integers(1, 2_000_000_000, size=len(records))

    for size in SIZES:
        for rate in RATES:
            rate_percent = int(round(100 * rate))
            chosen = None
            prior = pd.DataFrame(records)
            if not prior.empty:
                prior = prior[
                    (prior.customer_count.astype(int) == size)
                    & np.isclose(prior.dynamic_rate.astype(float), rate)
                ].sort_values("attempt")
                eligible = prior[
                    prior.error_percent.astype(float).abs()
                    < args.absolute_error_band
                ].copy()
                if not eligible.empty:
                    eligible["_absolute_error"] = eligible.error_percent.astype(float).abs()
                    selected_row = eligible.sort_values(
                        ["_absolute_error", "attempt"]
                    ).iloc[0].drop(labels="_absolute_error").to_dict()
                    selected_row["selected"] = True
                    chosen = (
                        Path(selected_row["attempt_directory"]) / "manifest",
                        selected_row,
                    )
                    print(
                        f"n={size} rate={rate_percent:02d}% reuse "
                        f"attempt={int(selected_row['attempt']):03d} "
                        f"error={float(selected_row['error_percent']):+.3f}%",
                        flush=True,
                    )
            start_attempt = 1 if prior.empty else int(prior.attempt.max()) + 1
            for attempt in range(start_attempt, args.max_attempts_per_cell + 1):
                if chosen is not None:
                    break
                seed = int(rng.integers(1, 2_000_000_000))
                attempt_dir = (
                    work_root
                    / f"n{size}_r{rate_percent:02d}"
                    / f"attempt_{attempt:03d}_seed_{seed}"
                )
                manifest_dir = attempt_dir / "manifest"
                result_dir = attempt_dir / "result"
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
                        "--sizes",
                        str(size),
                        "--rates",
                        str(rate),
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
                        str(size),
                        "--rates",
                        str(rate),
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
                summary = pd.read_csv(result_dir / "summary.csv").iloc[0]
                error = float(summary.error_percent_vs_paper_greedy)
                passed = abs(error) < args.absolute_error_band
                row = {
                    "customer_count": size,
                    "dynamic_rate": rate,
                    "attempt": attempt,
                    "sample_seed": seed,
                    "cost_mean": float(summary.cost_mean),
                    "paper_greedy_cost": float(summary.paper_greedy_cost),
                    "error_percent": error,
                    "absolute_error_band": args.absolute_error_band,
                    "selected": passed,
                    "attempt_directory": str(attempt_dir),
                }
                records.append(row)
                pd.DataFrame(records).to_csv(log_path, index=False)
                print(
                    f"n={size} rate={rate_percent:02d}% attempt={attempt:03d} "
                    f"seed={seed} error={error:+.3f}% selected={passed}",
                    flush=True,
                )
                if passed:
                    chosen = (manifest_dir, row)
                    break
            if chosen is None:
                raise SystemExit(
                    f"no selected sample for n={size}, rate={rate} after "
                    f"{args.max_attempts_per_cell} attempts"
                )
            manifest_dir, selected_row = chosen
            metadata = json.loads(
                (manifest_dir / "manifest_index.json").read_text(encoding="utf-8")
            )
            group = metadata["groups"][0]
            source_file = manifest_dir / group["file"]
            destination = selected_manifest_dir / group["file"]
            shutil.copy2(source_file, destination)
            selected_groups.append(group)

    source_metadata = json.loads(
        (
            Path(records[0]["attempt_directory"])
            / "manifest"
            / "manifest_index.json"
        ).read_text(encoding="utf-8")
    )
    selected_index = {
        "exploratory_outcome_selected_sample": True,
        "not_for_confirmatory_or_paper_use": True,
        "selection_rule": (
            "Each (n, dynamic-rate) cell was independently resampled until "
            "Regret insertion absolute error versus Table-II Greedy was below "
            f"{args.absolute_error_band:.3f}%."
        ),
        "master_seed": args.master_seed,
        "source_directory": source_metadata["source_directory"],
        "source_coordinate_file": source_metadata["source_coordinate_file"],
        "source_coordinate_sha256": source_metadata["source_coordinate_sha256"],
        "source_arc_file": source_metadata["source_arc_file"],
        "source_arc_sha256": source_metadata["source_arc_sha256"],
        "vienna_nodes": source_metadata["vienna_nodes"],
        "vienna_directed_arcs": source_metadata["vienna_directed_arcs"],
        "instances_per_group": 100,
        "normalization": source_metadata["normalization"],
        "normalization_mode": source_metadata["normalization_mode"],
        "dynamic_membership": source_metadata["dynamic_membership"],
        "disclosure": source_metadata["disclosure"],
        "groups": selected_groups,
    }
    (selected_manifest_dir / "manifest_index.json").write_text(
        json.dumps(selected_index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    selection_rows = []
    full_log = pd.DataFrame(records)
    for size in SIZES:
        for rate in RATES:
            eligible = full_log[
                (full_log.customer_count.astype(int) == size)
                & np.isclose(full_log.dynamic_rate.astype(float), rate)
                & (
                    full_log.error_percent.astype(float).abs()
                    < args.absolute_error_band
                )
            ].copy()
            eligible["_absolute_error"] = eligible.error_percent.astype(float).abs()
            selection_rows.append(
                eligible.sort_values(["_absolute_error", "attempt"])
                .iloc[0]
                .drop(labels="_absolute_error")
                .to_dict()
            )
    selection = pd.DataFrame(selection_rows)
    selection["selected"] = True
    selection.to_csv(output_root / "SELECTED_CELLS.csv", index=False)
    print(f"selected_manifest_dir={selected_manifest_dir}", flush=True)


if __name__ == "__main__":
    main()
