"""Run strict classical DCVRP baselines on the local Vienna instances."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from experiments.reviewer_study.paper_multivehicle_env import (
    PaperInstance,
    paper_vehicle_count,
)
from experiments.reviewer_study.run_paper_multivehicle_classical import (
    _worker,
    summarize,
)


METHODS = ("Regret insertion", "Tabu Search", "Adaptive LNS", "OR-Tools")
AUDIT_METHODS = METHODS + ("Greedy (same instances)",)
SIZES = (20, 35, 50)
RATES = (0.10, 0.25, 0.50, 0.75)
PAPER_GREEDY = {
    (0.10, 20): (5.57, 0.86, 99.85),
    (0.10, 35): (9.18, 1.13, 99.95),
    (0.10, 50): (12.70, 1.48, 99.95),
    (0.25, 20): (6.09, 0.98, 99.85),
    (0.25, 35): (10.13, 1.30, 100.00),
    (0.25, 50): (13.95, 1.52, 99.95),
    (0.50, 20): (7.11, 1.02, 99.90),
    (0.50, 35): (12.03, 1.47, 99.95),
    (0.50, 50): (16.50, 1.71, 99.95),
    (0.75, 20): (7.77, 0.88, 99.90),
    (0.75, 35): (13.37, 1.43, 99.85),
    (0.75, 50): (18.90, 1.69, 99.90),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=AUDIT_METHODS, default=list(METHODS))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--rates", nargs="+", type=float, default=list(RATES))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--ortools-time-ms", type=int, default=1_000)
    parser.add_argument("--alns-time-ms", type=int, default=1_000)
    parser.add_argument("--algorithm-seed", type=int, default=314159)
    parser.add_argument(
        "--assignment-policy",
        choices=("free", "fixed-greedy"),
        default="fixed-greedy",
        help=(
            "fixed-greedy preserves the common multi-vehicle task allocation "
            "and replaces only each assigned route optimizer"
        ),
    )
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--environment-mode",
        choices=("executed-path-paper", "continuous-single-trip"),
        default="continuous-single-trip",
        help=(
            "The Table-II protocol uses ten synchronized periodic planning "
            "boundaries and one fleet-level depot closure; the newer "
            "mid-edge executed-path protocol is retained as a diagnostic mode"
        ),
    )
    return parser.parse_args()


def load_manifest_index(manifest_dir: Path) -> tuple[dict, dict[tuple[int, float], dict]]:
    metadata = json.loads(
        (manifest_dir / "manifest_index.json").read_text(encoding="utf-8")
    )
    groups = {
        (int(row["customer_count"]), round(float(row["nominal_dynamic_rate"]), 6)): row
        for row in metadata["groups"]
    }
    return metadata, groups


def load_group(manifest_dir: Path, group: dict) -> list[PaperInstance]:
    with np.load(manifest_dir / group["file"]) as data:
        coordinates = data["coordinates"]
        demands = data["demands"]
        service = data["service_minutes"]
        disclosure = data["disclosure_minutes"]
    return [
        PaperInstance(
            coordinates=coordinates[index],
            demands=demands[index],
            service_minutes=service[index],
            disclosure_minutes=disclosure[index],
        )
        for index in range(coordinates.shape[0])
    ]


def add_paper_error(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["paper_greedy_cost"] = [
        PAPER_GREEDY[(float(rate), int(size))][0]
        for rate, size in zip(result.dynamic_rate, result.customer_count)
    ]
    result["paper_greedy_cost_sd"] = [
        PAPER_GREEDY[(float(rate), int(size))][1]
        for rate, size in zip(result.dynamic_rate, result.customer_count)
    ]
    result["paper_greedy_qos_percent"] = [
        PAPER_GREEDY[(float(rate), int(size))][2]
        for rate, size in zip(result.dynamic_rate, result.customer_count)
    ]
    result["error_percent_vs_paper_greedy"] = (
        100.0 * (result.cost_mean / result.paper_greedy_cost - 1.0)
    )
    return result


def acceptance_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, frame in summary.groupby("method", sort=False):
        absolute = frame.error_percent_vs_paper_greedy.abs()
        rows.append(
            {
                "method": method,
                "cells": len(frame),
                "mape_percent": float(absolute.mean()),
                "max_absolute_error_percent": float(absolute.max()),
                "mape_below_3_percent": bool(absolute.mean() < 3.0),
                "max_error_below_5_percent": bool(absolute.max() < 5.0),
                "accepted": bool(absolute.mean() < 3.0 and absolute.max() < 5.0),
            }
        )
    return pd.DataFrame(rows)


def write_table(summary: pd.DataFrame, acceptance: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Vienna real-world strict classical baselines",
        "",
        (
            "Cost and QoS are mean ± sample SD over 100 instances. Time(100) "
            "is the sum of end-to-end per-instance elapsed times, including "
            "all ten interval updates and terminal closure."
        ),
        "",
    ]
    for size in SIZES:
        selected = summary[summary.customer_count == size]
        if selected.empty:
            continue
        lines.extend(
            [
                f"## n={size}, m={paper_vehicle_count(size)}",
                "",
                "| Rate | Method | Cost ↓ | QoS (%) ↑ | Time(100) s ↓ | Paper Greedy | Error % |",
                "|---:|:---|---:|---:|---:|---:|---:|",
            ]
        )
        for rate in RATES:
            for method in METHODS:
                cell = selected[
                    (selected.dynamic_rate == rate) & (selected.method == method)
                ]
                if cell.empty:
                    continue
                row = cell.iloc[0]
                lines.append(
                    f"| {100 * rate:.0f}% | {method} | "
                    f"{row.cost_mean:.2f} ± {row.cost_sd:.2f} | "
                    f"{row.qos_mean_percent:.2f} ± {row.qos_sd_percent:.2f} | "
                    f"{row.time_100_s:.2f} | {row.paper_greedy_cost:.2f} | "
                    f"{row.error_percent_vs_paper_greedy:+.2f}% |"
                )
        lines.append("")
    lines.extend(
        [
            "## Reproduction acceptance",
            "",
            "| Method | MAPE | Maximum absolute error | Accepted |",
            "|:---|---:|---:|:---:|",
        ]
    )
    for row in acceptance.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.mape_percent:.2f}% | "
            f"{row.max_absolute_error_percent:.2f}% | "
            f"{'PASS' if row.accepted else 'FAIL'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_dir = args.manifest_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, groups = load_manifest_index(manifest_dir)
    raw_path = output_dir / "raw_instances.csv"
    existing = pd.read_csv(raw_path) if args.resume and raw_path.exists() else pd.DataFrame()
    rows = [] if existing.empty else existing.to_dict("records")
    completed = (
        set()
        if existing.empty
        else set(
            zip(
                existing.method,
                existing.customer_count.astype(int),
                existing.dynamic_rate.round(6),
                existing.instance_id.astype(int),
            )
        )
    )
    experiment_started = time.perf_counter()

    for method in args.methods:
        for size in args.sizes:
            for rate in args.rates:
                group = groups[(size, round(rate, 6))]
                instances = load_group(manifest_dir, group)
                if args.max_instances is not None:
                    if args.max_instances <= 0:
                        raise ValueError("--max-instances must be positive")
                    instances = instances[: args.max_instances]
                payloads = []
                for instance_id, instance in enumerate(instances, 1):
                    key = (method, size, round(rate, 6), instance_id)
                    if key in completed:
                        continue
                    payloads.append(
                        (
                            instance,
                            method,
                            size,
                            rate,
                            instance_id,
                            args.ortools_time_ms,
                            args.alns_time_ms,
                            args.algorithm_seed,
                            args.environment_mode,
                            args.assignment_policy,
                            float(group["travel_minutes_per_normalized_unit"]),
                        )
                    )
                if not payloads:
                    print(f"skip {method} n={size} rate={rate}", flush=True)
                    continue
                group_started = time.perf_counter()
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = [executor.submit(_worker, payload) for payload in payloads]
                    for count, future in enumerate(as_completed(futures), 1):
                        result = future.result()
                        rows.append(result)
                        completed.add(
                            (
                                result["method"], int(result["customer_count"]),
                                round(float(result["dynamic_rate"]), 6),
                                int(result["instance_id"]),
                            )
                        )
                        if count % 10 == 0 or count == len(payloads):
                            current = [
                                row for row in rows
                                if row["method"] == method
                                and int(row["customer_count"]) == size
                                and math.isclose(float(row["dynamic_rate"]), rate)
                            ]
                            print(
                                f"{method:<16} n={size} rate={100 * rate:.0f}% "
                                f"done={len(current):03d}/100 "
                                f"cost={np.mean([row['cost'] for row in current]):.3f} "
                                f"qos={np.mean([row['qos_percent'] for row in current]):.2f}% "
                                f"wall={time.perf_counter() - group_started:.1f}s",
                                flush=True,
                            )
                            pd.DataFrame(rows).sort_values(
                                ["method", "customer_count", "dynamic_rate", "instance_id"]
                            ).to_csv(raw_path, index=False)

    raw = pd.DataFrame(rows).sort_values(
        ["method", "customer_count", "dynamic_rate", "instance_id"]
    )
    summary = add_paper_error(summarize(raw))
    acceptance = acceptance_table(summary)
    raw.to_csv(raw_path, index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    acceptance.to_csv(output_dir / "acceptance.csv", index=False)
    write_table(summary, acceptance, output_dir / "TABLE.md")
    config = {
        "environment": (
            "paper_multivehicle_v2"
            if args.environment_mode == "continuous-single-trip"
            else "time_driven_executed_path_multivehicle_v9"
        ),
        "environment_mode": args.environment_mode,
        "source_manifest": str(manifest_dir),
        "source_metadata": metadata,
        "methods": list(args.methods),
        "instances_per_cell": 100 if args.max_instances is None else args.max_instances,
        "sizes": list(args.sizes),
        "rates": list(args.rates),
        "assignment_policy": args.assignment_policy,
        "short_customer_fallback": None,
        "tabu_implementation": (
            "standalone deterministic Python tabu search with swap, relocate, "
            "and 2-opt neighbourhoods; it does not call OR-Tools"
        ),
        "tabu_time_limit_ms_per_active_static_update": args.ortools_time_ms,
        "ortools_failure_policy": "raise error; never substitute another method",
        "ortools_metaheuristic": "GUIDED_LOCAL_SEARCH",
        "ortools_time_limit_ms_per_active_static_update": args.ortools_time_ms,
        "alns_time_limit_ms_per_active_static_update": args.alns_time_ms,
        "workers": args.workers,
        "timing": "sum of 100 end-to-end per-instance elapsed times",
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "ortools_version": package_version("ortools"),
        "alns_version": package_version("alns"),
        "experiment_wall_time_s": time.perf_counter() - experiment_started,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(acceptance.to_string(index=False), flush=True)
    print(f"completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
