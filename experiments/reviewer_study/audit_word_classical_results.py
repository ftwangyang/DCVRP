"""Audit and format the Word-spec classical baseline rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .run_paper_multivehicle_classical import PAPER_GREEDY


METHODS = (
    "Regret insertion",
    "Tabu Search",
    "Adaptive LNS",
    "OR-Tools",
)
SIZES = (20, 35, 50)
RATES = (0.10, 0.25, 0.50, 0.75)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    raw = pd.read_csv(result_dir / "raw_instances.csv")
    summary = pd.read_csv(result_dir / "summary.csv")

    key = ["method", "customer_count", "dynamic_rate", "instance_id"]
    groups = raw.groupby(key[:-1]).size()
    checks = {
        "row_count": int(len(raw)),
        "expected_row_count": 5 * 3 * 4 * 100,
        "group_count_min": int(groups.min()),
        "group_count_max": int(groups.max()),
        "duplicate_keys": int(raw.duplicated(key).sum()),
        "environment_tags": sorted(raw.environment.unique().tolist()),
        "planning_calls": sorted(raw.planning_calls.unique().astype(int).tolist()),
        "all_finite": bool(
            np.isfinite(raw[["cost", "qos_percent", "solve_time_s"]]).all().all()
        ),
        "qos_min": float(raw.qos_percent.min()),
        "qos_max": float(raw.qos_percent.max()),
        "all_fleet_sizes_used": bool(
            (raw.vehicles_used == raw.vehicle_count).all()
        ),
    }
    checks["passed"] = bool(
        checks["row_count"] == checks["expected_row_count"]
        and checks["group_count_min"] == 100
        and checks["group_count_max"] == 100
        and checks["duplicate_keys"] == 0
        and checks["environment_tags"]
        == ["word_periodic_static_multivehicle_v5"]
        and checks["planning_calls"] == [11]
        and checks["all_finite"]
        and checks["all_fleet_sizes_used"]
    )
    if not checks["passed"]:
        raise RuntimeError(f"result audit failed: {checks}")
    (result_dir / "audit.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    indexed = summary.set_index(["method", "customer_count", "dynamic_rate"])
    lines = [
        "# Word-spec rerun with Greedy references",
        "",
        (
            "Every classical cell uses the same 100 instances and the same "
            "Word-spec environment. Paper Greedy is copied only as an external "
            "legacy reference and was not rerun under this environment."
        ),
        "",
    ]
    for size in SIZES:
        lines.extend(
            [
                f"## n={size}, m={size // 5}",
                "",
                "| Rate | Method | Cost (mean ± SD) | QoS (%) | Time(100) s | Δ vs same-env Greedy | Δ vs paper Greedy |",
                "|---:|:---|---:|---:|---:|---:|---:|",
            ]
        )
        for rate in RATES:
            same = indexed.loc[("Greedy (same instances)", size, rate)]
            paper_cost = PAPER_GREEDY[(rate, size)][0]
            lines.append(
                f"| {rate * 100:.0f}% | Greedy (same environment) | "
                f"{same.cost_mean:.2f} ± {same.cost_sd:.2f} | "
                f"{same.qos_mean_percent:.2f} | {same.time_100_s:.2f} | "
                "0.0% | "
                f"{100 * (same.cost_mean / paper_cost - 1):+.1f}% |"
            )
            for method in METHODS:
                row = indexed.loc[(method, size, rate)]
                lines.append(
                    f"| {rate * 100:.0f}% | {method} | "
                    f"{row.cost_mean:.2f} ± {row.cost_sd:.2f} | "
                    f"{row.qos_mean_percent:.2f} | {row.time_100_s:.2f} | "
                    f"{100 * (row.cost_mean / same.cost_mean - 1):+.1f}% | "
                    f"{100 * (row.cost_mean / paper_cost - 1):+.1f}% |"
                )
        lines.append("")
    (result_dir / "COMPARISON.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
