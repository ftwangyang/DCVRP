"""Audit split timing without treating a desired runtime trend as data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["method", "customer_count", "dynamic_rate", "instance_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--reference-raw", type=Path, required=True)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    raw = pd.read_csv(result_dir / "raw_instances.csv")
    reference = pd.read_csv(args.reference_raw.resolve())
    reference = reference[reference["method"].isin(raw["method"].unique())]
    merged = raw.merge(reference, on=KEYS, suffixes=("", "_reference"), validate="one_to_one")

    cost_difference = np.abs(merged["cost"] - merged["cost_reference"])
    qos_difference = np.abs(
        merged["qos_percent"] - merged["qos_percent_reference"]
    )
    timing_sum_error = np.abs(
        raw["planning_time_s"]
        - raw["initial_planning_time_s"]
        - raw["online_replanning_time_s"]
    )
    summary = pd.read_csv(result_dir / "summary.csv")

    call_sequences = []
    online_time_sequences = []
    for (method, size), group in summary.groupby(["method", "customer_count"]):
        ordered = group.sort_values("dynamic_rate")
        calls = ordered["active_online_replanning_calls_100"].to_numpy()
        online = ordered["online_replanning_time_100_s"].to_numpy()
        call_sequences.append(
            {
                "method": method,
                "n": int(size),
                "values": calls.astype(int).tolist(),
                "strictly_increasing": bool(np.all(np.diff(calls) > 0)),
            }
        )
        online_time_sequences.append(
            {
                "method": method,
                "n": int(size),
                "values_s": online.tolist(),
                "strictly_increasing": bool(np.all(np.diff(online) > 0)),
            }
        )

    failures = []
    if len(raw) != 4 * 3 * 4 * 100:
        failures.append(f"unexpected raw row count: {len(raw)}")
    if len(merged) != len(raw):
        failures.append("reference merge did not cover every timing row")
    if float(cost_difference.max()) > 1.0e-12:
        failures.append("Cost changed relative to accepted reference")
    if float(qos_difference.max()) > 1.0e-12:
        failures.append("QoS changed relative to accepted reference")
    if float(timing_sum_error.max()) > 1.0e-9:
        failures.append("initial plus online planning time does not equal planning time")
    if not (raw["solve_time_s"] + 1.0e-12 >= raw["planning_time_s"]).all():
        failures.append("planning time exceeds end-to-end time")
    if not all(row["strictly_increasing"] for row in call_sequences):
        failures.append("active online call count is not strictly increasing")

    report = {
        "passed": not failures,
        "failures": failures,
        "raw_rows": int(len(raw)),
        "max_cost_change": float(cost_difference.max()),
        "max_qos_change": float(qos_difference.max()),
        "max_timing_decomposition_error_s": float(timing_sum_error.max()),
        "active_online_call_sequences": call_sequences,
        "measured_online_time_sequences": online_time_sequences,
        "interpretation": (
            "Active update counts increase with dynamic rate, while measured CPU "
            "time need not increase because each periodic static subproblem becomes smaller."
        ),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    (result_dir / "TIMING_AUDIT.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
