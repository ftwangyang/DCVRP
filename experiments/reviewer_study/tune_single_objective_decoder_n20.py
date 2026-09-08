"""Validation-only calibration of the DVNDA-SO near-tie vehicle decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from experiments.reviewer_study.paper_dcvrp import generate_paired_paper_datasets
from experiments.reviewer_study.run_operational_metric_baselines_n20 import (
    DYNAMIC_RATES,
    evaluate_neural,
    set_seed,
)


METRICS = (
    "response_wait_min",
    "completion_delay_min",
    "vehicle_utilization_percent",
    "route_balance_cv",
    "cost",
    "qos_percent",
)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.validation_seed)
    datasets = generate_paired_paper_datasets(
        args.instances,
        DYNAMIC_RATES,
        customer_count=20,
        vehicle_count=4,
    )
    frames = []
    for tolerance in args.tolerances:
        for rate in DYNAMIC_RATES:
            frame, _ = evaluate_neural(
                "DVNDA-SO",
                checkpoint,
                datasets[rate],
                rate,
                device,
                timing_repetitions=1,
                so_tie_tolerance=tolerance,
            )
            frame.insert(0, "tie_tolerance", tolerance)
            frames.append(frame)
        print(f"validated tie_tolerance={tolerance:g}", flush=True)

    raw = pd.concat(frames, ignore_index=True)
    summary = (
        raw.groupby(["tie_tolerance", "dynamic_rate"], as_index=False)[
            list(METRICS)
        ]
        .mean()
        .sort_values(["tie_tolerance", "dynamic_rate"])
    )
    macro = summary.groupby("tie_tolerance", as_index=False)[list(METRICS)].mean()
    baseline = macro.loc[macro.tie_tolerance == 0.0].iloc[0]
    macro["cost_change_percent"] = 100.0 * (
        macro.cost / baseline.cost - 1.0
    )
    macro["qos_change_point"] = macro.qos_percent - baseline.qos_percent
    macro["cost_qos_guard"] = (
        (macro.cost <= baseline.cost * (1.0 + args.max_cost_increase))
        & (
            macro.qos_percent
            >= baseline.qos_percent - args.max_qos_drop_point
        )
    )
    macro["operational_score"] = (
        macro.response_wait_min / baseline.response_wait_min
        + macro.completion_delay_min / baseline.completion_delay_min
        + macro.route_balance_cv / baseline.route_balance_cv
        - macro.vehicle_utilization_percent
        / baseline.vehicle_utilization_percent
    )
    feasible = macro[macro.cost_qos_guard]
    selected = feasible.sort_values(
        ["operational_score", "cost", "tie_tolerance"]
    ).iloc[0]
    selected_tolerance = float(selected.tie_tolerance)

    raw.to_csv(output_dir / "raw_validation_metrics.csv", index=False)
    summary.to_csv(output_dir / "validation_by_dynamic_rate.csv", index=False)
    macro.to_csv(output_dir / "validation_macro.csv", index=False)
    decision = {
        "validation_seed": args.validation_seed,
        "instances_per_rate": args.instances,
        "checkpoint": str(checkpoint),
        "tolerances": args.tolerances,
        "cost_guard_relative": args.max_cost_increase,
        "qos_guard_percentage_point": args.max_qos_drop_point,
        "selected_tie_tolerance": selected_tolerance,
        "selection_rule": (
            "Among Cost/QoS-guarded candidates, minimize the sum of normalized "
            "response time, completion delay, and route-balance CV minus "
            "normalized utilization. Test data are not used."
        ),
    }
    (output_dir / "selection.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(macro.to_string(index=False), flush=True)
    print(f"selected tie_tolerance={selected_tolerance:g}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-seed", type=int, default=9901)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.25, 0.50, 1.00],
    )
    parser.add_argument("--max-cost-increase", type=float, default=0.01)
    parser.add_argument("--max-qos-drop-point", type=float, default=0.01)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "experiments/checkpoints/single_objective_refined_n20_m4/best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/results/single_objective_refined_n20_m4/decoder_validation"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
