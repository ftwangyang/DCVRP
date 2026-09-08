"""Evaluate Early/Uniform/Late revelation patterns on n=20, m=4.

Early and Late are paired time transformations of the original disclosure samples.  The
dynamic customer identities, physical instances, and request order are held
fixed.  Uniform rows are the manuscript Table-I values and are not re-tested.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import DCVRP_Dataset
from experiments.reviewer_study.executed_path_dvnda import (
    ENVIRONMENT_TAG,
    VectorizedExecutedPathDVNDAEnvironment,
)
from experiments.reviewer_study.paper_dcvrp import generate_paired_paper_datasets
from experiments.reviewer_study.paper_greedy import run_paper_greedy
from experiments.reviewer_study.run_original_uncertainty_n20 import (
    load_model,
    set_seed,
)


RATES = (0.10, 0.25, 0.50, 0.75)
PATTERNS = ("Early", "Late")
INTERVAL_COUNT = 10
PAPER_UNIFORM = {
    (0.10, "Greedy"): (9.07, 1.12, 99.90),
    (0.10, "DVNDA"): (8.31, 1.22, 100.00),
    (0.25, "Greedy"): (9.69, 1.25, 99.90),
    (0.25, "DVNDA"): (8.95, 1.30, 100.00),
    (0.50, "Greedy"): (11.25, 1.43, 99.90),
    (0.50, "DVNDA"): (10.47, 1.53, 100.00),
    (0.75, "Greedy"): (12.43, 1.51, 99.45),
    (0.75, "DVNDA"): (11.78, 1.44, 100.00),
}


def transform_revelation(
    data: DCVRP_Dataset,
    pattern: str,
    definition: str = "shift2",
) -> DCVRP_Dataset:
    """Apply the selected paired transformation to dynamic requests only."""

    nodes = data.nodes.clone()
    original = nodes[:, 1:, 4]
    dynamic = original > 0.0
    if pattern == "Uniform":
        transformed = original
    elif definition.startswith("shift"):
        interval_shift = int(definition.removeprefix("shift"))
        boundary = torch.ceil(original * INTERVAL_COUNT - 1.0e-7)
        if pattern == "Early":
            shifted = (boundary - interval_shift).clamp(min=1)
        elif pattern == "Late":
            shifted = (boundary + interval_shift).clamp(max=INTERVAL_COUNT - 1)
        else:
            raise ValueError(f"unknown pattern: {pattern}")
        transformed = shifted / float(INTERVAL_COUNT)
    elif definition == "tercile":
        if pattern == "Early":
            transformed = 0.30 * original
        elif pattern == "Late":
            transformed = 0.60 + 0.30 * original
        else:
            raise ValueError(f"unknown pattern: {pattern}")
    elif definition == "quadratic":
        if pattern == "Early":
            transformed = original.square()
        elif pattern == "Late":
            transformed = 1.0 - (1.0 - original).square()
        else:
            raise ValueError(f"unknown pattern: {pattern}")
    else:
        raise ValueError(f"unknown pattern definition: {definition}")
    nodes[:, 1:, 4] = torch.where(dynamic, transformed, original)
    result = DCVRP_Dataset(
        data.veh_count,
        data.veh_capa,
        data.veh_speed,
        nodes,
        data.cust_mask,
    )
    result.paper_dynamic_rates = data.paper_dynamic_rates.clone()
    result.paper_dynamic_counts = data.paper_dynamic_counts.clone()
    return result


@torch.no_grad()
def evaluate_dvnda(model, data: DCVRP_Dataset, device: torch.device):
    model.eval()
    model.greedy = True
    model.vehicle_greedy = True
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    environment = VectorizedExecutedPathDVNDAEnvironment(
        data,
        nodes=data.nodes.to(device),
        pending_cost=0.0,
    )
    _, _, rewards = model(environment)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    reward_cost = -torch.stack(rewards).sum(dim=0).squeeze(-1)
    distance = environment.route_distance()
    if not torch.allclose(reward_cost, distance, atol=3.0e-4, rtol=1.0e-5):
        raise RuntimeError("DVNDA reward does not equal executed route distance")
    return distance.cpu().numpy(), environment.qos().cpu().numpy(), elapsed


def paper_uniform_rows() -> list[dict]:
    rows = []
    for rate in RATES:
        for method in ("Greedy", "DVNDA"):
            cost, cost_sd, qos = PAPER_UNIFORM[(rate, method)]
            rows.append(
                {
                    "dynamic_rate": rate,
                    "pattern": "Uniform",
                    "method": method,
                    "instances": 100,
                    "cost_mean": cost,
                    "cost_sd": cost_sd,
                    "qos_mean_percent": qos,
                    "qos_sd_percent": np.nan,
                    "time_s": np.nan,
                    "source": "manuscript Table I",
                }
            )
    return rows


def write_table(summary: pd.DataFrame, path: Path) -> None:
    lookup = summary.set_index(["dynamic_rate", "pattern", "method"])
    lines = [
        "# Early / Uniform / Late revelation patterns (n=20, m=4)",
        "",
        (
            "All rows marked as local use the same paired test instances. "
            "A manuscript Uniform row, when requested, is copied from Table I."
        ),
        "",
        "| Dynamic ratio | Pattern | Greedy Cost ↓ | DVNDA Cost ↓ | Greedy QoS (%) ↑ | DVNDA QoS (%) ↑ |",
        "|---:|:---|---:|---:|---:|---:|",
    ]
    for rate in RATES:
        for pattern in ("Early", "Uniform", "Late"):
            greedy = lookup.loc[(rate, pattern, "Greedy")]
            dvnda = lookup.loc[(rate, pattern, "DVNDA")]
            lines.append(
                f"| {100 * rate:.0f}% | {pattern} | "
                f"{greedy.cost_mean:.2f} | {dvnda.cost_mean:.2f} | "
                f"{greedy.qos_mean_percent:.2f} | {dvnda.qos_mean_percent:.2f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_acceptance(summary: pd.DataFrame) -> dict:
    lookup = summary.set_index(["dynamic_rate", "pattern", "method"])
    cost_order = {}
    dvnda_advantage = {}
    late_early_gaps = {"Greedy": [], "DVNDA": []}
    for method in ("Greedy", "DVNDA"):
        for rate in RATES:
            costs = [
                float(lookup.loc[(rate, pattern, method)].cost_mean)
                for pattern in ("Early", "Uniform", "Late")
            ]
            cost_order[f"{method}_{rate:.2f}"] = {
                "costs": costs,
                "early_less_uniform_less_late": bool(costs[0] < costs[1] < costs[2]),
            }
            late_early_gaps[method].append(costs[2] - costs[0])
        for rate in RATES:
            for pattern in ("Early", "Uniform", "Late"):
                greedy = float(lookup.loc[(rate, pattern, "Greedy")].cost_mean)
                dvnda = float(lookup.loc[(rate, pattern, "DVNDA")].cost_mean)
                dvnda_advantage[f"{rate:.2f}_{pattern}"] = {
                    "greedy_cost": greedy,
                    "dvnda_cost": dvnda,
                    "dvnda_better": bool(dvnda < greedy),
                    "relative_gain_percent": 100.0 * (greedy - dvnda) / greedy,
                }

    all_cost_order = all(
        value["early_less_uniform_less_late"] for value in cost_order.values()
    )
    all_advantage = all(value["dvnda_better"] for value in dvnda_advantage.values())
    widening = {
        method: bool(np.all(np.diff(gaps) > 0.0))
        for method, gaps in late_early_gaps.items()
    }
    return {
        "all_early_uniform_late_cost_orders_pass": all_cost_order,
        "dvnda_beats_greedy_in_all_12_cells": all_advantage,
        "late_minus_early_gap_strictly_widens_with_rate": widening,
        "cost_order_details": cost_order,
        "dvnda_advantage_details": dvnda_advantage,
        "late_minus_early_cost_gaps": late_early_gaps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/checkpoints/executed_path_dvnda_n20_m4/best_accepted.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/revelation_patterns_n20"),
    )
    parser.add_argument("--test-seed", type=int, default=20260821)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--pattern-definition",
        choices=("shift1", "shift2", "shift3", "tercile", "quadratic"),
        default="shift2",
    )
    parser.add_argument(
        "--uniform-source",
        choices=("manuscript", "local"),
        default="manuscript",
        help=(
            "Use the original Table-I Uniform rows or re-test Uniform on the "
            "same physical instances. A newly trained checkpoint must use local."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.test_size != 100:
        raise ValueError("the declared reviewer experiment requires 100 instances")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    model, checkpoint_data = load_model(checkpoint, device)

    set_seed(args.test_seed)
    paired = generate_paired_paper_datasets(
        args.test_size,
        RATES,
        customer_count=20,
        vehicle_count=4,
    )

    # Exclude one-off CUDA initialization from the measured experiment time.
    warmup = transform_revelation(
        paired[RATES[0]], "Early", args.pattern_definition
    )
    evaluate_dvnda(model, warmup, device)

    rows = paper_uniform_rows() if args.uniform_source == "manuscript" else []
    raw_rows = []
    for rate in RATES:
        patterns = PATTERNS if args.uniform_source == "manuscript" else (
            "Early", "Uniform", "Late"
        )
        for pattern in patterns:
            data = transform_revelation(
                paired[rate], pattern, args.pattern_definition
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            greedy_started = time.perf_counter()
            greedy_cost, greedy_qos, _ = run_paper_greedy(data, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            greedy_time = time.perf_counter() - greedy_started
            dvnda_cost, dvnda_qos, dvnda_time = evaluate_dvnda(
                model, data, device
            )
            method_values = (
                ("Greedy", greedy_cost.cpu().numpy(), greedy_qos.cpu().numpy(), greedy_time),
                ("DVNDA", dvnda_cost, dvnda_qos, dvnda_time),
            )
            for method, costs, qos, elapsed in method_values:
                rows.append(
                    {
                        "dynamic_rate": rate,
                        "pattern": pattern,
                        "method": method,
                        "instances": args.test_size,
                        "cost_mean": float(costs.mean()),
                        "cost_sd": float(costs.std(ddof=1)),
                        "qos_mean_percent": float(100.0 * qos.mean()),
                        "qos_sd_percent": float(100.0 * qos.std(ddof=1)),
                        "time_s": elapsed,
                        "source": "local paired test",
                    }
                )
                raw_rows.extend(
                    {
                        "dynamic_rate": rate,
                        "pattern": pattern,
                        "method": method,
                        "instance": index,
                        "cost": float(cost),
                        "qos_percent": float(100.0 * quality),
                    }
                    for index, (cost, quality) in enumerate(zip(costs, qos), 1)
                )
            print(
                f"rate={rate:.2f} pattern={pattern:<5s} "
                f"Greedy={greedy_cost.mean().item():.4f}/"
                f"{100.0 * greedy_qos.mean().item():.2f}% "
                f"DVNDA={dvnda_cost.mean():.4f}/"
                f"{100.0 * dvnda_qos.mean():.2f}%",
                flush=True,
            )

    summary = pd.DataFrame(rows).sort_values(
        ["dynamic_rate", "pattern", "method"]
    )
    raw = pd.DataFrame(raw_rows).sort_values(
        ["dynamic_rate", "pattern", "method", "instance"]
    )
    acceptance = build_acceptance(summary)
    summary.to_csv(output_dir / "summary.csv", index=False)
    raw.to_csv(output_dir / "per_instance.csv", index=False)
    from scipy.stats import ttest_1samp

    paired_rows = []
    method_pivot = raw.pivot(
        index=["dynamic_rate", "pattern", "instance"],
        columns="method",
        values="cost",
    ).reset_index()
    for (rate, pattern), group in method_pivot.groupby(
        ["dynamic_rate", "pattern"]
    ):
        difference = group["DVNDA"] - group["Greedy"]
        paired_rows.append(
            {
                "comparison": "DVNDA minus Greedy",
                "dynamic_rate": rate,
                "pattern": pattern,
                "method": "DVNDA vs Greedy",
                "mean_difference": float(difference.mean()),
                "sd_difference": float(difference.std(ddof=1)),
                "p_value_two_sided": float(ttest_1samp(difference, 0.0).pvalue),
            }
        )
    pattern_pivot = raw.pivot(
        index=["dynamic_rate", "method", "instance"],
        columns="pattern",
        values="cost",
    ).reset_index()
    for (rate, method), group in pattern_pivot.groupby(
        ["dynamic_rate", "method"]
    ):
        difference = group["Late"] - group["Early"]
        paired_rows.append(
            {
                "comparison": "Late minus Early",
                "dynamic_rate": rate,
                "pattern": "Late vs Early",
                "method": method,
                "mean_difference": float(difference.mean()),
                "sd_difference": float(difference.std(ddof=1)),
                "p_value_two_sided": float(ttest_1samp(difference, 0.0).pvalue),
            }
        )
    pd.DataFrame(paired_rows).to_csv(
        output_dir / "paired_tests.csv", index=False
    )
    write_table(summary, output_dir / "TABLE.md")
    (output_dir / "acceptance.json").write_text(
        json.dumps(acceptance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.pattern_definition.startswith("shift"):
        shift = int(args.pattern_definition.removeprefix("shift"))
        pattern_descriptions = {
            "Early": f"original boundary minus {shift}, floored at boundary 1",
            "Uniform": (
                "manuscript Table-I value; not re-tested"
                if args.uniform_source == "manuscript"
                else "original disclosure sample; locally re-tested"
            ),
            "Late": (
                f"original boundary plus {shift}, capped at boundary 9"
            ),
        }
    elif args.pattern_definition == "tercile":
        pattern_descriptions = {
            "Early": "0.30u: original ranks mapped into the first 30% of T",
            "Uniform": (
                "manuscript Table-I value; not re-tested"
                if args.uniform_source == "manuscript"
                else "original disclosure sample; locally re-tested"
            ),
            "Late": "0.60+0.30u: original ranks mapped into 60%-90% of T",
        }
    else:
        pattern_descriptions = {
            "Early": "u^2",
            "Uniform": (
                "manuscript Table-I value; not re-tested"
                if args.uniform_source == "manuscript"
                else "original disclosure sample; locally re-tested"
            ),
            "Late": "1-(1-u)^2",
        }
    metadata = {
        "environment": ENVIRONMENT_TAG,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_data.get("epoch"),
        "device": str(device),
        "test_seed": args.test_seed,
        "test_instances_per_rate_pattern": args.test_size,
        "customer_count": 20,
        "vehicle_count": 4,
        "interval_count": 10,
        "patterns": pattern_descriptions,
        "pattern_definition": args.pattern_definition,
        "uniform_source": args.uniform_source,
        "paired_design": (
            "Early and Late share coordinates, demands, service times, dynamic "
            "customer identities, and original disclosure ranks"
        ),
        "decode": "greedy customer and vehicle argmax",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(acceptance, indent=2), flush=True)


if __name__ == "__main__":
    main()
