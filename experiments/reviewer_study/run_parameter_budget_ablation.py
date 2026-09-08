"""Matched parameterization ablation in the executed-path DCVRP environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from .executed_path_dvnda import VectorizedExecutedPathDVNDAEnvironment
from .paper_dcvrp import generate_paired_paper_datasets
from .run_original_uncertainty_n20 import load_model, set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DYNAMIC_RATES = (0.10, 0.25, 0.50, 0.75)
METHODS = ("DVNDA", "Shared-DVNDA", "PM-Shared-DVNDA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=20260826)
    parser.add_argument("--timing-repetitions", type=int, default=10)
    parser.add_argument(
        "--dvnda-checkpoint",
        type=Path,
        default=Path("experiments/checkpoints/executed_path_dvnda_n20_m4/best.pt"),
    )
    parser.add_argument(
        "--shared-checkpoint",
        type=Path,
        default=Path("experiments/checkpoints/latest_shared_dvnda_n20_m4/best.pt"),
    )
    parser.add_argument(
        "--pm-shared-checkpoint",
        type=Path,
        default=Path("experiments/checkpoints/latest_pm_shared_dvnda_n20_m4/best.pt"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/results/parameter_budget_ablation_executed_path"),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


@torch.inference_mode()
def evaluate(model, data, device: torch.device):
    environment = VectorizedExecutedPathDVNDAEnvironment(
        data,
        nodes=data.nodes.to(device),
        pending_cost=0.0,
    )
    model.greedy = True
    model.vehicle_greedy = True
    model(environment)
    return (
        environment.route_distance().cpu().numpy(),
        (100.0 * environment.qos()).cpu().numpy(),
    )


@torch.inference_mode()
def timed_batch(model, data, device: torch.device, repetitions: int):
    values = []
    for _ in range(repetitions):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        evaluate(model, data, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values.append(time.perf_counter() - started)
    return values


def paired_test(raw: pd.DataFrame, reference: str, comparator: str) -> list[dict]:
    rows = []
    for dynamic_rate in DYNAMIC_RATES:
        rate = raw[raw.dynamic_rate == dynamic_rate]
        left = rate[rate.method == reference].sort_values("instance")
        right = rate[rate.method == comparator].sort_values("instance")
        delta = right.cost.to_numpy() - left.cost.to_numpy()
        qos_delta = right.qos_percent.to_numpy() - left.qos_percent.to_numpy()
        sem = delta.std(ddof=1) / math.sqrt(len(delta))
        critical = stats.t.ppf(0.975, len(delta) - 1)
        rows.append(
            {
                "dynamic_rate": dynamic_rate,
                "reference": reference,
                "comparator": comparator,
                "paired_instances": len(delta),
                "comparator_minus_reference_cost": delta.mean(),
                "cost_delta_ci95_low": delta.mean() - critical * sem,
                "cost_delta_ci95_high": delta.mean() + critical * sem,
                "paired_t_p_two_sided": stats.ttest_rel(
                    right.cost.to_numpy(), left.cost.to_numpy()
                ).pvalue,
                "comparator_minus_reference_qos_pp": qos_delta.mean(),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    output_root = resolve(args.output_root)
    raw_root = output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "DVNDA": resolve(args.dvnda_checkpoint),
        "Shared-DVNDA": resolve(args.shared_checkpoint),
        "PM-Shared-DVNDA": resolve(args.pm_shared_checkpoint),
    }
    models = {}
    checkpoints = {}
    for method, path in paths.items():
        model, saved = load_model(path, device)
        model.eval()
        models[method] = model
        checkpoints[method] = saved

    resource_rows = []
    for method, model in models.items():
        selector_parameters = parameter_count(model.selector)
        total_parameters = parameter_count(model)
        resource_rows.append(
            {
                "method": method,
                "total_parameters": total_parameters,
                "selector_parameters": selector_parameters,
                "non_selector_parameters": total_parameters - selector_parameters,
            }
        )
    resources = pd.DataFrame(resource_rows)
    non_selector_counts = resources.non_selector_parameters.unique()
    if len(non_selector_counts) != 1:
        raise RuntimeError("non-selector parameter counts are not matched")
    independent_total = resources.loc[
        resources.method == "DVNDA", "total_parameters"
    ].iloc[0]
    pm_total = resources.loc[
        resources.method == "PM-Shared-DVNDA", "total_parameters"
    ].iloc[0]
    if abs(pm_total - independent_total) / independent_total >= 0.01:
        raise RuntimeError("PM-Shared-DVNDA differs from DVNDA by at least 1%")

    set_seed(args.test_seed)
    datasets = generate_paired_paper_datasets(
        args.instances,
        DYNAMIC_RATES,
        customer_count=20,
        vehicle_count=4,
    )
    raw_rows = []
    timing_rows = []
    for dynamic_rate in DYNAMIC_RATES:
        data = datasets[dynamic_rate]
        for method in METHODS:
            distance, qos = evaluate(models[method], data, device)
            for instance, (cost, qos_percent) in enumerate(zip(distance, qos)):
                raw_rows.append(
                    {
                        "dynamic_rate": dynamic_rate,
                        "method": method,
                        "instance": instance,
                        "cost": float(cost),
                        "qos_percent": float(qos_percent),
                    }
                )
            for repetition, elapsed in enumerate(
                timed_batch(
                    models[method], data, device, args.timing_repetitions
                )
            ):
                timing_rows.append(
                    {
                        "dynamic_rate": dynamic_rate,
                        "method": method,
                        "repetition": repetition,
                        "seconds_per_100_instances": elapsed,
                    }
                )
            print(f"rate={dynamic_rate:.2f}, method={method}", flush=True)

    raw = pd.DataFrame(raw_rows)
    timings = pd.DataFrame(timing_rows)
    summary = (
        raw.groupby(["dynamic_rate", "method"], as_index=False)
        .agg(
            instances=("instance", "nunique"),
            cost_mean=("cost", "mean"),
            cost_sd=("cost", "std"),
            qos_mean_percent=("qos_percent", "mean"),
            qos_sd_percent=("qos_percent", "std"),
        )
        .merge(
            timings.groupby(["dynamic_rate", "method"], as_index=False).agg(
                time_100_mean_s=("seconds_per_100_instances", "mean"),
                time_100_sd_s=("seconds_per_100_instances", "std"),
            ),
            on=["dynamic_rate", "method"],
        )
    )
    paired = pd.DataFrame(
        paired_test(raw, "DVNDA", "Shared-DVNDA")
        + paired_test(raw, "DVNDA", "PM-Shared-DVNDA")
    )

    raw.to_csv(raw_root / "parameter_budget_instances.csv", index=False)
    timings.to_csv(raw_root / "parameter_budget_timings.csv", index=False)
    resources.to_csv(output_root / "parameter_counts.csv", index=False)
    summary.to_csv(output_root / "parameter_budget_summary.csv", index=False)
    paired.to_csv(output_root / "paired_statistics.csv", index=False)

    lines = [
        "# Independent/shared parameter-budget ablation",
        "",
        "All methods receive the same candidate-vehicle state, complete fleet state, visible-customer features, and feasibility masks. PM-Shared-DVNDA widens the single shared scorer so that its total parameter count differs from DVNDA by less than 1%.",
        "",
        "| Dynamic rate | Method | Parameters | Cost $\\downarrow$ | QoS (\\%) $\\uparrow$ | Time / 100 (s) $\\downarrow$ |",
        "| ---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    resource_map = resources.set_index("method").total_parameters.to_dict()
    for dynamic_rate in DYNAMIC_RATES:
        for method in METHODS:
            row = summary[
                (summary.dynamic_rate == dynamic_rate) & (summary.method == method)
            ].iloc[0]
            lines.append(
                f"| {100 * dynamic_rate:.0f}% | {method} | "
                f"{resource_map[method]:,} | "
                f"{row.cost_mean:.2f} $\\pm$ {row.cost_sd:.2f} | "
                f"{row.qos_mean_percent:.2f} $\\pm$ {row.qos_sd_percent:.2f} | "
                f"{row.time_100_mean_s:.3f} $\\pm$ {row.time_100_sd_s:.3f} |"
            )
    (output_root / "TABLE_PARAMETER_BUDGET.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    metadata = {
        "environment": "time_driven_executed_path_v4",
        "customer_count": 20,
        "vehicle_count": 4,
        "decision_intervals": 10,
        "instances_per_dynamic_rate": args.instances,
        "test_seed": args.test_seed,
        "training_seed_per_method": {
            method: checkpoints[method].get("config", {}).get("train_seed")
            for method in METHODS
        },
        "checkpoints": {
            method: {
                "path": str(paths[method]),
                "sha256": sha256(paths[method]),
                "epoch": checkpoints[method].get("epoch"),
                "selector": checkpoints[method].get("config", {}).get("selector"),
                "train_vehicle_policy": checkpoints[method]
                .get("config", {})
                .get("train_vehicle_policy"),
            }
            for method in METHODS
        },
        "claim_limit": "One matched training seed; 100 paired test instances quantify conditional test-set variation but not across-training-seed uncertainty.",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    (output_root / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(paired.to_string(index=False), flush=True)
    print(f"wrote results to {output_root}", flush=True)


if __name__ == "__main__":
    main()
