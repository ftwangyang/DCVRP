"""Matched independent-weight versus shared-weight reviewer ablation.

Both models use the same customer decoder and expose the vehicle scorer to the
same candidate-vehicle state, full fleet state, visible customer features, and
feasibility masks.  The only experimental factor is whether the four candidate
vehicles are scored by four independent parameter sets or one shared set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import DCVRP_Dataset  # noqa: E402
from experiments.reviewer_study.paper_dcvrp import (  # noqa: E402
    PAPER_INTERVAL_COUNT,
    VectorizedPaperDCVRPEnvironment,
    generate_paired_paper_datasets,
)
from experiments.reviewer_study.run_original_uncertainty_n20 import (  # noqa: E402
    DYNAMIC_RATES,
    load_model,
)
from experiments.reviewer_study.selectors import (  # noqa: E402
    IndependentSelector,
    SharedSelector,
)


METHODS = ("DVNDA", "Shared-DVNDA")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def architecture_checks(models: dict[str, torch.nn.Module]) -> dict:
    independent = models["DVNDA"]
    shared = models["Shared-DVNDA"]
    if not isinstance(independent.selector, IndependentSelector):
        raise TypeError("DVNDA checkpoint does not contain IndependentSelector")
    if not isinstance(shared.selector, SharedSelector):
        raise TypeError("Shared-DVNDA checkpoint does not contain SharedSelector")

    independent_network = independent.selector.vehicle_networks[0]
    shared_network = shared.selector.network
    independent_shapes = {
        name: tuple(value.shape)
        for name, value in independent_network.state_dict().items()
    }
    shared_shapes = {
        name: tuple(value.shape)
        for name, value in shared_network.state_dict().items()
    }
    non_selector_independent = (
        parameter_count(independent) - parameter_count(independent.selector)
    )
    non_selector_shared = parameter_count(shared) - parameter_count(shared.selector)
    checks = {
        "same_vehicle_network_layer_shapes": independent_shapes == shared_shapes,
        "same_non_selector_parameter_count": (
            non_selector_independent == non_selector_shared
        ),
        "independent_selector_is_four_exact_copies": (
            parameter_count(independent.selector)
            == 4 * parameter_count(shared.selector)
        ),
        "non_selector_parameters_dvnda": non_selector_independent,
        "non_selector_parameters_shared": non_selector_shared,
    }
    if not all(
        checks[key]
        for key in (
            "same_vehicle_network_layer_shapes",
            "same_non_selector_parameter_count",
            "independent_selector_is_four_exact_copies",
        )
    ):
        raise RuntimeError(f"architecture control failed: {checks}")
    return checks


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    data: DCVRP_Dataset,
    device: torch.device,
    timing_repetitions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    model.eval()
    model.greedy = True
    model.vehicle_greedy = True
    model.include_vehicle_log_probability = False
    model.selector.evaluation_rule = "argmax"
    distance = qos = penalized_cost = None
    timings: list[float] = []
    for repetition in range(timing_repetitions):
        environment = VectorizedPaperDCVRPEnvironment(
            data,
            nodes=data.nodes.to(device),
            pending_cost=5.0,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        _, _, reward_steps = model(environment)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
        current_distance = environment.route_distance().detach().cpu().numpy()
        current_qos = (100.0 * environment.qos()).detach().cpu().numpy()
        current_penalized = (
            -torch.stack(reward_steps).sum(dim=0).squeeze(-1)
        ).detach().cpu().numpy()
        if repetition == 0:
            distance = current_distance
            qos = current_qos
            penalized_cost = current_penalized
        else:
            np.testing.assert_allclose(current_distance, distance, rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(current_qos, qos, rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(
                current_penalized, penalized_cost, rtol=0.0, atol=1e-6
            )
    assert distance is not None and qos is not None and penalized_cost is not None
    return distance, qos, penalized_cost, timings


def mean_sd(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1))


def paired_statistics(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    comparison_sets: list[tuple[str, pd.DataFrame]] = [
        (f"{100 * rate:.0f}%", raw[raw.dynamic_rate == rate])
        for rate in DYNAMIC_RATES
    ]
    macro = (
        raw.groupby(["method", "instance_id"], as_index=False)
        .agg(
            cost=("cost", "mean"),
            qos_percent=("qos_percent", "mean"),
            penalized_cost=("penalized_cost", "mean"),
        )
        .assign(dynamic_rate=np.nan)
    )
    comparison_sets.append(("Macro average", macro))
    for label, frame in comparison_sets:
        independent = frame[frame.method == "DVNDA"].sort_values("instance_id")
        shared = frame[frame.method == "Shared-DVNDA"].sort_values("instance_id")
        delta = (
            shared.penalized_cost.to_numpy()
            - independent.penalized_cost.to_numpy()
        )
        standard_error = delta.std(ddof=1) / math.sqrt(len(delta))
        critical = stats.t.ppf(0.975, len(delta) - 1)
        test = stats.ttest_rel(
            shared.penalized_cost,
            independent.penalized_cost,
            alternative="greater",
        )
        rows.append(
            {
                "scope": label,
                "paired_instances": len(delta),
                "shared_minus_dvnda_cost": float(
                    shared.cost.mean() - independent.cost.mean()
                ),
                "shared_minus_dvnda_qos_percentage_points": float(
                    shared.qos_percent.mean() - independent.qos_percent.mean()
                ),
                "shared_minus_dvnda_penalized_cost": float(delta.mean()),
                "penalized_cost_delta_ci95_low": float(
                    delta.mean() - critical * standard_error
                ),
                "penalized_cost_delta_ci95_high": float(
                    delta.mean() + critical * standard_error
                ),
                "one_sided_p_shared_worse": float(test.pvalue),
            }
        )
    return pd.DataFrame(rows)


def make_overall_summary(raw: pd.DataFrame, timing: pd.DataFrame) -> pd.DataFrame:
    per_instance = (
        raw.groupby(["method", "instance_id"], as_index=False)
        .agg(cost=("cost", "mean"), qos_percent=("qos_percent", "mean"))
    )
    performance = (
        per_instance.groupby("method", as_index=False)
        .agg(
            cost_mean=("cost", "mean"),
            cost_sd=("cost", "std"),
            qos_mean_percent=("qos_percent", "mean"),
            qos_sd_percent=("qos_percent", "std"),
        )
    )
    speed = (
        timing.groupby("method", as_index=False)
        .agg(
            solve_time_100_mean_s=("solve_time_100_s", "mean"),
            solve_time_100_sd_s=("solve_time_100_s", "std"),
        )
    )
    return performance.merge(speed, on="method")


def write_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Independent- versus shared-weight ablation (n=20, m=4)",
        "",
        "Each cell uses 100 paired test instances. Cost and QoS are mean ± sample SD across instances; time is mean ± SD across 10 timed GPU runs of one 100-instance batch.",
        "",
        "| Dynamic rate | Method | Cost ↓ | QoS (%) ↑ | Solve time / 100 (s) ↓ |",
        "|---:|:---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {100 * row.dynamic_rate:.0f}% | {row.method} | "
            f"{row.cost_mean:.2f} ± {row.cost_sd:.2f} | "
            f"{row.qos_mean_percent:.2f} ± {row.qos_sd_percent:.2f} | "
            f"{row.solve_time_100_mean_s:.3f} ± "
            f"{row.solve_time_100_sd_s:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.instances != 100:
        raise ValueError("reviewer protocol requires exactly 100 instances per rate")
    if args.timing_repetitions < 2:
        raise ValueError("timing_repetitions must be at least 2")
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "DVNDA": args.independent_checkpoint.resolve(),
        "Shared-DVNDA": args.shared_checkpoint.resolve(),
    }
    models = {}
    checkpoint_data = {}
    for method, checkpoint in checkpoints.items():
        models[method], checkpoint_data[method] = load_model(checkpoint, device)
    checks = architecture_checks(models)

    set_seed(args.test_seed)
    datasets = generate_paired_paper_datasets(
        args.instances,
        DYNAMIC_RATES,
        customer_count=20,
        vehicle_count=4,
    )
    # Per-model warm-up excludes lazy CUDA setup from reported inference time.
    for method, model in models.items():
        evaluate_model(model, datasets[DYNAMIC_RATES[0]], device, 1)

    raw_rows: list[dict] = []
    timing_rows: list[dict] = []
    summary_rows: list[dict] = []
    for dynamic_rate in DYNAMIC_RATES:
        data = datasets[dynamic_rate]
        for method in METHODS:
            distance, qos, penalized_cost, timings = evaluate_model(
                models[method], data, device, args.timing_repetitions
            )
            cost_mean, cost_sd = mean_sd(distance)
            qos_mean, qos_sd = mean_sd(qos)
            time_mean, time_sd = mean_sd(np.asarray(timings))
            summary_rows.append(
                {
                    "dynamic_rate": dynamic_rate,
                    "method": method,
                    "n_instances": args.instances,
                    "cost_mean": cost_mean,
                    "cost_sd": cost_sd,
                    "qos_mean_percent": qos_mean,
                    "qos_sd_percent": qos_sd,
                    "solve_time_100_mean_s": time_mean,
                    "solve_time_100_sd_s": time_sd,
                }
            )
            for instance_id, values in enumerate(
                zip(distance, qos, penalized_cost), start=1
            ):
                raw_rows.append(
                    {
                        "dynamic_rate": dynamic_rate,
                        "method": method,
                        "instance_id": instance_id,
                        "cost": float(values[0]),
                        "qos_percent": float(values[1]),
                        "penalized_cost": float(values[2]),
                    }
                )
            for repetition, elapsed in enumerate(timings, start=1):
                timing_rows.append(
                    {
                        "dynamic_rate": dynamic_rate,
                        "method": method,
                        "repetition": repetition,
                        "solve_time_100_s": elapsed,
                    }
                )
            print(
                f"rate={100 * dynamic_rate:.0f}% method={method:<12} "
                f"cost={cost_mean:.3f}±{cost_sd:.3f} "
                f"qos={qos_mean:.2f}±{qos_sd:.2f}% "
                f"time100={time_mean:.3f}±{time_sd:.3f}s",
                flush=True,
            )

    raw = pd.DataFrame(raw_rows)
    timing = pd.DataFrame(timing_rows)
    summary = pd.DataFrame(summary_rows)
    summary["method"] = pd.Categorical(summary.method, METHODS, ordered=True)
    summary = summary.sort_values(["dynamic_rate", "method"]).reset_index(drop=True)
    summary["method"] = summary.method.astype(str)
    overall = make_overall_summary(raw, timing)
    paired = paired_statistics(raw)
    resources = pd.DataFrame(
        [
            {
                "method": method,
                "total_parameters": parameter_count(model),
                "selector_parameters": parameter_count(model.selector),
                "non_selector_parameters": (
                    parameter_count(model) - parameter_count(model.selector)
                ),
            }
            for method, model in models.items()
        ]
    )

    raw.to_csv(output_dir / "raw_instance_results.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    timing.to_csv(output_dir / "timing_repetitions.csv", index=False)
    summary.to_csv(output_dir / "summary_by_dynamic_rate.csv", index=False)
    overall.to_csv(output_dir / "summary_macro_average.csv", index=False)
    paired.to_csv(output_dir / "paired_statistics.csv", index=False)
    resources.to_csv(output_dir / "parameter_counts.csv", index=False)
    write_table(summary, output_dir / "TABLE_SHARED_VS_INDEPENDENT.md")

    metadata = {
        "environment": "VectorizedPaperDCVRPEnvironment",
        "customer_count": 20,
        "vehicle_count": 4,
        "interval_count": PAPER_INTERVAL_COUNT,
        "dynamic_rates": list(DYNAMIC_RATES),
        "instances_per_rate": args.instances,
        "paired_across_methods_and_rates": True,
        "test_seed": args.test_seed,
        "timing_repetitions": args.timing_repetitions,
        "decode": "greedy customer and argmax vehicle for both models",
        "cost_definition": "executed Euclidean route distance in normalized coordinate units",
        "qos_definition": "100 * served requests / 20",
        "controlled_input_representation": (
            "candidate vehicle state + full fleet state + visible customer features + per-vehicle feasibility mask"
        ),
        "only_factor_changed": "four independent VehicleSelectionNetwork weights versus one shared VehicleSelectionNetwork weight set",
        "architecture_checks": checks,
        "checkpoints": {
            method: {
                "path": str(path),
                "sha256": sha256(path),
                "epoch": int(checkpoint_data[method].get("epoch", -1)),
                "training_config": checkpoint_data[method].get("config", {}),
            }
            for method, path in checkpoints.items()
        },
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "claim_limit": (
            "One matched training seed isolates the architecture on these instances but does not estimate across-training-seed variance."
        ),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote matched ablation to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=20260823)
    parser.add_argument("--timing-repetitions", type=int, default=10)
    parser.add_argument(
        "--independent-checkpoint",
        type=Path,
        default=Path("experiments/checkpoints/latest_aggregation_n20_m4/best.pt"),
    )
    parser.add_argument(
        "--shared-checkpoint",
        type=Path,
        default=Path("experiments/checkpoints/latest_shared_dvnda_n20_m4/best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/latest_shared_vs_independent_n20_m4"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
