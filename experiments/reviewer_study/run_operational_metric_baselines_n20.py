"""Operational-metric comparison on paired synthetic n=20, m=4 instances.

The script keeps the manuscript's ten-boundary executed-path environment and
reports only the five reviewer-requested operational metrics in the paper
table.  Cost, QoS, and unserved counts are nevertheless retained in the raw
audit file so that a low response time cannot be produced by silently dropping
customers.

MARDAM, MAAM, LiDRL, and AMCVN are paper-guided reimplementations because no
official local implementation/checkpoint was supplied.  They share DVNDA's
customer encoder/decoder and training protocol and differ only in the vehicle
selection mechanism described in the manuscript.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import DCVRP_Dataset
from experiments.reviewer_study.executed_path_multivehicle_env import (
    ExecutedPathFleet,
)
from experiments.reviewer_study.paper_dcvrp import (
    PAPER_HORIZON_MINUTES,
    PAPER_INTERVAL_COUNT,
    generate_paired_paper_datasets,
)
from experiments.reviewer_study.paper_multivehicle_env import (
    PaperInstance,
    Plan,
    SimulationResult,
    VEHICLE_CAPACITY,
    disclosure_boundaries,
)
from experiments.reviewer_study.paper_multivehicle_solvers import make_planner
from experiments.reviewer_study.road_environment import (
    VectorizedRoadPaperDCVRPEnvironment,
)
from experiments.reviewer_study.model import OperationalAttentionLearner
from experiments.reviewer_study.run_original_uncertainty_n20 import load_model
from experiments.reviewer_study.selectors import build_selector


DYNAMIC_RATES = (0.10, 0.25, 0.50, 0.75)
METHOD_ORDER_BASE = (
    "Greedy",
    "Regret insertion",
    "Tabu Search",
    "ALNS",
    "OR-Tools",
    "MARDAM",
    "MAAM",
    "LiDRL",
    "AMCVN",
)
METHOD_ORDER = METHOD_ORDER_BASE + ("DVNDA",)
CLASSICAL_SOLVER_NAMES = {
    "Regret insertion": "Regret insertion",
    "Tabu Search": "Tabu Search",
    "ALNS": "Adaptive LNS",
    "OR-Tools": "OR-Tools",
}
DEFAULT_CHECKPOINTS = {
    "MARDAM": Path("experiments/checkpoints/reviewer3_mardam_n20_m4/best.pt"),
    "MAAM": Path("experiments/checkpoints/reviewer3_maam_n20_m4/best.pt"),
    "LiDRL": Path("experiments/checkpoints/reviewer3_lidrl_n20_m4/best.pt"),
    "AMCVN": Path("experiments/checkpoints/reviewer3_amcvn_n20_m4/best.pt"),
    "DVNDA": Path("experiments/checkpoints/latest_aggregation_n20_m4/best.pt"),
    "DVNDA-MO": Path("experiments/checkpoints/latest_aggregation_n20_m4/best.pt"),
    "DVNDA-SO": Path(
        "experiments/checkpoints/single_objective_refined_n20_m4/best.pt"
    ),
}

DVNDA_MO_CALIBRATION = {
    "selector_timeline_weight": 10_000.0,
    "selector_route_balance_weight": 50.0,
    "customer_spatial_dispersion_weight": 15.0,
    "customer_service_weight": 150.0,
    "customer_age_weight": 0.0,
    "validation_seed": 8821,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def euclidean_matrices(
    data: DCVRP_Dataset, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = data.nodes[:, :, :2].to(device)
    distance = torch.cdist(coordinates, coordinates)
    # Internal time is normalized to a 480-minute horizon.  At physical speed
    # one, one normalized coordinate unit takes one minute.
    travel_time = distance / PAPER_HORIZON_MINUTES
    return distance, travel_time


def to_paper_instances(data: DCVRP_Dataset) -> list[PaperInstance]:
    nodes = data.nodes.detach().cpu().numpy()
    return [
        PaperInstance(
            coordinates=instance[:, :2].astype(float, copy=True),
            demands=np.rint(
                instance[1:, 2] * VEHICLE_CAPACITY
            ).astype(int),
            service_minutes=(
                instance[1:, 3] * PAPER_HORIZON_MINUTES
            ).astype(float),
            disclosure_minutes=(
                instance[1:, 4] * PAPER_HORIZON_MINUTES
            ).astype(float),
        )
        for instance in nodes
    ]


def metrics_from_result(
    instance: PaperInstance, result: SimulationResult
) -> dict[str, float]:
    reveal = disclosure_boundaries(instance.disclosure_minutes)
    response: list[float] = []
    completion: list[float] = []
    busy = np.zeros(4, dtype=float)
    distance = np.zeros(4, dtype=float)
    for event in result.events:
        busy[event.vehicle] += max(0.0, event.completion_minute - event.start_minute)
        distance[event.vehicle] += float(event.distance)
        if event.customer is not None:
            response.append(
                max(0.0, event.arrival_minute - reveal[event.customer])
            )
            completion.append(
                max(0.0, event.completion_minute - reveal[event.customer])
            )
    makespan = max(float(result.final_completion_minute), 1.0e-12)
    mean_distance = float(distance.mean())
    return {
        "response_wait_min": float(np.mean(response)) if response else 0.0,
        "completion_delay_min": (
            float(np.mean(completion)) if completion else 0.0
        ),
        "vehicle_utilization_percent": float(
            np.clip(100.0 * busy.sum() / (4.0 * makespan), 0.0, 100.0)
        ),
        "route_balance_cv": (
            float(distance.std(ddof=0) / mean_distance)
            if mean_distance > 0.0
            else 0.0
        ),
        "cost": float(result.cost),
        "qos_percent": float(result.qos_percent),
        "unserved_requests": float(result.unserved_customers),
    }


def event_driven_greedy(instance: PaperInstance) -> SimulationResult:
    """Nearest visible task whenever a vehicle becomes idle."""

    from experiments.reviewer_study.paper_multivehicle_env import DispatchEvent

    depot = np.asarray(instance.coordinates[0], dtype=float)
    customer_coordinates = np.asarray(instance.coordinates[1:], dtype=float)
    reveal = disclosure_boundaries(instance.disclosure_minutes)
    positions = np.repeat(depot[None, :], 4, axis=0)
    capacities = np.full(4, VEHICLE_CAPACITY, dtype=int)
    available = np.zeros(4, dtype=float)
    assigned = np.zeros(instance.customer_count, dtype=bool)
    counts = np.zeros(4, dtype=int)
    events = []
    total_distance = 0.0
    current_time = 0.0

    for _ in range(instance.customer_count * 20):
        visible = (reveal <= current_time + 1.0e-12) & (~assigned)
        idle = np.flatnonzero(available <= current_time + 1.0e-12)
        for vehicle in idle:
            feasible = np.flatnonzero(
                visible & (instance.demands <= capacities[vehicle])
            )
            if feasible.size == 0:
                continue
            distances = np.linalg.norm(
                customer_coordinates[feasible] - positions[vehicle], axis=1
            )
            customer = int(feasible[int(np.argmin(distances))])
            leg = float(np.linalg.norm(
                customer_coordinates[customer] - positions[vehicle]
            ))
            arrival = current_time + leg
            completion = arrival + float(instance.service_minutes[customer])
            events.append(
                DispatchEvent(
                    interval_index=min(
                        PAPER_INTERVAL_COUNT,
                        int(current_time // (PAPER_HORIZON_MINUTES / PAPER_INTERVAL_COUNT)),
                    ),
                    vehicle=int(vehicle),
                    customer=customer,
                    start_minute=float(current_time),
                    arrival_minute=arrival,
                    completion_minute=completion,
                    distance=leg,
                )
            )
            total_distance += leg
            positions[vehicle] = customer_coordinates[customer]
            capacities[vehicle] -= int(instance.demands[customer])
            available[vehicle] = completion
            assigned[customer] = True
            visible[customer] = False
            counts[vehicle] += 1

        if assigned.all():
            break
        future = []
        future_reveal = reveal[(~assigned) & (reveal > current_time + 1.0e-12)]
        future_idle = available[available > current_time + 1.0e-12]
        if future_reveal.size:
            future.append(float(future_reveal.min()))
        if future_idle.size:
            future.append(float(future_idle.min()))
        if not future:
            break
        next_time = min(future)
        if next_time > PAPER_HORIZON_MINUTES + 1.0e-12:
            break
        current_time = next_time

    for vehicle in range(4):
        start = float(available[vehicle])
        leg = float(np.linalg.norm(positions[vehicle] - depot))
        arrival = start + leg
        events.append(
            DispatchEvent(
                interval_index=PAPER_INTERVAL_COUNT + 1,
                vehicle=vehicle,
                customer=None,
                start_minute=start,
                arrival_minute=arrival,
                completion_minute=arrival,
                distance=leg,
            )
        )
        total_distance += leg
        available[vehicle] = arrival

    served = int(assigned.sum())
    return SimulationResult(
        cost=total_distance,
        qos_percent=100.0 * served / instance.customer_count,
        served_customers=served,
        unserved_customers=instance.customer_count - served,
        final_completion_minute=float(available.max()),
        planning_calls=PAPER_INTERVAL_COUNT,
        events=tuple(events),
        vehicle_customer_counts=tuple(int(value) for value in counts),
    )


def evaluate_neural(
    method: str,
    checkpoint: Path,
    data: DCVRP_Dataset,
    dynamic_rate: float,
    device: torch.device,
    timing_repetitions: int,
    so_tie_tolerance: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    if method == "DVNDA-MO":
        checkpoint_data = torch.load(
            checkpoint, map_location=device, weights_only=False
        )
        selector = build_selector(
            "operational_independent",
            vehicle_count=4,
            selector_size=64,
            selector_heads=4,
            evaluation_rule="argmax",
        )
        model = OperationalAttentionLearner(
            selector=selector,
            customer_feature_size=5,
            vehicle_state_size=4,
            model_size=128,
            layer_count=3,
            head_count=8,
            ff_size=512,
            tanh_exploration=10,
            customer_spatial_dispersion_weight=DVNDA_MO_CALIBRATION[
                "customer_spatial_dispersion_weight"
            ],
            customer_service_weight=DVNDA_MO_CALIBRATION[
                "customer_service_weight"
            ],
            customer_age_weight=DVNDA_MO_CALIBRATION[
                "customer_age_weight"
            ],
        ).to(device)
        model.load_state_dict(checkpoint_data["model"], strict=True)
    else:
        model, checkpoint_data = load_model(checkpoint, device)
        if method == "DVNDA-SO":
            if not hasattr(model.selector, "tie_tolerance"):
                raise RuntimeError(
                    "DVNDA-SO checkpoint must use single_objective_independent"
                )
            model.selector.tie_tolerance = float(so_tie_tolerance)
    model.eval()
    model.greedy = True
    model.vehicle_greedy = True
    model.include_vehicle_log_probability = False
    distance, travel = euclidean_matrices(data, device)

    def run_once():
        environment = VectorizedRoadPaperDCVRPEnvironment(
            data,
            nodes=data.nodes.to(device),
            base_distance_matrix=distance,
            base_travel_time_matrix=travel,
            base_physical_distance_matrix=distance,
            pending_cost=5.0,
        )
        model(environment)
        operational = environment.operational_metrics()
        return environment, operational

    # Unmeasured warm-up removes CUDA lazy initialization from timing.
    with torch.inference_mode():
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed_values = []
    selected_environment = selected_operational = None
    for _ in range(timing_repetitions):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            environment, operational = run_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_values.append(time.perf_counter() - started)
        selected_environment = environment
        selected_operational = operational
    assert selected_environment is not None and selected_operational is not None
    per_epoch_ms = (
        1000.0 * float(np.mean(elapsed_values))
        / len(data)
        / PAPER_INTERVAL_COUNT
    )
    frame = pd.DataFrame(
        {
            "dynamic_rate": dynamic_rate,
            "method": method,
            "instance": np.arange(1, len(data) + 1),
            "response_wait_min": selected_operational[
                "response_wait_time_min"
            ].cpu().numpy(),
            "completion_delay_min": selected_operational[
                "completion_delay_min"
            ].cpu().numpy(),
            "vehicle_utilization_percent": selected_operational[
                "vehicle_utilization_percent"
            ].cpu().numpy(),
            "route_balance_cv": selected_operational[
                "route_balance_cv"
            ].cpu().numpy(),
            "time_per_epoch_ms": per_epoch_ms,
            "cost": selected_environment.route_distance().cpu().numpy(),
            "qos_percent": (
                100.0 * selected_environment.qos()
            ).cpu().numpy(),
            "unserved_requests": selected_operational[
                "uncompleted_requests"
            ].cpu().numpy(),
        }
    )
    return frame, checkpoint_data


def evaluate_classical(
    method: str,
    instances: list[PaperInstance],
    dynamic_rate: float,
    solver_time_ms: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for instance_index, instance in enumerate(instances, start=1):
        started = time.perf_counter()
        if method == "Greedy":
            result = event_driven_greedy(instance)
        else:
            call_index = 0

            def planner(state) -> Plan:
                nonlocal call_index
                call_index += 1
                selected = make_planner(
                    CLASSICAL_SOLVER_NAMES[method],
                    ortools_time_ms=solver_time_ms,
                    alns_time_ms=solver_time_ms,
                    seed=(
                        seed
                        + int(round(100 * dynamic_rate)) * 100_000
                        + instance_index * 100
                        + call_index
                    ),
                    assignment_policy="free",
                )
                return selected(state)

            result = ExecutedPathFleet(
                4,
                travel_minutes_per_normalized_unit=1.0,
                completed_route_return="commit-if-started",
            ).run(instance, planner)
        elapsed = time.perf_counter() - started
        row = metrics_from_result(instance, result)
        row.update(
            {
                "dynamic_rate": dynamic_rate,
                "method": method,
                "instance": instance_index,
                "time_per_epoch_ms": (
                    1000.0 * elapsed / PAPER_INTERVAL_COUNT
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(
    raw: pd.DataFrame, method_order: tuple[str, ...] = METHOD_ORDER
) -> pd.DataFrame:
    metrics = (
        "response_wait_min",
        "completion_delay_min",
        "vehicle_utilization_percent",
        "route_balance_cv",
        "time_per_epoch_ms",
        "cost",
        "qos_percent",
        "unserved_requests",
    )
    rows = []
    for (rate, method), frame in raw.groupby(["dynamic_rate", "method"], sort=False):
        row = {
            "dynamic_rate": rate,
            "method": method,
            "n": len(frame),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_sd"] = float(frame[metric].std(ddof=1))
        rows.append(row)
    result = pd.DataFrame(rows)
    result["method"] = pd.Categorical(
        result.method, categories=method_order, ordered=True
    )
    result = result.sort_values(["dynamic_rate", "method"]).reset_index(drop=True)
    result["method"] = result.method.astype(str)
    return result


def mean_sd(row, metric: str, digits: int = 2) -> str:
    return (
        f"{getattr(row, metric + '_mean'):.{digits}f} ± "
        f"{getattr(row, metric + '_sd'):.{digits}f}"
    )


def latex_mean_sd(row, metric: str, digits: int = 2) -> str:
    return mean_sd(row, metric, digits).replace("±", "$\\pm$")


def write_tables(summary: pd.DataFrame, output_dir: Path) -> None:
    markdown = [
        "# Operational metrics on n=20, m=4 synthetic instances",
        "",
        "Each row contains 100 fixed paired test instances. Values are mean ± sample SD. Time/epoch is amortized over the ten nominal decision intervals.",
        "",
        "| Dynamic rate | Method | Response/Waiting (min) ↓ | Completion delay (min) ↓ | Vehicle utilization (%) ↑ | Time/epoch (ms) ↓ | Route balance CV ↓ |",
        "|---:|:---|---:|---:|---:|---:|---:|",
    ]
    latex_rows = []
    for row in summary.itertuples(index=False):
        markdown.append(
            f"| {100 * row.dynamic_rate:.0f}% | {row.method} | "
            f"{mean_sd(row, 'response_wait_min')} | "
            f"{mean_sd(row, 'completion_delay_min')} | "
            f"{mean_sd(row, 'vehicle_utilization_percent')} | "
            f"{row.time_per_epoch_ms_mean:.2f} | "
            f"{mean_sd(row, 'route_balance_cv')} |"
        )
        latex_rows.append(
            f"{100 * row.dynamic_rate:.0f}\\% & {row.method} & "
            f"{latex_mean_sd(row, 'response_wait_min')} & "
            f"{latex_mean_sd(row, 'completion_delay_min')} & "
            f"{latex_mean_sd(row, 'vehicle_utilization_percent')} & "
            f"{row.time_per_epoch_ms_mean:.2f} & "
            f"{latex_mean_sd(row, 'route_balance_cv')} \\\\" 
        )
    (output_dir / "TABLE_OPERATIONAL_METRICS.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    latex = [
        "\\begin{tabular}{clccccc}",
        "\\toprule",
        "Dynamic rate & Method & Response/Waiting & Completion delay & Utilization (\\%) & Time/epoch (ms) & Route balance CV \\\\",
        "\\midrule",
        *latex_rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (output_dir / "TABLE_OPERATIONAL_METRICS.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )

    complete_markdown = [
        "# Complete performance table on n=20, m=4 synthetic instances",
        "",
        "Each row contains the same 100 fixed paired test instances. Values are mean ± sample SD; Time/epoch is the mean wall-clock time.",
        "",
        "| Dynamic rate | Method | Cost ↓ | QoS (%) ↑ | Response/Waiting (min) ↓ | Completion delay (min) ↓ | Vehicle utilization (%) ↑ | Time/epoch (ms) ↓ | Route balance CV ↓ |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    complete_latex_rows = []
    for row in summary.itertuples(index=False):
        complete_markdown.append(
            f"| {100 * row.dynamic_rate:.0f}% | {row.method} | "
            f"{mean_sd(row, 'cost')} | {mean_sd(row, 'qos_percent')} | "
            f"{mean_sd(row, 'response_wait_min')} | "
            f"{mean_sd(row, 'completion_delay_min')} | "
            f"{mean_sd(row, 'vehicle_utilization_percent')} | "
            f"{row.time_per_epoch_ms_mean:.2f} | "
            f"{mean_sd(row, 'route_balance_cv')} |"
        )
        complete_latex_rows.append(
            f"{100 * row.dynamic_rate:.0f}\\% & {row.method} & "
            f"{latex_mean_sd(row, 'cost')} & "
            f"{latex_mean_sd(row, 'qos_percent')} & "
            f"{latex_mean_sd(row, 'response_wait_min')} & "
            f"{latex_mean_sd(row, 'completion_delay_min')} & "
            f"{latex_mean_sd(row, 'vehicle_utilization_percent')} & "
            f"{row.time_per_epoch_ms_mean:.2f} & "
            f"{latex_mean_sd(row, 'route_balance_cv')} \\\\"
        )
    (output_dir / "TABLE_ALL_METRICS.md").write_text(
        "\n".join(complete_markdown) + "\n", encoding="utf-8"
    )
    complete_latex = [
        "\\begin{tabular}{clccccccc}",
        "\\toprule",
        "Dynamic rate & Method & Cost & QoS (\\%) & Response/Waiting & Completion delay & Utilization (\\%) & Time/epoch (ms) & Route balance CV \\\\",
        "\\midrule",
        *complete_latex_rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (output_dir / "TABLE_ALL_METRICS.tex").write_text(
        "\n".join(complete_latex) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    if args.instances != 100 and not args.allow_nonreviewer_size:
        raise ValueError("reviewer protocol requires exactly 100 instances")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dvnda_method = {
        "original": "DVNDA",
        "mo": "DVNDA-MO",
        "so": "DVNDA-SO",
    }[args.dvnda_mode]
    method_order = METHOD_ORDER_BASE + (dvnda_method,)
    checkpoint_paths = {
        "MARDAM": args.mardam_checkpoint.resolve(),
        "MAAM": args.maam_checkpoint.resolve(),
        "LiDRL": args.lidrl_checkpoint.resolve(),
        "AMCVN": args.amcvn_checkpoint.resolve(),
        dvnda_method: args.dvnda_checkpoint.resolve(),
    }
    missing = [str(path) for path in checkpoint_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing neural checkpoints: " + ", ".join(missing))

    set_seed(args.test_seed)
    datasets = generate_paired_paper_datasets(
        args.instances,
        DYNAMIC_RATES,
        customer_count=20,
        vehicle_count=4,
    )
    frames = []
    checkpoint_metadata = {}
    for dynamic_rate in DYNAMIC_RATES:
        data = datasets[dynamic_rate]
        instances = to_paper_instances(data)
        for method in method_order[:5]:
            frame = evaluate_classical(
                method,
                instances,
                dynamic_rate,
                args.solver_time_ms,
                args.algorithm_seed,
            )
            frames.append(frame)
            print(
                f"rate={100*dynamic_rate:.0f}% method={method:<17} "
                f"response={frame.response_wait_min.mean():.2f} "
                f"completion={frame.completion_delay_min.mean():.2f} "
                f"qos={frame.qos_percent.mean():.2f}%",
                flush=True,
            )
        for method in method_order[5:]:
            frame, metadata = evaluate_neural(
                method,
                checkpoint_paths[method],
                data,
                dynamic_rate,
                device,
                args.timing_repetitions,
                args.so_tie_tolerance,
            )
            frames.append(frame)
            checkpoint_metadata[method] = {
                "path": str(checkpoint_paths[method]),
                "epoch": int(metadata.get("epoch", -1)),
                "selector": (
                    "operational_independent"
                    if method == "DVNDA-MO"
                    else metadata.get("config", {}).get("selector", "independent")
                ),
            }
            print(
                f"rate={100*dynamic_rate:.0f}% method={method:<17} "
                f"response={frame.response_wait_min.mean():.2f} "
                f"completion={frame.completion_delay_min.mean():.2f} "
                f"qos={frame.qos_percent.mean():.2f}%",
                flush=True,
            )

    raw = pd.concat(frames, ignore_index=True)
    summary = summarize(raw, method_order)
    if not np.isfinite(raw.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("non-finite result detected")
    raw.to_csv(output_dir / "raw_instance_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_by_dynamic_rate.csv", index=False)
    write_tables(summary, output_dir)

    diagnostics = summary[
        [
            "dynamic_rate",
            "method",
            "cost_mean",
            "cost_sd",
            "qos_percent_mean",
            "qos_percent_sd",
            "unserved_requests_mean",
            "unserved_requests_sd",
        ]
    ]
    diagnostics.to_csv(output_dir / "SERVICE_COMPLETENESS_AUDIT.csv", index=False)
    config = {
        "environment": "time_driven_executed_path_v4/v9",
        "customer_count": 20,
        "vehicle_count": 4,
        "decision_intervals": PAPER_INTERVAL_COUNT,
        "horizon_minutes": PAPER_HORIZON_MINUTES,
        "instances_per_rate": args.instances,
        "dynamic_rates": DYNAMIC_RATES,
        "paired_instances_across_methods_and_rates": True,
        "test_seed": args.test_seed,
        "classical_static_solver_budget_ms_per_update": args.solver_time_ms,
        "neural_timing_repetitions": args.timing_repetitions,
        "metric_scope": "committed/actually executed events only",
        "mardam_maam_lidrl_amcvn_status": "paper-guided reimplementations, not official checkpoints",
        "neural_checkpoints": checkpoint_metadata,
        "dvnda_mode": args.dvnda_mode,
        "dvnda_single_objective": args.dvnda_mode in {"original", "so"},
        "hardware": {
            "neural_device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "classical_device": "CPU",
        },
        "claim_rule": (
            "The checkpoint was selected on validation Cost/QoS only. The "
            "fixed test set is evaluated without instance screening."
            if args.dvnda_mode in {"original", "so"}
            else "Calibration used validation data only. The fixed test set "
            "is evaluated without instance screening."
        ),
    }
    if args.dvnda_mode == "mo":
        config["dvnda_mo_calibration"] = DVNDA_MO_CALIBRATION
    if args.dvnda_mode == "so":
        config["dvnda_so_tie_tolerance"] = args.so_tie_tolerance
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote results to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--allow-nonreviewer-size", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-seed", type=int, default=20260828)
    parser.add_argument("--algorithm-seed", type=int, default=314159)
    parser.add_argument("--solver-time-ms", type=int, default=100)
    parser.add_argument("--timing-repetitions", type=int, default=5)
    parser.add_argument("--mardam-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["MARDAM"])
    parser.add_argument("--maam-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["MAAM"])
    parser.add_argument("--lidrl-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["LiDRL"])
    parser.add_argument("--amcvn-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["AMCVN"])
    parser.add_argument(
        "--dvnda-mode", choices=("original", "mo", "so"), default="original"
    )
    parser.add_argument("--so-tie-tolerance", type=float, default=0.0)
    parser.add_argument("--dvnda-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["DVNDA"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/reviewer3_operational_baselines_n20_m4"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
