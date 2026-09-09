"""Evaluation script for Dynamic Capacitated Vehicle Routing Problem (DCVRP).

Evaluates trained deep reinforcement learning models and the Greedy baseline
on DCVRP instances across various degrees of dynamism (phi in {0.10, 0.25, 0.50, 0.75}).
Strictly conforms to Section IV-A and Table I of the manuscript:
"DVNDA: Deep Reinforcement Learning with Dual-Attention for Dynamic Capacitated Vehicle Routing Problem"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t as student_t

from env import (
    DEFAULT_CUSTOMER_COUNT,
    DEFAULT_DYNAMIC_RATES,
    DEFAULT_VEHICLE_COUNT,
    DCVRPEnvironment,
    generate_evaluation_split,
    run_greedy,
)
from models import AttentionLearner, build_selector

AVAILABLE_METHODS = ["Greedy", "DVNDA", "AMCVN", "LiDRL", "MAAM", "MARDAM"]

# Official Table I results reported in the manuscript (mean_cost, sd_cost, qos_percent, inference_time)
TABLE1_BENCHMARK = {
    20: {
        0.10: {"Greedy": (9.07, 1.12, 99.90, "—"), "MARDAM": (8.91, 1.29, 99.95, "1s"), "MAAM": (8.83, 1.22, 99.80, "1s"), "LiDRL": (8.67, 1.27, 100.0, "1s"), "AMCVN": (8.39, 1.29, 100.0, "1s"), "DVNDA": (8.31, 1.22, 100.0, "1s")},
        0.25: {"Greedy": (9.69, 1.25, 99.90, "—"), "MARDAM": (9.85, 1.27, 99.80, "1s"), "MAAM": (9.68, 1.55, 99.65, "1s"), "LiDRL": (9.45, 1.24, 100.0, "1s"), "AMCVN": (9.23, 1.23, 100.0, "1s"), "DVNDA": (8.95, 1.30, 100.0, "1s")},
        0.50: {"Greedy": (11.25, 1.43, 99.90, "—"), "MARDAM": (11.45, 1.37, 99.85, "1s"), "MAAM": (11.32, 1.30, 99.40, "1s"), "LiDRL": (11.01, 1.45, 100.0, "1s"), "AMCVN": (10.75, 1.54, 100.0, "1s"), "DVNDA": (10.47, 1.53, 100.0, "1s")},
        0.75: {"Greedy": (12.43, 1.51, 99.45, "—"), "MARDAM": (12.81, 1.49, 99.85, "1s"), "MAAM": (12.72, 1.53, 99.95, "1s"), "LiDRL": (12.52, 1.62, 100.0, "1s"), "AMCVN": (12.03, 1.47, 100.0, "1s"), "DVNDA": (11.78, 1.44, 100.0, "1s")},
    },
    35: {
        0.10: {"Greedy": (15.95, 1.44, 99.95, "—"), "MARDAM": (15.86, 2.32, 99.95, "1s"), "MAAM": (15.54, 2.37, 99.95, "1s"), "LiDRL": (15.13, 2.42, 100.0, "2s"), "AMCVN": (15.03, 2.42, 100.0, "2s"), "DVNDA": (14.94, 2.00, 100.0, "3s")},
        0.25: {"Greedy": (16.96, 1.95, 99.95, "—"), "MARDAM": (16.90, 2.42, 99.95, "1s"), "MAAM": (16.83, 2.46, 99.80, "1s"), "LiDRL": (16.71, 2.53, 100.0, "2s"), "AMCVN": (16.42, 2.90, 100.0, "2s"), "DVNDA": (16.14, 2.14, 100.0, "3s")},
        0.50: {"Greedy": (19.63, 2.01, 99.95, "—"), "MARDAM": (19.79, 2.55, 99.90, "1s"), "MAAM": (19.68, 2.68, 99.65, "1s"), "LiDRL": (19.16, 2.49, 100.0, "2s"), "AMCVN": (19.04, 2.46, 100.0, "2s"), "DVNDA": (18.90, 2.24, 100.0, "3s")},
        0.75: {"Greedy": (21.55, 2.21, 99.95, "—"), "MARDAM": (21.84, 2.71, 99.85, "1s"), "MAAM": (21.70, 2.11, 99.65, "1s"), "LiDRL": (21.47, 2.85, 100.0, "2s"), "AMCVN": (21.19, 2.62, 100.0, "2s"), "DVNDA": (20.98, 2.26, 100.0, "3s")},
    },
    50: {
        0.10: {"Greedy": (21.41, 2.11, 99.95, "—"), "MARDAM": (21.24, 2.96, 99.95, "3s"), "MAAM": (19.81, 2.24, 99.90, "3s"), "LiDRL": (19.38, 2.31, 100.0, "4s"), "AMCVN": (18.94, 2.75, 100.0, "4s"), "DVNDA": (18.89, 2.27, 100.0, "5s")},
        0.25: {"Greedy": (22.84, 2.10, 99.85, "—"), "MARDAM": (22.75, 2.81, 99.95, "3s"), "MAAM": (22.33, 2.69, 99.95, "3s"), "LiDRL": (21.78, 2.75, 100.0, "4s"), "AMCVN": (21.56, 2.51, 100.0, "4s"), "DVNDA": (21.21, 2.66, 100.0, "5s")},
        0.50: {"Greedy": (26.71, 2.48, 99.95, "—"), "MARDAM": (26.91, 2.87, 99.95, "3s"), "MAAM": (26.76, 2.78, 99.80, "3s"), "LiDRL": (25.65, 2.98, 100.0, "4s"), "AMCVN": (25.49, 2.89, 100.0, "4s"), "DVNDA": (25.31, 2.81, 100.0, "5s")},
        0.75: {"Greedy": (30.19, 2.77, 99.85, "—"), "MARDAM": (30.57, 2.48, 99.90, "3s"), "MAAM": (30.30, 3.11, 99.20, "3s"), "LiDRL": (29.87, 3.07, 100.0, "4s"), "AMCVN": (29.33, 3.05, 100.0, "4s"), "DVNDA": (29.00, 2.69, 100.0, "5s")},
    },
}


def evaluate_greedy(
    split: dict[float, object],
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
) -> list[dict]:
    """Evaluate the event-driven nearest-available Greedy heuristic baseline."""
    rows = []
    for rate, dataset in split.items():
        start_time = time.perf_counter()
        distances_t, qos_t = run_greedy(dataset, reveal="continuous")
        elapsed = time.perf_counter() - start_time

        distances = distances_t.numpy()
        qos_values = (100.0 * qos_t).numpy()

        dist_mean = float(distances.mean())
        dist_sd = float(distances.std(ddof=1))
        dist_sem = dist_sd / np.sqrt(len(distances))
        ci95 = float(student_t.ppf(0.975, df=len(distances) - 1) * dist_sem)
        qos_mean = float(qos_values.mean())

        rows.append({
            "method": "Greedy",
            "rate": rate,
            "distance_mean": dist_mean,
            "distance_sd": dist_sd,
            "ci95_low": dist_mean - ci95,
            "ci95_high": dist_mean + ci95,
            "qos_mean": qos_mean,
            "elapsed_s": elapsed,
            "customer_count": customer_count,
        })
    return rows


def evaluate_neural(
    method: str,
    checkpoint_path: Path,
    split: dict[float, object],
    device: torch.device,
    vehicle_count: int = DEFAULT_VEHICLE_COUNT,
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
) -> list[dict]:
    """Evaluate a trained neural model checkpoint on DCVRP test instances."""
    selector = build_selector(method, vehicle_count=vehicle_count)
    model = AttentionLearner(selector)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    compat_state = {
        k: v for k, v in checkpoint["model"].items()
        if k in model.state_dict() and v.shape == model.state_dict()[k].shape
    }
    model.load_state_dict(compat_state, strict=False)
    model.eval()
    model.greedy = True
    model.vehicle_greedy = True
    model.include_vehicle_log_probability = False
    model = model.to(device)

    # Warm-up pass to initialize CUDA context and kernel caches
    if len(split) > 0 and device.type == "cuda":
        first_dataset = next(iter(split.values()))
        warmup_env = DCVRPEnvironment(
            first_dataset,
            nodes=first_dataset.nodes[:2].to(device),
            pending_cost=0.0,
        )
        with torch.no_grad():
            model(warmup_env)
        torch.cuda.synchronize(device)

    rows = []
    for rate, dataset in split.items():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()

        env = DCVRPEnvironment(
            dataset,
            nodes=dataset.nodes.to(device),
            pending_cost=0.0,
        )
        with torch.no_grad():
            model(env)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time

        distances = env.route_distance().detach().cpu().numpy()
        qos_values = (100.0 * env.qos()).detach().cpu().numpy()

        dist_mean = float(distances.mean())
        dist_sd = float(distances.std(ddof=1))
        dist_sem = dist_sd / np.sqrt(len(distances))
        ci95 = float(student_t.ppf(0.975, df=len(distances) - 1) * dist_sem)
        qos_mean = float(qos_values.mean())

        rows.append({
            "method": method,
            "rate": rate,
            "distance_mean": dist_mean,
            "distance_sd": dist_sd,
            "ci95_low": dist_mean - ci95,
            "ci95_high": dist_mean + ci95,
            "qos_mean": qos_mean,
            "elapsed_s": elapsed,
            "customer_count": customer_count,
        })
    return rows


def print_results(rows: list[dict]):
    """Format and print standard evaluation metrics matching Table I layout."""
    print("=" * 82)
    print(
        f"{'Method':<10} | {'Dynamic Rate':<12} | {'Distance (Mean +/- SD)':<24} | "
        f"{'QoS (%)':<10} | {'Time':<8}"
    )
    print("-" * 82)
    for r in rows:
        dist_str = f"{r['distance_mean']:.2f} +/- {r['distance_sd']:.2f}"
        qos_str = f"{r['qos_mean']:.2f}%" if r["qos_mean"] < 100.0 else "100.0%"
        if r["method"] == "Greedy":
            time_str = "-"
        else:
            time_sec = max(1, int(round(r["elapsed_s"])))
            time_str = f"{time_sec}s"
        rate_str = f"phi = {r['rate']*100:>4.0f}%"
        print(
            f"{r['method']:<10} | {rate_str:<12} | {dist_str:<24} | "
            f"{qos_str:<10} | {time_str:<8}"
        )
    print("=" * 82)


def print_comparison_table(rows: list[dict], customer_count: int):
    """Print a side-by-side comparison between measured results and Table I targets."""
    scale_targets = TABLE1_BENCHMARK.get(customer_count)
    if not scale_targets:
        print(f"\nNote: No manuscript Table I baseline targets available for scale n={customer_count}.")
        return

    print(f"\n{'=' * 96}")
    print(f"  Table I Reproduction Verification (n = {customer_count} Customers)")
    print(f"{'=' * 96}")
    print(
        f"{'Method':<8} | {'phi':<5} | {'Measured Cost':<18} | {'Table I Cost':<18} | "
        f"{'Cost Gap':<9} | {'Measured QoS':<12} | {'Table I QoS':<11}"
    )
    print(f"{'-' * 96}")

    for r in rows:
        method = r["method"]
        rate = round(r["rate"], 2)
        target_info = scale_targets.get(rate, {}).get(method)
        if target_info:
            target_mean, target_sd, target_qos, _ = target_info
            gap = (r["distance_mean"] - target_mean) / target_mean * 100.0
            gap_str = f"{gap:+.2f}%"
            meas_cost = f"{r['distance_mean']:.2f} +/- {r['distance_sd']:.2f}"
            targ_cost = f"{target_mean:.2f} +/- {target_sd:.2f}"
            meas_qos = f"{r['qos_mean']:.2f}%"
            targ_qos = f"{target_qos:.2f}%" if target_qos < 100.0 else "100%"
            rate_label = f"{int(rate * 100)}%"
            print(
                f"{method:<8} | {rate_label:<5} | {meas_cost:<18} | {targ_cost:<18} | "
                f"{gap_str:<9} | {meas_qos:<12} | {targ_qos:<11}"
            )
    print(f"{'=' * 96}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DCVRP algorithms strictly following manuscript Table I."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="DVNDA",
        choices=AVAILABLE_METHODS + ["all"],
        help="Method to evaluate: Greedy, DVNDA, AMCVN, LiDRL, MAAM, MARDAM, or 'all'.",
    )
    parser.add_argument(
        "-n",
        "--customer-count",
        type=int,
        default=DEFAULT_CUSTOMER_COUNT,
        help=f"Number of customer locations (default: {DEFAULT_CUSTOMER_COUNT}).",
    )
    parser.add_argument(
        "-m",
        "--vehicle-count",
        type=int,
        default=None,
        help="Number of vehicles (default: auto-computed as n / 5).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Custom path to model checkpoint. Defaults to checkpoints/{method}[_n{n}].pt.",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=100,
        help="Number of evaluation instances per dynamic rate (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260821,
        help="Random seed for test instance generation (default: 20260821).",
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=list(DEFAULT_DYNAMIC_RATES),
        help="Dynamic customer rates to evaluate (default: 0.10 0.25 0.50 0.75).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--compare-table1",
        action="store_true",
        help="Display side-by-side comparison with published Table I results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.vehicle_count is None:
        args.vehicle_count = max(1, round(args.customer_count / 5))

    print(
        f"Loading test instances (n={args.customer_count}, m={args.vehicle_count}, "
        f"instances={args.instances}, seed={args.seed})..."
    )
    split = generate_evaluation_split(
        instances=args.instances,
        dynamic_rates=args.rates,
        customer_count=args.customer_count,
        vehicle_count=args.vehicle_count,
        seed=args.seed,
    )

    methods = AVAILABLE_METHODS if args.method == "all" else [args.method]
    all_rows = []

    for method in methods:
        if method == "Greedy":
            print(f"Evaluating Greedy heuristic baseline (CPU sequential)...")
            rows = evaluate_greedy(split, customer_count=args.customer_count)
            all_rows.extend(rows)
            continue

        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            scale_candidate = Path("checkpoints") / f"{method}_n{args.customer_count}.pt"
            default_path = Path("checkpoints") / f"{method}.pt"
            if args.customer_count != DEFAULT_CUSTOMER_COUNT and scale_candidate.exists():
                checkpoint_path = scale_candidate
            elif default_path.exists():
                checkpoint_path = default_path
            else:
                checkpoint_path = scale_candidate

        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Skipping {method}.")
            continue

        print(f"Evaluating {method} on {device} (checkpoint: {checkpoint_path})...")
        rows = evaluate_neural(
            method,
            checkpoint_path,
            split,
            device,
            vehicle_count=args.vehicle_count,
            customer_count=args.customer_count,
        )
        all_rows.extend(rows)

    if all_rows:
        print_results(all_rows)
        if args.compare_table1:
            print_comparison_table(all_rows, args.customer_count)


if __name__ == "__main__":
    main()
