"""Evaluation script for Dynamic Capacitated Vehicle Routing Problem (DCVRP).

Evaluates trained deep reinforcement learning models on DCVRP instances across
various dynamic degrees of dynamism (phi in {0.10, 0.25, 0.50, 0.75}).
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
)
from models import AttentionLearner, build_selector

AVAILABLE_METHODS = ["DVNDA", "AMCVN", "LiDRL", "MAAM", "MARDAM"]


def evaluate_method(
    method: str,
    checkpoint_path: Path,
    split: dict[float, object],
    device: torch.device,
    vehicle_count: int = DEFAULT_VEHICLE_COUNT,
) -> list[dict]:
    """Evaluate a trained model checkpoint on test instance batches."""
    selector = build_selector(method, vehicle_count=vehicle_count)
    model = AttentionLearner(selector)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if "regimes" in checkpoint and hasattr(model.selector, "regimes"):
        model.selector.regimes = {
            k: {p: v.to(device) for p, v in params.items()}
            for k, params in checkpoint["regimes"].items()
        }
    model.eval()
    model.greedy = True
    model.vehicle_greedy = True
    model.include_vehicle_log_probability = False
    model = model.to(device)

    # Warm-up pass to initialize CUDA context and kernel caches
    if len(split) > 0:
        first_dataset = next(iter(split.values()))
        warmup_env = DCVRPEnvironment(
            first_dataset,
            nodes=first_dataset.nodes[:2].to(device),
            pending_cost=0.0,
        )
        with torch.no_grad():
            model(warmup_env)
        if device.type == "cuda":
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
        })
    return rows


def print_results(rows: list[dict]):
    """Format and print evaluation metrics."""
    print("=" * 78)
    print(f"{'Method':<10} | {'Dynamic Rate':<12} | {'Distance (Mean +/- SD)':<24} | {'QoS (%)':<10} | {'Time (s)':<8}")
    print("-" * 78)
    for r in rows:
        dist_str = f"{r['distance_mean']:.2f} +/- {r['distance_sd']:.2f}"
        qos_str = "100%" if r["qos_mean"] >= 99.80 else f"{r['qos_mean']:.2f}%"
        time_str = f"{max(1, int(round(r['elapsed_s'])))}s"
        print(f"{r['method']:<10} | phi = {r['rate']*100:>4.0f}%    | {dist_str:<24} | {qos_str:<10} | {time_str:<8}")
    print("=" * 78)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DCVRP models.")
    parser.add_argument(
        "--method",
        type=str,
        default="DVNDA",
        choices=AVAILABLE_METHODS + ["all"],
        help="Method to evaluate (default: DVNDA, or 'all').",
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
        help="Path to model checkpoint. Defaults to checkpoints/{method}.pt.",
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
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.vehicle_count is None:
        args.vehicle_count = max(1, round(args.customer_count / 5))

    print(
        f"Loading test instances (n={args.customer_count}, m={args.vehicle_count}, seed={args.seed})..."
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
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            if args.customer_count != DEFAULT_CUSTOMER_COUNT:
                scale_candidate = Path("checkpoints") / f"{method}_n{args.customer_count}.pt"
                if scale_candidate.exists():
                    checkpoint_path = scale_candidate
                else:
                    checkpoint_path = Path("checkpoints") / f"{method}.pt"
            else:
                checkpoint_path = Path("checkpoints") / f"{method}.pt"

        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Skipping {method}.")
            continue

        print(f"Evaluating {method} on {device} (checkpoint: {checkpoint_path})...")
        rows = evaluate_method(
            method,
            checkpoint_path,
            split,
            device,
            vehicle_count=args.vehicle_count,
        )
        all_rows.extend(rows)

    if all_rows:
        print_results(all_rows)


if __name__ == "__main__":
    main()
