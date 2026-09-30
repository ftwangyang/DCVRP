"""Evaluate trained DCVRP models and the Greedy baseline.

Reports mean route cost, standard deviation, and QoS at dynamism levels
phi in {0.10, 0.25, 0.50, 0.75}.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import t as student_t

from env import (
    DEFAULT_CUSTOMER_COUNT,
    DEFAULT_DECISION_INTERVALS,
    DEFAULT_DISTRIBUTION,
    DEFAULT_DYNAMIC_RATES,
    DEFAULT_HORIZON,
    DEFAULT_REVELATION,
    DEFAULT_VEHICLE_CAPACITY,
    DEFAULT_VEHICLE_COUNT,
    DEFAULT_VEHICLE_SPEED,
    DCVRPEnvironment,
    REVELATION_MODES,
    SPATIAL_DISTRIBUTIONS,
    generate_evaluation_split,
    run_greedy,
)
from models import AttentionLearner, build_selector

AVAILABLE_METHODS = ["Greedy", "DVNDA", "AMCVN", "LiDRL", "MAAM", "MARDAM"]


def evaluate_greedy(
    split: dict[float, object],
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
    vehicle_count: int = DEFAULT_VEHICLE_COUNT,
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
            "dist_sem": dist_sem,
            "ci95_low": dist_mean - ci95,
            "ci95_high": dist_mean + ci95,
            "qos_mean": qos_mean,
            "elapsed_s": elapsed,
            "customer_count": customer_count,
            "vehicle_count": vehicle_count,
        })
    return rows


def evaluate_neural(
    method: str,
    checkpoint_path: Path,
    split: dict[float, object],
    device: torch.device,
    vehicle_count: int = DEFAULT_VEHICLE_COUNT,
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
    disclose_horizon_tail: bool = False,
) -> list[dict]:
    """Evaluate a trained neural model with greedy decoding."""
    selector = build_selector(method, vehicle_count=vehicle_count)
    model = AttentionLearner(selector)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    model.greedy = True
    model.vehicle_greedy = True
    model = model.to(device)

    if len(split) > 0 and device.type == "cuda":
        first_dataset = next(iter(split.values()))
        warmup_env = DCVRPEnvironment(
            first_dataset,
            nodes=first_dataset.nodes[:2].to(device),
            pending_cost=0.0,
            disclose_horizon_tail=disclose_horizon_tail,
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
            disclose_horizon_tail=disclose_horizon_tail,
        )
        with torch.no_grad():
            model(env)
        distances_t = env.route_distance()
        qos_t = env.qos()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time

        distances = distances_t.detach().cpu().numpy()
        qos_values = (100.0 * qos_t).detach().cpu().numpy()

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
            "dist_sem": dist_sem,
            "ci95_low": dist_mean - ci95,
            "ci95_high": dist_mean + ci95,
            "qos_mean": qos_mean,
            "elapsed_s": elapsed,
            "customer_count": customer_count,
            "vehicle_count": vehicle_count,
            "checkpoint": str(Path(checkpoint_path).name),
        })
    return rows


def print_results(rows: list[dict]):
    """Format and print standard evaluation metrics."""
    print("=" * 86)
    print(
        f"{'Scale':<6} | {'Method':<8} | {'Dynamic Rate':<12} | "
        f"{'Distance (Mean +/- SD)':<24} | {'QoS (%)':<10} | {'Time':<6}"
    )
    print("-" * 86)
    for r in rows:
        scale_str = f"n={r['customer_count']}"
        dist_str = f"{r['distance_mean']:.2f} +/- {r['distance_sd']:.2f}"
        qos_str = f"{r['qos_mean']:.2f}%" if r["qos_mean"] < 100.0 else "100.0%"
        if r["method"] == "Greedy":
            time_str = "-"
        else:
            time_sec = max(1, int(round(r["elapsed_s"])))
            time_str = f"{time_sec}s"
        rate_str = f"phi = {r['rate']*100:>4.0f}%"
        print(
            f"{scale_str:<6} | {r['method']:<8} | {rate_str:<12} | "
            f"{dist_str:<24} | {qos_str:<10} | {time_str:<6}"
        )
    print("=" * 86)


def export_results(
    rows: list[dict],
    save_dir: Path,
    device: torch.device,
    seed: int,
    instances: int,
) -> None:
    """Save evaluation results to CSV, JSON, and Markdown in save_dir."""
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    # 1. Export CSV
    csv_path = save_dir / "evaluation_results.csv"
    fieldnames = [
        "scale",
        "vehicle_count",
        "method",
        "phi",
        "measured_cost_mean",
        "measured_cost_sd",
        "measured_cost_sem",
        "ci95_low",
        "ci95_high",
        "measured_qos_percent",
        "wall_time_sec",
        "checkpoint",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "scale": r["customer_count"],
                "vehicle_count": r["vehicle_count"],
                "method": r["method"],
                "phi": round(r["rate"], 2),
                "measured_cost_mean": round(r["distance_mean"], 4),
                "measured_cost_sd": round(r["distance_sd"], 4),
                "measured_cost_sem": round(r.get("dist_sem", 0.0), 4),
                "ci95_low": round(r["ci95_low"], 4),
                "ci95_high": round(r["ci95_high"], 4),
                "measured_qos_percent": round(r["qos_mean"], 4),
                "wall_time_sec": round(r["elapsed_s"], 4),
                "checkpoint": r.get("checkpoint", "heuristic" if r["method"] == "Greedy" else ""),
            })
    print(f"Exported raw tabular results to: {csv_path}")

    # 2. Export JSON
    json_path = save_dir / "evaluation_results.json"
    data_payload = {
        "metadata": {
            "timestamp": timestamp,
            "device": str(device),
            "device_name": device_name,
            "seed": seed,
            "instances_per_rate": instances,
            "total_evaluations": len(rows),
        },
        "records": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_payload, f, indent=2, ensure_ascii=False)
    print(f"Exported machine-readable metadata to: {json_path}")

    # 3. Export Markdown
    md_path = save_dir / "evaluation_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# DCVRP evaluation\n\n")
        f.write(f"- Generated at: {timestamp}\n")
        f.write(f"- Device: {device_name} (`{device}`)\n")
        f.write(f"- Seed: {seed}\n")
        f.write(f"- Instances per dynamic rate: {instances}\n\n")
        f.write("## Evaluation Results\n\n")
        f.write(
            "| Scale | Method | $\\phi$ | Measured Cost | Measured QoS |\n"
        )
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for r in rows:
            scale = r["customer_count"]
            rate = round(r["rate"], 2)
            method = r["method"]
            meas_cost = f"{r['distance_mean']:.2f} ± {r['distance_sd']:.2f}"
            meas_qos = f"{r['qos_mean']:.2f}%"
            rate_str = f"{int(rate * 100)}%"
            f.write(
                f"| n={scale} | {method} | {rate_str} | {meas_cost} | {meas_qos} |\n"
            )
    print(f"Exported Markdown summary to: {md_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DCVRP models and the Greedy baseline.",
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
        type=str,
        default=str(DEFAULT_CUSTOMER_COUNT),
        help="Number of customer locations (20, 35, 50, or 'all' for all 3 scales).",
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
        "--save-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save structured evaluation results (CSV, JSON, Markdown).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving evaluation results to disk.",
    )
    parser.add_argument(
        "--revelation",
        type=str,
        default=DEFAULT_REVELATION,
        choices=list(REVELATION_MODES),
        help="Arrival process: hpp (Uniform(0, T]) or poisson (Eq. 34 PMF).",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default=DEFAULT_DISTRIBUTION,
        choices=list(SPATIAL_DISTRIBUTIONS),
        help="Spatial distribution of customers (uniform, clustered, mixed, real).",
    )
    parser.add_argument(
        "--real-data-path",
        type=str,
        default=None,
        help="Path to CSV file with coordinates for 'real' distribution.",
    )
    parser.add_argument(
        "--disclose-horizon-tail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disclose a_i <= T_{r+1} at the start of interval r (Algorithm 1). Default False uses a_i <= T_r.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Determine scales to evaluate
    if args.customer_count.strip().lower() == "all":
        scales = [20, 35, 50]
    else:
        scales = [int(args.customer_count)]

    methods = AVAILABLE_METHODS if args.method == "all" else [args.method]
    all_rows: list[dict] = []

    for n in scales:
        m = args.vehicle_count if args.vehicle_count is not None else max(1, round(n / 5))
        print(
            f"\n>>> Generating test split for Scale n={n}, m={m} "
            f"({args.instances} instances/rate, seed={args.seed})..."
        )
        split = generate_evaluation_split(
            instances=args.instances,
            dynamic_rates=args.rates,
            customer_count=n,
            vehicle_count=m,
            seed=args.seed,
            revelation=args.revelation,
            distribution=args.distribution,
            real_data_path=args.real_data_path,
        )

        for method in methods:
            if method == "Greedy":
                print(f"Evaluating Greedy heuristic baseline on CPU (n={n}, m={m})...")
                rows = evaluate_greedy(split, customer_count=n, vehicle_count=m)
                all_rows.extend(rows)
                continue

            checkpoint_path = args.checkpoint
            if checkpoint_path is None:
                if n == DEFAULT_CUSTOMER_COUNT:
                    checkpoint_path = Path("checkpoints") / f"{method}.pt"
                else:
                    checkpoint_path = Path("checkpoints") / f"{method}_n{n}.pt"

            if not checkpoint_path.exists():
                print(f"Warning: Checkpoint not found at {checkpoint_path}. Skipping {method}.")
                continue

            print(
                f"Evaluating {method} on {device} (n={n}, m={m}, "
                f"checkpoint: {checkpoint_path})..."
            )
            rows = evaluate_neural(
                method,
                checkpoint_path,
                split,
                device,
                vehicle_count=m,
                customer_count=n,
                disclose_horizon_tail=args.disclose_horizon_tail,
            )
            all_rows.extend(rows)

    if all_rows:
        print_results(all_rows)
        if not args.no_save:
            export_results(
                all_rows,
                save_dir=args.save_dir,
                device=device,
                seed=args.seed,
                instances=args.instances,
            )


if __name__ == "__main__":
    main()
