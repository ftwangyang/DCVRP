"""Run rolling-horizon classical baseline experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .classical_baselines import PLANNERS, simulate_instance
from .data_generation import generate_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/classical"))
    parser.add_argument("--methods", nargs="+", default=list(PLANNERS))
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--customers", type=int, default=20)
    parser.add_argument("--vehicles", type=int, default=4)
    parser.add_argument("--dynamic-ratio", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = generate_dataset(
        args.instances,
        customer_count=args.customers,
        vehicle_count=args.vehicles,
        dynamic_ratio=args.dynamic_ratio,
        arrival_profile="uniform",
        seed=args.seed,
    )
    rows = []
    for method in args.methods:
        print(f"CLASSICAL method={method}", flush=True)
        for instance in range(args.instances):
            result = simulate_instance(
                dataset.nodes[instance].numpy(),
                args.vehicles,
                method,
                seed=args.seed + instance,
            )
            rows.append({
                "method": method,
                "instance": instance,
                "seed": args.seed,
                "dynamic_ratio": args.dynamic_ratio,
                **asdict(result),
            })
    raw_dir = args.output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    data.to_csv(raw_dir / "classical_instances.csv", index=False)
    print(data.groupby("method")[[
        "cost_with_penalty", "distance", "qos", "response_time_min", "planning_time_ms"
    ]].mean())


if __name__ == "__main__":
    main()

