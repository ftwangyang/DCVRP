"""Compare classical planners with exact small-instance MIP references."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .classical_baselines import (
    alns_routes,
    local_search_routes,
    nearest_routes,
    ortools_routes,
    regret_routes,
    tabu_routes,
    _route_cost,
)
from .data_generation import generate_dataset
from .exact_mip import solve_cvrp_mip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/optimality"))
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--customers", type=int, default=8)
    parser.add_argument("--vehicles", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--mip-time-limit", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = generate_dataset(
        args.instances,
        customer_count=args.customers,
        vehicle_count=args.vehicles,
        dynamic_ratio=0.0,
        seed=args.seed,
    )
    rows = []
    for instance in range(args.instances):
        nodes = dataset.nodes[instance].numpy()
        coordinates = nodes[:, :2]
        demands = nodes[:, 2]
        exact_started = time.perf_counter()
        exact = solve_cvrp_mip(
            coordinates,
            demands,
            args.vehicles,
            time_limit_s=args.mip_time_limit,
        )
        exact_seconds = time.perf_counter() - exact_started
        if not np.isfinite(exact.objective):
            continue
        starts = np.repeat(coordinates[0][None, :], args.vehicles, axis=0)
        capacities = np.ones(args.vehicles)
        customers = list(range(1, len(coordinates)))
        methods = {
            "nearest": lambda: nearest_routes(starts, capacities, customers, coordinates, demands, coordinates[0]),
            "regret": lambda: regret_routes(starts, capacities, customers, coordinates, demands, coordinates[0]),
            "local_search": lambda: local_search_routes(starts, capacities, customers, coordinates, demands, coordinates[0]),
            "tabu": lambda: tabu_routes(starts, capacities, customers, coordinates, demands, coordinates[0]),
            "alns": lambda: alns_routes(
                starts, capacities, customers, coordinates, demands, coordinates[0],
                np.random.default_rng(args.seed + instance),
            ),
            "ortools": lambda: ortools_routes(
                starts, capacities, customers, coordinates, demands, coordinates[0]
            ),
        }
        for method, planner in methods.items():
            started = time.perf_counter()
            routes = planner()
            seconds = time.perf_counter() - started
            cost = sum(
                _route_cost(starts[vehicle], route, coordinates, coordinates[0])
                for vehicle, route in enumerate(routes)
            )
            rows.append({
                "instance": instance,
                "method": method,
                "exact_cost": exact.objective,
                "method_cost": cost,
                "optimality_gap_percent": 100.0 * (cost / exact.objective - 1.0),
                "method_seconds": seconds,
                "mip_seconds": exact_seconds,
                "mip_success": exact.success,
                "mip_gap": exact.mip_gap,
            })
    raw_dir = args.output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    data.to_csv(raw_dir / "optimality_gap_instances.csv", index=False)
    print(data.groupby("method")[["optimality_gap_percent", "method_seconds"]].mean())


if __name__ == "__main__":
    main()
