"""Measure selector parameter, latency, and memory scaling to 100 vehicles."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .selectors import build_selector


def benchmark_selector(
    selector_name: str,
    vehicle_count: int,
    customer_count: int,
    device: torch.device,
    repeats: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    selector = build_selector(
        selector_name,
        vehicle_count,
        selector_size=32,
        selector_heads=4,
    ).to(device).eval()
    vehicles = torch.rand((1, vehicle_count, 4), device=device)
    customers = torch.rand((1, customer_count + 1, 5), device=device)
    vehicle_done = torch.zeros((1, vehicle_count), dtype=torch.bool, device=device)
    customer_mask = torch.zeros((1, customer_count + 1), dtype=torch.bool, device=device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(3):
            selector.select(vehicles, customers, vehicle_done, customer_mask, greedy=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            selector.select(vehicles, customers, vehicle_done, customer_mask, greedy=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    parameters = sum(parameter.numel() for parameter in selector.parameters())
    return {
        "selector": selector_name,
        "vehicle_count": vehicle_count,
        "customer_count": customer_count,
        "parameter_count": parameters,
        "model_size_mb_fp32": parameters * 4 / (1024 ** 2),
        "latency_mean_ms": float(np.mean(timings)),
        "latency_p95_ms": float(np.quantile(timings, 0.95)),
        "latency_std_ms": float(np.std(timings, ddof=1)),
        "peak_memory_mb": peak_memory / (1024 ** 2),
        "repeats": repeats,
        "device": str(device),
        "seed": seed,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/scaling"))
    parser.add_argument("--selectors", nargs="+", default=["shared", "centralized", "independent"])
    parser.add_argument("--fleet-sizes", nargs="+", type=int, default=[4, 10, 20, 50, 100])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345, 3456])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    rows = []
    for selector in args.selectors:
        for vehicle_count in args.fleet_sizes:
            customer_count = 5 * vehicle_count
            for seed in args.seeds:
                print(
                    f"SCALE selector={selector} vehicles={vehicle_count} seed={seed}",
                    flush=True,
                )
                rows.append(benchmark_selector(
                    selector,
                    vehicle_count,
                    customer_count,
                    device,
                    repeats=args.repeats,
                    seed=seed,
                ))
    raw_dir = args.output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    data.to_csv(raw_dir / "selector_scaling.csv", index=False)
    print(data.groupby(["selector", "vehicle_count"])[
        ["parameter_count", "latency_mean_ms", "peak_memory_mb"]
    ].mean())


if __name__ == "__main__":
    main()

