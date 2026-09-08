"""Write CPU-baseline manifests from the same generator as DVNDA testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from .paper_dcvrp import (
    PAPER_HORIZON_MINUTES,
    PAPER_VEHICLE_CAPACITY,
    generate_paired_paper_datasets,
)


RATES = (0.10, 0.25, 0.50, 0.75)
SIZES = (20, 35, 50)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "generator": "generate_paired_paper_datasets",
        "instances_per_rate": args.instances,
        "base_seed": args.seed,
        "rates": list(RATES),
        "sizes": list(SIZES),
        "coordinate_range": [0.0, 1.0],
        "vehicle_capacity": PAPER_VEHICLE_CAPACITY,
        "physical_speed_normalized_units_per_minute": 1.0,
        "horizon_minutes": PAPER_HORIZON_MINUTES,
        "nested_dynamic_sets_within_size": True,
        "size_seeds": {},
    }
    for customer_count in SIZES:
        size_seed = args.seed + 10_000 * customer_count
        set_seed(size_seed)
        datasets = generate_paired_paper_datasets(
            args.instances,
            RATES,
            customer_count=customer_count,
            vehicle_count=customer_count // 5,
        )
        metadata["size_seeds"][str(customer_count)] = size_seed
        for rate, dataset in datasets.items():
            nodes = dataset.nodes.detach().cpu().numpy()
            path = args.output_dir / (
                f"release_n{customer_count}_r{int(round(100 * rate)):02d}.npz"
            )
            np.savez_compressed(
                path,
                coordinates=nodes[:, :, :2].astype(np.float64),
                demands=np.rint(
                    nodes[:, 1:, 2] * PAPER_VEHICLE_CAPACITY
                ).astype(np.int64),
                service_minutes=np.rint(
                    nodes[:, 1:, 3] * PAPER_HORIZON_MINUTES
                ).astype(np.float64),
                disclosure_minutes=(
                    nodes[:, 1:, 4] * PAPER_HORIZON_MINUTES
                ).astype(np.float64),
            )
            print(path, flush=True)
    (args.output_dir / "manifest_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

