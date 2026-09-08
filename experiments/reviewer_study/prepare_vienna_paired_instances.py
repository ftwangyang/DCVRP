"""Prepare paired Vienna instances for the final classical comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.reviewer_study.paper_multivehicle_env import HORIZON_MINUTES, paper_vehicle_count
from experiments.reviewer_study.prepare_vienna_multivehicle_instances import sha256


SIZES = (20, 35, 50)
RATES = (0.10, 0.25, 0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("vienna_data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.instances != 100:
        raise ValueError("the final Vienna experiment requires 100 instances")
    data_dir = args.data_dir.resolve()
    coordinate_path = data_dir / "vienna_cordinates.csv"
    arc_path = data_dir / "vienna.d"
    frame = pd.read_csv(coordinate_path)
    city_coordinates = frame[["xcoords", "ycoords"]].to_numpy(dtype=np.float64)
    node_ids = frame["id"].to_numpy(dtype=np.int64)
    if city_coordinates.shape != (16080, 2):
        raise ValueError("the Vienna coordinate file must contain 16,080 nodes")
    with arc_path.open("r", encoding="utf-8") as stream:
        arc_count = int(stream.readline())
    if arc_count != 36424:
        raise ValueError("the Vienna arc file must contain 36,424 directed arcs")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for size in SIZES:
        group_seed = args.seed + 10_000 * size
        rng = np.random.default_rng(group_seed)
        selections = np.stack(
            [
                rng.choice(len(node_ids), size=size + 1, replace=False)
                for _ in range(args.instances)
            ]
        )
        raw = city_coordinates[selections]
        # The manuscript normalizes all Vienna nodes before sampling test
        # instances.  Independent x/y min-max scaling is the conventional
        # interpretation of mapping city coordinates to [0, 1]^2.
        minimum = city_coordinates.min(axis=0)
        span = city_coordinates.max(axis=0) - minimum
        if np.any(span <= 0.0):
            raise RuntimeError("degenerate Vienna coordinate range")
        coordinates = (raw - minimum.reshape(1, 1, 2)) / span.reshape(1, 1, 2)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(group_seed + 1)
        demands = torch.randint(
            5, 41, (args.instances, size), generator=generator, dtype=torch.int64
        )
        service = torch.randint(
            10, 31, (args.instances, size), generator=generator, dtype=torch.int64
        ).to(torch.float64)
        poisson_rates = torch.linspace(
            1.0, HORIZON_MINUTES, size, dtype=torch.float64
        )
        # Match data.py: create_logistics_distribution is called without a
        # batch_size, so one Poisson vector is shared by the 100 instances.
        disclosure_sample = torch.poisson(
            poisson_rates, generator=generator
        ).clamp_(min=1.0, max=HORIZON_MINUTES)
        order = torch.rand(
            (args.instances, size), generator=generator, dtype=torch.float64
        ).argsort(dim=1)
        rank = torch.empty_like(order)
        rank.scatter_(
            1,
            order,
            torch.arange(size, dtype=torch.int64).view(1, size).expand(args.instances, -1),
        )

        for rate in RATES:
            dynamic_count = int(round(rate * size))
            dynamic = rank < dynamic_count
            disclosure = (
                dynamic.to(torch.float64) * disclosure_sample
            ).numpy()
            filename = f"vienna_n{size}_r{int(round(100 * rate)):02d}.npz"
            path = output_dir / filename
            np.savez_compressed(
                path,
                coordinates=coordinates.astype(np.float64),
                demands=demands.numpy().astype(np.int64),
                service_minutes=service.numpy().astype(np.float64),
                disclosure_minutes=disclosure.astype(np.float64),
                node_ids=node_ids[selections],
            )
            records.append(
                {
                    "customer_count": size,
                    "vehicle_count": paper_vehicle_count(size),
                    "nominal_dynamic_rate": rate,
                    "realized_dynamic_rate": float(dynamic.float().mean()),
                    "dynamic_customers_per_instance": dynamic_count,
                    "instances": args.instances,
                    "seed": group_seed,
                    "coordinate_minimum": minimum.tolist(),
                    "coordinate_minimum": minimum.tolist(),
                    "coordinate_scale": span.tolist(),
                    "travel_minutes_per_normalized_unit": 100.0,
                    "file": filename,
                    "sha256": sha256(path),
                }
            )
            print(
                f"saved {filename}: dynamic={dynamic_count}/{size}, "
                f"scale_x={span[0]:.6f}, scale_y={span[1]:.6f}",
                flush=True,
            )

    metadata = {
        "source_directory": str(data_dir),
        "source_coordinate_file": str(coordinate_path),
        "source_coordinate_sha256": sha256(coordinate_path),
        "source_arc_file": str(arc_path),
        "source_arc_sha256": sha256(arc_path),
        "vienna_nodes": 16080,
        "vienna_directed_arcs": arc_count,
        "instances_per_group": args.instances,
        "base_seed": args.seed,
        "normalization_mode": (
            "global per-axis min-max over all 16,080 Vienna nodes before "
            "sampling"
        ),
        "dynamic_design": (
            "paired physical instances with exact nested dynamic customer sets "
            "for 10%, 25%, 50%, and 75%"
        ),
        "disclosure": (
            "one shared Poisson vector with rates linspace(1,480,n), matching "
            "the released data.py batch behavior"
        ),
        "groups": records,
    }
    (output_dir / "manifest_index.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
