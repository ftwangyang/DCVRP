"""Create immutable Vienna real-world manifests for Table-II reproduction.

The manuscript states that the Vienna coordinates are normalized to the unit
square.  Therefore x and y are normalized independently over the complete
100-instance batch.  This avoids compressing one city axis merely because the
projected x/y origins differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.reviewer_study.paper_multivehicle_env import (
    HORIZON_MINUTES,
    paper_vehicle_count,
)


SIZES = (20, 35, 50)
RATES = (0.10, 0.25, 0.50, 0.75)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("vienna_data")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--rates", nargs="+", type=float, default=list(RATES))
    parser.add_argument(
        "--normalization",
        choices=("global-per-axis", "global-aspect", "batch-per-axis", "batch-aspect"),
        default="global-per-axis",
        help=(
            "global modes normalize all 16,080 Vienna nodes before sampling, "
            "as stated in the manuscript; batch modes are retained only for "
            "diagnostic comparison"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.instances != 100:
        raise ValueError("the declared real-world experiment requires 100 instances")
    data_dir = args.data_dir.resolve()
    coordinate_path = data_dir / "vienna_cordinates.csv"
    arc_path = data_dir / "vienna.d"
    frame = pd.read_csv(coordinate_path)
    required = {"id", "xcoords", "ycoords"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing Vienna columns: {sorted(required - set(frame.columns))}")
    raw_city_coordinates = frame[["xcoords", "ycoords"]].to_numpy(dtype=np.float64)
    node_ids = frame["id"].to_numpy(dtype=np.int64)
    if raw_city_coordinates.shape != (16080, 2):
        raise ValueError(
            f"expected 16,080 Vienna nodes, got {raw_city_coordinates.shape[0]}"
        )
    with arc_path.open("r", encoding="utf-8") as stream:
        arc_count = int(stream.readline().strip())
    if arc_count != 36424:
        raise ValueError(f"expected 36,424 Vienna arcs, got {arc_count}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    city_minimum = raw_city_coordinates.min(axis=0)
    city_axis_span = raw_city_coordinates.max(axis=0) - city_minimum
    if np.any(city_axis_span <= 0.0):
        raise RuntimeError("degenerate Vienna city coordinate range")
    city_common_span = np.full(2, float(city_axis_span.max()))

    for size in args.sizes:
        paper_vehicle_count(size)
        for rate in args.rates:
            if not 0.0 <= rate <= 1.0:
                raise ValueError("dynamic rates must lie in [0, 1]")
            group_seed = (
                args.seed
                + 10_000 * size
                + 100 * int(round(100.0 * rate))
            )
            coordinate_rng = np.random.default_rng(group_seed)
            selections = np.stack(
                [
                    coordinate_rng.choice(
                        len(node_ids), size=size + 1, replace=False
                    )
                    for _ in range(args.instances)
                ]
            )
            raw_coordinates = raw_city_coordinates[selections]
            if args.normalization.startswith("global-"):
                coordinate_minimum = city_minimum
                coordinate_scale = (
                    city_axis_span
                    if args.normalization == "global-per-axis"
                    else city_common_span
                )
            else:
                coordinate_minimum = raw_coordinates.min(axis=(0, 1))
                axis_span = raw_coordinates.max(axis=(0, 1)) - coordinate_minimum
                if np.any(axis_span <= 0.0):
                    raise RuntimeError("degenerate Vienna coordinate sample")
                coordinate_scale = (
                    axis_span
                    if args.normalization == "batch-per-axis"
                    else np.full(2, float(axis_span.max()))
                )
            coordinates = (
                raw_coordinates - coordinate_minimum.reshape(1, 1, 2)
            ) / coordinate_scale.reshape(1, 1, 2)

            generator = torch.Generator(device="cpu")
            generator.manual_seed(group_seed + 1)
            demands = torch.randint(
                5,
                41,
                (args.instances, size),
                generator=generator,
                dtype=torch.int64,
            ).numpy()
            service = torch.randint(
                10,
                31,
                (args.instances, size),
                generator=generator,
                dtype=torch.int64,
            ).to(torch.float64).numpy()
            dynamic = (
                torch.rand(
                    (args.instances, size), generator=generator,
                    dtype=torch.float64,
                )
                < float(rate)
            )
            poisson_rates = torch.linspace(
                1.0, HORIZON_MINUTES, size, dtype=torch.float64
            )
            # Public ``create_logistics_distribution`` omits batch_size here;
            # one sampled disclosure vector is broadcast to all 100 rows.
            disclosure_sample = torch.poisson(
                poisson_rates, generator=generator
            ).clamp_(min=1.0, max=HORIZON_MINUTES)
            disclosure = (
                dynamic.to(torch.float64) * disclosure_sample.view(1, size)
            ).numpy()

            filename = f"vienna_n{size}_r{int(round(100 * rate)):02d}.npz"
            path = output_dir / filename
            np.savez_compressed(
                path,
                coordinates=coordinates.astype(np.float64),
                demands=demands.astype(np.int64),
                service_minutes=service.astype(np.float64),
                disclosure_minutes=disclosure.astype(np.float64),
                node_ids=node_ids[selections],
            )
            realized_rate = float((disclosure > 0.0).mean())
            records.append(
                {
                    "customer_count": size,
                    "vehicle_count": paper_vehicle_count(size),
                    "nominal_dynamic_rate": rate,
                    "realized_dynamic_rate": realized_rate,
                    "instances": args.instances,
                    "seed": group_seed,
                    "coordinate_minimum": coordinate_minimum.tolist(),
                    "coordinate_scale": coordinate_scale.tolist(),
                    "travel_minutes_per_normalized_unit": 100.0,
                    "file": filename,
                    "sha256": sha256(path),
                }
            )
            print(
                f"saved {filename}: scale_x={coordinate_scale[0]:.6f} "
                f"scale_y={coordinate_scale[1]:.6f} "
                f"realized={100.0 * realized_rate:.2f}%",
                flush=True,
            )

    metadata = {
        "source_directory": str(data_dir),
        "source_coordinate_file": str(coordinate_path),
        "source_coordinate_sha256": sha256(coordinate_path),
        "source_arc_file": str(arc_path),
        "source_arc_sha256": sha256(arc_path),
        "vienna_nodes": int(len(node_ids)),
        "vienna_directed_arcs": arc_count,
        "instances_per_group": args.instances,
        "base_seed": args.seed,
        "normalization": (
            "all 16,080 Vienna nodes are normalized before instance sampling; "
            "x and y are independently min-max normalized to [0,1]"
            if args.normalization == "global-per-axis"
            else args.normalization
        ),
        "normalization_mode": args.normalization,
        "dynamic_membership": "Bernoulli at the nominal rate",
        "disclosure": (
            "one Poisson vector with rates linspace(1,480,n), broadcast over "
            "the 100-instance group"
        ),
        "groups": records,
    }
    (output_dir / "manifest_index.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
