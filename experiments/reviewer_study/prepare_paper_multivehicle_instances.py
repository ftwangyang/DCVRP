"""Prepare immutable released-generator test manifests for CPU solvers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paper_multivehicle_env import generate_release_batch, paper_vehicle_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 35, 50])
    parser.add_argument(
        "--rates", nargs="+", type=float, default=[0.10, 0.25, 0.50, 0.75]
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--seed-policy",
        choices=("indexed", "fixed"),
        default="indexed",
        help=(
            "indexed gives every size/rate a derived seed; fixed resets the "
            "published validation seed for every group"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for customer_count in args.sizes:
        paper_vehicle_count(customer_count)
        for dynamic_rate in args.rates:
            group_seed = int(args.seed)
            if args.seed_policy == "indexed":
                group_seed += (
                    10_000 * int(customer_count)
                    + 100 * int(round(100.0 * dynamic_rate))
                )
            instances = generate_release_batch(
                customer_count,
                args.instances,
                dynamic_rate,
                group_seed,
            )
            path = output_dir / (
                f"release_n{customer_count}_r{int(round(100 * dynamic_rate)):02d}.npz"
            )
            np.savez_compressed(
                path,
                coordinates=np.stack([item.coordinates for item in instances]),
                demands=np.stack([item.demands for item in instances]),
                service_minutes=np.stack(
                    [item.service_minutes for item in instances]
                ),
                disclosure_minutes=np.stack(
                    [item.disclosure_minutes for item in instances]
                ),
            )
            realized = np.stack(
                [item.disclosure_minutes > 0.0 for item in instances]
            ).mean()
            manifest_rows.append(
                {
                    "customer_count": customer_count,
                    "vehicle_count": paper_vehicle_count(customer_count),
                    "nominal_dynamic_rate": dynamic_rate,
                    "realized_dynamic_rate": float(realized),
                    "instances": args.instances,
                    "seed": group_seed,
                    "seed_policy": args.seed_policy,
                    "file": path.name,
                }
            )
            print(
                f"saved {path.name}: realized dynamic rate={100 * realized:.2f}%",
                flush=True,
            )
    (output_dir / "manifest_index.json").write_text(
        json.dumps(manifest_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
