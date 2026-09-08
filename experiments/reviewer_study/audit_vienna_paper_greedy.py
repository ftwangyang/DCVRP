"""Audit the local Vienna manifests against the manuscript Greedy Table II."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import DCVRP_Dataset
from experiments.reviewer_study.paper_greedy import run_paper_greedy
from experiments.reviewer_study.paper_multivehicle_env import paper_vehicle_count


SIZES = (20, 35, 50)
RATES = (0.10, 0.25, 0.50, 0.75)
PAPER_GREEDY = {
    (0.10, 20): (5.57, 0.86, 99.85),
    (0.10, 35): (9.18, 1.13, 99.95),
    (0.10, 50): (12.70, 1.48, 99.95),
    (0.25, 20): (6.09, 0.98, 99.85),
    (0.25, 35): (10.13, 1.30, 100.00),
    (0.25, 50): (13.95, 1.52, 99.95),
    (0.50, 20): (7.11, 1.02, 99.90),
    (0.50, 35): (12.03, 1.47, 99.95),
    (0.50, 50): (16.50, 1.71, 99.95),
    (0.75, 20): (7.77, 0.88, 99.90),
    (0.75, 35): (13.37, 1.43, 99.85),
    (0.75, 50): (18.90, 1.69, 99.90),
}


def load_dataset(path: Path, customer_count: int) -> DCVRP_Dataset:
    with np.load(path) as values:
        coordinates = values["coordinates"]
        demands = values["demands"]
        service = values["service_minutes"]
        disclosure = values["disclosure_minutes"]
    batch = coordinates.shape[0]
    customers = np.concatenate(
        [
            coordinates[:, 1:, :],
            (demands / 150.0)[..., None],
            (service / 480.0)[..., None],
            (disclosure / 480.0)[..., None],
        ],
        axis=2,
    ).astype(np.float32)
    depot = np.zeros((batch, 1, 5), dtype=np.float32)
    depot[:, 0, :2] = coordinates[:, 0, :]
    nodes = torch.tensor(np.concatenate([depot, customers], axis=1))
    # One normalized distance unit represents 100 physical travel minutes.
    return DCVRP_Dataset(
        paper_vehicle_count(customer_count), 1.0, 4.8, nodes, None
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_dir = args.manifest_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in SIZES:
        for rate in RATES:
            path = manifest_dir / f"vienna_n{size}_r{int(round(100 * rate)):02d}.npz"
            data = load_dataset(path, size)
            cost, qos, _ = run_paper_greedy(data, torch.device("cpu"))
            target, target_sd, target_qos = PAPER_GREEDY[(rate, size)]
            mean = float(cost.mean())
            rows.append(
                {
                    "customer_count": size,
                    "vehicle_count": paper_vehicle_count(size),
                    "dynamic_rate": rate,
                    "instances": len(data),
                    "cost_mean": mean,
                    "cost_sd": float(cost.std(unbiased=True)),
                    "qos_mean_percent": float(100.0 * qos.mean()),
                    "qos_sd_percent": float(100.0 * qos.std(unbiased=True)),
                    "paper_cost_mean": target,
                    "paper_cost_sd": target_sd,
                    "paper_qos_percent": target_qos,
                    "error_percent": 100.0 * (mean / target - 1.0),
                }
            )
            print(
                f"n={size} rate={100 * rate:.0f}% local={mean:.4f} "
                f"paper={target:.4f} error={rows[-1]['error_percent']:+.2f}%",
                flush=True,
            )
    result = pd.DataFrame(rows)
    absolute = result.error_percent.abs()
    acceptance = {
        "mape_percent": float(absolute.mean()),
        "maximum_absolute_error_percent": float(absolute.max()),
        "mape_below_3_percent": bool(absolute.mean() < 3.0),
        "maximum_error_below_5_percent": bool(absolute.max() < 5.0),
        "accepted": bool(absolute.mean() < 3.0 and absolute.max() < 5.0),
    }
    result.to_csv(output_dir / "greedy_audit.csv", index=False)
    (output_dir / "acceptance.json").write_text(
        json.dumps(acceptance, indent=2), encoding="utf-8"
    )
    print(json.dumps(acceptance, indent=2), flush=True)


if __name__ == "__main__":
    main()
