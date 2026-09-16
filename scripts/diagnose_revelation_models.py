"""Compare paper-literal and process-based revelation-time interpretations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import DCVRPDataset, run_greedy, set_seed
from eval import TABLE1_BENCHMARK, evaluate_neural


RATES = (0.10, 0.25, 0.50, 0.75)


def make_split(instances: int, mode: str, seed: int):
    set_seed(seed)
    n, horizon, capacity = 20, 480.0, 150.0
    xy = torch.rand(instances, n + 1, 2)
    demand = torch.randint(5, 42, (instances, n, 1)).float()
    service = torch.randint(10, 32, (instances, n, 1)).float()
    if mode == "literal_poisson":
        disclosure = torch.poisson(torch.full((instances, n, 1), 240.5)).clamp_(1, 480)
    elif mode == "conditional_process":
        # Conditional on n arrivals in (0,T], homogeneous Poisson-process
        # event times are the order statistics of n independent uniforms.
        disclosure = torch.rand(instances, n, 1).mul_(horizon).clamp_min_(1.0)
    elif mode == "indexed_poisson":
        means = torch.linspace(1.0, horizon, n).view(1, n, 1).expand(instances, -1, -1)
        disclosure = torch.poisson(means).clamp_(1, 480)
    else:
        raise ValueError(mode)

    order = torch.argsort(torch.rand(instances, n), dim=1)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(n).view(1, -1).expand(instances, -1))
    split = {}
    for rate in RATES:
        release = torch.where(
            (ranks < round(n * rate)).unsqueeze(-1), disclosure, torch.zeros_like(disclosure)
        )
        customers = torch.cat((xy[:, 1:], demand / capacity, service / horizon, release / horizon), dim=2)
        depot = torch.zeros(instances, 1, 5)
        depot[:, :, :2] = xy[:, :1]
        split[rate] = DCVRPDataset(4, 1.0, 480.0, torch.cat((depot, customers), dim=1))
    return split, disclosure


def summarize(split, disclosure, checkpoint: Path | None):
    output = {
        "revelation_mean": float(disclosure.mean()),
        "revelation_sd": float(disclosure.std()),
        "greedy": [],
    }
    for rate, data in split.items():
        costs, qos = run_greedy(data, reveal="continuous")
        target = TABLE1_BENCHMARK[20][rate]["Greedy"][0]
        mean = float(costs.mean())
        output["greedy"].append({
            "phi": rate, "cost": mean, "qos": float(qos.mean() * 100),
            "gap": (mean / target - 1.0) * 100.0,
        })
    if checkpoint is not None:
        output["dvnda"] = evaluate_neural("DVNDA", checkpoint, split, torch.device("cpu"))
        for row in output["dvnda"]:
            target = TABLE1_BENCHMARK[20][round(float(row["rate"]), 2)]["DVNDA"][0]
            row["gap"] = (float(row["distance_mean"]) / target - 1.0) * 100.0
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for mode in ("literal_poisson", "conditional_process", "indexed_poisson"):
        split, disclosure = make_split(args.instances, mode, args.seed)
        result[mode] = summarize(split, disclosure, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
