"""Evaluate aggregation rules and vehicle-identity permutation sensitivity."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .protocol import evaluate_one, load_model
from .selectors import build_selector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=Path("experiments/checkpoints/architecture"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/aggregation_ordering"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345, 3456])
    parser.add_argument("--test-size", type=int, default=60)
    parser.add_argument("--stochastic-repeats", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    frames = []
    test_seed = 20_267_000
    for train_seed in args.seeds:
        checkpoint = args.checkpoint_root / f"independent_seed{train_seed}.pt"
        model, config, _, _ = load_model(checkpoint, device)
        config = replace(config, test_size=args.test_size)
        original_networks = list(model.selector.vehicle_networks)

        # Current parameter-free argmax aggregation.
        model.selector.evaluation_rule = "argmax"
        frame = evaluate_one(
            model, "independent", train_seed, config, device,
            test_seed=test_seed, scenario="aggregation_argmax",
        )
        frame["replicate"] = 0
        frames.append(frame)

        # Softmax sampling over the same independent scores.
        model.selector.evaluation_rule = "softmax"
        for replicate in range(args.stochastic_repeats):
            torch.manual_seed(80_000 + train_seed + replicate)
            frame = evaluate_one(
                model, "independent", train_seed, config, device,
                test_seed=test_seed, scenario="aggregation_softmax",
            )
            frame["replicate"] = replicate
            frames.append(frame)

        # Random vehicle selection keeps the trained node decoder unchanged.
        learned_selector = model.selector
        model.selector = build_selector("random", config.vehicle_count).to(device)
        for replicate in range(args.stochastic_repeats):
            torch.manual_seed(90_000 + train_seed + replicate)
            frame = evaluate_one(
                model, "independent", train_seed, config, device,
                test_seed=test_seed, scenario="aggregation_random",
            )
            frame["replicate"] = replicate
            frames.append(frame)
        model.selector = learned_selector

        # Shuffle the mapping between trained subnetworks and vehicle IDs.
        model.selector.evaluation_rule = "argmax"
        rng = np.random.default_rng(100_000 + train_seed)
        for replicate in range(args.permutations):
            permutation = rng.permutation(config.vehicle_count).tolist()
            model.selector.vehicle_networks = nn.ModuleList(
                [original_networks[index] for index in permutation]
            )
            frame = evaluate_one(
                model, "independent", train_seed, config, device,
                test_seed=test_seed, scenario="vehicle_id_permuted",
            )
            frame["replicate"] = replicate
            frame["vehicle_network_permutation"] = "-".join(map(str, permutation))
            frames.append(frame)
        model.selector.vehicle_networks = nn.ModuleList(original_networks)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw_dir = args.output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(raw_dir / "aggregation_ordering_instances.csv", index=False)
    print(raw.groupby("scenario")[[
        "cost_with_penalty", "distance", "qos", "response_time_min"
    ]].mean())


if __name__ == "__main__":
    main()

