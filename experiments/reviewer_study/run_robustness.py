"""Run dynamic-scheduling, arrival-process, and uncertainty experiments."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import DCVRP_Dataset

from .data_generation import generate_dataset, mean_revelation_time
from .environment import ExperimentalEnvironment
from .protocol import StudyConfig, evaluate_one, load_model


def _boundaries(nodes: torch.Tensor, mode: str, fixed_segments: int) -> list[float]:
    fixed = np.linspace(0.0, 1.0, fixed_segments + 1).tolist()
    reveal = nodes[0, 1:, 4].detach().cpu().numpy()
    event = [0.0] + sorted(float(value) for value in reveal if value > 0.0) + [1.0]
    if mode == "fixed":
        return fixed
    if mode == "event":
        return sorted(set(event))
    if mode == "hybrid":
        return sorted(set(fixed + event))
    raise ValueError(f"Unknown scheduling mode: {mode}")


def evaluate_schedule(
    model,
    selector_name: str,
    train_seed: int,
    config: StudyConfig,
    device: torch.device,
    mode: str,
    test_seed: int,
) -> pd.DataFrame:
    dataset = generate_dataset(
        config.test_size,
        customer_count=config.customer_count,
        vehicle_count=config.vehicle_count,
        dynamic_ratio=config.dynamic_ratio,
        arrival_profile=config.arrival_profile,
        spatial=config.spatial,
        seed=test_seed,
    )
    frames = []
    model.eval()
    model.greedy = True
    for instance in range(config.test_size):
        one = DCVRP_Dataset(
            dataset.veh_count,
            dataset.veh_capa,
            dataset.veh_speed,
            dataset.nodes[instance:instance + 1].clone(),
        )
        boundaries = _boundaries(one.nodes, mode, config.segment_count)
        environment = ExperimentalEnvironment(
            one,
            nodes=one.nodes.to(device),
            pending_cost=config.pending_cost,
            horizon=1.0,
            segment_boundaries=boundaries,
            stochastic_seed=test_seed + instance,
        )
        started = time.perf_counter()
        with torch.no_grad():
            _, _, rewards = model(environment)
        elapsed = time.perf_counter() - started
        row = pd.DataFrame(environment.metrics())
        row["cost_with_penalty"] = float(-torch.stack(rewards).sum().cpu())
        row["penalty"] = row["cost_with_penalty"] - row["distance"]
        row["wall_time_ms_per_instance"] = elapsed * 1000.0
        row["boundary_count"] = len(boundaries) - 1
        row["instance"] = instance
        row["scenario"] = f"schedule_{mode}"
        row["train_seed"] = train_seed
        row["selector"] = selector_name
        row["dynamic_ratio"] = config.dynamic_ratio
        row["mean_revelation_time"] = mean_revelation_time(one)
        row["arrival_profile"] = config.arrival_profile
        row["spatial"] = config.spatial
        row["segment_count"] = config.segment_count
        row["schedule_mode"] = mode
        row["travel_noise"] = 0.0
        row["service_noise"] = 0.0
        row["congestion"] = 0.0
        frames.append(row)
    return pd.concat(frames, ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=Path("experiments/checkpoints/architecture"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/robustness"))
    parser.add_argument("--selector", default="independent")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345, 3456])
    parser.add_argument("--test-size", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    raw_dir = args.output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for seed in args.seeds:
        checkpoint = args.checkpoint_root / f"{args.selector}_seed{seed}.pt"
        model, base_config, selector_name, train_seed = load_model(checkpoint, device)
        base_config = replace(base_config, test_size=args.test_size)

        # 1. Fixed interval sensitivity.
        for segments in (6, 8, 10, 12, 14):
            config = replace(base_config, segment_count=segments)
            frames.append(evaluate_one(
                model, selector_name, train_seed, config, device,
                scenario=f"interval_{segments}", test_seed=20_260_000 + segments,
            ))

        # 2. Temporal arrival profiles at the same dynamic ratio.
        for profile_index, profile in enumerate(
            ("early", "uniform", "late", "two_peak", "clustered", "nhpp")
        ):
            config = replace(base_config, arrival_profile=profile)
            frames.append(evaluate_one(
                model, selector_name, train_seed, config, device,
                scenario=f"arrival_{profile}", test_seed=20_261_000 + profile_index,
            ))

        # 3. Dynamic-ratio boundary, including the manuscript's 90% case.
        for ratio_index, ratio in enumerate((0.10, 0.25, 0.50, 0.75, 0.90)):
            config = replace(base_config, dynamic_ratio=ratio)
            frames.append(evaluate_one(
                model, selector_name, train_seed, config, device,
                scenario=f"dynamic_{ratio:.2f}", test_seed=20_262_000 + ratio_index,
            ))

        # 3b. Decoder compatibility scaling C (manuscript default C=10).
        original_scale = model.tanh_exploration
        for scale in (2.0, 5.0, 10.0, 20.0):
            model.tanh_exploration = scale
            frames.append(evaluate_one(
                model, selector_name, train_seed, base_config, device,
                scenario=f"tanh_scale_{scale:g}", test_seed=20_262_500,
            ))
        model.tanh_exploration = original_scale

        # 4. Operational uncertainty and peak congestion.
        uncertainty = [
            ("deterministic", 0.0, 0.0, 0.0),
            ("travel_noise", 0.20, 0.0, 0.0),
            ("service_noise", 0.0, 0.20, 0.0),
            ("joint_noise", 0.20, 0.20, 0.0),
            ("peak_congestion", 0.10, 0.10, 0.45),
        ]
        for scenario_index, (name, travel, service, congestion) in enumerate(uncertainty):
            frames.append(evaluate_one(
                model, selector_name, train_seed, base_config, device,
                scenario=f"uncertainty_{name}",
                test_seed=20_263_000 + scenario_index,
                travel_noise=travel,
                service_noise=service,
                congestion=congestion,
            ))

        # 5. Fair single-instance scheduling comparison.
        schedule_config = replace(base_config, test_size=min(args.test_size, 40))
        for mode in ("fixed", "event", "hybrid"):
            frames.append(evaluate_schedule(
                model,
                selector_name,
                train_seed,
                schedule_config,
                device,
                mode=mode,
                test_seed=20_264_000,
            ))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(raw_dir / "robustness_instances.csv", index=False)
    print(raw.groupby("scenario")[["cost_with_penalty", "distance", "qos", "response_time_min"]].mean())


if __name__ == "__main__":
    main()
