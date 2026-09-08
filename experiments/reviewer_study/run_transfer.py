"""Fleet-size transfer and m=10 to m=100 fine-tuning experiment."""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_

from .protocol import (
    StudyConfig,
    _episode,
    build_model,
    evaluate_one,
    set_seed,
    train_one,
)


def transfer_state(source, target, selector_name: str, source_vehicles: int) -> None:
    """Load size-compatible state and deterministically expand independent heads."""
    source_state = source.state_dict()
    target_state = target.state_dict()
    copied = {}
    for name, value in target_state.items():
        if name in source_state and source_state[name].shape == value.shape:
            copied[name] = source_state[name]
            continue
        prefix = "selector.vehicle_networks."
        if selector_name == "independent" and name.startswith(prefix):
            remainder = name[len(prefix):]
            vehicle_text, suffix = remainder.split(".", 1)
            source_name = f"{prefix}{int(vehicle_text) % source_vehicles}.{suffix}"
            if source_name in source_state and source_state[source_name].shape == value.shape:
                copied[name] = source_state[source_name]
    missing, unexpected = target.load_state_dict(copied, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected transferred parameters: {unexpected}")
    required_missing = [
        name for name in missing
        if not (
            selector_name == "independent"
            and name.startswith("selector.vehicle_networks.")
        )
    ]
    if required_missing:
        raise RuntimeError(f"Non-selector transfer mismatch: {required_missing}")


def fine_tune(model, config: StudyConfig, seed: int, device: torch.device,
              steps: int) -> tuple[float, int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(steps):
        _, _, cost, log_probability = _episode(
            model,
            config,
            device,
            data_seed=70_000_000 + seed * 1000 + step,
            greedy=False,
        )
        standardized = (cost - cost.mean()) / cost.std().clamp_min(1.0e-6)
        loss = (standardized.detach() * log_probability).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return elapsed, peak


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/transfer"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("experiments/checkpoints/transfer"))
    parser.add_argument("--selectors", nargs="+", default=["shared", "independent"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345, 3456])
    parser.add_argument("--source-vehicles", type=int, default=10)
    parser.add_argument("--fleet-sizes", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--customers", type=int, default=50)
    parser.add_argument("--source-epochs", type=int, default=5)
    parser.add_argument("--source-steps", type=int, default=8)
    parser.add_argument("--fine-tune-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    raw_dir = args.output_root / "raw"
    history_dir = args.output_root / "history"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    resources = []
    for selector_name in args.selectors:
        for seed in args.seeds:
            source_config = StudyConfig(
                customer_count=args.customers,
                vehicle_count=args.source_vehicles,
                epochs=args.source_epochs,
                steps_per_epoch=args.source_steps,
                batch_size=args.batch_size,
                test_size=args.test_size,
                validation_size=min(16, args.test_size),
                learning_rate=2.0e-4,
            )
            print(f"TRANSFER source={selector_name} seed={seed}", flush=True)
            source, _, source_resource = train_one(
                selector_name,
                seed,
                source_config,
                args.checkpoint_root,
                history_dir,
                device,
            )
            source_resource["stage"] = "source_training"
            resources.append(source_resource)

            for fleet_size in args.fleet_sizes:
                target_test_size = max(
                    8,
                    min(
                        args.test_size,
                        int(args.test_size * args.source_vehicles / fleet_size),
                    ),
                )
                target_config = replace(
                    source_config,
                    vehicle_count=fleet_size,
                    epochs=1,
                    steps_per_epoch=(1 if fleet_size >= 100 else args.fine_tune_steps),
                    batch_size=min(args.batch_size, 4 if fleet_size < 100 else 2),
                    test_size=target_test_size,
                    validation_size=min(8, target_test_size),
                )
                print(
                    f"TRANSFER target={selector_name} seed={seed} "
                    f"m={fleet_size} test={target_test_size}",
                    flush=True,
                )
                set_seed(seed + fleet_size)
                target = build_model(selector_name, target_config, device)
                transfer_state(source, target, selector_name, args.source_vehicles)
                zero_shot = evaluate_one(
                    target,
                    selector_name,
                    seed,
                    target_config,
                    device,
                    test_seed=20_268_021,
                    scenario=f"m{args.source_vehicles}_to_m{fleet_size}_zero_shot",
                )
                zero_shot["stage"] = "zero_shot"
                zero_shot["source_vehicle_count"] = args.source_vehicles
                zero_shot["target_vehicle_count"] = fleet_size
                rows.append(zero_shot)

                fine_tune_steps = 1 if fleet_size >= 100 else args.fine_tune_steps
                seconds, peak = fine_tune(
                    target,
                    target_config,
                    seed + fleet_size,
                    device,
                    fine_tune_steps,
                )
                tuned = evaluate_one(
                    target,
                    selector_name,
                    seed,
                    target_config,
                    device,
                    test_seed=20_268_021,
                    scenario=f"m{args.source_vehicles}_to_m{fleet_size}_fine_tuned",
                )
                tuned["stage"] = "fine_tuned"
                tuned["source_vehicle_count"] = args.source_vehicles
                tuned["target_vehicle_count"] = fleet_size
                rows.append(tuned)
                print(
                    f"TRANSFER done={selector_name} seed={seed} m={fleet_size} "
                    f"zero_cost={zero_shot.cost_with_penalty.mean():.3f} "
                    f"tuned_cost={tuned.cost_with_penalty.mean():.3f}",
                    flush=True,
                )
                resources.append({
                    "selector": selector_name,
                    "seed": seed,
                    "stage": "target_fine_tuning",
                    "target_vehicle_count": fleet_size,
                    "parameter_count": sum(p.numel() for p in target.parameters()),
                    "training_seconds": seconds,
                    "peak_memory_bytes": peak,
                    "fine_tune_steps": fine_tune_steps,
                })
                checkpoint = args.checkpoint_root / f"{selector_name}_m{fleet_size}_seed{seed}.pt"
                torch.save({
                    "selector": selector_name,
                    "seed": seed,
                    "config": target_config.__dict__,
                    "model": target.state_dict(),
                }, checkpoint)
                # Preserve every completed stress-test cell so an m=100
                # resource failure cannot erase smaller-fleet evidence.
                pd.concat(rows, ignore_index=True).to_csv(
                    raw_dir / "transfer_instances.csv", index=False
                )
                pd.DataFrame(resources).to_csv(
                    raw_dir / "transfer_resources.csv", index=False
                )
                del target
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            del source
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(raw_dir / "transfer_instances.csv", index=False)
    pd.DataFrame(resources).to_csv(raw_dir / "transfer_resources.csv", index=False)
    print(raw.groupby(["selector", "target_vehicle_count", "stage"])[
        ["cost_with_penalty", "distance", "qos", "wall_time_ms_per_instance"]
    ].mean())


if __name__ == "__main__":
    main()
