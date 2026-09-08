"""Run controlled selector-architecture ablations in a fixed order."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from .protocol import StudyConfig, evaluate_one, save_config, train_one


DEFAULT_SELECTORS = [
    "shared",
    "shared_wide",
    "shared_id",
    "centralized",
    "attention_aggregation",
    "independent_narrow",
    "independent",
    "joint_pair",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("experiments/checkpoints"))
    parser.add_argument("--selectors", nargs="+", default=DEFAULT_SELECTORS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345, 3456])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--model-size", type=int, default=64)
    parser.add_argument("--layer-count", type=int, default=2)
    parser.add_argument("--ff-size", type=int, default=128)
    parser.add_argument("--selector-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    config = replace(
        StudyConfig(),
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        test_size=args.test_size,
        learning_rate=args.learning_rate,
        model_size=args.model_size,
        layer_count=args.layer_count,
        ff_size=args.ff_size,
        selector_size=args.selector_size,
    )
    raw_dir = args.output_root / "raw"
    summary_dir = args.output_root / "summary"
    history_dir = args.output_root / "history"
    for directory in (raw_dir, summary_dir, history_dir, args.checkpoint_root):
        directory.mkdir(parents=True, exist_ok=True)
    save_config(config, args.output_root / "architecture_config.json")

    raw_frames = []
    resource_rows = []
    for selector in args.selectors:
        for seed in args.seeds:
            print(f"TRAIN selector={selector} seed={seed}", flush=True)
            model, _, resources = train_one(
                selector,
                seed,
                config,
                checkpoint_dir=args.checkpoint_root,
                history_dir=history_dir,
                device=device,
            )
            print(f"EVAL selector={selector} seed={seed}", flush=True)
            raw_frames.append(evaluate_one(
                model, selector, seed, config, device=device
            ))
            resource_rows.append(resources)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    raw = pd.concat(raw_frames, ignore_index=True)
    raw.to_csv(raw_dir / "architecture_instances.csv", index=False)
    resources = pd.DataFrame(resource_rows)
    resources.to_csv(raw_dir / "architecture_resources.csv", index=False)
    print(raw.groupby("selector")[["distance", "qos", "response_time_min"]].mean())


if __name__ == "__main__":
    main()
