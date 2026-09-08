"""Continue the main method beyond the equal-budget architecture screening."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_

from .protocol import _episode, evaluate_one, load_model, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoints", type=Path, default=Path("experiments/checkpoints/architecture"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("experiments/checkpoints/production"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/production"))
    parser.add_argument("--selector", default="independent")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345, 3456])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    history_dir = args.output_root / "history"
    raw_dir = args.output_root / "raw"
    history_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    resources = []
    for seed in args.seeds:
        source = args.source_checkpoints / f"{args.selector}_seed{seed}.pt"
        model, source_config, selector_name, train_seed = load_model(source, device)
        config = replace(
            source_config,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            test_size=args.test_size,
            validation_size=max(source_config.validation_size, 32),
            learning_rate=args.learning_rate,
        )
        set_seed(seed + 600_000)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        best_cost = float("inf")
        best_epoch = 0
        best_state = None
        rows = []
        updates_completed = 0
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(config.epochs + 1):
            if epoch > 0:
                model.train()
                train_costs = []
                for step in range(config.steps_per_epoch):
                    _, _, cost, log_probability = _episode(
                        model,
                        config,
                        device,
                        data_seed=60_000_000 + seed * 10_000 + epoch * config.steps_per_epoch + step,
                        greedy=False,
                    )
                    normalized = (cost - cost.mean()) / cost.std().clamp_min(1.0e-6)
                    loss = (normalized.detach() * log_probability).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    clip_grad_norm_(model.parameters(), config.gradient_clip)
                    optimizer.step()
                    updates_completed += 1
                    train_costs.append(float(cost.mean().detach().cpu()))
            else:
                train_costs = [np.nan]

            model.eval()
            with torch.no_grad():
                _, validation_environment, validation_cost, _ = _episode(
                    model,
                    config,
                    device,
                    data_seed=89_000_000 + seed,
                    greedy=True,
                    batch_size_override=config.validation_size,
                )
            metrics = validation_environment.metrics()
            validation_mean = float(validation_cost.mean().cpu())
            if validation_mean < best_cost:
                best_cost = validation_mean
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            rows.append({
                "selector": selector_name,
                "seed": seed,
                "continuation_epoch": epoch,
                "train_cost_mean": float(np.nanmean(train_costs)) if epoch else np.nan,
                "validation_cost_mean": validation_mean,
                "validation_distance_mean": float(metrics["distance"].mean()),
                "validation_qos_mean": float(metrics["qos"].mean()),
            })
            print(
                f"PRODUCTION seed={seed} epoch={epoch}/{config.epochs} "
                f"validation={validation_mean:.3f} qos={metrics['qos'].mean():.3f}",
                flush=True,
            )
            if epoch > 0 and epoch - best_epoch >= args.patience:
                print(
                    f"PRODUCTION seed={seed} early_stop={epoch} best={best_epoch}",
                    flush=True,
                )
                break
        if best_state is None:
            raise RuntimeError("No production validation state was selected")
        model.load_state_dict(best_state)
        checkpoint = args.checkpoint_root / f"{selector_name}_seed{seed}.pt"
        torch.save({
            "selector": selector_name,
            "seed": seed,
            "config": asdict(config),
            "model": model.state_dict(),
        }, checkpoint)
        pd.DataFrame(rows).to_csv(
            history_dir / f"{selector_name}_seed{seed}.csv", index=False
        )
        frame = evaluate_one(model, selector_name, train_seed, config, device)
        frame["best_continuation_epoch"] = best_epoch
        frames.append(frame)
        resources.append({
            "selector": selector_name,
            "seed": seed,
            "additional_training_seconds": time.perf_counter() - started,
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0
            ),
            "best_continuation_epoch": best_epoch,
            "best_validation_cost": best_cost,
            "additional_updates": updates_completed,
            "checkpoint": str(checkpoint),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(raw_dir / "production_instances.csv", index=False)
    pd.DataFrame(resources).to_csv(raw_dir / "production_resources.csv", index=False)
    print(raw.groupby("selector")[[
        "cost_with_penalty", "distance", "qos", "response_time_min"
    ]].mean())


if __name__ == "__main__":
    main()
