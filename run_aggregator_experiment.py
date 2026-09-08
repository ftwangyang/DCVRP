"""Aggregator Comparison Experiment: Argmax vs MLP vs Attention.

Trains three models with identical hyperparameters, differing only in
the vehicle-selection aggregation method inside DVNDA.  The independent
VehicleSelectionNetwork per vehicle is kept unchanged.

Usage:
    python run_aggregator_experiment.py [--device cuda|cpu] [--epochs 20]
"""

import os
import sys
import time
import json
import copy
import random
import argparse
from pathlib import Path
from itertools import chain, repeat

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from data import DCVRP_Dataset
from learner import AttentionLearner, DCVRP_Environment
from rollout import RolloutBaseline
from args import parse_args


# ─── Experiment configuration ───────────────────────────────────────────────

AGGREGATORS = ["argmax", "mlp", "attention"]
DEFAULT_SEED = 1234


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reinforce_loss(logprobs, rewards, baseline=None, weights=None,
                   discount=1.0, reduction='mean'):
    if weights is None:
        weights = repeat(1.0)

    if isinstance(rewards, torch.Tensor):
        if baseline is None:
            baseline = torch.zeros_like(rewards)
        loss = torch.stack([-logp * w for logp, w in zip(logprobs, weights)]).sum(dim=0)
        loss *= (rewards - baseline.detach())
        if baseline.requires_grad:
            loss += F.smooth_l1_loss(baseline, rewards)
    else:
        if baseline is None:
            baseline = repeat(torch.zeros_like(rewards[0]))
        cumul = torch.zeros_like(rewards[0])
        vals = []
        for r in reversed(rewards):
            cumul = r + discount * cumul
            vals.append(cumul)
        vals.reverse()
        loss = []
        bl_loss = []
        for val, logp, bl, w in zip(vals, logprobs, baseline, weights):
            loss.append(-logp * (val - bl.detach()) * w)
            if bl.requires_grad:
                bl_loss.append(F.smooth_l1_loss(bl, val))
        loss = torch.stack(loss).sum(dim=0)
        if bl_loss:
            loss += torch.stack(bl_loss).sum(dim=0)

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()


def train_one_epoch(learner, baseline, train_data, env_params, optimizer,
                    device, batch_size, iter_count, max_grad_norm, epoch_idx, total_epochs):
    learner.train()
    loader = DataLoader(train_data, batch_size, shuffle=True)

    ep_loss = ep_val = 0
    step = 0
    for minibatch in loader:
        if train_data.cust_mask is None:
            custs, mask = minibatch.to(device), None
        else:
            custs, mask = minibatch[0].to(device), minibatch[1].to(device)

        dyna = DCVRP_Environment(train_data, custs, mask, *env_params)
        actions, logps, rewards, bl_vals = baseline(dyna)
        loss = reinforce_loss(logps, rewards, bl_vals)

        val = rewards.mean() if isinstance(rewards, torch.Tensor) else torch.stack(rewards).sum(0).mean()

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            clip_grad_norm_(
                chain.from_iterable(grp["params"] for grp in optimizer.param_groups),
                max_grad_norm
            )
        optimizer.step()

        ep_loss += loss.item()
        ep_val += val.item()
        step += 1

    return ep_loss / max(step, 1), ep_val / max(step, 1)


def train_aggregator(aggregator_type, device, config):
    """Train a single model with the given aggregator type, return val history."""
    set_seed(config["seed"])

    print(f"\n{'='*60}")
    print(f"  Training aggregator: {aggregator_type.upper()}")
    print(f"{'='*60}")

    # Generate data
    gen_params = [
        config["customers_count"], config["vehicles_count"],
        config["veh_capa"], config["veh_speed"],
        None,  # min_cust_count
        (0, 101), (5, 41),  # loc_range, dem_range
    ]
    train_data = DCVRP_Dataset.generate(
        config["iter_count"] * config["batch_size"], *gen_params
    )
    train_data.normalize()

    # Build model
    learner = AttentionLearner(
        DCVRP_Dataset.CUST_FEAT_SIZE,
        DCVRP_Environment.VEH_STATE_SIZE,
        config["model_size"],
        config["layer_count"],
        config["head_count"],
        config["ff_size"],
        config["tanh_xplor"],
        veh_count=config["vehicles_count"],
        aggregator_type=aggregator_type,
    )
    learner.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in learner.parameters())
    dvnda_params = sum(p.numel() for p in learner.dvnda.parameters())
    print(f"  Total params: {total_params:,}")
    print(f"  DVNDA params: {dvnda_params:,}")

    # Baseline
    baseline = RolloutBaseline(learner, config["rollout_count"], config["rollout_threshold"])
    baseline.to(device)

    # Optimizer
    optimizer = Adam(learner.parameters(), config["learning_rate"])

    env_params = [config["pending_cost"]]
    val_history = []
    timing_history = []

    for epoch in range(config["epoch_count"]):
        t0 = time.time()
        ep_loss, ep_val = train_one_epoch(
            learner, baseline, train_data, env_params, optimizer,
            device, config["batch_size"], config["iter_count"],
            config["max_grad_norm"], epoch, config["epoch_count"]
        )
        elapsed = time.time() - t0
        val_history.append(ep_val)
        timing_history.append(elapsed)
        print(f"  Epoch {epoch+1:3d}/{config['epoch_count']} | "
              f"loss={ep_loss:.4f} | val={ep_val:.4f} | time={elapsed:.1f}s")

    return {
        "aggregator": aggregator_type,
        "val_history": val_history,
        "timing_history": timing_history,
        "total_params": total_params,
        "dvnda_params": dvnda_params,
        "final_val": val_history[-1] if val_history else float('nan'),
        "best_val": max(val_history) if val_history else float('nan'),
        "mean_epoch_time": np.mean(timing_history) if timing_history else 0,
    }


def format_results_table(results):
    """Generate a formatted comparison table."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("  AGGREGATOR COMPARISON RESULTS")
    lines.append("=" * 80)

    # Summary table
    header = f"{'Aggregator':>12} | {'Best Val':>10} | {'Final Val':>10} | {'DVNDA Params':>12} | {'Total Params':>12} | {'Avg Time/Ep':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        lines.append(
            f"{r['aggregator']:>12} | {r['best_val']:>10.4f} | {r['final_val']:>10.4f} | "
            f"{r['dvnda_params']:>12,} | {r['total_params']:>12,} | {r['mean_epoch_time']:>10.1f}s"
        )

    lines.append("")

    # Val history comparison
    lines.append("Val History (per epoch):")
    n_epochs = len(results[0]["val_history"])
    header2 = f"{'Epoch':>6} | " + " | ".join(f"{r['aggregator']:>12}" for r in results)
    lines.append(header2)
    lines.append("-" * len(header2))
    for ep in range(n_epochs):
        row = f"{ep+1:>6} | " + " | ".join(
            f"{r['val_history'][ep]:>12.4f}" for r in results
        )
        lines.append(row)

    lines.append("")

    # Analysis
    argmax_r = next(r for r in results if r["aggregator"] == "argmax")
    for r in results:
        if r["aggregator"] != "argmax":
            diff = r["best_val"] - argmax_r["best_val"]
            pct = (diff / abs(argmax_r["best_val"])) * 100 if argmax_r["best_val"] != 0 else 0
            verdict = "WORSE" if diff < 0 else "BETTER" if diff > 0 else "SAME"
            lines.append(
                f"  {r['aggregator'].upper()} vs ARGMAX: "
                f"best_val diff = {diff:+.4f} ({pct:+.2f}%) → {verdict}"
            )

    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregator Comparison Experiment")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--iter-count", type=int, default=500,
                        help="Iterations per epoch")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/results/aggregator_comparison",
                        help="Output directory for results")
    args = parser.parse_args()

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Common config for all runs
    config = {
        "seed": args.seed,
        "customers_count": 20,
        "vehicles_count": 4,
        "veh_capa": 150,
        "veh_speed": 1,
        "model_size": 128,
        "layer_count": 3,
        "head_count": 8,
        "ff_size": 512,
        "tanh_xplor": 10,
        "epoch_count": args.epochs,
        "iter_count": args.iter_count,
        "batch_size": args.batch_size,
        "learning_rate": 0.0001,
        "max_grad_norm": 2,
        "pending_cost": 5,
        "rollout_count": 3,
        "rollout_threshold": 0.05,
    }

    print("\nExperiment Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Train each aggregator
    all_results = []
    for agg_type in AGGREGATORS:
        result = train_aggregator(agg_type, device, config)
        all_results.append(result)

    # Format and print results
    table = format_results_table(all_results)
    print(table)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save text results
    with open(output_dir / "results.txt", "w", encoding="utf-8") as f:
        f.write(f"Aggregator Comparison Experiment\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n\n")
        f.write(f"Configuration:\n")
        f.write(json.dumps(config, indent=2))
        f.write(f"\n\n")
        f.write(table)

    # Save JSON results
    json_results = []
    for r in all_results:
        jr = dict(r)
        jr["val_history"] = [float(v) for v in jr["val_history"]]
        jr["timing_history"] = [float(t) for t in jr["timing_history"]]
        json_results.append(jr)

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": config,
            "results": json_results,
            "device": str(device),
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
