"""Training script for Dynamic Capacitated Vehicle Routing Problem (DCVRP).

Trains deep reinforcement learning models using REINFORCE with a Rollout Baseline.
Hyperparameters strictly follow Section IV-A of the manuscript:
- Problem: n=20 customers, m=4 vehicles, Q=150 capacity, T=480 horizon, beta=10 intervals.
- Network: 3-layer 8-head Transformer encoder (d=128, ff=512), C=10 tanh exploration.
- Optimization: Adam optimizer with learning rate 1e-4, gradient norm clipping 2.0.
- RL Algorithm: REINFORCE with Rollout Baseline (update threshold alpha = 0.05).
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import torch
from scipy.stats import ttest_rel
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from env import (
    DEFAULT_CUSTOMER_COUNT,
    DEFAULT_DECISION_INTERVALS,
    DEFAULT_DYNAMIC_RATES,
    DEFAULT_HORIZON,
    DEFAULT_VEHICLE_CAPACITY,
    DEFAULT_VEHICLE_COUNT,
    DEFAULT_VEHICLE_SPEED,
    DCVRPEnvironment,
    generate_dataset,
    set_seed,
)
from models import AttentionLearner, build_selector


class RolloutBaseline:
    """Greedy rollout baseline for REINFORCE variance reduction."""

    def __init__(self, model: AttentionLearner, dataset, device: torch.device, method: str):
        selector = build_selector(method, vehicle_count=DEFAULT_VEHICLE_COUNT)
        self.model = AttentionLearner(selector).to(device)
        self.model.load_state_dict(model.state_dict())
        self.model.eval()
        self.model.greedy = True
        self.model.vehicle_greedy = True
        self.dataset = dataset
        self.device = device
        self.method = method

    @torch.no_grad()
    def eval(self, data, device: torch.device) -> torch.Tensor:
        env = DCVRPEnvironment(data, nodes=data.nodes.to(device), pending_cost=5.0)
        _, _, rewards = self.model(env)
        return torch.stack(rewards).sum(dim=0)

    @torch.no_grad()
    def validate_and_update(
        self, candidate: AttentionLearner, val_data, alpha: float = 0.05
    ) -> bool:
        selector = build_selector(self.method, vehicle_count=DEFAULT_VEHICLE_COUNT)
        candidate_eval = AttentionLearner(selector).to(self.device)
        candidate_eval.load_state_dict(candidate.state_dict())
        candidate_eval.eval()
        candidate_eval.greedy = True
        candidate_eval.vehicle_greedy = True

        env_c = DCVRPEnvironment(val_data, nodes=val_data.nodes.to(self.device), pending_cost=5.0)
        _, _, r_c = candidate_eval(env_c)
        returns_candidate = torch.stack(r_c).sum(dim=0).squeeze(-1).cpu().numpy()

        env_b = DCVRPEnvironment(val_data, nodes=val_data.nodes.to(self.device), pending_cost=5.0)
        _, _, r_b = self.model(env_b)
        returns_baseline = torch.stack(r_b).sum(dim=0).squeeze(-1).cpu().numpy()

        # Higher return (lower cost) is better
        t_stat, p_val = ttest_rel(returns_candidate, returns_baseline)
        if returns_candidate.mean() > returns_baseline.mean() and p_val < alpha:
            self.model.load_state_dict(candidate_eval.state_dict())
            return True
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Train DCVRP models.")
    parser.add_argument("--method", type=str, default="DVNDA", help="Vehicle selection method (default: DVNDA).")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100).")
    parser.add_argument("--steps-per-epoch", type=int, default=100, help="Steps per epoch (default: 100).")
    parser.add_argument("--batch-size", type=int, default=100, help="Training batch size (default: 100).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4).")
    parser.add_argument("--max-grad-norm", type=float, default=2.0, help="Max gradient norm (default: 2.0).")
    parser.add_argument("--val-size", type=int, default=100, help="Validation set size (default: 100).")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234).")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"), help="Directory to save checkpoints.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing {args.method} model on {device}...")
    selector = build_selector(args.method, vehicle_count=DEFAULT_VEHICLE_COUNT)
    model = AttentionLearner(
        selector=selector,
        customer_feature_size=5,
        vehicle_state_size=4,
        model_size=128,
        layer_count=3,
        head_count=8,
        ff_size=512,
        tanh_exploration=10.0,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    val_dataset = generate_dataset(
        batch_size=args.val_size,
        dynamic_rate=0.5,
        seed=args.seed + 9999,
    )
    baseline = RolloutBaseline(model, val_dataset, device, args.method)

    best_val_return = -float("inf")

    print(f"Starting training for {args.epochs} epochs ({args.steps_per_epoch} steps/epoch, batch size {args.batch_size})...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.greedy = False
        epoch_start = time.perf_counter()
        total_loss = 0.0

        for step in range(1, args.steps_per_epoch + 1):
            rate = float(DEFAULT_DYNAMIC_RATES[step % len(DEFAULT_DYNAMIC_RATES)])
            data = generate_dataset(batch_size=args.batch_size, dynamic_rate=rate)

            env = DCVRPEnvironment(data, nodes=data.nodes.to(device), pending_cost=5.0)
            _, log_probabilities, rewards = model(env)

            policy_returns = torch.stack(rewards).sum(dim=0)  # (B, 1)

            with torch.no_grad():
                baseline_returns = baseline.eval(data, device)

            advantage = policy_returns - baseline_returns
            log_prob = torch.stack(log_probabilities).sum(dim=0)
            loss = -(log_prob * advantage.detach()).mean()

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.perf_counter() - epoch_start
        avg_loss = total_loss / args.steps_per_epoch

        # Evaluate on validation dataset
        updated = baseline.validate_and_update(model, val_dataset)

        with torch.no_grad():
            env_val = DCVRPEnvironment(val_dataset, nodes=val_dataset.nodes.to(device), pending_cost=0.0)
            eval_sel = build_selector(args.method, vehicle_count=DEFAULT_VEHICLE_COUNT)
            model_eval = AttentionLearner(eval_sel).to(device)
            model_eval.load_state_dict(model.state_dict())
            model_eval.eval()
            model_eval.greedy = True
            model_eval.vehicle_greedy = True
            model_eval(env_val)
            val_dist = env_val.route_distance().mean().item()
            val_qos = env_val.qos().mean().item() * 100.0

        update_str = " [Baseline Updated]" if updated else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:8.4f} | Val Dist: {val_dist:6.2f} | Val QoS: {val_qos:5.1f}% | Time: {epoch_time:5.1f}s{update_str}")

        # Save checkpoint
        checkpoint_path = args.output_dir / f"{args.method}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_distance": val_dist,
                "val_qos": val_qos,
            },
            checkpoint_path,
        )


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()