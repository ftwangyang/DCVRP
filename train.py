"""Training script for Dynamic Capacitated Vehicle Routing Problem (DCVRP).

Trains deep reinforcement learning models using REINFORCE with a Rollout Baseline.
Hyperparameters strictly follow Section III-D (Algorithm 2) and Section IV-A of the manuscript:
- Problem: n in {20, 35, 50}, m = n / 5, Q = 150, T = 480, beta = 10 synchronized intervals.
- Iterations:
    * n = 20: batch size 100, 1000 steps/epoch, 100 epochs.
    * n = 35, 50: batch size 50, 500 steps/epoch, 100 epochs.
- Network: 3-layer 8-head Transformer encoder (d=128, ff=512), C=10 tanh exploration.
- Optimization: Adam optimizer with learning rate 1e-4, gradient norm clipping 2.0.
- RL Algorithm: REINFORCE with Rollout Baseline (paired t-test update threshold alpha = 0.05).
- Penalty: alpha = 5.0 for unserved customers (Eq. 14).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
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
    generate_evaluation_split,
    set_seed,
)
from models import AttentionLearner, build_selector


class RolloutBaseline:
    """Greedy rollout baseline for REINFORCE variance reduction (Algorithm 2)."""

    def __init__(
        self,
        model: AttentionLearner,
        device: torch.device,
        method: str,
        vehicle_count: int = DEFAULT_VEHICLE_COUNT,
    ):
        selector = build_selector(method, vehicle_count=vehicle_count)
        self.model = AttentionLearner(selector).to(device)
        self.model.load_state_dict(model.state_dict())
        self.model.eval()
        self.model.greedy = True
        self.model.vehicle_greedy = True
        self.device = device
        self.method = method
        self.vehicle_count = vehicle_count

    @torch.no_grad()
    def eval(self, data, device: torch.device) -> torch.Tensor:
        """Compute baseline return on a batch of training instances."""
        env = DCVRPEnvironment(data, nodes=data.nodes.to(device), pending_cost=5.0)
        _, _, rewards = self.model(env)
        return torch.stack(rewards).sum(dim=0)

    @torch.no_grad()
    def validate_and_update(
        self,
        candidate: AttentionLearner,
        val_dataset,
        alpha: float = 0.05,
    ) -> bool:
        """Paired t-test comparison between candidate policy and baseline policy."""
        selector = build_selector(self.method, vehicle_count=self.vehicle_count)
        candidate_eval = AttentionLearner(selector).to(self.device)
        candidate_eval.load_state_dict(candidate.state_dict())
        candidate_eval.eval()
        candidate_eval.greedy = True
        candidate_eval.vehicle_greedy = True

        env_c = DCVRPEnvironment(val_dataset, nodes=val_dataset.nodes.to(self.device), pending_cost=5.0)
        _, _, r_c = candidate_eval(env_c)
        returns_candidate = torch.stack(r_c).sum(dim=0).squeeze(-1).cpu().numpy()

        env_b = DCVRPEnvironment(val_dataset, nodes=val_dataset.nodes.to(self.device), pending_cost=5.0)
        _, _, r_b = self.model(env_b)
        returns_baseline = torch.stack(r_b).sum(dim=0).squeeze(-1).cpu().numpy()

        # Higher return (less negative cost + penalty) indicates a superior policy
        t_stat, p_val = ttest_rel(returns_candidate, returns_baseline)
        if returns_candidate.mean() > returns_baseline.mean() and p_val < alpha:
            self.model.load_state_dict(candidate_eval.state_dict())
            return True
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DCVRP models following Section III-D and Section IV-A."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="DVNDA",
        choices=["DVNDA", "AMCVN", "LiDRL", "MAAM", "MARDAM"],
        help="Vehicle selection policy architecture (default: DVNDA).",
    )
    parser.add_argument(
        "-n",
        "--customer-count",
        type=int,
        default=DEFAULT_CUSTOMER_COUNT,
        help=f"Number of customer locations (default: {DEFAULT_CUSTOMER_COUNT}).",
    )
    parser.add_argument(
        "-m",
        "--vehicle-count",
        type=int,
        default=None,
        help="Number of vehicles (default: auto-computed as n / 5).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Total training epochs (default: 100).",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Iterations per epoch (default: 1000 for n=20, 500 for n=35/50 per manuscript).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: 100 for n=20, 50 for n=35/50 per manuscript).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Adam learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=2.0,
        help="Gradient norm clipping threshold (default: 2.0).",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=100,
        help="Validation set size per dynamic rate (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed (default: 1234).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to an existing checkpoint to resume training from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to save trained model checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda' or 'cpu').",
    )
    return parser.parse_args()


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.vehicle_count is None:
        args.vehicle_count = max(1, round(args.customer_count / 5))

    # Manuscript defaults: n=20 -> (B=100, N=1000); n=35,50 -> (B=50, N=500)
    if args.batch_size is None:
        args.batch_size = 100 if args.customer_count == 20 else 50
    if args.steps_per_epoch is None:
        args.steps_per_epoch = 1000 if args.customer_count == 20 else 500

    suffix = f"_n{args.customer_count}" if args.customer_count != DEFAULT_CUSTOMER_COUNT else ""
    checkpoint_path = args.output_dir / f"{args.method}{suffix}.pt"
    best_checkpoint_path = args.output_dir / f"{args.method}{suffix}_best.pt"

    print("=" * 76)
    print(f"  Training DCVRP Model: {args.method}")
    print(f"  Scale: n = {args.customer_count} customers, m = {args.vehicle_count} vehicles")
    print(f"  Hyperparameters: {args.epochs} epochs, {args.steps_per_epoch} steps/epoch, batch size {args.batch_size}")
    print(f"  Optimizer: Adam (lr={args.lr}, clip={args.max_grad_norm}), Device: {device}")
    print("=" * 76)

    selector = build_selector(args.method, vehicle_count=args.vehicle_count)
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

    start_epoch = 1
    best_val_dist = float("inf")

    if args.resume and args.resume.exists():
        print(f"Resuming training from checkpoint: {args.resume}")
        chk = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(chk["model"])
        if "optimizer" in chk:
            optimizer.load_state_dict(chk["optimizer"])
        if chk.get("customer_count") != args.customer_count:
            print(f"Transfer learning from scale n={chk.get('customer_count')} to n={args.customer_count}")
            start_epoch = 1
            best_val_dist = float("inf")
        else:
            start_epoch = chk.get("epoch", 0) + 1
            best_val_dist = chk.get("val_distance", float("inf"))
        print(f"Resumed weights loaded (best val dist: {best_val_dist:.2f})")

    # Fixed validation dataset across dynamic rates
    val_split = generate_evaluation_split(
        instances=args.val_size,
        dynamic_rates=DEFAULT_DYNAMIC_RATES,
        customer_count=args.customer_count,
        vehicle_count=args.vehicle_count,
        seed=args.seed + 9999,
    )
    baseline_dataset = val_split[0.50]
    baseline = RolloutBaseline(
        model,
        device=device,
        method=args.method,
        vehicle_count=args.vehicle_count,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        model.greedy = False
        epoch_start = time.perf_counter()
        total_loss = 0.0

        for step in range(1, args.steps_per_epoch + 1):
            rate = float(DEFAULT_DYNAMIC_RATES[step % len(DEFAULT_DYNAMIC_RATES)])
            data = generate_dataset(
                batch_size=args.batch_size,
                customer_count=args.customer_count,
                vehicle_count=args.vehicle_count,
                dynamic_rate=rate,
            )

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

        # Update baseline via paired t-test on validation data
        baseline_updated = baseline.validate_and_update(model, baseline_dataset, alpha=0.05)

        # Full validation across all dynamic rates
        val_dists = []
        val_qoss = []
        with torch.no_grad():
            eval_sel = build_selector(args.method, vehicle_count=args.vehicle_count)
            model_eval = AttentionLearner(eval_sel).to(device)
            model_eval.load_state_dict(model.state_dict())
            model_eval.eval()
            model_eval.greedy = True
            model_eval.vehicle_greedy = True

            for v_rate, v_data in val_split.items():
                env_val = DCVRPEnvironment(v_data, nodes=v_data.nodes.to(device), pending_cost=0.0)
                model_eval(env_val)
                val_dists.append(env_val.route_distance().mean().item())
                val_qoss.append(env_val.qos().mean().item() * 100.0)

        mean_val_dist = float(np.mean(val_dists))
        mean_val_qos = float(np.mean(val_qoss))

        update_tag = " [Baseline Updated]" if baseline_updated else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:8.4f} | "
            f"Val Dist: {mean_val_dist:6.2f} | Val QoS: {mean_val_qos:5.1f}% | "
            f"Time: {epoch_time:5.1f}s{update_tag}"
        )

        # Save standard clean checkpoint
        checkpoint_dict = {
            "epoch": epoch,
            "customer_count": args.customer_count,
            "vehicle_count": args.vehicle_count,
            "method": args.method,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_distance": mean_val_dist,
            "val_qos": mean_val_qos,
        }
        torch.save(checkpoint_dict, checkpoint_path)

        if mean_val_dist < best_val_dist:
            best_val_dist = mean_val_dist
            torch.save(checkpoint_dict, best_checkpoint_path)

    print(f"\nTraining completed. Final checkpoint saved to: {checkpoint_path}")
    if best_checkpoint_path.exists():
        print(f"Best validation checkpoint saved to: {best_checkpoint_path} (Val Dist: {best_val_dist:.2f})")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()