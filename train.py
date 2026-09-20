"""Train DCVRP policies with REINFORCE and a greedy rollout baseline.

Hyperparameters follow Section III-D (Algorithm 2) and Section IV-A:
n in {20, 35, 50}, m = n/5, Q = 150, T = 480, beta = 10 intervals.
n = 20 uses batch size 100 and 1000 steps per epoch; n = 35 and 50 use
batch size 50 and 500 steps. The encoder is a 3-layer 8-head Transformer
(d = 128, ff = 512) with C = 10 tanh clipping. Adam starts at 1e-4 and
cosine-anneals to 1e-5, with gradient clipping 2.0. During training both
the vehicle and the customer are sampled; evaluation uses argmax for the
vehicle (Eq. 27) and greedy or sampled decoding for customers.
Unserved customers are penalized with alpha = 5 (Eq. 14).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_rel
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from env import (
    DEFAULT_CUSTOMER_COUNT,
    DEFAULT_DYNAMIC_RATES,
    DEFAULT_REVELATION,
    DEFAULT_VEHICLE_COUNT,
    DCVRPDataset,
    DCVRPEnvironment,
    REVELATION_MODES,
    generate_dataset,
    generate_evaluation_split,
    set_seed,
)
from eval import TABLE1_CELL_GAP_LIMIT, table1_gap_report
from models import AttentionLearner, build_selector


def repeat_rollout_batch(data: DCVRPDataset, repeats: int) -> DCVRPDataset:
    """Expand a batch for parallel independent policy rollouts.

    Averaging the REINFORCE loss over this expanded batch is exactly the
    arithmetic mean of ``repeats`` serial rollout losses; only execution is
    vectorized. Instance rows remain adjacent so baseline returns can use the
    same ``repeat_interleave`` ordering.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    mask = None
    if data.cust_mask is not None:
        mask = data.cust_mask.repeat_interleave(repeats, dim=0)
    return DCVRPDataset(
        vehicle_count=data.veh_count,
        vehicle_capacity=data.veh_capa,
        vehicle_speed=data.veh_speed,
        nodes=data.nodes.repeat_interleave(repeats, dim=0),
        customer_mask=mask,
        revelation=getattr(data, "revelation", DEFAULT_REVELATION),
    )


class RolloutBaseline:
    """Greedy rollout baseline for REINFORCE variance reduction (Algorithm 2)."""

    def __init__(
        self,
        model: AttentionLearner,
        device: torch.device,
        method: str,
        vehicle_count: int = DEFAULT_VEHICLE_COUNT,
        disclose_horizon_tail: bool = False,
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
        self.disclose_horizon_tail = bool(disclose_horizon_tail)

    def _env(self, data, pending_cost: float) -> DCVRPEnvironment:
        return DCVRPEnvironment(
            data,
            nodes=data.nodes.to(self.device),
            pending_cost=pending_cost,
            record_trace=False,
            disclose_horizon_tail=self.disclose_horizon_tail,
        )

    @torch.no_grad()
    def eval(self, data, device: torch.device) -> torch.Tensor:
        """Greedy rollout return used as R_{pi_BL} in Eq. 33.

        A deterministic greedy policy yields identical repeats, so one greedy
        rollout is exactly the mean of the paper's three baseline rollouts.
        """
        env = self._env(data, pending_cost=5.0)
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

        env_c = self._env(val_dataset, pending_cost=5.0)
        _, _, r_c = candidate_eval(env_c)
        returns_candidate = torch.stack(r_c).sum(dim=0).squeeze(-1).cpu().numpy()

        env_b = self._env(val_dataset, pending_cost=5.0)
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
        help="Iterations per epoch (default: 1000 for n=20, 500 for n=35/50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: 100 for n=20, 50 for n=35/50).",
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
    parser.add_argument(
        "--rollouts",
        type=int,
        default=3,
        help="Sampled policy rollouts per training instance (paper: 3).",
    )
    parser.add_argument(
        "--early-stop-epoch",
        type=int,
        default=0,
        help="If >0, stop once every reported Table I cell is within --early-stop-mae.",
    )
    parser.add_argument(
        "--early-stop-mae",
        type=float,
        default=TABLE1_CELL_GAP_LIMIT,
        help="Maximum absolute percent gap versus Table I used with --early-stop-epoch.",
    )
    parser.add_argument(
        "--revelation",
        type=str,
        default=DEFAULT_REVELATION,
        choices=list(REVELATION_MODES),
        help="Arrival process: hpp (Uniform(0, T]) or poisson (Eq. 34 PMF).",
    )
    parser.add_argument(
        "--disclose-horizon-tail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, disclose a_i <= T_{r+1}. Default discloses a_i <= T_r.",
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
    table1_best_checkpoint_path = args.output_dir / f"{args.method}{suffix}_table1_best.pt"

    print("=" * 76)
    print(f"  Training DCVRP Model: {args.method}")
    print(f"  Scale: n = {args.customer_count} customers, m = {args.vehicle_count} vehicles")
    print(f"  Hyperparameters: {args.epochs} epochs, {args.steps_per_epoch} steps/epoch, batch size {args.batch_size}")
    print(f"  Policy rollouts: {args.rollouts} | Vehicle: sample train / argmax eval | Baseline: greedy rollout + t-test 0.05")
    print(
        f"  Arrivals: {args.revelation} | disclose a_i<=T_r={not args.disclose_horizon_tail} | "
        f"phi={DEFAULT_DYNAMIC_RATES}"
    )
    print(f"  Seed: {args.seed} | Device: {device}")
    print(f"  Optimizer: Adam (lr={args.lr} cosine to 1e-5, clip={args.max_grad_norm})")
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
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-5)

    start_epoch = 1
    best_val_dist = float("inf")
    best_val_objective = float("inf")
    best_table1_max = float("inf")
    last_table1_all_within = False

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
            best_val_objective = float("inf")
        else:
            start_epoch = chk.get("epoch", 0) + 1
            prev_val = chk.get("val_distance", float("inf"))
            best_val_dist = float("inf") if prev_val == 0.0 else float(prev_val)
            best_val_objective = float(chk.get("val_objective", float("inf")))
            prev_table1 = chk.get("table1_max_gap")
            if prev_table1 is not None:
                best_table1_max = float(prev_table1)
            last_table1_all_within = bool(chk.get("table1_all_within", False))
        print(
            f"Resumed weights loaded (last val distance: {best_val_dist:.2f}, "
            f"objective: {best_val_objective:.2f})"
        )

    # Fixed validation dataset across dynamic rates
    val_split = generate_evaluation_split(
        instances=args.val_size,
        dynamic_rates=DEFAULT_DYNAMIC_RATES,
        customer_count=args.customer_count,
        vehicle_count=args.vehicle_count,
        seed=args.seed + 9999,
        revelation=args.revelation,
    )
    baseline_dataset = val_split[0.50]
    table1_split = generate_evaluation_split(
        instances=args.val_size,
        dynamic_rates=DEFAULT_DYNAMIC_RATES,
        customer_count=args.customer_count,
        vehicle_count=args.vehicle_count,
        seed=20260821,
        revelation=args.revelation,
    )
    set_seed(args.seed)
    for _ in range(start_epoch - 1):
        scheduler.step()
    baseline = RolloutBaseline(
        model,
        device=device,
        method=args.method,
        vehicle_count=args.vehicle_count,
        disclose_horizon_tail=args.disclose_horizon_tail,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        model.greedy = False
        model.vehicle_greedy = False
        epoch_start = time.perf_counter()
        total_loss = 0.0

        for step in range(1, args.steps_per_epoch + 1):
            rate = float(DEFAULT_DYNAMIC_RATES[step % len(DEFAULT_DYNAMIC_RATES)])
            data = generate_dataset(
                batch_size=args.batch_size,
                customer_count=args.customer_count,
                vehicle_count=args.vehicle_count,
                dynamic_rate=rate,
                revelation=args.revelation,
            )

            with torch.no_grad():
                baseline_returns = baseline.eval(data, device)

            optimizer.zero_grad()
            rollout_data = repeat_rollout_batch(data, args.rollouts)
            env = DCVRPEnvironment(
                rollout_data,
                nodes=rollout_data.nodes.to(device),
                pending_cost=5.0,
                record_trace=False,
                disclose_horizon_tail=args.disclose_horizon_tail,
            )
            _, log_probabilities, rewards = model(env)
            # Joint log-probability of the vehicle and the selected node.
            log_prob = torch.stack(log_probabilities).sum(0)
            policy_returns = torch.stack(rewards).sum(0)
            repeated_baseline = baseline_returns.repeat_interleave(
                args.rollouts, dim=0
            )
            advantage = (policy_returns - repeated_baseline).detach()
            adv_std = advantage.std()
            if float(adv_std) > 1e-6:
                advantage = (advantage - advantage.mean()) / adv_std
            rollout_loss = -(log_prob * advantage).mean()
            rollout_loss.backward()
            step_loss = rollout_loss.item()

            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += step_loss
            if step == 1 or step % 10 == 0 or step == args.steps_per_epoch:
                print(
                    f"         step {step}/{args.steps_per_epoch} "
                    f"loss {step_loss:.4f}",
                    flush=True,
                )

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
                env_val = DCVRPEnvironment(
                    v_data,
                    nodes=v_data.nodes.to(device),
                    pending_cost=0.0,
                    record_trace=False,
                    disclose_horizon_tail=args.disclose_horizon_tail,
                )
                model_eval(env_val)
                val_dists.append(env_val.route_distance().mean().item())
                val_qoss.append(env_val.qos().mean().item() * 100.0)

        mean_val_dist = float(np.mean(val_dists))
        mean_val_qos = float(np.mean(val_qoss))
        mean_val_unserved = args.customer_count * (1.0 - mean_val_qos / 100.0)
        mean_val_objective = mean_val_dist + 5.0 * mean_val_unserved

        update_tag = " [Baseline Updated]" if baseline_updated else ""
        rate_str = " ".join(
            f"{float(rate):.2f}:{dist:.2f}"
            for rate, dist in zip(val_split.keys(), val_dists)
        )
        print(
            f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:8.4f} | "
            f"Val Dist: {mean_val_dist:6.2f} | Val QoS: {mean_val_qos:5.1f}% | "
            f"Val Obj: {mean_val_objective:6.2f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {epoch_time:5.1f}s{update_tag}",
            flush=True,
        )
        print(f"         val-by-phi {rate_str}", flush=True)
        qos_str = " ".join(
            f"{float(rate):.2f}:{qos:.1f}%"
            for rate, qos in zip(val_split.keys(), val_qoss)
        )
        print(f"         val-qos-by-phi {qos_str}", flush=True)

        table1_measured: dict[float, float] = {}
        table1_qos: dict[float, float] = {}
        with torch.no_grad():
            for t_rate, t_data in table1_split.items():
                env_t = DCVRPEnvironment(
                    t_data,
                    nodes=t_data.nodes.to(device),
                    pending_cost=0.0,
                    record_trace=False,
                    disclose_horizon_tail=args.disclose_horizon_tail,
                )
                model_eval(env_t)
                key = round(float(t_rate), 2)
                table1_measured[key] = env_t.route_distance().mean().item()
                table1_qos[key] = env_t.qos().mean().item() * 100.0
        table1_report = table1_gap_report(
            args.customer_count,
            args.method,
            table1_measured,
            cell_limit=float(args.early_stop_mae),
            qos_by_rate=table1_qos,
            min_qos_percent=100.0,
        )
        table1_mae = table1_report["mae"]
        table1_max = table1_report["max_abs"]
        last_table1_all_within = bool(table1_report["all_within"])
        table1_all_within = last_table1_all_within
        if table1_report["parts"]:
            print(
                f"         table1-phi {' '.join(table1_report['parts'])} | "
                f"MAE {table1_mae:.2f}% max {table1_max:.2f}%",
                flush=True,
            )
            qos_phi = " ".join(
                f"{rate:.2f}:{qos:.1f}%" for rate, qos in table1_qos.items()
            )
            print(f"         table1-qos-by-phi {qos_phi}", flush=True)

        param_count = int(sum(p.numel() for p in model.parameters()))
        checkpoint_dict = {
            "epoch": epoch,
            "customer_count": args.customer_count,
            "vehicle_count": args.vehicle_count,
            "method": args.method,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_distance": float(mean_val_dist),
            "val_qos": float(mean_val_qos),
            "val_unserved": float(mean_val_unserved),
            "val_objective": float(mean_val_objective),
            "val_distance_by_rate": {
                str(rate): float(dist)
                for rate, dist in zip(val_split.keys(), val_dists)
            },
            "val_qos_by_rate": {
                str(rate): float(qos)
                for rate, qos in zip(val_split.keys(), val_qoss)
            },
            "parameter_count": param_count,
            "rollouts": int(args.rollouts),
            "seed": int(args.seed),
            "revelation": str(args.revelation),
            "disclose_horizon_tail": bool(args.disclose_horizon_tail),
            "table1_mae": table1_mae,
            "table1_max_gap": table1_max,
            "table1_gap_by_rate": {
                str(rate): float(gap) for rate, gap in table1_report["signed"].items()
            },
            "table1_cost_by_rate": {
                str(rate): float(cost) for rate, cost in table1_measured.items()
            },
            "table1_qos_by_rate": {
                str(rate): float(qos) for rate, qos in table1_qos.items()
            },
            "table1_all_within": table1_all_within,
            "table1_cell_limit": float(args.early_stop_mae),
        }
        torch.save(checkpoint_dict, checkpoint_path)

        publish = {
            key: value
            for key, value in checkpoint_dict.items()
            if key != "optimizer"
        }
        if mean_val_objective < best_val_objective:
            best_val_objective = mean_val_objective
            best_val_dist = mean_val_dist
            torch.save(publish, best_checkpoint_path)
        if table1_max is not None and table1_max < best_table1_max:
            best_table1_max = table1_max
            torch.save(publish, table1_best_checkpoint_path)
        scheduler.step()
        cost_within = (
            table1_max is not None
            and len(table1_report.get("signed", {})) >= 4
            and table1_max <= args.early_stop_mae
        )
        if cost_within and args.early_stop_epoch > 0:
            print(
                f"         Early stop: every Table I cell within "
                f"{args.early_stop_mae:.1f}%.",
                flush=True,
            )
            break
        if (
            args.early_stop_epoch > 0
            and epoch == args.early_stop_epoch
            and table1_max is not None
            and table1_max > args.early_stop_mae
        ):
            print(
                f"         Early stop at epoch {epoch}: max Table I gap "
                f"{table1_max:.2f}% > {args.early_stop_mae:.1f}%.",
                flush=True,
            )
            break

    print(f"\nTraining completed. Final checkpoint saved to: {checkpoint_path}")
    if best_checkpoint_path.exists():
        print(
            f"Best validation checkpoint saved to: {best_checkpoint_path} "
            f"(objective: {best_val_objective:.2f}, distance: {best_val_dist:.2f})"
        )
    if table1_best_checkpoint_path.exists():
        print(
            f"Closest Table I checkpoint saved to: {table1_best_checkpoint_path} "
            f"(max gap: {best_table1_max:.2f}%)"
        )
    return {
        "seed": int(args.seed),
        "best_val_dist": None if best_val_dist == float("inf") else float(best_val_dist),
        "best_val_objective": None if best_val_objective == float("inf") else float(best_val_objective),
        "best_table1_max_gap": None if best_table1_max == float("inf") else float(best_table1_max),
        "table1_all_within": bool(last_table1_all_within),
        "checkpoint": str(checkpoint_path),
    }


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
