"""Robust DVNDA training over Early/Uniform/Late revelation patterns.

This runner keeps the n=20, m=4 DVNDA architecture and the executed-path
time-driven environment unchanged.  Only the training distribution is
augmented: each physical instance is assigned one of three predeclared
revelation patterns.  Checkpoint selection uses a fixed validation set; the
held-out reviewer test set is never inspected during training.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_

from data import DCVRP_Dataset
from experiments.reviewer_study.paper_dcvrp import (
    generate_paired_paper_datasets,
    generate_paper_dataset,
)
from experiments.reviewer_study.paper_greedy import run_paper_greedy
from experiments.reviewer_study.run_original_uncertainty_n20 import (
    DYNAMIC_RATES,
    build_model,
    configure_training_policy,
    index_dataset,
    load_model,
    make_training_environment,
    repeat_dataset,
    save_checkpoint,
    set_seed,
    validation_metrics,
)
from experiments.reviewer_study.run_revelation_patterns_n20 import (
    transform_revelation,
)


PATTERNS = ("Early", "Uniform", "Late")
PATTERN_TO_CODE = {name: index for index, name in enumerate(PATTERNS)}
ENVIRONMENT_TAG = "time_driven_executed_path_v4"


def transform_revelation_per_instance(
    data: DCVRP_Dataset,
    pattern_codes: torch.Tensor,
) -> DCVRP_Dataset:
    """Apply the symmetric first-third/last-third transformation by row."""

    if pattern_codes.ndim != 1 or pattern_codes.numel() != len(data):
        raise ValueError("pattern_codes must contain one code per instance")
    nodes = data.nodes.clone()
    original = nodes[:, 1:, 4]
    dynamic = original > 0.0
    codes = pattern_codes.to(original.device).view(-1, 1)
    early = 0.30 * original
    late = 0.60 + 0.30 * original
    transformed = torch.where(codes == PATTERN_TO_CODE["Early"], early, original)
    transformed = torch.where(codes == PATTERN_TO_CODE["Late"], late, transformed)
    nodes[:, 1:, 4] = torch.where(dynamic, transformed, original)
    result = DCVRP_Dataset(
        data.veh_count,
        data.veh_capa,
        data.veh_speed,
        nodes,
        data.cust_mask,
    )
    if hasattr(data, "paper_dynamic_rates"):
        result.paper_dynamic_rates = data.paper_dynamic_rates.clone()
    if hasattr(data, "paper_dynamic_counts"):
        result.paper_dynamic_counts = data.paper_dynamic_counts.clone()
    return result


def balanced_pattern_codes(size: int) -> torch.Tensor:
    codes = torch.arange(size, dtype=torch.int64) % len(PATTERNS)
    return codes[torch.randperm(size)]


def evaluate_validation_grid(
    model,
    validation_by_rate: dict[float, DCVRP_Dataset],
    greedy_grid: dict[tuple[float, str], tuple[float, float]],
    device: torch.device,
) -> list[dict]:
    rows: list[dict] = []
    for rate in DYNAMIC_RATES:
        for pattern in PATTERNS:
            data = transform_revelation(
                validation_by_rate[rate], pattern, "tercile"
            )
            penalized, distance, qos = validation_metrics(
                model, data, device, rollouts=1, sample_customers=False
            )
            greedy_distance, greedy_qos = greedy_grid[(rate, pattern)]
            rows.append(
                {
                    "dynamic_rate": rate,
                    "pattern": pattern,
                    "penalized_cost": penalized,
                    "distance": distance,
                    "qos": qos,
                    "greedy_distance": greedy_distance,
                    "greedy_qos": greedy_qos,
                    "relative_gain": 1.0 - distance / greedy_distance,
                }
            )
    return rows


def validation_score(rows: list[dict]) -> tuple[float, bool]:
    """Rank checkpoints on validation robustness without a trend-order gate."""

    distance = np.asarray([row["distance"] for row in rows])
    greedy = np.asarray([row["greedy_distance"] for row in rows])
    qos = np.asarray([row["qos"] for row in rows])
    greedy_qos = np.asarray([row["greedy_qos"] for row in rows])
    excess = np.maximum(distance - greedy, 0.0)
    qos_deficit = np.maximum(greedy_qos - qos - 0.001, 0.0)
    score = float(distance.mean() + 10.0 * excess.mean() + 50.0 * qos_deficit.mean())
    accepted = bool(
        np.all(distance < greedy)
        and np.all(qos + 0.001 >= greedy_qos)
        and np.min(qos) >= 0.995
    )
    return score, accepted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-seed", type=int, default=7319)
    parser.add_argument("--validation-seed", type=int, default=910241)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path(
            "experiments/checkpoints/executed_path_dvnda_n20_m4/"
            "best_accepted.pt"
        ),
        help="Warm-start checkpoint. Use an empty string to train from scratch.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "experiments/checkpoints/revelation_robust_n20_m4/best.pt"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    set_seed(args.train_seed)

    if str(args.init_checkpoint):
        model, initial = load_model(args.init_checkpoint.resolve(), device)
        initial_epoch = initial.get("epoch")
    else:
        model = build_model(device, "independent")
        initial_epoch = None
    configure_training_policy(model, "sample_logp")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    baseline = copy.deepcopy(model).to(device)
    baseline.eval()
    configure_training_policy(baseline, "sample_logp")

    set_seed(args.validation_seed)
    validation_by_rate = generate_paired_paper_datasets(
        args.validation_size,
        DYNAMIC_RATES,
        customer_count=20,
        vehicle_count=4,
    )
    greedy_grid: dict[tuple[float, str], tuple[float, float]] = {}
    for rate in DYNAMIC_RATES:
        for pattern in PATTERNS:
            data = transform_revelation(
                validation_by_rate[rate], pattern, "tercile"
            )
            cost, qos, _ = run_paper_greedy(data, device)
            greedy_grid[(rate, pattern)] = (
                float(cost.mean().item()), float(qos.mean().item())
            )

    set_seed(args.train_seed)
    pool_size = args.steps_per_epoch * args.batch_size
    base_pool = generate_paper_dataset(
        pool_size,
        DYNAMIC_RATES,
        customer_count=20,
        vehicle_count=4,
    )
    pattern_codes = balanced_pattern_codes(pool_size)
    training_pool = transform_revelation_per_instance(base_pool, pattern_codes)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    accepted_path = args.checkpoint.with_name(
        args.checkpoint.stem + "_accepted.pt"
    )
    last_path = args.checkpoint.with_name(args.checkpoint.stem + "_last.pt")
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": ENVIRONMENT_TAG,
        "customer_count": 20,
        "vehicle_count": 4,
        "interval_count": 10,
        "patterns": {
            "Early": "0.30u (first 30% of horizon)",
            "Uniform": "original paper disclosure sample u",
            "Late": "0.60+0.30u (60%-90% of horizon)",
        },
        "training_pattern_allocation": "balanced per fixed training pool",
        "checkpoint_selection": (
            "fixed validation set; mean distance plus penalties for Greedy/QoS deficits"
        ),
        "test_set_used_for_selection": False,
        "warm_start_epoch": initial_epoch,
    }

    best_score = math.inf
    best_accepted_score = math.inf
    history: list[dict] = []
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        configure_training_policy(model, "sample_logp")
        order = torch.randperm(pool_size)
        loss_sum = reward_sum = 0.0
        epoch_started = time.perf_counter()
        policy_rewards: list[torch.Tensor] = []
        baseline_rewards: list[torch.Tensor] = []

        for step in range(1, args.steps_per_epoch + 1):
            offset = (step - 1) * args.batch_size
            indices = order[offset : offset + args.batch_size]
            data = index_dataset(training_pool, indices)
            environment = make_training_environment(data, device)
            _, log_probabilities, reward_steps = model(environment)
            rewards = torch.stack(reward_steps).sum(dim=0)
            with torch.no_grad():
                baseline_data = repeat_dataset(data, 3)
                baseline_environment = make_training_environment(
                    baseline_data, device
                )
                _, _, baseline_steps = baseline(baseline_environment)
                baseline_value = torch.stack(baseline_steps).sum(dim=0).reshape(
                    3, args.batch_size, 1
                ).mean(dim=0)
            advantage = rewards - baseline_value
            normalized = (advantage - advantage.mean()) / advantage.std().clamp_min(1.0e-6)
            sequence_logp = torch.stack(log_probabilities).sum(dim=0)
            loss = -(sequence_logp * normalized.detach()).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            loss_sum += float(loss.item())
            reward_sum += float(rewards.mean().item())
            policy_rewards.append(rewards.detach().cpu().flatten())
            baseline_rewards.append(baseline_value.detach().cpu().flatten())

            if step == 1 or step % args.log_every == 0 or step == args.steps_per_epoch:
                print(
                    f"epoch={epoch:02d}/{args.epochs:02d} "
                    f"step={step:03d}/{args.steps_per_epoch:03d} "
                    f"loss={loss_sum / step:.4f} reward={reward_sum / step:.4f} "
                    f"elapsed={time.perf_counter() - epoch_started:.1f}s",
                    flush=True,
                )

        improvement = torch.cat(policy_rewards) - torch.cat(baseline_rewards)
        if float(improvement.mean()) > 0.0:
            baseline.load_state_dict(model.state_dict())

        rows = evaluate_validation_grid(model, validation_by_rate, greedy_grid, device)
        score, accepted = validation_score(rows)
        mean_distance = float(np.mean([row["distance"] for row in rows]))
        min_qos = float(np.min([row["qos"] for row in rows]))
        minimum_gain = float(np.min([row["relative_gain"] for row in rows]))
        row = {
            "epoch": epoch,
            "loss": loss_sum / args.steps_per_epoch,
            "reward": reward_sum / args.steps_per_epoch,
            "validation_score": score,
            "validation_distance_mean": mean_distance,
            "validation_qos_min": min_qos,
            "validation_relative_gain_min": minimum_gain,
            "accepted": accepted,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        for cell in rows:
            label = f"r{int(100 * cell['dynamic_rate'])}_{cell['pattern'].lower()}"
            row[f"{label}_distance"] = cell["distance"]
            row[f"{label}_qos"] = cell["qos"]
            row[f"{label}_greedy"] = cell["greedy_distance"]
            row[f"{label}_gain"] = cell["relative_gain"]
        history.append(row)

        if score < best_score:
            best_score = score
            save_checkpoint(
                args.checkpoint, model, optimizer, epoch, score,
                mean_distance, min_qos, config
            )
        if accepted and score < best_accepted_score:
            best_accepted_score = score
            accepted_config = dict(config)
            accepted_config["validation_cells"] = rows
            save_checkpoint(
                accepted_path, model, optimizer, epoch, score,
                mean_distance, min_qos, accepted_config
            )
        save_checkpoint(
            last_path, model, optimizer, epoch, best_score,
            mean_distance, min_qos, config
        )
        pd.DataFrame(history).to_csv(
            args.checkpoint.parent / "training_history.csv", index=False
        )
        print(
            f"VALIDATION epoch={epoch:02d} score={score:.4f} "
            f"distance={mean_distance:.4f} min_qos={100 * min_qos:.2f}% "
            f"min_gain={100 * minimum_gain:.2f}% "
            f"accepted={'YES' if accepted else 'NO'}",
            flush=True,
        )

    config["training_seconds"] = time.perf_counter() - training_started
    config["best_validation_score"] = best_score
    config["best_accepted_score"] = (
        None if math.isinf(best_accepted_score) else best_accepted_score
    )
    (args.checkpoint.parent / "training_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"best checkpoint: {args.checkpoint.resolve()}", flush=True)
    print(
        f"accepted checkpoint: "
        f"{accepted_path.resolve() if accepted_path.exists() else 'none'}",
        flush=True,
    )


if __name__ == "__main__":
    main()

