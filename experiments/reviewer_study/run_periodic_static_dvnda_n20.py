"""Train and evaluate DVNDA in the revised periodic-static environment.

The historical event-driven nearest-task Greedy implementation is kept
unchanged and evaluated on the same 100 paired test instances at each dynamic
rate.  DVNDA alone uses the ten-interval periodic-static execution protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import DCVRP_Dataset  # noqa: E402
from experiments.reviewer_study.paper_dcvrp import (  # noqa: E402
    generate_paired_paper_datasets,
    generate_paper_dataset,
)
from experiments.reviewer_study.paper_greedy import run_paper_greedy  # noqa: E402
from experiments.reviewer_study.periodic_static_dvnda import (  # noqa: E402
    ENVIRONMENT_TAG,
    MANUSCRIPT_NORMALIZED_SPEED,
    VectorizedPeriodicStaticDVNDAEnvironment,
)
from experiments.reviewer_study.run_original_uncertainty_n20 import (  # noqa: E402
    PAPER_DETERMINISTIC,
    PAPER_GREEDY,
    build_model,
    configure_training_policy,
    index_dataset,
    repeat_dataset,
)


DYNAMIC_RATES = (0.10, 0.25, 0.50, 0.75)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_environment(
    data: DCVRP_Dataset,
    device: torch.device,
    *,
    pending_cost: float = 5.0,
) -> VectorizedPeriodicStaticDVNDAEnvironment:
    return VectorizedPeriodicStaticDVNDAEnvironment(
        data,
        nodes=data.nodes.to(device),
        pending_cost=pending_cost,
    )


@torch.no_grad()
def evaluate_dvnda(
    model,
    data: DCVRP_Dataset,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    model.eval()
    model.greedy = True
    model.vehicle_greedy = True
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    environment = make_environment(data, device, pending_cost=0.0)
    _, _, rewards = model(environment)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    reward_cost = -torch.stack(rewards).sum(dim=0).squeeze(-1)
    distance = environment.route_distance()
    if not torch.allclose(reward_cost, distance, atol=3.0e-4, rtol=1.0e-5):
        raise RuntimeError("DVNDA reward and executed route distance diverged")
    return {
        "cost": distance.detach().cpu().numpy(),
        "qos": environment.qos().detach().cpu().numpy(),
        "time_s": elapsed,
        "mandatory_return_distance": environment.mandatory_return_distance
        .detach()
        .cpu()
        .numpy(),
        "mandatory_return_count": environment.mandatory_return_count.sum(dim=1)
        .detach()
        .cpu()
        .numpy(),
    }


@torch.no_grad()
def validation_score(model, datasets, device: torch.device) -> tuple[float, float, list[dict]]:
    rows = []
    for rate in DYNAMIC_RATES:
        values = evaluate_dvnda(model, datasets[rate], device)
        rows.append(
            {
                "dynamic_rate": rate,
                "cost": float(np.mean(values["cost"])),
                "qos": float(np.mean(values["qos"])),
            }
        )
    return (
        float(np.mean([row["cost"] for row in rows])),
        float(np.mean([row["qos"] for row in rows])),
        rows,
    )


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train(args: argparse.Namespace, device: torch.device) -> None:
    set_seed(args.train_seed)
    model = build_model(device, "independent")
    configure_training_policy(model, "sample_logp")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    baseline = copy.deepcopy(model).to(device)
    baseline.eval()
    configure_training_policy(baseline, "sample_logp")

    checkpoint = args.checkpoint
    last_checkpoint = checkpoint.with_name(checkpoint.stem + "_last.pt")
    start_epoch = 1
    best_cost = math.inf
    best_qos = -math.inf
    history: list[dict] = []
    if args.resume and last_checkpoint.exists():
        saved = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        baseline.load_state_dict(saved.get("baseline", saved["model"]))
        start_epoch = int(saved["epoch"]) + 1
        best_cost = float(saved.get("best_cost", math.inf))
        best_qos = float(saved.get("best_qos", -math.inf))
        history_path = checkpoint.parent / "training_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")
        print(f"resumed epoch {start_epoch - 1} from {last_checkpoint}", flush=True)

    set_seed(args.validation_seed)
    validation_data = generate_paired_paper_datasets(
        args.validation_size,
        DYNAMIC_RATES,
        customer_count=20,
        vehicle_count=4,
    )
    set_seed(args.train_seed)
    pool_size = args.steps_per_epoch * args.batch_size
    print(f"generating fixed training pool: {pool_size} n=20,m=4 instances", flush=True)
    training_pool = generate_paper_dataset(
        pool_size,
        DYNAMIC_RATES,
        customer_count=20,
        vehicle_count=4,
    )

    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": ENVIRONMENT_TAG,
        "normalized_vehicle_speed": MANUSCRIPT_NORMALIZED_SPEED,
        "physical_speed_coordinate_units_per_minute": 1.0,
        "customer_count": 20,
        "vehicle_count": 4,
        "interval_count": 10,
        "training_vehicle_policy": "sample_logp",
        "evaluation_policy": "argmax/greedy",
        "model_initialization": "from_scratch" if start_epoch == 1 else "resumed",
        "device_resolved": str(device),
    }

    training_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        configure_training_policy(model, "sample_logp")
        order = torch.randperm(pool_size)
        epoch_started = time.perf_counter()
        loss_sum = reward_sum = gradient_sum = selector_gradient_sum = 0.0
        policy_rewards: list[torch.Tensor] = []
        baseline_rewards: list[torch.Tensor] = []

        for step in range(1, args.steps_per_epoch + 1):
            offset = (step - 1) * args.batch_size
            data = index_dataset(
                training_pool, order[offset : offset + args.batch_size]
            )
            environment = make_environment(data, device)
            _, log_probabilities, reward_steps = model(environment)
            rewards = torch.stack(reward_steps).sum(dim=0)

            with torch.no_grad():
                repeated = repeat_dataset(data, args.baseline_rollouts)
                baseline_environment = make_environment(repeated, device)
                _, _, baseline_steps = baseline(baseline_environment)
                baseline_value = torch.stack(baseline_steps).sum(dim=0).reshape(
                    args.baseline_rollouts, args.batch_size, 1
                ).mean(dim=0)

            advantage = rewards - baseline_value
            advantage = (
                advantage - advantage.mean()
            ) / advantage.std().clamp_min(1.0e-6)
            sequence_logp = torch.stack(log_probabilities).sum(dim=0)
            loss = -(sequence_logp * advantage.detach()).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            selector_terms = [
                parameter.grad.detach().square().sum()
                for parameter in model.selector.parameters()
                if parameter.grad is not None
            ]
            selector_gradient = (
                torch.sqrt(torch.stack(selector_terms).sum())
                if selector_terms
                else loss.new_zeros(())
            )
            gradient = clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            loss_sum += float(loss.item())
            reward_sum += float(rewards.mean().item())
            gradient_sum += float(gradient)
            selector_gradient_sum += float(selector_gradient)
            policy_rewards.append(rewards.detach().cpu().flatten())
            baseline_rewards.append(baseline_value.detach().cpu().flatten())

            if step == 1 or step % args.log_every == 0 or step == args.steps_per_epoch:
                print(
                    f"epoch={epoch:02d}/{args.epochs:02d} "
                    f"step={step:03d}/{args.steps_per_epoch:03d} "
                    f"loss={loss_sum / step:.4f} "
                    f"reward={reward_sum / step:.4f} "
                    f"grad={gradient_sum / step:.3f} "
                    f"selector_grad={selector_gradient_sum / step:.3f} "
                    f"elapsed={time.perf_counter() - epoch_started:.1f}s",
                    flush=True,
                )

        validation_cost, validation_qos, rate_rows = validation_score(
            model, validation_data, device
        )
        policy_mean = float(torch.cat(policy_rewards).mean())
        baseline_mean = float(torch.cat(baseline_rewards).mean())
        baseline_updated = policy_mean > baseline_mean
        if baseline_updated:
            baseline.load_state_dict(model.state_dict())

        epoch_seconds = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch,
            "loss": loss_sum / args.steps_per_epoch,
            "reward": reward_sum / args.steps_per_epoch,
            "gradient_norm": gradient_sum / args.steps_per_epoch,
            "selector_gradient_norm": selector_gradient_sum / args.steps_per_epoch,
            "validation_cost": validation_cost,
            "validation_qos": validation_qos,
            "baseline_updated": baseline_updated,
            "epoch_seconds": epoch_seconds,
        }
        for rate_row in rate_rows:
            label = int(round(100 * rate_row["dynamic_rate"]))
            row[f"rate_{label}_cost"] = rate_row["cost"]
            row[f"rate_{label}_qos"] = rate_row["qos"]
        history.append(row)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(
            checkpoint.parent / "training_history.csv", index=False
        )

        is_better = (
            validation_qos > best_qos + 1.0e-8
            or (
                abs(validation_qos - best_qos) <= 1.0e-8
                and validation_cost < best_cost
            )
        )
        if is_better:
            best_cost, best_qos = validation_cost, validation_qos
            save_checkpoint(
                checkpoint,
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "baseline": baseline.state_dict(),
                    "best_cost": best_cost,
                    "best_qos": best_qos,
                    "config": config,
                },
            )
        save_checkpoint(
            last_checkpoint,
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "baseline": baseline.state_dict(),
                "best_cost": best_cost,
                "best_qos": best_qos,
                "config": config,
            },
        )
        per_rate = " ".join(
            f"r{int(100*r['dynamic_rate'])}={r['cost']:.3f}/{100*r['qos']:.2f}%"
            for r in rate_rows
        )
        print(
            f"epoch={epoch:02d} validation={validation_cost:.4f}/"
            f"{100*validation_qos:.2f}% baseline_update={baseline_updated} "
            f"{per_rate} epoch_seconds={epoch_seconds:.1f}",
            flush=True,
        )

    config["training_seconds_this_run"] = time.perf_counter() - training_started
    config["parameter_updates"] = args.epochs * args.steps_per_epoch
    config["best_validation_cost"] = best_cost
    config["best_validation_qos"] = best_qos
    (checkpoint.parent / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"best checkpoint: {checkpoint}", flush=True)


def load_model(checkpoint: Path, device: torch.device):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(device, "independent")
    model.load_state_dict(saved["model"])
    model.eval()
    model.greedy = True
    model.vehicle_greedy = True
    return model, saved


def evaluate(args: argparse.Namespace, device: torch.device) -> None:
    model, saved = load_model(args.checkpoint, device)
    set_seed(args.test_seed)
    datasets = generate_paired_paper_datasets(
        args.test_size,
        DYNAMIC_RATES,
        customer_count=20,
        vehicle_count=4,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    raw_rows: list[dict] = []

    # Exclude one-off CUDA context/kernel initialization from the solve time.
    # The warm-up uses deterministic greedy decoding and does not alter either
    # the checkpoint or the subsequently reported instances.
    if device.type == "cuda":
        evaluate_dvnda(model, datasets[DYNAMIC_RATES[0]], device)

    for rate in DYNAMIC_RATES:
        data = datasets[rate]
        dvnda = evaluate_dvnda(model, data, device)
        greedy_started = time.perf_counter()
        greedy_cost, greedy_qos, _ = run_paper_greedy(data, torch.device("cpu"))
        greedy_elapsed = time.perf_counter() - greedy_started
        methods = (
            (
                "DVNDA-v3",
                np.asarray(dvnda["cost"]),
                np.asarray(dvnda["qos"]),
                float(dvnda["time_s"]),
                PAPER_DETERMINISTIC[rate],
            ),
            (
                "Greedy-event",
                greedy_cost.cpu().numpy(),
                greedy_qos.cpu().numpy(),
                greedy_elapsed,
                PAPER_GREEDY[rate],
            ),
        )
        for method, cost, qos, elapsed, paper in methods:
            row = {
                "dynamic_rate": rate,
                "method": method,
                "instances": args.test_size,
                "cost_mean": float(cost.mean()),
                "cost_sd": float(cost.std(ddof=1)),
                "qos_mean": float(qos.mean()),
                "qos_sd": float(qos.std(ddof=1)),
                "time_s": elapsed,
                "paper_cost_mean": float(paper["cost_mean"]),
                "paper_cost_sd": float(paper["cost_sd"]),
                "paper_qos_mean": float(paper["qos_mean"]),
                "cost_gap_vs_paper": float(cost.mean() - paper["cost_mean"]),
                "checkpoint_epoch": int(saved["epoch"]),
            }
            if method == "DVNDA-v3":
                mandatory_return_mean = float(
                    np.mean(dvnda["mandatory_return_distance"])
                )
                row["mandatory_return_distance_mean"] = mandatory_return_mean
                row["mandatory_return_count_mean"] = float(
                    np.mean(dvnda["mandatory_return_count"])
                )
                row["cost_excluding_mandatory_returns"] = (
                    row["cost_mean"] - mandatory_return_mean
                )
                row["nonreturn_gap_vs_paper"] = (
                    row["cost_excluding_mandatory_returns"]
                    - row["paper_cost_mean"]
                )
            rows.append(row)
            for instance, (instance_cost, instance_qos) in enumerate(
                zip(cost, qos)
            ):
                raw_rows.append(
                    {
                        "dynamic_rate": rate,
                        "method": method,
                        "instance": instance,
                        "cost": float(instance_cost),
                        "qos": float(instance_qos),
                    }
                )
        print(
            f"rate={100*rate:.0f}% "
            f"DVNDA={np.mean(dvnda['cost']):.3f}±{np.std(dvnda['cost'], ddof=1):.3f}/"
            f"{100*np.mean(dvnda['qos']):.2f}% "
            f"Greedy={greedy_cost.mean():.3f}±{greedy_cost.std(unbiased=True):.3f}/"
            f"{100*greedy_qos.mean():.2f}%",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    raw = pd.DataFrame(raw_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    raw.to_csv(output_dir / "per_instance.csv", index=False)
    metadata = {
        "environment": ENVIRONMENT_TAG,
        "dvnda_protocol": "10-interval periodic static re-optimization",
        "greedy_protocol": "event-driven nearest visible feasible task when idle",
        "same_test_instances": True,
        "same_travel_speed": True,
        "normalized_vehicle_speed": MANUSCRIPT_NORMALIZED_SPEED,
        "physical_speed_coordinate_units_per_minute": 1.0,
        "test_seed": args.test_seed,
        "test_instances_per_rate": args.test_size,
        "timing": "one unreported CUDA warm-up batch, then wall-clock batch time",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(saved["epoch"]),
        "comparison_caveat": (
            "DVNDA and Greedy use the requested different execution protocols; "
            "therefore this is not an environment-identical algorithm comparison."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    lines = [
        "# Periodic-static DVNDA n=20, m=4",
        "",
        f"Checkpoint epoch: {int(saved['epoch'])}; paired test instances per rate: {args.test_size}.",
        "",
        "| Dynamic rate | Method | Cost (mean ± SD) | QoS (mean ± SD) | Time for 100 (s) | Paper cost | Gap |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {100*row.dynamic_rate:.0f}% | {row.method} | "
            f"{row.cost_mean:.2f} ± {row.cost_sd:.2f} | "
            f"{100*row.qos_mean:.2f}% ± {100*row.qos_sd:.2f}% | "
            f"{row.time_s:.3f} | {row.paper_cost_mean:.2f} ± {row.paper_cost_sd:.2f} | "
            f"{row.cost_gap_vs_paper:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## DVNDA cost decomposition",
            "",
            "| Dynamic rate | Total cost | Mandatory interval returns | Cost excluding those returns | Paper DVNDA | Non-return gap |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary[summary.method == "DVNDA-v3"].iterrows():
        lines.append(
            f"| {100*row.dynamic_rate:.0f}% | {row.cost_mean:.2f} | "
            f"{row.mandatory_return_distance_mean:.2f} | "
            f"{row.cost_excluding_mandatory_returns:.2f} | "
            f"{row.paper_cost_mean:.2f} | {row.nonreturn_gap_vs_paper:+.2f} |"
        )
    lines.extend(
        [
            "",
            "DVNDA uses the revised periodic-static protocol; Greedy is the unchanged event-driven nearest-task baseline. Because the requested execution protocols differ, their row-to-row difference is not a strictly environment-identical algorithm effect.",
        ]
    )
    (output_dir / "TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train_eval", "train", "eval"), default="train_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--baseline-rollouts", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--train-seed", type=int, default=1234)
    parser.add_argument("--validation-seed", type=int, default=4321)
    parser.add_argument("--test-seed", type=int, default=20260821)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/checkpoints/periodic_static_dvnda_n20_m4/best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/periodic_static_dvnda_n20_m4"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    print(
        f"device={device} gpu={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'} "
        f"environment={ENVIRONMENT_TAG}",
        flush=True,
    )
    if args.mode in ("train_eval", "train"):
        train(args, device)
    if args.mode in ("train_eval", "eval"):
        evaluate(args, device)


if __name__ == "__main__":
    main()
