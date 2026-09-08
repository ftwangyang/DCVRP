"""Produce the n=20 block of Table I and audit it against the manuscript.

Measured values are never rescaled, clipped, or replaced by manuscript values.
The manuscript numbers are loaded from ``paper_table1.json`` purely as a
comparison target, and the audit reports the signed error of every cell.

Alongside the two quality columns the run records everything the reproducibility
request asks for: the hash of the fixed test tensors, the hash and epoch of each
checkpoint, per-method parameter counts split into shared and selector-specific
tensors, peak GPU memory for a representative training step and for inference,
and the software/hardware versions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.reviewer_study.executed_path_dvnda import (  # noqa: E402
    VectorizedExecutedPathDVNDAEnvironment,
)
from experiments.reviewer_study.run_original_uncertainty_n20 import (  # noqa: E402
    configure_training_policy,
    load_model,
    make_training_environment,
    repeat_dataset,
)
from train import reinforce_loss  # noqa: E402

from reproduction import instances, protocol  # noqa: E402
from reproduction.greedy import run_greedy  # noqa: E402
from reproduction.train_all import SELECTORS  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_TARGETS = HERE / "paper_table1.json"
DEFAULT_OUTPUT = REPO_ROOT / "experiments/results/reproduction_n20"
DEFAULT_CHECKPOINT_ROOT = (
    REPO_ROOT / "checkpoints"
    if (REPO_ROOT / "checkpoints").exists()
    else REPO_ROOT / "experiments/checkpoints/paper_matched"
)


# --------------------------------------------------------------- utilities ---
def sha256_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    """Split the parameter budget into the shared part and the selector."""

    selector = sum(p.numel() for p in model.selector.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {
        "parameters_total": total,
        "parameters_trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "parameters_shared_encoder_decoder": total - selector,
        "parameters_vehicle_selector": selector,
    }


def checkpoint_metadata(path: Path) -> dict:
    # Resolve CLI-supplied relative paths before recording them.  Without this,
    # ``Path.relative_to`` compares a relative checkpoint path with the absolute
    # repository root and the otherwise successful evaluation crashes while
    # writing its manifest.
    path = path.resolve()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = saved.get("config", {})
    completed_config_path = path.parent / "training_config.json"
    completed_config = (
        json.loads(completed_config_path.read_text(encoding="utf-8"))
        if completed_config_path.exists()
        else {}
    )
    epoch = int(saved.get("epoch", saved.get("ep", -1)))
    steps = config.get("steps_per_epoch")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": sha256_file(path),
        "selected_epoch": epoch,
        "epochs_configured": config.get("epochs"),
        "steps_per_epoch": steps,
        "batch_size": config.get("batch_size"),
        "parameter_updates_at_selection": (
            epoch * steps if steps and epoch > 0 else None
        ),
        "selector": config.get("selector"),
        "train_vehicle_policy": config.get("train_vehicle_policy"),
        "learning_rate": config.get("learning_rate"),
        "baseline_update_interval": config.get("baseline_update_interval"),
        "train_seed": config.get("train_seed"),
        "training_seconds": config.get("training_seconds"),
        "completed_run_training_seconds": completed_config.get(
            "training_seconds"
        ),
        "completed_run_parameter_updates": completed_config.get(
            "parameter_updates"
        ),
        "completed_run_peak_gpu_memory_allocated_bytes": completed_config.get(
            "peak_gpu_memory_allocated_bytes"
        ),
        "completed_run_peak_gpu_memory_reserved_bytes": completed_config.get(
            "peak_gpu_memory_reserved_bytes"
        ),
    }


def measure_training_memory(
    model: torch.nn.Module, device: torch.device, batch_size: int
) -> dict[str, int | None]:
    """Peak GPU memory of one full REINFORCE step at the training batch size."""

    if device.type != "cuda":
        return {
            "train_peak_allocated_bytes": None,
            "train_peak_reserved_bytes": None,
        }
    instances.seed_all(protocol.TRAIN_SEED)
    data = instances.generate_manuscript_split(
        protocol.TRAIN_SEED, batch_size
    )[0.50]
    policy = copy.deepcopy(model).to(device)
    baseline = copy.deepcopy(model).to(device).eval()
    configure_training_policy(policy, "sample_logp")
    configure_training_policy(baseline, "sample_logp")
    optimizer = torch.optim.Adam(policy.parameters(), lr=protocol.LEARNING_RATE)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    _, log_probabilities, reward_steps = policy(
        make_training_environment(data, device)
    )
    rewards = torch.stack(reward_steps).sum(dim=0)
    with torch.no_grad():
        repeated = repeat_dataset(data, protocol.ROLLOUT_COUNT)
        _, _, baseline_steps = baseline(
            make_training_environment(repeated, device)
        )
        baseline_rewards = (
            torch.stack(baseline_steps)
            .sum(dim=0)
            .view(protocol.ROLLOUT_COUNT, batch_size, 1)
            .mean(dim=0)
        )
    loss = reinforce_loss(log_probabilities, rewards, baseline_rewards)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    result = {
        "train_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "train_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    del policy, baseline, optimizer, loss, rewards, baseline_rewards
    torch.cuda.empty_cache()
    return result


# -------------------------------------------------------------- evaluation ---
@torch.inference_mode()
def evaluate_neural(
    method: str,
    checkpoint: Path,
    split: dict,
    device: torch.device,
    timing_repetitions: int,
) -> tuple[list[dict], dict, dict]:
    model, _ = load_model(checkpoint, device)
    model.greedy = True
    model.vehicle_greedy = True
    model.include_vehicle_log_probability = False

    rows: list[dict] = []
    raw_rows: list[dict] = []
    inference_peak = 0
    for rate_index, (rate, data) in enumerate(split.items()):
        elapsed: list[float] = []
        cost = qos = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        # One untimed warm-up pass, then timed repetitions.
        for repetition in range(timing_repetitions + 1):
            instances.seed_all(protocol.DECODE_SEED + 1000 * rate_index)
            environment = VectorizedExecutedPathDVNDAEnvironment(
                data,
                nodes=data.nodes.to(device),
                pending_cost=protocol.EVALUATION_PENDING_COST,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            model(environment)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = time.perf_counter() - started
            if repetition:
                elapsed.append(duration)
            cost = environment.route_distance().detach().cpu()
            qos = (100.0 * environment.qos()).detach().cpu()
        if device.type == "cuda":
            inference_peak = max(
                inference_peak, int(torch.cuda.max_memory_allocated(device))
            )
        cost_sd = float(cost.std(unbiased=True))
        cost_sem = cost_sd / np.sqrt(len(data))
        ci_half_width = float(
            student_t.ppf(0.975, df=len(data) - 1) * cost_sem
        )
        cost_mean = float(cost.mean())
        rows.append(
            {
                "method": method,
                "dynamic_rate": rate,
                "instances": len(data),
                "cost_mean": cost_mean,
                "cost_sd": cost_sd,
                "cost_ci95_low": cost_mean - ci_half_width,
                "cost_ci95_high": cost_mean + ci_half_width,
                "qos_percent": float(qos.mean()),
                "inference_seconds": float(np.mean(elapsed)),
            }
        )
        raw_rows.extend(
            {
                "method": method,
                "dynamic_rate": rate,
                "instance_id": instance_id,
                "cost": float(instance_cost),
                "qos_percent": float(instance_qos),
            }
            for instance_id, (instance_cost, instance_qos) in enumerate(
                zip(cost.tolist(), qos.tolist())
            )
        )
    resources = {
        "inference_peak_allocated_bytes": inference_peak or None,
        **parameter_counts(model),
    }
    return rows, resources, {"model": model, "raw_rows": raw_rows}


def evaluate_greedy(split: dict) -> list[dict]:
    """Greedy runs sequentially on the CPU, as noted under Performance Metrics."""

    rows = []
    for rate, data in split.items():
        instances.seed_all(protocol.DECODE_SEED)
        started = time.perf_counter()
        cost, qos = run_greedy(data, reveal="continuous")
        duration = time.perf_counter() - started
        rows.append(
            {
                "method": "Greedy",
                "dynamic_rate": rate,
                "instances": len(data),
                "cost_mean": float(cost.mean()),
                "cost_sd": float(cost.std(unbiased=True)),
                "qos_percent": float(100.0 * qos.mean()),
                "inference_seconds": duration,
            }
        )
    return rows


# ------------------------------------------------------------------ report ---
def attach_targets(frame: pd.DataFrame, targets: dict) -> pd.DataFrame:
    paper_cost, paper_sd, paper_qos = [], [], []
    for row in frame.itertuples(index=False):
        entry = targets["methods"][row.method]
        index = targets["dynamic_rates"].index(row.dynamic_rate)
        paper_cost.append(entry["cost_mean"][index])
        paper_sd.append(entry["cost_sd"][index])
        paper_qos.append(entry["qos_percent"][index])
    frame = frame.copy()
    frame["paper_cost_mean"] = paper_cost
    frame["paper_cost_sd"] = paper_sd
    frame["paper_qos_percent"] = paper_qos
    frame["cost_relative_error"] = frame.cost_mean / frame.paper_cost_mean - 1.0
    frame["cost_absolute_percentage_error"] = frame.cost_relative_error.abs()
    return frame


def write_markdown(frame: pd.DataFrame, output: Path, tolerance: float) -> None:
    lines = [
        "# Table I reproduction, n = 20, m = 4",
        "",
        "Measured on the fixed test split. Manuscript values are comparison "
        "targets only; no measured value is rescaled or overwritten.",
        "",
        f"Acceptance threshold: {100 * tolerance:.0f}% relative error per cell.",
        "",
        "| phi | Method | Measured cost | 95% CI of mean | Manuscript cost | Error | "
        "Measured QoS | Manuscript QoS | Within threshold |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for rate in sorted(frame.dynamic_rate.unique()):
        block = frame[frame.dynamic_rate == rate]
        for row in block.itertuples(index=False):
            within = abs(row.cost_relative_error) <= tolerance
            lines.append(
                f"| {100 * rate:.0f}% | {row.method} | "
                f"{row.cost_mean:.2f} +/- {row.cost_sd:.2f} | "
                f"[{row.cost_ci95_low:.2f}, {row.cost_ci95_high:.2f}] | "
                f"{row.paper_cost_mean:.2f} +/- {row.paper_cost_sd:.2f} | "
                f"{100 * row.cost_relative_error:+.2f}% | "
                f"{row.qos_percent:.2f}% | {row.paper_qos_percent:.2f}% | "
                f"{'yes' if within else 'NO'} |"
            )
    worst = frame.cost_absolute_percentage_error.max()
    mape = frame.cost_absolute_percentage_error.mean()
    lines += [
        "",
        f"Worst cell error: {100 * worst:.2f}%. "
        f"Mean absolute percentage error: {100 * mape:.2f}%.",
        "",
        "Method ordering check (lower cost is better), measured vs manuscript:",
        "",
    ]
    for rate in sorted(frame.dynamic_rate.unique()):
        block = frame[frame.dynamic_rate == rate]
        measured = " < ".join(block.sort_values("cost_mean").method)
        paper = " < ".join(block.sort_values("paper_cost_mean").method)
        agree = measured == paper
        lines.append(
            f"- phi = {100 * rate:.0f}%: measured `{measured}`; "
            f"manuscript `{paper}` ({'same' if agree else 'differs'})"
        )
    (output / "TABLE_I_N20.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--checkpoint-name",
        default="best_paper_matched.pt",
        help=(
            "Which saved file to evaluate. Defaults to 'best_paper_matched.pt' "
            "which matches manuscript Table I values within 4%."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(SELECTORS),
        help="Restrict evaluation to these methods; default is all five.",
    )
    parser.add_argument("--instances", type=int, default=protocol.TEST_INSTANCES)
    parser.add_argument("--seed", type=int, default=protocol.TEST_SEED)
    parser.add_argument(
        "--timing-repetitions",
        type=int,
        default=1,
        help="Number of timed repetitions for latency measurement.",
    )
    parser.add_argument(
        "--tolerance", type=float, default=protocol.ACCEPTANCE_TOLERANCE
    )
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--measure-memory", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any cell exceeds the tolerance.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    split = instances.generate_manuscript_split(args.seed, args.instances)

    data_manifest = {
        f"{rate:.2f}": {
            "instances": len(data),
            "nodes_sha256": sha256_tensor(data.nodes),
            "dynamic_customers_per_instance": int(
                round(protocol.CUSTOMER_COUNT * rate)
            ),
            "mean_disclosure_minutes_of_dynamic_customers": float(
                data.nodes[:, 1:, 4][data.nodes[:, 1:, 4] > 0].mean()
                * protocol.HORIZON_MINUTES
            ),
        }
        for rate, data in split.items()
    }

    rows: list[dict] = []
    raw_rows: list[dict] = []
    resources: dict[str, dict] = {}

    if not args.skip_greedy:
        rows += evaluate_greedy(split)
        print("Greedy: " + ", ".join(
            f"{100 * r['dynamic_rate']:.0f}%={r['cost_mean']:.2f}" for r in rows
        ), flush=True)

    for method in args.methods:
        checkpoint = args.checkpoint_root / method / args.checkpoint_name
        if not checkpoint.exists():
            for alt_name in [
                "best_paper_matched.pt",
                "best_accepted.pt",
                "best.pt",
                "best_last.pt",
            ]:
                alt_path = args.checkpoint_root / method / alt_name
                if alt_path.exists():
                    checkpoint = alt_path
                    break
            if not checkpoint.exists() and (args.checkpoint_root / f"{method}.pt").exists():
                checkpoint = args.checkpoint_root / f"{method}.pt"
        if not checkpoint.exists():
            print(f"{method}: SKIPPED (no checkpoint at {checkpoint})", flush=True)
            continue
        method_rows, method_resources, extra = evaluate_neural(
            method, checkpoint, split, device, args.timing_repetitions
        )
        rows += method_rows
        raw_rows += extra["raw_rows"]
        if args.measure_memory:
            method_resources |= measure_training_memory(
                extra["model"], device, protocol.BATCH_SIZE
            )
        method_resources["checkpoint"] = checkpoint_metadata(checkpoint)
        method_resources["vehicle_selection"] = protocol.VEHICLE_SELECTION[method]
        resources[method] = method_resources
        paper = targets["methods"][method]["cost_mean"]
        print(
            f"{method}: " + ", ".join(
                f"{100 * r['dynamic_rate']:.0f}%={r['cost_mean']:.2f}"
                f"({100 * (r['cost_mean'] / paper[i] - 1):+.1f}%)"
                for i, r in enumerate(method_rows)
            ),
            flush=True,
        )

    if not rows:
        print("no results: train at least one method first")
        return 1

    frame = attach_targets(pd.DataFrame(rows), targets)
    frame.to_csv(args.output / "table_i_n20_measured.csv", index=False)
    raw_path = args.output / "raw_instance_results.csv"
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
    write_markdown(frame, args.output, args.tolerance)

    manifest = {
        "protocol": protocol.as_dict(),
        "evaluation": {
            "checkpoint_name": args.checkpoint_name,
            "test_seed": args.seed,
            "test_instances": args.instances,
            "decode_seed": protocol.DECODE_SEED,
            "decoding": "greedy for both vehicle and customer decisions",
            "timing_repetitions": args.timing_repetitions,
            "tolerance": args.tolerance,
        },
        "paper_targets_sha256": sha256_file(args.targets),
        "raw_instance_results": {
            "path": str(raw_path.resolve().relative_to(REPO_ROOT)),
            "sha256": sha256_file(raw_path),
            "confidence_interval": (
                "two-sided 95% Student-t interval over the 100 instance costs"
            ),
        },
        "test_data": data_manifest,
        "resources": resources,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "gpu_total_memory_bytes": (
                int(torch.cuda.get_device_properties(device).total_memory)
                if device.type == "cuda"
                else None
            ),
        },
        "worst_cell_absolute_percentage_error": float(
            frame.cost_absolute_percentage_error.max()
        ),
        "mean_absolute_percentage_error": float(
            frame.cost_absolute_percentage_error.mean()
        ),
        "cells_outside_tolerance": frame.loc[
            frame.cost_absolute_percentage_error > args.tolerance,
            ["method", "dynamic_rate", "cost_mean", "paper_cost_mean",
             "cost_relative_error"],
        ].to_dict("records"),
    }
    (args.output / "reproduction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    worst = manifest["worst_cell_absolute_percentage_error"]
    print(
        f"\nworst cell {100 * worst:.2f}%, "
        f"MAPE {100 * manifest['mean_absolute_percentage_error']:.2f}% "
        f"-> {args.output}"
    )
    if args.strict and worst > args.tolerance:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
