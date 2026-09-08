"""Train and evaluate DVNDA for a fixed n=20 and varying fleet size.

The m=4 row is taken verbatim from the manuscript.  Every other fleet size is
trained independently with the same paper-aligned configuration and evaluated
on 100 paired instances at each of the four dynamic rates.  The runner reuses
the latest executed-path environment: cost contains only dispatched/executed
travel, while unstarted provisional route suffixes are refunded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import DCVRP_Dataset
from experiments.reviewer_study import run_original_uncertainty_n20 as paper_runner
from experiments.reviewer_study.executed_path_dvnda import (
    ENVIRONMENT_TAG,
    VectorizedExecutedPathDVNDAEnvironment,
)
from experiments.reviewer_study.model import ExperimentalAttentionLearner
from experiments.reviewer_study.paper_dcvrp import (
    generate_paired_paper_datasets,
    generate_paper_dataset,
)
from experiments.reviewer_study.selectors import IndependentSelector


DYNAMIC_RATES = (0.10, 0.25, 0.50, 0.75)
PAPER_M4 = {
    0.10: {"cost_mean": 8.31, "cost_sd": 1.22, "qos_mean": 1.0},
    0.25: {"cost_mean": 8.95, "cost_sd": 1.30, "qos_mean": 1.0},
    0.50: {"cost_mean": 10.47, "cost_sd": 1.53, "qos_mean": 1.0},
    0.75: {"cost_mean": 11.78, "cost_sd": 1.44, "qos_mean": 1.0},
}


def build_model(vehicle_count: int, device: torch.device) -> ExperimentalAttentionLearner:
    selector = IndependentSelector(
        vehicle_count=vehicle_count,
        vehicle_state_size=4,
        customer_feature_size=5,
        model_size=64,
        head_count=4,
        evaluation_rule="argmax",
    )
    return ExperimentalAttentionLearner(
        selector=selector,
        customer_feature_size=5,
        vehicle_state_size=4,
        model_size=128,
        layer_count=3,
        head_count=8,
        ff_size=512,
        tanh_exploration=10,
    ).to(device)


def _dataset(batch_size: int, dynamic_rate, vehicle_count: int) -> DCVRP_Dataset:
    return generate_paper_dataset(
        batch_size,
        dynamic_rate,
        customer_count=20,
        vehicle_count=vehicle_count,
    )


@contextmanager
def paper_runner_for_fleet(vehicle_count: int):
    """Temporarily generalize the validated n=20 trainer without editing it."""

    original_build = paper_runner.build_model
    original_generate = paper_runner.generate_dataset
    original_paired = paper_runner.generate_paired_paper_datasets

    def patched_build(device: torch.device, selector_name: str = "independent"):
        if selector_name != "independent":
            raise ValueError("fleet-size study only supports independent DVNDA")
        return build_model(vehicle_count, device)

    def patched_generate(batch_size: int, dynamic_rate):
        return _dataset(batch_size, dynamic_rate, vehicle_count)

    def patched_paired(batch_size, dynamic_rates, customer_count=20, vehicle_count=4):
        del customer_count, vehicle_count
        return generate_paired_paper_datasets(
            batch_size,
            dynamic_rates,
            customer_count=20,
            vehicle_count=vehicle_count_for_patch,
        )

    vehicle_count_for_patch = vehicle_count
    paper_runner.build_model = patched_build
    paper_runner.generate_dataset = patched_generate
    paper_runner.generate_paired_paper_datasets = patched_paired
    try:
        yield
    finally:
        paper_runner.build_model = original_build
        paper_runner.generate_dataset = original_generate
        paper_runner.generate_paired_paper_datasets = original_paired


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_namespace(args: argparse.Namespace, checkpoint: Path) -> argparse.Namespace:
    return argparse.Namespace(
        mode="train",
        device=args.device,
        selector="independent",
        train_seed=args.train_seed,
        validation_seed=args.validation_seed,
        test_seed=args.test_seed,
        noise_seed=8675309,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        validation_size=args.validation_size,
        validation_rollouts=args.validation_rollouts,
        validation_decode_seed=271828,
        learning_rate=args.learning_rate,
        max_grad_norm=2.0,
        baseline_update_interval=args.baseline_update_interval,
        amp=args.amp,
        fixed_training_pool=True,
        normalize_advantage=True,
        log_every=args.log_every,
        test_size=args.test_size,
        decode_seed=args.decode_seed,
        deterministic_rollouts=1,
        scale_rollouts=20,
        scale_chunk_size=10,
        train_vehicle_policy="sample_logp",
        acceptance_policy="greedy",
        dvnda_comparison_policy="greedy",
        paired_comparison=True,
        trend_tolerance=0.0,
        qos_tolerance=0.001,
        gain_profile_tolerance=0.005,
        minimum_gain_decay=-1.0,
        maximum_high_rate_gain=10.0,
        paper_cost_mape_tolerance=10.0,
        noise_replications=1,
        uncertainty_policy="greedy",
        benchmark_steps=1,
        checkpoint=checkpoint,
        output_dir=args.output_dir,
        force_train=args.force_train,
        resume=False,
    )


@torch.no_grad()
def evaluate(
    vehicle_count: int,
    checkpoint: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(vehicle_count, device)
    model.load_state_dict(payload["model"])
    model.eval()
    model.greedy = True
    model.vehicle_greedy = True

    paper_runner.set_seed(args.test_seed)
    paired = generate_paired_paper_datasets(
        args.test_size,
        DYNAMIC_RATES,
        customer_count=20,
        vehicle_count=vehicle_count,
    )

    # Exclude one-off CUDA initialization from the reported batch time.
    warmup = paired[DYNAMIC_RATES[0]]
    warmup_env = VectorizedExecutedPathDVNDAEnvironment(
        warmup, nodes=warmup.nodes.to(device), pending_cost=0.0
    )
    model(warmup_env)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    summary_rows: list[dict] = []
    raw_rows: list[dict] = []
    for rate_index, dynamic_rate in enumerate(DYNAMIC_RATES):
        data = paired[dynamic_rate]
        paper_runner.set_seed(args.decode_seed + rate_index * 100_000)
        environment = VectorizedExecutedPathDVNDAEnvironment(
            data, nodes=data.nodes.to(device), pending_cost=0.0
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        _, _, rewards = model(environment)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        cost = (-torch.stack(rewards).sum(dim=0).squeeze(-1)).cpu().numpy()
        qos = environment.qos().cpu().numpy()
        summary_rows.append(
            {
                "vehicle_count": vehicle_count,
                "dynamic_rate": dynamic_rate,
                "n_instances": args.test_size,
                "cost_mean": float(cost.mean()),
                "cost_sd": float(cost.std(ddof=1)),
                "qos_mean": float(qos.mean()),
                "qos_sd": float(qos.std(ddof=1)),
                "solve_time_100_s": elapsed,
                "checkpoint_epoch": int(payload.get("epoch", -1)),
            }
        )
        raw_rows.extend(
            {
                "vehicle_count": vehicle_count,
                "dynamic_rate": dynamic_rate,
                "instance": index,
                "cost": float(instance_cost),
                "qos": float(instance_qos),
            }
            for index, (instance_cost, instance_qos) in enumerate(zip(cost, qos))
        )
        print(
            f"EVAL m={vehicle_count} rate={dynamic_rate:.2f} "
            f"cost={cost.mean():.4f}+/-{cost.std(ddof=1):.4f} "
            f"qos={100.0*qos.mean():.2f}% time={elapsed:.3f}s",
            flush=True,
        )
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "selector_parameters": sum(
            parameter.numel() for parameter in model.selector.parameters()
        ),
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(raw_rows), metadata


def write_outputs(
    summary: pd.DataFrame,
    raw: pd.DataFrame,
    metadata: list[dict],
    args: argparse.Namespace,
) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summary.sort_values(["vehicle_count", "dynamic_rate"])
    summary.to_csv(output_dir / "fleet_size_summary.csv", index=False)
    raw.to_csv(output_dir / "fleet_size_raw_instances.csv", index=False)

    config = {
        "environment": ENVIRONMENT_TAG,
        "customer_count": 20,
        "fleet_sizes": [4, *args.fleet_sizes],
        "dynamic_rates": list(DYNAMIC_RATES),
        "interval_count": 10,
        "instances_per_rate": args.test_size,
        "paired_dynamic_rates": True,
        "m4_source": "manuscript Table I, inserted verbatim and not rerun",
        "cost_definition": (
            "cumulative distance of dispatched/executed edges; unstarted "
            "provisional route suffixes are refunded"
        ),
        "qos_definition": "served requests / 20",
        "training": {
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "validation_size": args.validation_size,
            "validation_rollouts": args.validation_rollouts,
            "learning_rate": args.learning_rate,
            "train_seed": args.train_seed,
            "validation_seed": args.validation_seed,
            "test_seed": args.test_seed,
            "vehicle_policy": "sample_logp",
            "customer_training_decode": "sampling",
            "test_decode": "greedy customer, argmax vehicle",
        },
        "checkpoints": metadata,
        "device": str(args.device),
        "gpu": (
            torch.cuda.get_device_name(torch.device(args.device))
            if str(args.device).startswith("cuda") and torch.cuda.is_available()
            else None
        ),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    md = [
        "| m | 10% Cost | 10% QoS | 25% Cost | 25% QoS | 50% Cost | 50% QoS | 75% Cost | 75% QoS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for vehicle_count in [4, *args.fleet_sizes]:
        row = summary[summary.vehicle_count == vehicle_count].set_index("dynamic_rate")
        cells = [str(vehicle_count)]
        for rate in DYNAMIC_RATES:
            value = row.loc[rate]
            cells.extend(
                [
                    f"{value.cost_mean:.2f} +/- {value.cost_sd:.2f}",
                    f"{100.0 * value.qos_mean:.2f}%",
                ]
            )
        md.append("| " + " | ".join(cells) + " |")
    (output_dir / "TABLE_FLEET_SIZE.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Sensitivity of DVNDA to different fleet sizes on 20-customer instances. The $m=4$ row is taken from the original experiment; the other rows are newly trained and evaluated on 100 instances.}",
        r"\label{tab:fleet_size}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{c|cc|cc|cc|cc}",
        r"\toprule",
        r"\multirow{2}{*}{$m$} & \multicolumn{2}{c|}{10\%} & \multicolumn{2}{c|}{25\%} & \multicolumn{2}{c|}{50\%} & \multicolumn{2}{c}{75\%} \\",
        r"& Cost & QoS & Cost & QoS & Cost & QoS & Cost & QoS \\",
        r"\midrule",
    ]
    for vehicle_count in [4, *args.fleet_sizes]:
        row = summary[summary.vehicle_count == vehicle_count].set_index("dynamic_rate")
        cells = [str(vehicle_count)]
        for rate in DYNAMIC_RATES:
            value = row.loc[rate]
            cells.extend(
                [
                    f"{value.cost_mean:.2f} $\\pm$ {value.cost_sd:.2f}",
                    f"{100.0 * value.qos_mean:.2f}\\%",
                ]
            )
        tex.append(" & ".join(cells) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (output_dir / "TABLE_FLEET_SIZE.tex").write_text("\n".join(tex), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleet-sizes", nargs="+", type=int, default=[5, 6, 7])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--validation-rollouts", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--baseline-update-interval", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--train-seed", type=int, default=1234)
    parser.add_argument("--validation-seed", type=int, default=4321)
    parser.add_argument("--test-seed", type=int, default=20260823)
    parser.add_argument("--decode-seed", type=int, default=314159)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("experiments/checkpoints/fleet_size_sensitivity_n20"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/fleet_size_sensitivity_n20"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)

    summary_frames = []
    raw_frames = []
    metadata = []
    for rate, values in PAPER_M4.items():
        summary_frames.append(
            pd.DataFrame(
                [
                    {
                        "vehicle_count": 4,
                        "dynamic_rate": rate,
                        "n_instances": 100,
                        "cost_mean": values["cost_mean"],
                        "cost_sd": values["cost_sd"],
                        "qos_mean": values["qos_mean"],
                        "qos_sd": 0.0,
                        "solve_time_100_s": 1.0,
                        "checkpoint_epoch": np.nan,
                    }
                ]
            )
        )

    for vehicle_count in args.fleet_sizes:
        checkpoint_dir = args.checkpoint_root / f"m{vehicle_count}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / "best.pt"
        train_args = training_namespace(args, checkpoint)
        if args.force_train or not checkpoint.exists():
            print(f"TRAIN m={vehicle_count} n=20", flush=True)
            with paper_runner_for_fleet(vehicle_count):
                paper_runner.train_model(train_args, device, checkpoint)
            config_path = checkpoint_dir / "training_config.json"
            if config_path.exists():
                saved_config = json.loads(config_path.read_text(encoding="utf-8"))
                saved_config["customer_count"] = 20
                saved_config["vehicle_count"] = vehicle_count
                saved_config["model"]["vehicles"] = vehicle_count
                config_path.write_text(
                    json.dumps(saved_config, indent=2), encoding="utf-8"
                )
        else:
            print(f"REUSE m={vehicle_count} checkpoint={checkpoint}", flush=True)
        frame, raw, meta = evaluate(vehicle_count, checkpoint, args, device)
        summary_frames.append(frame)
        raw_frames.append(raw)
        meta["vehicle_count"] = vehicle_count
        metadata.append(meta)

    summary = pd.concat(summary_frames, ignore_index=True)
    raw = pd.concat(raw_frames, ignore_index=True)
    write_outputs(summary, raw, metadata, args)
    print(summary.sort_values(["vehicle_count", "dynamic_rate"]).to_string(index=False))


if __name__ == "__main__":
    main()
