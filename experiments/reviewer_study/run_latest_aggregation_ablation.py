"""Evaluate Argmax, Softmax, and Random vehicle aggregation fairly.

The three rules share one independently parameterized vehicle scorer and one
customer decoder.  The scorer is trained with stochastic vehicle sampling so
that REINFORCE can propagate gradients through the vehicle decision.  At test
time only the rule that converts the vehicle scores to a vehicle index changes:

* Argmax: select the feasible vehicle with the largest learned score.
* Softmax: sample a feasible vehicle from softmax(score), temperature 1.
* Random: sample uniformly from the feasible vehicles.

This common-checkpoint design isolates aggregation from training seed and model
quality.  Evaluation uses the latest normalized, time-driven paper environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import DCVRP_Dataset  # noqa: E402
from experiments.reviewer_study.paper_dcvrp import (  # noqa: E402
    PAPER_INTERVAL_COUNT,
    VectorizedPaperDCVRPEnvironment,
    generate_paired_paper_datasets,
)
from experiments.reviewer_study.run_original_uncertainty_n20 import (  # noqa: E402
    DYNAMIC_RATES,
    build_model,
)
from experiments.reviewer_study.selectors import RandomSelector  # noqa: E402


METHODS = ("Argmax", "Softmax", "Random")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trained_model(checkpoint: Path, device: torch.device):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(device)
    model.load_state_dict(saved["model"])
    model.eval()
    model.greedy = True
    model.vehicle_greedy = True
    model.include_vehicle_log_probability = False
    return model, saved


def configure_method(model, method: str, trained_selector) -> None:
    """Change only the score-to-vehicle decision used at inference."""

    if method == "Argmax":
        model.selector = trained_selector
        model.selector.evaluation_rule = "argmax"
    elif method == "Softmax":
        model.selector = trained_selector
        model.selector.evaluation_rule = "softmax"
    elif method == "Random":
        model.selector = RandomSelector(
            trained_selector.vehicle_count,
            evaluation_rule="softmax",
        ).to(next(model.parameters()).device)
    else:
        raise ValueError(f"unknown aggregation method: {method}")


@torch.inference_mode()
def evaluate_once(
    model,
    data: DCVRP_Dataset,
    method: str,
    trained_selector,
    device: torch.device,
    decode_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    configure_method(model, method, trained_selector)
    set_seed(decode_seed)
    environment = VectorizedPaperDCVRPEnvironment(
        data,
        nodes=data.nodes.to(device),
        pending_cost=5.0,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    _, _, reward_steps = model(environment)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    distance = environment.route_distance().detach().cpu().numpy()
    qos_percent = (100.0 * environment.qos()).detach().cpu().numpy()
    penalized_cost = (
        -torch.stack(reward_steps).sum(dim=0).squeeze(-1)
    ).detach().cpu().numpy()
    return distance, qos_percent, penalized_cost, elapsed


def mean_sd(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def format_mean_sd(mean: float, sd: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def write_markdown_table(summary: pd.DataFrame, output: Path) -> None:
    lines = [
        "# Latest time-driven aggregation ablation (n=20, m=4)",
        "",
        (
            "Each dynamic-rate cell contains the same 100 paired test instances. "
            "Cost and QoS are mean ± sample SD across instances after averaging "
            "independent policy draws; solve time is mean ± SD for one GPU batch of 100."
        ),
        "",
        "| Dynamic rate | Aggregation | Cost ↓ | QoS (%) ↑ | Solve time / 100 (s) ↓ |",
        "|---:|:---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {rate:.0f}% | {method} | {cost} | {qos} | {timing} |".format(
                rate=100.0 * row["dynamic_rate"],
                method=row["method"],
                cost=format_mean_sd(row["cost_mean"], row["cost_sd"]),
                qos=format_mean_sd(row["qos_mean_percent"], row["qos_sd_percent"]),
                timing=format_mean_sd(
                    row["solve_time_100_mean_s"],
                    row["solve_time_100_sd_s"],
                    digits=3,
                ),
            )
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(summary: pd.DataFrame, output: Path) -> None:
    rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Aggregation-rule ablation on the time-driven DCVRP environment ($n=20$, $m=4$, 100 paired instances per dynamic rate). Values are mean $\pm$ standard deviation.}",
        r"\label{tab:aggregation_ablation}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        "Dynamic rate & Rule & Cost $\\downarrow$ & QoS (\\%) $\\uparrow$ "
        "& Time/100 (s) $\\downarrow$ \\\\",
        r"\midrule",
    ]
    previous_rate = None
    for _, row in summary.iterrows():
        rate = int(round(100.0 * row["dynamic_rate"]))
        if previous_rate is not None and rate != previous_rate:
            rows.append(r"\midrule")
        rows.append(
            f"{rate}\% & {row['method']} & "
            f"{row['cost_mean']:.2f} $\\pm$ {row['cost_sd']:.2f} & "
            f"{row['qos_mean_percent']:.2f} $\\pm$ {row['qos_sd_percent']:.2f} & "
            f"{row['solve_time_100_mean_s']:.3f} $\\pm$ {row['solve_time_100_sd_s']:.3f} \\\\" 
        )
        previous_rate = rate
    rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def paired_comparisons(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rate in DYNAMIC_RATES:
        subset = raw[raw["dynamic_rate"] == rate]
        reference = subset[subset["method"] == "Argmax"].sort_values("instance_id")
        for method in ("Softmax", "Random"):
            candidate = subset[subset["method"] == method].sort_values("instance_id")
            cost_delta = candidate["cost"].to_numpy() - reference["cost"].to_numpy()
            qos_delta = (
                candidate["qos_percent"].to_numpy()
                - reference["qos_percent"].to_numpy()
            )
            test = ttest_rel(
                candidate["penalized_cost"].to_numpy(),
                reference["penalized_cost"].to_numpy(),
            )
            rows.append(
                {
                    "dynamic_rate": rate,
                    "comparison": f"{method} - Argmax",
                    "mean_cost_delta": float(cost_delta.mean()),
                    "mean_qos_delta_percent": float(qos_delta.mean()),
                    "mean_penalized_cost_delta": float(
                        candidate["penalized_cost"].mean()
                        - reference["penalized_cost"].mean()
                    ),
                    "paired_t_p_two_sided_penalized_cost": float(test.pvalue),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.test_size != 100:
        raise ValueError("This reviewer table is fixed to exactly 100 instances per rate")
    if args.policy_replications < 2:
        raise ValueError(
            "policy_replications must be at least 2 to estimate stochastic-policy means"
        )

    device = torch.device(args.device)
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, saved = load_trained_model(checkpoint, device)
    trained_selector = model.selector

    set_seed(args.test_seed)
    datasets = generate_paired_paper_datasets(
        args.test_size,
        DYNAMIC_RATES,
        customer_count=20,
        vehicle_count=4,
    )

    # Compile/cache warm-up is excluded from timing.
    evaluate_once(
        model,
        datasets[DYNAMIC_RATES[0]],
        "Argmax",
        trained_selector,
        device,
        args.decode_seed,
    )

    raw_rows: list[dict] = []
    summary_rows: list[dict] = []
    rollout_rows: list[dict] = []
    timing_rows: list[dict] = []
    for rate_index, dynamic_rate in enumerate(DYNAMIC_RATES):
        data = datasets[dynamic_rate]
        for method_index, method in enumerate(METHODS):
            base_decode_seed = (
                args.decode_seed
                + 100_000 * rate_index
                + 10_000 * method_index
            )
            distance_draws = []
            qos_draws = []
            penalized_draws = []
            timings = []
            for repetition in range(args.policy_replications):
                decode_seed = base_decode_seed + repetition
                current_distance, current_qos, current_penalized, elapsed = evaluate_once(
                    model,
                    data,
                    method,
                    trained_selector,
                    device,
                    decode_seed,
                )
                timings.append(elapsed)
                distance_draws.append(current_distance)
                qos_draws.append(current_qos)
                penalized_draws.append(current_penalized)
                timing_rows.append(
                    {
                        "dynamic_rate": dynamic_rate,
                        "method": method,
                        "repetition": repetition + 1,
                        "solve_time_100_s": elapsed,
                    }
                )
                for instance_id, (cost, quality, penalized_value) in enumerate(
                    zip(current_distance, current_qos, current_penalized), start=1
                ):
                    rollout_rows.append(
                        {
                            "dynamic_rate": dynamic_rate,
                            "method": method,
                            "policy_replication": repetition + 1,
                            "instance_id": instance_id,
                            "cost": float(cost),
                            "qos_percent": float(quality),
                            "penalized_cost": float(penalized_value),
                        }
                    )

            # Keep 100 independent experimental units.  For stochastic rules,
            # each instance value estimates its expectation over policy draws;
            # the reported SD remains the between-instance sample SD.
            distances = np.stack(distance_draws, axis=0).mean(axis=0)
            qos = np.stack(qos_draws, axis=0).mean(axis=0)
            penalized = np.stack(penalized_draws, axis=0).mean(axis=0)
            cost_mean, cost_sd = mean_sd(distances)
            qos_mean, qos_sd = mean_sd(qos)
            penalized_mean, penalized_sd = mean_sd(penalized)
            timing_mean, timing_sd = mean_sd(np.asarray(timings))
            summary_rows.append(
                {
                    "dynamic_rate": dynamic_rate,
                    "method": method,
                    "n_instances": args.test_size,
                    "cost_mean": cost_mean,
                    "cost_sd": cost_sd,
                    "qos_mean_percent": qos_mean,
                    "qos_sd_percent": qos_sd,
                    "penalized_cost_mean": penalized_mean,
                    "penalized_cost_sd": penalized_sd,
                    "solve_time_100_mean_s": timing_mean,
                    "solve_time_100_sd_s": timing_sd,
                    "solve_time_per_instance_ms": 1000.0 * timing_mean / args.test_size,
                }
            )
            for instance_id, (cost, quality, penalized_value) in enumerate(
                zip(distances, qos, penalized), start=1
            ):
                raw_rows.append(
                    {
                        "dynamic_rate": dynamic_rate,
                        "method": method,
                        "instance_id": instance_id,
                        "cost": float(cost),
                        "qos_percent": float(quality),
                        "penalized_cost": float(penalized_value),
                    }
                )
            print(
                f"rate={100 * dynamic_rate:.0f}% method={method:<7} "
                f"cost={cost_mean:.3f}±{cost_sd:.3f} "
                f"qos={qos_mean:.2f}±{qos_sd:.2f}% "
                f"time100={timing_mean:.3f}±{timing_sd:.3f}s",
                flush=True,
            )

    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summary_rows)
    summary["method"] = pd.Categorical(summary["method"], METHODS, ordered=True)
    summary = summary.sort_values(["dynamic_rate", "method"]).reset_index(drop=True)
    summary["method"] = summary["method"].astype(str)
    rollout_raw = pd.DataFrame(rollout_rows)
    timing = pd.DataFrame(timing_rows)
    comparisons = paired_comparisons(raw)

    raw.to_csv(output_dir / "raw_instance_results.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    rollout_raw.to_csv(
        output_dir / "raw_policy_rollouts.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    summary.to_csv(output_dir / "summary_by_dynamic_rate.csv", index=False)
    timing.to_csv(output_dir / "timing_repetitions.csv", index=False)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    write_markdown_table(summary, output_dir / "TABLE_AGGREGATION.md")
    write_latex_table(summary, output_dir / "TABLE_AGGREGATION.tex")

    config = {
        "environment": "VectorizedPaperDCVRPEnvironment",
        "customer_count": 20,
        "vehicle_count": 4,
        "dynamic_rates": list(DYNAMIC_RATES),
        "interval_count": PAPER_INTERVAL_COUNT,
        "test_instances_per_rate": args.test_size,
        "paired_across_dynamic_rates": True,
        "test_seed": args.test_seed,
        "decode_seed": args.decode_seed,
        "policy_replications_per_instance": args.policy_replications,
        "cost_definition": "executed Euclidean route distance in normalized coordinate units",
        "qos_definition": "100 * served requests / 20",
        "softmax_temperature": 1.0,
        "random_definition": "uniform sample over currently feasible vehicles",
        "customer_decoding": "greedy for all three rules",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": int(saved.get("epoch", -1)),
        "checkpoint_training_config": saved.get("config", {}),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote reviewer table and raw results to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/checkpoints/latest_aggregation_n20_m4/best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/latest_aggregation_n20_m4"),
    )
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=20260823)
    parser.add_argument("--decode-seed", type=int, default=314159)
    parser.add_argument("--policy-replications", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
