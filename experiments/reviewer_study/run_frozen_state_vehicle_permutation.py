"""Vehicle-module permutation test on frozen, state-diverged DCVRP snapshots.

The earlier episode-level relabeling test restarted every rollout from four
identical depot states.  In a homogeneous fleet that protocol can produce an
exact vehicle-label relabeling and therefore identical aggregate route cost,
even when the independent modules implement different scoring functions.

This script runs the identity mapping to selected interval boundaries, freezes
the complete physical environment state, and then changes only the mapping
from the four independent selector subnetworks to the four fixed vehicle-state
slots.  It reports both immediate score/vehicle-choice sensitivity and the
route Cost/QoS obtained by continuing every mapping from the same snapshot.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .executed_path_dvnda import VectorizedExecutedPathDVNDAEnvironment
from .paper_dcvrp import generate_paired_paper_datasets
from .run_original_uncertainty_n20 import load_model, set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DYNAMIC_RATES = (0.10, 0.25, 0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/checkpoints/executed_path_dvnda_n20_m4/best.pt"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/results/frozen_state_vehicle_permutation"),
    )
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--snapshot-intervals", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def effective_vehicle_done(environment) -> torch.Tensor:
    """Reproduce the feasibility mask used by PaperDCVRPEnvironment."""

    feasible_customer = (~environment.mask[:, :, 1:]).any(dim=2)
    feasible_vehicle = feasible_customer & (~environment.veh_done)
    row_has_feasible_vehicle = feasible_vehicle.any(dim=1)
    effective_done = torch.where(
        row_has_feasible_vehicle[:, None],
        environment.veh_done | (~feasible_vehicle),
        environment.veh_done,
    )
    safe_done = effective_done.clone()
    all_done = safe_done.all(dim=1)
    safe_done[all_done, 0] = False
    return safe_done


def encode_current_customers(model, environment) -> None:
    model._encode_customers(environment.nodes, environment.cust_mask)
    environment.new_customers = False


@torch.no_grad()
def take_greedy_customer_step(model, environment) -> None:
    if environment.new_customers:
        encode_current_customers(model, environment)
    customer_log_probability = model._customer_log_probability(environment)
    customer_index = customer_log_probability.argmax(dim=1, keepdim=True)
    environment.step(customer_index)


@torch.no_grad()
def run_to_interval(model, data, device: torch.device, target_interval: int):
    environment = VectorizedExecutedPathDVNDAEnvironment(
        data,
        nodes=data.nodes.to(device),
        pending_cost=0.0,
    )
    environment.selector = model.selector
    environment.policy_greedy = True
    environment.reset()
    encode_current_customers(model, environment)
    while not environment.done and environment.current_segment < target_interval:
        take_greedy_customer_step(model, environment)
    if environment.current_segment != target_interval:
        raise RuntimeError(
            f"failed to reach interval {target_interval}; "
            f"stopped at {environment.current_segment}, done={environment.done}"
        )
    # Do not duplicate the model inside every environment snapshot.
    environment.selector = None
    return copy.deepcopy(environment)


def state_diverged(vehicles: torch.Tensor, tolerance: float = 1e-7) -> torch.Tensor:
    spread = vehicles.amax(dim=1) - vehicles.amin(dim=1)
    return spread.abs().amax(dim=1) > tolerance


@torch.no_grad()
def selector_scores(model, environment, original_networks, permutation):
    model.selector.vehicle_networks = nn.ModuleList(
        [original_networks[index] for index in permutation]
    )
    done = effective_vehicle_done(environment)
    logits = model.selector.scores(
        environment.vehicles,
        environment.nodes,
        done,
        environment.mask,
    ).masked_fill(done, -torch.inf)
    return logits, logits.argmax(dim=1)


@torch.no_grad()
def continue_from_snapshot(model, snapshot, original_networks, permutation):
    environment = copy.deepcopy(snapshot)
    model.selector.vehicle_networks = nn.ModuleList(
        [original_networks[index] for index in permutation]
    )
    environment.selector = model.selector
    environment.policy_greedy = True
    environment._update_cur_veh()
    encode_current_customers(model, environment)
    while not environment.done:
        take_greedy_customer_step(model, environment)
    return (
        environment.route_distance().detach().cpu().numpy(),
        (100.0 * environment.qos()).detach().cpu().numpy(),
    )


def snapshot_diagnostics(
    model,
    snapshot,
    original_networks,
    permutations,
    dynamic_rate: float,
    interval: int,
) -> list[dict]:
    identity = tuple(range(len(original_networks)))
    identity_logits, identity_choice = selector_scores(
        model, snapshot, original_networks, identity
    )
    diverged = state_diverged(snapshot.vehicles)
    feasible_count = (~effective_vehicle_done(snapshot)).sum(dim=1)
    eligible = diverged & (feasible_count >= 2)
    state_spread = (
        snapshot.vehicles.amax(dim=1) - snapshot.vehicles.amin(dim=1)
    ).abs().amax(dim=1)
    rows = []
    for permutation_index, permutation in enumerate(permutations):
        logits, choice = selector_scores(model, snapshot, original_networks, permutation)
        finite = torch.isfinite(identity_logits) & torch.isfinite(logits)
        delta = torch.where(finite, logits - identity_logits, torch.zeros_like(logits))
        finite_count = finite.sum(dim=1).clamp_min(1)
        rms_delta = torch.sqrt(delta.square().sum(dim=1) / finite_count)
        max_delta = delta.abs().amax(dim=1)
        for instance in range(snapshot.minibatch_size):
            rows.append(
                {
                    "dynamic_rate": dynamic_rate,
                    "snapshot_interval": interval,
                    "permutation_index": permutation_index,
                    "permutation": "-".join(map(str, permutation)),
                    "identity": permutation == identity,
                    "instance": instance,
                    "state_diverged": bool(diverged[instance].item()),
                    "feasible_vehicle_count": int(feasible_count[instance].item()),
                    "eligible": bool(eligible[instance].item()),
                    "state_spread": float(state_spread[instance].item()),
                    "identity_vehicle": int(identity_choice[instance].item()),
                    "permuted_vehicle": int(choice[instance].item()),
                    "vehicle_choice_changed": bool(
                        choice[instance].item() != identity_choice[instance].item()
                    ),
                    "score_rms_difference": float(rms_delta[instance].item()),
                    "max_abs_score_difference": float(max_delta[instance].item()),
                }
            )
    model.selector.vehicle_networks = nn.ModuleList(original_networks)
    return rows


def continuation_results(
    model,
    snapshot,
    original_networks,
    permutations,
    dynamic_rate: float,
    interval: int,
) -> list[dict]:
    identity = tuple(range(len(original_networks)))
    rows = []
    for permutation_index, permutation in enumerate(permutations):
        distance, qos = continue_from_snapshot(
            model, snapshot, original_networks, permutation
        )
        for instance, (instance_distance, instance_qos) in enumerate(
            zip(distance, qos)
        ):
            rows.append(
                {
                    "dynamic_rate": dynamic_rate,
                    "snapshot_interval": interval,
                    "permutation_index": permutation_index,
                    "permutation": "-".join(map(str, permutation)),
                    "identity": permutation == identity,
                    "instance": instance,
                    "cost": float(instance_distance),
                    "qos_percent": float(instance_qos),
                }
            )
    model.selector.vehicle_networks = nn.ModuleList(original_networks)
    return rows


def summarize(score_raw: pd.DataFrame, continuation_raw: pd.DataFrame) -> pd.DataFrame:
    identity = continuation_raw[continuation_raw.identity].set_index(
        ["dynamic_rate", "snapshot_interval", "instance"]
    )
    compared = continuation_raw[~continuation_raw.identity].merge(
        identity[["cost", "qos_percent"]],
        left_on=["dynamic_rate", "snapshot_interval", "instance"],
        right_index=True,
        suffixes=("_permuted", "_identity"),
    )
    compared["cost_difference"] = compared.cost_permuted - compared.cost_identity
    compared["qos_difference"] = (
        compared.qos_percent_permuted - compared.qos_percent_identity
    )

    rows = []
    for dynamic_rate in DYNAMIC_RATES:
        scores = score_raw[
            (score_raw.dynamic_rate == dynamic_rate)
            & (~score_raw.identity)
            & (score_raw.eligible)
        ]
        continuation = compared[compared.dynamic_rate == dynamic_rate]
        identity_rate = identity.loc[dynamic_rate]
        rows.append(
            {
                "dynamic_rate": dynamic_rate,
                "instances": int(identity_rate.index.get_level_values("instance").nunique()),
                "snapshot_intervals": ",".join(
                    map(
                        str,
                        sorted(
                            identity_rate.index.get_level_values(
                                "snapshot_interval"
                            ).unique()
                        ),
                    )
                ),
                "permutations": int(
                    continuation_raw[continuation_raw.dynamic_rate == dynamic_rate][
                        "permutation_index"
                    ].nunique()
                ),
                "eligible_frozen_state_comparisons": int(len(scores)),
                "vehicle_choice_change_percent": 100.0
                * scores.vehicle_choice_changed.mean(),
                "mean_score_rms_difference": scores.score_rms_difference.mean(),
                "max_abs_score_difference": scores.max_abs_score_difference.max(),
                "mean_cost_difference": continuation.cost_difference.mean(),
                "mean_abs_cost_difference": continuation.cost_difference.abs().mean(),
                "max_abs_cost_difference": continuation.cost_difference.abs().max(),
                "mean_qos_difference_pp": continuation.qos_difference.mean(),
                "mean_abs_qos_difference_pp": continuation.qos_difference.abs().mean(),
                "max_abs_qos_difference_pp": continuation.qos_difference.abs().max(),
            }
        )
    return pd.DataFrame(rows), compared


def write_markdown(summary: pd.DataFrame, output_root: Path) -> None:
    lines = [
        "# Frozen-state vehicle-module permutation test",
        "",
        "The identity rollout is stopped at intervals 3, 5, and 7. The complete physical environment state is cloned, and only the mapping between the four independent selector subnetworks and the four fixed vehicle-state slots is changed. All 24 mappings are evaluated from each identical snapshot.",
        "",
        "| Dynamic rate | Instances | Frozen intervals | Mappings | Vehicle-choice change (%) | Mean score RMS $\\Delta$ | Mean $\\Delta$Cost | Mean $|\\Delta$Cost| | Max $|\\Delta$Cost| | Mean $\\Delta$QoS (pp) | Max $|\\Delta$QoS|$ (pp) |",
        "| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {100 * row.dynamic_rate:.0f}% | {row.instances} | "
            f"{row.snapshot_intervals} | {row.permutations} | "
            f"{row.vehicle_choice_change_percent:.2f} | "
            f"{row.mean_score_rms_difference:.4f} | "
            f"{row.mean_cost_difference:+.4f} | "
            f"{row.mean_abs_cost_difference:.4f} | "
            f"{row.max_abs_cost_difference:.4f} | "
            f"{row.mean_qos_difference_pp:+.4f} | "
            f"{row.max_abs_qos_difference_pp:.4f} |"
        )
    (output_root / "TABLE_FROZEN_STATE_PERMUTATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if any(interval <= 0 or interval >= 10 for interval in args.snapshot_intervals):
        raise ValueError("snapshot intervals must be between 1 and 9")
    device = torch.device(args.device)
    checkpoint = resolve(args.checkpoint)
    output_root = resolve(args.output_root)
    raw_root = output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    model, checkpoint_data = load_model(checkpoint, device)
    if not hasattr(model.selector, "vehicle_networks"):
        raise TypeError("the checkpoint does not contain independent vehicle networks")
    model.eval()
    model.greedy = True
    model.vehicle_greedy = True
    original_networks = list(model.selector.vehicle_networks)
    permutations = list(itertools.permutations(range(len(original_networks))))
    if len(original_networks) != 4 or len(permutations) != math.factorial(4):
        raise ValueError("this reviewer experiment requires exactly four vehicles")

    set_seed(args.seed)
    datasets = generate_paired_paper_datasets(
        args.instances,
        DYNAMIC_RATES,
        customer_count=20,
        vehicle_count=4,
    )

    score_rows = []
    continuation_rows = []
    started = time.perf_counter()
    for dynamic_rate in DYNAMIC_RATES:
        data = datasets[dynamic_rate]
        for interval in sorted(set(args.snapshot_intervals)):
            snapshot = run_to_interval(model, data, device, interval)
            score_rows.extend(
                snapshot_diagnostics(
                    model,
                    snapshot,
                    original_networks,
                    permutations,
                    dynamic_rate,
                    interval,
                )
            )
            continuation_rows.extend(
                continuation_results(
                    model,
                    snapshot,
                    original_networks,
                    permutations,
                    dynamic_rate,
                    interval,
                )
            )
            print(
                f"rate={dynamic_rate:.2f}, interval={interval}: "
                f"evaluated {len(permutations)} frozen-state mappings",
                flush=True,
            )

    score_raw = pd.DataFrame(score_rows)
    continuation_raw = pd.DataFrame(continuation_rows)
    summary, paired = summarize(score_raw, continuation_raw)
    score_raw.to_csv(raw_root / "frozen_state_scores.csv", index=False)
    continuation_raw.to_csv(raw_root / "frozen_state_continuations.csv", index=False)
    paired.to_csv(raw_root / "paired_continuation_differences.csv", index=False)
    summary.to_csv(output_root / "frozen_state_permutation_summary.csv", index=False)
    write_markdown(summary, output_root)
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": checkpoint_data.get("epoch"),
        "checkpoint_train_vehicle_policy": checkpoint_data.get("config", {}).get(
            "train_vehicle_policy"
        ),
        "environment": "time_driven_executed_path_v4",
        "customer_count": 20,
        "vehicle_count": 4,
        "instances_per_dynamic_rate": args.instances,
        "dynamic_rates": DYNAMIC_RATES,
        "snapshot_intervals": sorted(set(args.snapshot_intervals)),
        "permutations": len(permutations),
        "mapping_intervention": "vehicle states fixed; only selector subnetwork-to-slot assignment permuted",
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    (output_root / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"wrote results to {output_root}", flush=True)


if __name__ == "__main__":
    main()
