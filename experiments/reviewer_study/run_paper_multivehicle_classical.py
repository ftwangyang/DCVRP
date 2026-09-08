"""Run classical baselines on the rewritten paper-faithful multi-vehicle fleet."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from .paper_multivehicle_env import (
    PaperInstance,
    PaperTimeDrivenFleet,
    paper_vehicle_count,
)
from .periodic_static_multivehicle_env import (
    ENVIRONMENT_TAG as PERIODIC_STATIC_ENVIRONMENT_TAG,
    PeriodicStaticFleet,
)
from .executed_path_multivehicle_env import (
    ENVIRONMENT_TAG as EXECUTED_PATH_ENVIRONMENT_TAG,
    ExecutedPathFleet,
)
from .word_periodic_multivehicle_env import (
    ENVIRONMENT_TAG as WORD_PERIODIC_ENVIRONMENT_TAG,
    WordPeriodicStaticFleet,
)
from .paper_multivehicle_solvers import make_planner


METHODS = (
    "Greedy (same instances)",
    "Regret insertion",
    "Tabu Search",
    "Adaptive LNS",
    "OR-Tools",
)
SIZES = (20, 35, 50)
RATES = (0.10, 0.25, 0.50, 0.75)

PAPER_GREEDY = {
    (0.10, 20): (9.07, 1.12, 99.90),
    (0.10, 35): (15.95, 1.44, 99.95),
    (0.10, 50): (21.41, 2.11, 99.95),
    (0.25, 20): (9.69, 1.25, 99.90),
    (0.25, 35): (16.96, 1.95, 99.95),
    (0.25, 50): (22.84, 2.10, 99.85),
    (0.50, 20): (11.25, 1.43, 99.90),
    (0.50, 35): (19.63, 2.01, 99.95),
    (0.50, 50): (26.71, 2.48, 99.95),
    (0.75, 20): (12.43, 1.51, 99.45),
    (0.75, 35): (21.55, 2.21, 99.95),
    (0.75, 50): (30.19, 2.77, 99.85),
}

PAPER_DVNDA = {
    (0.10, 20): (8.31, 1.22, 100.0, 1.0),
    (0.10, 35): (14.94, 2.00, 100.0, 3.0),
    (0.10, 50): (18.89, 2.27, 100.0, 5.0),
    (0.25, 20): (8.95, 1.30, 100.0, 1.0),
    (0.25, 35): (16.14, 2.14, 100.0, 3.0),
    (0.25, 50): (21.21, 2.66, 100.0, 5.0),
    (0.50, 20): (10.47, 1.53, 100.0, 1.0),
    (0.50, 35): (18.90, 2.24, 100.0, 3.0),
    (0.50, 50): (25.31, 2.81, 100.0, 5.0),
    (0.75, 20): (11.78, 1.44, 100.0, 1.0),
    (0.75, 35): (20.98, 2.26, 100.0, 3.0),
    (0.75, 50): (29.00, 2.69, 100.0, 5.0),
}


def load_group(
    manifest_dir: Path, customer_count: int, dynamic_rate: float
) -> list[PaperInstance]:
    path = manifest_dir / (
        f"release_n{customer_count}_r{int(round(100 * dynamic_rate)):02d}.npz"
    )
    with np.load(path) as data:
        coordinates = data["coordinates"]
        demands = data["demands"]
        service = data["service_minutes"]
        disclosure = data["disclosure_minutes"]
    return [
        PaperInstance(
            coordinates=coordinates[index],
            demands=demands[index],
            service_minutes=service[index],
            disclosure_minutes=disclosure[index],
        )
        for index in range(coordinates.shape[0])
    ]


def _worker(payload: tuple) -> dict:
    base_payload = payload[:10]
    (
        instance,
        method,
        customer_count,
        dynamic_rate,
        instance_id,
        ortools_time_ms,
        alns_time_ms,
        algorithm_seed,
        environment_mode,
        assignment_policy,
    ) = base_payload
    travel_minutes_per_normalized_unit = (
        float(payload[10]) if len(payload) > 10 else None
    )
    started = time.perf_counter()
    planning_call = 0
    planning_time_s = 0.0
    initial_planning_time_s = 0.0
    online_replanning_time_s = 0.0
    active_online_replanning_time_s = 0.0
    online_replanning_calls = 0
    active_online_replanning_calls = 0

    def planner(state):
        nonlocal planning_call, planning_time_s
        nonlocal initial_planning_time_s, online_replanning_time_s
        nonlocal active_online_replanning_time_s
        nonlocal online_replanning_calls, active_online_replanning_calls
        planning_call += 1
        call_started = time.perf_counter()
        selected = make_planner(
            method,
            ortools_time_ms=ortools_time_ms,
            alns_time_ms=alns_time_ms,
            seed=(
                algorithm_seed
                + customer_count * 1_000_000
                + int(round(100 * dynamic_rate)) * 10_000
                + instance_id * 100
                + planning_call
            ),
            assignment_policy=assignment_policy,
        )
        plan = selected(state)
        elapsed = time.perf_counter() - call_started
        planning_time_s += elapsed
        if state.current_minute <= 1.0e-12:
            initial_planning_time_s += elapsed
        else:
            online_replanning_time_s += elapsed
            online_replanning_calls += 1
            if state.customer_ids:
                active_online_replanning_time_s += elapsed
                active_online_replanning_calls += 1
        return plan

    if environment_mode == "continuous-single-trip":
        fleet = PaperTimeDrivenFleet(paper_vehicle_count(customer_count))
        environment_tag = "paper_multivehicle_v2"
    elif environment_mode == "periodic-static-return":
        fleet = PeriodicStaticFleet(paper_vehicle_count(customer_count))
        environment_tag = PERIODIC_STATIC_ENVIRONMENT_TAG
    elif environment_mode == "executed-path-paper":
        fleet_kwargs = {}
        if travel_minutes_per_normalized_unit is not None:
            fleet_kwargs["travel_minutes_per_normalized_unit"] = (
                travel_minutes_per_normalized_unit
            )
        fleet = ExecutedPathFleet(
            paper_vehicle_count(customer_count), **fleet_kwargs
        )
        environment_tag = EXECUTED_PATH_ENVIRONMENT_TAG
    elif environment_mode == "word-periodic-static":
        fleet = WordPeriodicStaticFleet(paper_vehicle_count(customer_count))
        environment_tag = WORD_PERIODIC_ENVIRONMENT_TAG
    else:
        raise ValueError(f"unknown environment mode: {environment_mode}")
    result = fleet.run(instance, planner)
    vehicles_used = sum(value > 0 for value in result.vehicle_customer_counts)
    if result.served_customers and vehicles_used == 0:
        raise RuntimeError("served customers without an active vehicle")
    return {
        "method": method,
        "customer_count": customer_count,
        "vehicle_count": paper_vehicle_count(customer_count),
        "dynamic_rate": dynamic_rate,
        "instance_id": instance_id,
        "cost": result.cost,
        "qos_percent": result.qos_percent,
        "served_customers": result.served_customers,
        "unserved_customers": result.unserved_customers,
        "solve_time_s": time.perf_counter() - started,
        "planning_time_s": planning_time_s,
        "initial_planning_time_s": initial_planning_time_s,
        "online_replanning_time_s": online_replanning_time_s,
        "active_online_replanning_time_s": active_online_replanning_time_s,
        "online_replanning_calls": online_replanning_calls,
        "active_online_replanning_calls": active_online_replanning_calls,
        "planning_calls": result.planning_calls,
        "final_completion_minute": result.final_completion_minute,
        "vehicles_used": vehicles_used,
        "vehicle_customer_counts": ";".join(
            str(value) for value in result.vehicle_customer_counts
        ),
        "environment": environment_tag,
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(
            ["method", "customer_count", "vehicle_count", "dynamic_rate"],
            as_index=False,
        )
        .agg(
            instances=("instance_id", "count"),
            cost_mean=("cost", "mean"),
            cost_sd=("cost", "std"),
            qos_mean_percent=("qos_percent", "mean"),
            qos_sd_percent=("qos_percent", "std"),
            time_100_s=("solve_time_s", "sum"),
            time_per_instance_mean_s=("solve_time_s", "mean"),
            planning_time_100_s=("planning_time_s", "sum"),
            initial_planning_time_100_s=("initial_planning_time_s", "sum"),
            online_replanning_time_100_s=("online_replanning_time_s", "sum"),
            active_online_replanning_time_100_s=(
                "active_online_replanning_time_s", "sum"
            ),
            online_replanning_calls_100=("online_replanning_calls", "sum"),
            active_online_replanning_calls_100=(
                "active_online_replanning_calls", "sum"
            ),
            unserved_total=("unserved_customers", "sum"),
            vehicles_used_mean=("vehicles_used", "mean"),
            final_completion_mean_min=("final_completion_minute", "mean"),
        )
        .sort_values(["dynamic_rate", "method", "customer_count"])
        .reset_index(drop=True)
    )


def paper_rows() -> pd.DataFrame:
    rows = []
    for rate in RATES:
        for size in SIZES:
            for method, source in (
                ("Greedy (paper)", PAPER_GREEDY),
                ("DVNDA (paper)", PAPER_DVNDA),
            ):
                values = source[(rate, size)]
                rows.append(
                    {
                        "method": method,
                        "customer_count": size,
                        "vehicle_count": paper_vehicle_count(size),
                        "dynamic_rate": rate,
                        "instances": 100,
                        "cost_mean": values[0],
                        "cost_sd": values[1],
                        "qos_mean_percent": values[2],
                        "qos_sd_percent": np.nan,
                        "time_100_s": np.nan if len(values) == 3 else values[3],
                        "time_per_instance_mean_s": np.nan,
                        "unserved_total": np.nan,
                        "vehicles_used_mean": np.nan,
                        "final_completion_mean_min": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def write_table(
    combined: pd.DataFrame, path: Path, environment_mode: str
) -> None:
    order = (
        "Greedy (paper)",
        "Greedy (same instances)",
        "Regret insertion",
        "Tabu Search",
        "Adaptive LNS",
        "OR-Tools",
    )
    lines = [
        "# Time-driven periodic-static classical baselines",
        "",
        (
            "All values are measured on 100 instances. Initial is the t=0 "
            "static-planning time; Online is the cumulative time of the ten "
            "later boundary/terminal planning calls; Total is end-to-end "
            "elapsed time. Times are sums over 100 instances, not shortened "
            "parallel wall time."
        ),
        "",
    ]
    comparison = combined[combined.method.isin(order)].copy()

    def mean_sd(mean: float, sd: float) -> str:
        if pd.isna(sd):
            return f"{mean:.2f} (SD not reported)"
        return f"{mean:.2f} ± {sd:.2f}"

    def seconds(value: float, digits: int) -> str:
        return "not reported" if pd.isna(value) else f"{value:.{digits}f}"

    for size in SIZES:
        vehicle_count = paper_vehicle_count(size)
        lines.extend(
            [
                f"## n={size}, m={vehicle_count}",
                "",
                "| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Initial(100) s | Online(100) s | Total(100) s | Active online calls |",
                "|---:|:---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        selected = comparison[comparison.customer_count == size].set_index(
            ["dynamic_rate", "method"]
        )
        for rate in RATES:
            for method in order:
                key = (rate, method)
                if key not in selected.index:
                    continue
                row = selected.loc[key]
                lines.append(
                    f"| {100 * rate:.0f}% | {method} | "
                    f"{mean_sd(row.cost_mean, row.cost_sd)} | "
                    f"{mean_sd(row.qos_mean_percent, row.qos_sd_percent)} | "
                    f"{seconds(row.initial_planning_time_100_s, 2)} | "
                    f"{seconds(row.online_replanning_time_100_s, 2)} | "
                    f"{seconds(row.time_100_s, 2)} | "
                    f"{seconds(row.active_online_replanning_calls_100, 0)} |"
                )
        lines.append("")
    if environment_mode == "word-periodic-static":
        environment_note = (
            "Environment: Word-spec 480-minute horizon with 10 synchronized "
            "intervals; complete static multi-vehicle solve at each boundary; "
            "commit customer legs with start < next boundary; destroy unstarted "
            "suffixes without charging them; at most one physical depot return "
            "per active vehicle and interval; capacity is restored only after "
            "that return; complete terminal solve at T=480."
        )
    else:
        environment_note = (
            "Environment: 10 synchronized intervals; complete static "
            "multi-vehicle solve at each boundary; commit only customer legs "
            "with start < next boundary; destroy unstarted suffixes without "
            "charging them; terminal solve at T=480."
        )
    lines.append(environment_note)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--rates", nargs="+", type=float, default=list(RATES))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--ortools-time-ms", type=int, default=1_000)
    parser.add_argument("--alns-time-ms", type=int, default=1_000)
    parser.add_argument("--algorithm-seed", type=int, default=314159)
    parser.add_argument(
        "--assignment-policy",
        choices=("free", "fixed-greedy"),
        default="free",
        help=(
            "Use fixed-greedy to preserve the common vehicle-customer "
            "allocation and compare only within-vehicle route optimization."
        ),
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help=(
            "Optional deterministic prefix used only for smoke/pilot runs; "
            "the confirmatory experiment leaves this unset and uses all 100."
        ),
    )
    parser.add_argument(
        "--environment-mode",
        choices=(
            "continuous-single-trip",
            "periodic-static-return",
            "executed-path-paper",
            "word-periodic-static",
        ),
        default="executed-path-paper",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_dir = args.manifest_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_instances.csv"
    existing = pd.read_csv(raw_path) if args.resume and raw_path.exists() else pd.DataFrame()
    rows = [] if existing.empty else existing.to_dict("records")
    completed = (
        set()
        if existing.empty
        else set(
            zip(
                existing.method,
                existing.customer_count.astype(int),
                existing.dynamic_rate.round(6),
                existing.instance_id.astype(int),
            )
        )
    )
    experiment_started = time.perf_counter()

    for method in args.methods:
        for size in args.sizes:
            for rate in args.rates:
                instances = load_group(manifest_dir, size, rate)
                if args.max_instances is not None:
                    if args.max_instances <= 0:
                        raise ValueError("--max-instances must be positive")
                    instances = instances[: args.max_instances]
                payloads = []
                for index, instance in enumerate(instances, 1):
                    key = (method, size, round(rate, 6), index)
                    if key in completed:
                        continue
                    payloads.append(
                        (
                            instance,
                            method,
                            size,
                            rate,
                            index,
                            args.ortools_time_ms,
                            args.alns_time_ms,
                            args.algorithm_seed,
                            args.environment_mode,
                            args.assignment_policy,
                        )
                    )
                if not payloads:
                    print(f"skip {method} n={size} rate={rate}", flush=True)
                    continue
                group_started = time.perf_counter()
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = [executor.submit(_worker, payload) for payload in payloads]
                    for count, future in enumerate(as_completed(futures), 1):
                        result = future.result()
                        rows.append(result)
                        completed.add(
                            (
                                result["method"],
                                int(result["customer_count"]),
                                round(float(result["dynamic_rate"]), 6),
                                int(result["instance_id"]),
                            )
                        )
                        if count % 10 == 0 or count == len(payloads):
                            group = [
                                row
                                for row in rows
                                if row["method"] == method
                                and int(row["customer_count"]) == size
                                and math.isclose(float(row["dynamic_rate"]), rate)
                            ]
                            print(
                                f"{method:<16} n={size} rate={100*rate:.0f}% "
                                f"done={len(group):03d}/100 cost={np.mean([x['cost'] for x in group]):.3f} "
                                f"qos={np.mean([x['qos_percent'] for x in group]):.2f}% "
                                f"vehicles={np.mean([x['vehicles_used'] for x in group]):.2f} "
                                f"wall={time.perf_counter()-group_started:.1f}s",
                                flush=True,
                            )
                            pd.DataFrame(rows).sort_values(
                                ["method", "customer_count", "dynamic_rate", "instance_id"]
                            ).to_csv(raw_path, index=False)

    raw = pd.DataFrame(rows).sort_values(
        ["method", "customer_count", "dynamic_rate", "instance_id"]
    )
    summary = summarize(raw)
    combined = pd.concat([summary, paper_rows()], ignore_index=True, sort=False)
    raw.to_csv(raw_path, index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    combined.to_csv(output_dir / "with_paper.csv", index=False)
    write_table(combined, output_dir / "TABLE.md", args.environment_mode)
    environment_tag = (
        "paper_multivehicle_v2"
        if args.environment_mode == "continuous-single-trip"
        else (
            PERIODIC_STATIC_ENVIRONMENT_TAG
            if args.environment_mode == "periodic-static-return"
            else (
                EXECUTED_PATH_ENVIRONMENT_TAG
                if args.environment_mode == "executed-path-paper"
                else WORD_PERIODIC_ENVIRONMENT_TAG
            )
        )
    )
    metadata = {
        "environment": environment_tag,
        "environment_mode": args.environment_mode,
        "source_generator": (
            "immutable NPZ manifests; generator metadata stored beside manifests"
            if args.environment_mode in (
                "executed-path-paper",
                "word-periodic-static",
            )
            else "released data.py semantics"
        ),
        "vehicle_state": "independent position, remaining capacity, available time per vehicle",
        "first_stage_fleet_use": (
            "min(m, visible requests) nonempty routes at every static solve"
            if args.environment_mode == "executed-path-paper"
            else "all m vehicles receive a nonempty route when at least m requests are visible"
            if args.environment_mode == "word-periodic-static"
            else "solver-defined"
        ),
        "intervals": 10,
        "horizon_minutes": 480,
        "travel_minutes_per_normalized_unit": (
            100.0
            if args.environment_mode in (
                "continuous-single-trip",
                "executed-path-paper",
            )
            else 1.0
        ),
        "m_equals_n_over_5": True,
        "terminal_dispatch": (
            "boundary-only: customer leg start <= 480"
            if args.environment_mode == "continuous-single-trip"
            else "complete static closure at 480 after ten regular intervals"
        ),
        "depot_return": (
            "commit a completed-route depot leg when it starts before the "
            "boundary; restore capacity upon physical return; final closure "
            "for vehicles not already at the depot"
            if args.environment_mode == "executed-path-paper"
            else "once after final dispatch"
            if args.environment_mode == "continuous-single-trip"
            else "at most once per active interval; capacity restored only at physical return"
        ),
        "cost_definition": (
            "sum of committed customer edges with start < boundary plus one "
            "physical depot return for each active vehicle/interval; unstarted "
            "planned suffixes are not charged"
            if args.environment_mode == "word-periodic-static"
            else "sum of committed customer and depot edges with start < "
            "boundary plus terminal physical closure; unstarted planned "
            "suffixes are not charged"
        ),
        "methods": list(args.methods),
        "assignment_policy": args.assignment_policy,
        "short_route_fallback": None,
        "tabu_implementation": (
            "standalone deterministic Python tabu search with swap, relocate, "
            "and 2-opt neighbourhoods; it does not call OR-Tools"
        ),
        "tabu_time_ms_per_update": args.ortools_time_ms,
        "ortools_metaheuristic": "GUIDED_LOCAL_SEARCH",
        "timing_definition": {
            "initial_planning_time": "planner call at t=0",
            "online_replanning_time": (
                "sum of planner calls at t>0, including the terminal closure"
            ),
            "total_elapsed_time": (
                "end-to-end environment plus all planning calls per instance"
            ),
            "time_100_aggregation": (
                "sum of the 100 per-instance elapsed times, independent of "
                "parallel experiment wall time"
            ),
        },
        "instances_per_group": (
            100 if args.max_instances is None else args.max_instances
        ),
        "ortools_time_ms_per_update": args.ortools_time_ms,
        "alns_time_ms_per_nonempty_update": args.alns_time_ms,
        "workers": args.workers,
        "cpu": platform.processor(),
        "python_version": platform.python_version(),
        "alns_version": package_version("alns"),
        "ortools_version": package_version("ortools"),
        "logical_cpus": os.cpu_count(),
        "experiment_wall_time_s": time.perf_counter() - experiment_started,
    }
    (output_dir / "config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
