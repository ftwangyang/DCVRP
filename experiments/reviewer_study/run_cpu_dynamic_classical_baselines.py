"""CPU classical baselines in the manuscript's 10-interval DCVRP protocol.

The runner evaluates four planning rules on nested, paired synthetic instances:

* Regret insertion: deterministic regret-2 construction.
* Tabu Search: Google OR-Tools TABU_SEARCH.
* Adaptive LNS: the ``alns`` 7.0.0 package with adaptive roulette-wheel
  selection over random/worst removal and greedy/regret-2 repair.
* OR-Tools: Google OR-Tools GUIDED_LOCAL_SEARCH.

At each of the ten interval boundaries, only disclosed and unserved requests
are replanned.  Legs whose start time lies before the next boundary are
committed and the unstarted suffix is discarded.  A final dispatch at minute
480 handles requests disclosed in the last interval, matching paper_dcvrp.py.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxIterations
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


REPO_ROOT = Path(__file__).resolve().parents[2]
DYNAMIC_RATES = (0.10, 0.25, 0.50, 0.75)
CUSTOMER_SIZES = (20, 35, 50)
METHODS = ("Regret insertion", "Tabu Search", "Adaptive LNS", "OR-Tools")
HORIZON_MINUTES = 480.0
INTERVAL_COUNT = 10
VEHICLE_CAPACITY = 150
VEHICLE_SPEED = 1.0
DISTANCE_SCALE = 1_000_000
DROP_PENALTY_SCALED = 1_000_000_000
UNASSIGNED_PENALTY = 1_000.0


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


@dataclass(frozen=True)
class Instance:
    coordinates: np.ndarray
    demands: np.ndarray
    service_minutes: np.ndarray
    disclosure_minutes: np.ndarray


@dataclass
class PlanningContext:
    depot: np.ndarray
    starts: np.ndarray
    capacities: np.ndarray
    customer_ids: list[int]
    coordinates: np.ndarray
    demands: np.ndarray


@dataclass
class RouteState:
    context: PlanningContext
    routes: list[list[int]]
    unassigned: list[int]

    def copy(self) -> "RouteState":
        return RouteState(
            self.context,
            copy.deepcopy(self.routes),
            self.unassigned.copy(),
        )

    def objective(self) -> float:
        return solution_cost(self.context, self.routes) + (
            UNASSIGNED_PENALTY * len(self.unassigned)
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def vehicle_count(customer_count: int) -> int:
    if customer_count % 5:
        raise ValueError("paper sizes require m=n/5")
    return customer_count // 5


def generate_paired_instances(
    customer_count: int,
    instance_count: int,
    seed: int,
) -> dict[float, list[Instance]]:
    """Generate nested dynamic-rate variants from common physical instances."""

    rng = np.random.default_rng(seed)
    coordinates = rng.random((instance_count, customer_count + 1, 2))
    demands = rng.integers(5, 42, (instance_count, customer_count))
    service = rng.integers(10, 32, (instance_count, customer_count)).astype(float)
    rates = np.linspace(1.0, HORIZON_MINUTES, customer_count)
    sampled_disclosure = rng.poisson(
        np.broadcast_to(rates, (instance_count, customer_count))
    ).clip(1.0, HORIZON_MINUTES)
    maximum_count = int(np.rint(max(DYNAMIC_RATES) * customer_count))

    dynamic_order = np.full(
        (instance_count, customer_count), customer_count, dtype=int
    )
    for instance_index in range(instance_count):
        maximum_dynamic = rng.permutation(customer_count)[:maximum_count]
        nested_order = rng.permutation(maximum_dynamic)
        dynamic_order[instance_index, nested_order] = np.arange(maximum_count)

    paired: dict[float, list[Instance]] = {}
    for dynamic_rate in DYNAMIC_RATES:
        dynamic_count = int(np.rint(dynamic_rate * customer_count))
        disclosure = np.zeros((instance_count, customer_count), dtype=float)
        active = dynamic_order < dynamic_count
        disclosure[active] = sampled_disclosure[active]
        paired[dynamic_rate] = [
            Instance(
                coordinates=coordinates[index].copy(),
                demands=demands[index].copy(),
                service_minutes=service[index].copy(),
                disclosure_minutes=disclosure[index].copy(),
            )
            for index in range(instance_count)
        ]
    return paired


def point(context: PlanningContext, customer_id: int) -> np.ndarray:
    return context.coordinates[customer_id]


def route_cost(
    context: PlanningContext,
    vehicle_index: int,
    route: list[int],
) -> float:
    previous = context.starts[vehicle_index]
    cost = 0.0
    for customer in route:
        destination = point(context, customer)
        cost += float(np.linalg.norm(previous - destination))
        previous = destination
    cost += float(np.linalg.norm(previous - context.depot))
    return cost


def solution_cost(context: PlanningContext, routes: list[list[int]]) -> float:
    return sum(
        route_cost(context, vehicle_index, route)
        for vehicle_index, route in enumerate(routes)
    )


def insertion_cost(
    context: PlanningContext,
    vehicle_index: int,
    route: list[int],
    position: int,
    customer: int,
) -> float:
    predecessor = (
        context.starts[vehicle_index]
        if position == 0
        else point(context, route[position - 1])
    )
    successor = (
        context.depot
        if position == len(route)
        else point(context, route[position])
    )
    customer_point = point(context, customer)
    return float(
        np.linalg.norm(predecessor - customer_point)
        + np.linalg.norm(customer_point - successor)
        - np.linalg.norm(predecessor - successor)
    )


def route_load(context: PlanningContext, route: list[int]) -> int:
    return int(sum(int(context.demands[customer]) for customer in route))


def feasible_insertions(
    context: PlanningContext,
    routes: list[list[int]],
    customer: int,
) -> list[tuple[float, int, int]]:
    options = []
    demand = int(context.demands[customer])
    for vehicle_index, route in enumerate(routes):
        if route_load(context, route) + demand > int(context.capacities[vehicle_index]):
            continue
        for position in range(len(route) + 1):
            options.append(
                (
                    insertion_cost(
                        context, vehicle_index, route, position, customer
                    ),
                    vehicle_index,
                    position,
                )
            )
    return sorted(options)


def regret_insert_all(
    context: PlanningContext,
    routes: list[list[int]],
    unassigned: Iterable[int],
) -> tuple[list[list[int]], list[int]]:
    remaining = list(unassigned)
    while remaining:
        candidates = []
        infeasible = []
        for customer in remaining:
            options = feasible_insertions(context, routes, customer)
            if not options:
                infeasible.append(customer)
                continue
            best = options[0][0]
            regret = math.inf if len(options) == 1 else options[1][0] - best
            candidates.append((regret, -best, int(context.demands[customer]), customer, options[0]))
        if not candidates:
            return routes, remaining
        _, _, _, selected, option = max(candidates)
        _, vehicle_index, position = option
        routes[vehicle_index].insert(position, selected)
        remaining.remove(selected)
    return routes, []


def solve_regret(context: PlanningContext) -> tuple[list[list[int]], list[int]]:
    routes = [[] for _ in range(len(context.capacities))]
    return regret_insert_all(context, routes, context.customer_ids)


def solve_ortools(
    context: PlanningContext,
    metaheuristic: str,
    time_limit_ms: int,
) -> tuple[list[list[int]], list[int]]:
    customer_count = len(context.customer_ids)
    vehicle_total = len(context.capacities)
    node_coordinates = [context.depot]
    node_coordinates.extend(point(context, customer) for customer in context.customer_ids)
    node_coordinates.extend(context.starts[vehicle] for vehicle in range(vehicle_total))
    starts = list(range(customer_count + 1, customer_count + 1 + vehicle_total))
    manager = pywrapcp.RoutingIndexManager(
        len(node_coordinates), vehicle_total, starts, [0] * vehicle_total
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(
            round(
                DISTANCE_SCALE
                * float(
                    np.linalg.norm(
                        node_coordinates[from_node] - node_coordinates[to_node]
                    )
                )
            )
        )

    transit = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    local_to_global = {index + 1: customer for index, customer in enumerate(context.customer_ids)}

    def demand_callback(index: int) -> int:
        node = manager.IndexToNode(index)
        customer = local_to_global.get(node)
        return 0 if customer is None else int(context.demands[customer])

    demand = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand,
        0,
        [int(value) for value in context.capacities],
        True,
        "Capacity",
    )
    for local_node in range(1, customer_count + 1):
        routing.AddDisjunction(
            [manager.NodeToIndex(local_node)], DROP_PENALTY_SCALED
        )

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    if metaheuristic == "tabu":
        parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
        )
    elif metaheuristic == "gls":
        parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
    else:
        raise ValueError(metaheuristic)
    parameters.time_limit.FromMilliseconds(time_limit_ms)
    parameters.log_search = False
    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        return solve_regret(context)

    routes: list[list[int]] = [[] for _ in range(vehicle_total)]
    assigned = set()
    for vehicle in range(vehicle_total):
        index = routing.Start(vehicle)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            customer = local_to_global.get(node)
            if customer is not None:
                routes[vehicle].append(customer)
                assigned.add(customer)
            index = solution.Value(routing.NextVar(index))
    unassigned = [customer for customer in context.customer_ids if customer not in assigned]
    return routes, unassigned


def random_removal(state: RouteState, rng: np.random.Generator) -> RouteState:
    destroyed = state.copy()
    assigned = [customer for route in destroyed.routes for customer in route]
    if not assigned:
        return destroyed
    count = max(1, int(math.ceil(0.10 * len(assigned))))
    for customer in rng.choice(assigned, size=min(count, len(assigned)), replace=False):
        selected = int(customer)
        for route in destroyed.routes:
            if selected in route:
                route.remove(selected)
                destroyed.unassigned.append(selected)
                break
    return destroyed


def worst_removal(state: RouteState, rng: np.random.Generator) -> RouteState:
    del rng
    destroyed = state.copy()
    savings = []
    for vehicle_index, route in enumerate(destroyed.routes):
        for position, customer in enumerate(route):
            previous = (
                destroyed.context.starts[vehicle_index]
                if position == 0
                else point(destroyed.context, route[position - 1])
            )
            following = (
                destroyed.context.depot
                if position == len(route) - 1
                else point(destroyed.context, route[position + 1])
            )
            current = point(destroyed.context, customer)
            saving = (
                np.linalg.norm(previous - current)
                + np.linalg.norm(current - following)
                - np.linalg.norm(previous - following)
            )
            savings.append((float(saving), vehicle_index, customer))
    count = max(1, int(math.ceil(0.10 * len(savings)))) if savings else 0
    for _, vehicle_index, customer in sorted(savings, reverse=True)[:count]:
        destroyed.routes[vehicle_index].remove(customer)
        destroyed.unassigned.append(customer)
    return destroyed


def greedy_repair(state: RouteState, rng: np.random.Generator) -> RouteState:
    repaired = state.copy()
    rng.shuffle(repaired.unassigned)
    still_unassigned = []
    for customer in repaired.unassigned:
        options = feasible_insertions(repaired.context, repaired.routes, customer)
        if not options:
            still_unassigned.append(customer)
            continue
        _, vehicle_index, position = options[0]
        repaired.routes[vehicle_index].insert(position, customer)
    repaired.unassigned = still_unassigned
    return repaired


def regret_repair(state: RouteState, rng: np.random.Generator) -> RouteState:
    del rng
    repaired = state.copy()
    repaired.routes, repaired.unassigned = regret_insert_all(
        repaired.context, repaired.routes, repaired.unassigned
    )
    return repaired


def solve_alns(
    context: PlanningContext,
    iterations: int,
    seed: int,
) -> tuple[list[list[int]], list[int]]:
    routes, unassigned = solve_regret(context)
    initial = RouteState(context, routes, unassigned)
    if len(context.customer_ids) < 2 or iterations <= 0:
        return initial.routes, initial.unassigned
    engine = ALNS(np.random.default_rng(seed))
    engine.add_destroy_operator(random_removal)
    engine.add_destroy_operator(worst_removal)
    engine.add_repair_operator(greedy_repair)
    engine.add_repair_operator(regret_repair)
    selector = RouletteWheel([25, 5, 1, 0], 0.8, 2, 2)
    acceptance = RecordToRecordTravel.autofit(
        initial.objective(), 0.02, 0.0, iterations
    )
    result = engine.iterate(
        initial,
        selector,
        acceptance,
        MaxIterations(iterations),
    )
    best = result.best_state
    return best.routes, best.unassigned


def plan_routes(
    method: str,
    context: PlanningContext,
    ortools_time_ms: int,
    alns_iterations: int,
    seed: int,
) -> tuple[list[list[int]], list[int]]:
    if not context.customer_ids:
        return [[] for _ in range(len(context.capacities))], []
    if method == "Regret insertion":
        return solve_regret(context)
    if method == "Tabu Search":
        return solve_ortools(context, "tabu", ortools_time_ms)
    if method == "Adaptive LNS":
        return solve_alns(context, alns_iterations, seed)
    if method == "OR-Tools":
        return solve_ortools(context, "gls", ortools_time_ms)
    raise ValueError(method)


def simulate_dynamic_instance(
    instance: Instance,
    method: str,
    customer_count: int,
    dynamic_rate: float,
    instance_index: int,
    ortools_time_ms: int,
    alns_iterations: int,
    base_seed: int,
    cost_accounting: str,
) -> dict:
    started = time.perf_counter()
    depot = instance.coordinates[0].copy()
    customer_coordinates = instance.coordinates[1:]
    vehicle_total = vehicle_count(customer_count)
    vehicle_positions = np.repeat(depot[None, :], vehicle_total, axis=0)
    remaining_capacity = np.full(vehicle_total, VEHICLE_CAPACITY, dtype=int)
    vehicle_times = np.zeros(vehicle_total, dtype=float)
    served = np.zeros(customer_count, dtype=bool)
    total_distance = 0.0
    total_planned_distance = 0.0
    planning_calls = 0
    dropped_across_plans = 0

    def replan_and_execute(current_time: float, boundary: float | None) -> None:
        nonlocal total_distance, total_planned_distance, planning_calls, dropped_across_plans
        visible = np.flatnonzero(
            (instance.disclosure_minutes <= current_time) & (~served)
        ).tolist()
        context = PlanningContext(
            depot=depot,
            starts=vehicle_positions.copy(),
            capacities=remaining_capacity.copy(),
            customer_ids=visible,
            coordinates=customer_coordinates,
            demands=instance.demands,
        )
        planning_calls += 1
        interval_seed = (
            base_seed
            + 1_000_000 * customer_count
            + 10_000 * int(round(100 * dynamic_rate))
            + 100 * instance_index
            + planning_calls
        )
        routes, unassigned = plan_routes(
            method,
            context,
            ortools_time_ms,
            alns_iterations,
            interval_seed,
        )
        total_planned_distance += solution_cost(context, routes)
        dropped_across_plans += len(unassigned)
        assigned = [customer for route in routes for customer in route]
        if len(assigned) != len(set(assigned)):
            raise RuntimeError("solver returned duplicate customer assignments")
        for vehicle, route in enumerate(routes):
            sequence: list[int | None] = list(route) + [None]
            for customer in sequence:
                event_start = float(vehicle_times[vehicle])
                if boundary is not None and event_start >= boundary:
                    break
                destination = (
                    depot if customer is None else customer_coordinates[customer]
                )
                distance = float(
                    np.linalg.norm(vehicle_positions[vehicle] - destination)
                )
                total_distance += distance
                vehicle_times[vehicle] += distance / VEHICLE_SPEED
                vehicle_positions[vehicle] = destination
                if customer is not None:
                    if served[customer]:
                        raise RuntimeError("customer served more than once")
                    served[customer] = True
                    remaining_capacity[vehicle] -= int(instance.demands[customer])
                    if remaining_capacity[vehicle] < 0:
                        raise RuntimeError("capacity violation")
                    vehicle_times[vehicle] += float(
                        instance.service_minutes[customer]
                    )
        if boundary is not None:
            vehicle_times[:] = np.maximum(vehicle_times, boundary)

    interval = HORIZON_MINUTES / INTERVAL_COUNT
    for interval_index in range(INTERVAL_COUNT):
        current = interval_index * interval
        replan_and_execute(current, current + interval)
    replan_and_execute(HORIZON_MINUTES, None)

    elapsed = time.perf_counter() - started
    served_count = int(served.sum())
    if cost_accounting == "executed":
        reported_cost = total_distance
    elif cost_accounting == "planned":
        reported_cost = total_planned_distance
    else:
        raise ValueError(f"unknown cost accounting rule: {cost_accounting}")
    return {
        "method": method,
        "customer_count": customer_count,
        "vehicle_count": vehicle_total,
        "dynamic_rate": dynamic_rate,
        "instance_id": instance_index + 1,
        "cost": reported_cost,
        "executed_cost": total_distance,
        "planned_cost": total_planned_distance,
        "cost_accounting": cost_accounting,
        "qos_percent": 100.0 * served_count / customer_count,
        "served_customers": served_count,
        "unserved_customers": customer_count - served_count,
        "solve_time_s": elapsed,
        "planning_calls": planning_calls,
        "dropped_across_plans": dropped_across_plans,
        "final_completion_time_min": float(vehicle_times.max()),
    }


def _worker(payload: tuple) -> dict:
    return simulate_dynamic_instance(*payload)


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def completed_keys(frame: pd.DataFrame) -> set[tuple]:
    if frame.empty:
        return set()
    return set(
        zip(
            frame.method,
            frame.customer_count.astype(int),
            frame.dynamic_rate.round(6),
            frame.instance_id.astype(int),
        )
    )


def save_raw(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).sort_values(
        ["method", "customer_count", "dynamic_rate", "instance_id"]
    ).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def save_instance_manifests(
    datasets: dict[int, dict[float, list[Instance]]],
    output_dir: Path,
) -> None:
    """Persist every generated test field used by the confirmatory run."""

    manifest_dir = output_dir / "instance_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for customer_count, by_rate in datasets.items():
        reference = by_rate[min(by_rate)]
        payload: dict[str, np.ndarray] = {
            "coordinates": np.stack([item.coordinates for item in reference]),
            "demands": np.stack([item.demands for item in reference]),
            "service_minutes": np.stack(
                [item.service_minutes for item in reference]
            ),
        }
        for dynamic_rate, instances in by_rate.items():
            label = int(round(100.0 * dynamic_rate))
            payload[f"disclosure_minutes_rate_{label}"] = np.stack(
                [item.disclosure_minutes for item in instances]
            )
        np.savez_compressed(
            manifest_dir / f"paper_distribution_n{customer_count}.npz",
            **payload,
        )


def validate_instance_result(result: dict) -> None:
    customer_count = int(result["customer_count"])
    served = int(result["served_customers"])
    unserved = int(result["unserved_customers"])
    if served + unserved != customer_count:
        raise RuntimeError("served and unserved counts do not sum to n")
    expected_qos = 100.0 * served / customer_count
    if not math.isclose(float(result["qos_percent"]), expected_qos):
        raise RuntimeError("QoS is inconsistent with the service count")
    if not math.isfinite(float(result["cost"])) or float(result["cost"]) < 0.0:
        raise RuntimeError("invalid route cost")
    if not math.isfinite(float(result["solve_time_s"])) or float(
        result["solve_time_s"]
    ) < 0.0:
        raise RuntimeError("invalid solve time")


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
            time_per_instance_sd_s=("solve_time_s", "std"),
            unserved_total=("unserved_customers", "sum"),
        )
        .sort_values(["dynamic_rate", "method", "customer_count"])
        .reset_index(drop=True)
    )


def paper_rows() -> pd.DataFrame:
    rows = []
    for dynamic_rate in DYNAMIC_RATES:
        for customer_count in CUSTOMER_SIZES:
            dvnda = PAPER_DVNDA[(dynamic_rate, customer_count)]
            rows.append(
                {
                    "method": "DVNDA (paper)",
                    "customer_count": customer_count,
                    "vehicle_count": vehicle_count(customer_count),
                    "dynamic_rate": dynamic_rate,
                    "instances": 100,
                    "cost_mean": dvnda[0],
                    "cost_sd": dvnda[1],
                    "qos_mean_percent": dvnda[2],
                    "qos_sd_percent": np.nan,
                    "time_100_s": dvnda[3],
                    "time_per_instance_mean_s": dvnda[3] / 100.0,
                    "time_per_instance_sd_s": np.nan,
                    "unserved_total": 0,
                }
            )
            greedy = PAPER_GREEDY[(dynamic_rate, customer_count)]
            rows.append(
                {
                    "method": "Greedy (paper)",
                    "customer_count": customer_count,
                    "vehicle_count": vehicle_count(customer_count),
                    "dynamic_rate": dynamic_rate,
                    "instances": 100,
                    "cost_mean": greedy[0],
                    "cost_sd": greedy[1],
                    "qos_mean_percent": greedy[2],
                    "qos_sd_percent": np.nan,
                    "time_100_s": np.nan,
                    "time_per_instance_mean_s": np.nan,
                    "time_per_instance_sd_s": np.nan,
                    "unserved_total": np.nan,
                }
            )
    return pd.DataFrame(rows)


def value_cell(row: pd.Series, method: str) -> str:
    if method == "Greedy (paper)":
        timing = "--"
    else:
        timing = f"{row.time_100_s:.1f}s"
    qos = f"{row.qos_mean_percent:.2f}%"
    if pd.notna(row.qos_sd_percent):
        qos = f"{qos} ± {row.qos_sd_percent:.2f}"
    return (
        f"{row.cost_mean:.2f} ± {row.cost_sd:.2f}; "
        f"{qos}; {timing}"
    )


def write_table(
    combined: pd.DataFrame, path: Path, classical_instance_count: int
) -> None:
    order = [
        "Greedy (paper)",
        "Regret insertion",
        "Tabu Search",
        "Adaptive LNS",
        "OR-Tools",
        "DVNDA (paper)",
    ]
    indexed = combined.set_index(["dynamic_rate", "method", "customer_count"])
    lines = [
        "# CPU classical baselines in the 10-interval dynamic environment",
        "",
        f"Each classical cell reports Cost mean ± SD; QoS mean ± SD; and the sum of {classical_instance_count} per-instance end-to-end wall times. DVNDA and Greedy are copied verbatim from manuscript Table I (100 instances), where QoS SD was not reported.",
        "",
        "| Dynamic rate | Method | n=20, m=4 | n=35, m=7 | n=50, m=10 |",
        "|---:|:---|:---|:---|:---|",
    ]
    for dynamic_rate in DYNAMIC_RATES:
        for method in order:
            cells = []
            for size in CUSTOMER_SIZES:
                key = (dynamic_rate, method, size)
                cells.append(
                    value_cell(indexed.loc[key], method)
                    if key in indexed.index
                    else "not run"
                )
            lines.append(
                f"| {100 * dynamic_rate:.0f}% | {method} | "
                + " | ".join(cells)
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.instances != 100 and not args.allow_nonpaper_size:
        raise ValueError("paper table requires --instances 100")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_classical_instances.csv"
    config_path = output_dir / "experiment_config.json"
    prior_wall_time = 0.0
    if args.resume and config_path.exists():
        prior_wall_time = float(
            json.loads(config_path.read_text(encoding="utf-8")).get(
                "experiment_wall_time_s", 0.0
            )
        )
    existing = load_existing(raw_path) if args.resume else pd.DataFrame()
    if not existing.empty:
        if "cost_accounting" not in existing.columns:
            compatible = pd.Series(False, index=existing.index)
        else:
            compatible = existing["cost_accounting"].eq(args.cost_accounting)
        incompatible_count = int((~compatible).sum())
        if incompatible_count:
            print(
                f"discard {incompatible_count} resumed rows with missing or "
                f"incompatible cost_accounting; recomputing them",
                flush=True,
            )
            existing = existing.loc[compatible].copy()
    rows = existing.to_dict("records") if not existing.empty else []
    done = completed_keys(existing)
    selected_methods = tuple(args.methods)
    selected_sizes = tuple(args.sizes)
    selected_rates = tuple(args.rates)
    invalid_methods = set(selected_methods) - set(METHODS)
    if invalid_methods:
        raise ValueError(f"unknown methods: {sorted(invalid_methods)}")

    datasets = {
        customer_count: generate_paired_instances(
            customer_count,
            args.instances,
            args.data_seed + customer_count * 10_000,
        )
        for customer_count in selected_sizes
    }
    save_instance_manifests(datasets, output_dir)
    experiment_started = time.perf_counter()
    group_wall_path = output_dir / "parallel_group_wall_times.csv"
    if args.resume and group_wall_path.exists():
        group_wall = pd.read_csv(group_wall_path)
        compatible_groups = set(
            zip(
                existing.method,
                existing.customer_count.astype(int),
                existing.dynamic_rate.round(6),
            )
        )
        keep_group = [
            (
                row.method,
                int(row.customer_count),
                round(float(row.dynamic_rate), 6),
            )
            in compatible_groups
            for row in group_wall.itertuples(index=False)
        ]
        group_wall_rows = group_wall.loc[keep_group].to_dict("records")
    else:
        group_wall_rows = []
    for method in selected_methods:
        for customer_count in selected_sizes:
            for dynamic_rate in selected_rates:
                payloads = []
                for instance_index, instance in enumerate(
                    datasets[customer_count][dynamic_rate]
                ):
                    key = (
                        method,
                        customer_count,
                        round(dynamic_rate, 6),
                        instance_index + 1,
                    )
                    if key in done:
                        continue
                    payloads.append(
                        (
                            instance,
                            method,
                            customer_count,
                            dynamic_rate,
                            instance_index,
                            args.ortools_time_ms,
                            args.alns_iterations,
                            args.algorithm_seed,
                            args.cost_accounting,
                        )
                    )
                if not payloads:
                    print(
                        f"skip completed method={method} n={customer_count} rate={dynamic_rate}",
                        flush=True,
                    )
                    continue
                group_started = time.perf_counter()
                completed = 0
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = [executor.submit(_worker, payload) for payload in payloads]
                    for future in as_completed(futures):
                        result = future.result()
                        validate_instance_result(result)
                        rows.append(result)
                        done.add(
                            (
                                result["method"],
                                int(result["customer_count"]),
                                round(float(result["dynamic_rate"]), 6),
                                int(result["instance_id"]),
                            )
                        )
                        completed += 1
                        if completed % 10 == 0 or completed == len(payloads):
                            group_values = [
                                row
                                for row in rows
                                if row["method"] == method
                                and int(row["customer_count"]) == customer_count
                                and math.isclose(
                                    float(row["dynamic_rate"]), dynamic_rate
                                )
                            ]
                            print(
                                f"method={method:<16} n={customer_count:02d} "
                                f"rate={100 * dynamic_rate:02.0f}% "
                                f"done={len(group_values):03d}/{args.instances:03d} "
                                f"cost={np.mean([x['cost'] for x in group_values]):.3f} "
                                f"qos={np.mean([x['qos_percent'] for x in group_values]):.2f}% "
                                f"wall={time.perf_counter() - group_started:.1f}s",
                                flush=True,
                            )
                            save_raw(rows, raw_path)
                group_wall_rows.append(
                    {
                        "method": method,
                        "customer_count": customer_count,
                        "dynamic_rate": dynamic_rate,
                        "workers": args.workers,
                        "wall_time_s": time.perf_counter() - group_started,
                    }
                )

    raw = pd.DataFrame(rows)
    classical_summary = summarize(raw)
    combined = pd.concat(
        [classical_summary, paper_rows()], ignore_index=True, sort=False
    ).sort_values(["dynamic_rate", "method", "customer_count"])
    classical_summary.to_csv(output_dir / "classical_summary.csv", index=False)
    combined.to_csv(output_dir / "all_methods_with_paper.csv", index=False)
    pd.DataFrame(group_wall_rows).to_csv(group_wall_path, index=False)
    write_table(
        combined,
        output_dir / "TABLE_CPU_CLASSICAL_BASELINES.md",
        args.instances,
    )
    metadata = {
        "environment": {
            "horizon_minutes": HORIZON_MINUTES,
            "interval_count": INTERVAL_COUNT,
            "dynamic_rates": list(selected_rates),
            "customer_sizes": list(selected_sizes),
            "vehicle_counts": {
                str(size): vehicle_count(size) for size in selected_sizes
            },
            "capacity": VEHICLE_CAPACITY,
            "speed": VEHICLE_SPEED,
            "demand_range_inclusive": [5, 41],
            "service_minutes_range_inclusive": [10, 31],
            "coordinate_range": [0, 1],
            "commit_rule": "commit a planned leg iff its start time is before the next boundary; discard and replan the suffix",
            "terminal_rule": "one final full dispatch at minute 480",
            "cost_rule": (
                "sum Euclidean length of committed legs; unstarted suffixes excluded"
                if args.cost_accounting == "executed"
                else "sum complete planned-route lengths at every interval, matching the released environment reward accounting"
            ),
            "empty_interval_rule": "an empty interval still closes every active route at the depot; vehicles do not retain a free dispersed position",
        },
        "algorithms": {
            "Regret insertion": {
                "implementation": "NumPy deterministic regret-2 construction",
                "iterations": "not applicable (one constructive pass per update)",
            },
            "Tabu Search": {
                "library": "Google OR-Tools 9.10.4067",
                "first_solution": "PATH_CHEAPEST_ARC",
                "metaheuristic": "TABU_SEARCH",
                "time_limit_per_update_ms": args.ortools_time_ms,
                "other_parameters": "OR-Tools defaults",
            },
            "Adaptive LNS": {
                "library": "alns 7.0.0",
                "iterations_per_update": args.alns_iterations,
                "official_CVRP_example_iterations": 3000,
                "destroy": ["random removal 10%", "worst removal 10%"],
                "repair": ["greedy insertion", "regret-2 insertion"],
                "operator_selection": "RouletteWheel([25,5,1,0], decay=0.8)",
                "acceptance": "RecordToRecordTravel.autofit(start_gap=0.02,end_gap=0)",
            },
            "OR-Tools": {
                "library": "Google OR-Tools 9.10.4067",
                "first_solution": "PATH_CHEAPEST_ARC",
                "metaheuristic": "GUIDED_LOCAL_SEARCH",
                "time_limit_per_update_ms": args.ortools_time_ms,
                "other_parameters": "OR-Tools defaults",
            },
        },
        "instances_per_size_and_rate": args.instances,
        "common_random_numbers_across_methods_and_rates": True,
        "data_seed": args.data_seed,
        "algorithm_seed": args.algorithm_seed,
        "cost_accounting": args.cost_accounting,
        "workers_used_for_experiment_wall_time": args.workers,
        "reported_classical_time": "sum of 100 per-instance end-to-end wall times measured inside the CPU worker; 12-process experiment wall time is separate",
        "paper_rows": "DVNDA and Greedy copied verbatim from DCVRP.pdf Table I; not recomputed",
        "cpu": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version,
        "platform": platform.platform(),
        "experiment_wall_time_s": (
            prior_wall_time + time.perf_counter() - experiment_started
        ),
        "latest_invocation_wall_time_s": time.perf_counter() - experiment_started,
    }
    config_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"completed; results written to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument(
        "--methods", nargs="+", default=list(METHODS), choices=METHODS
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=list(CUSTOMER_SIZES)
    )
    parser.add_argument(
        "--rates", nargs="+", type=float, default=list(DYNAMIC_RATES)
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--ortools-time-ms", type=int, default=1000)
    parser.add_argument("--alns-iterations", type=int, default=3000)
    parser.add_argument("--data-seed", type=int, default=20260901)
    parser.add_argument("--algorithm-seed", type=int, default=314159)
    parser.add_argument(
        "--cost-accounting",
        choices=("executed", "planned"),
        default="executed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/cpu_dynamic_classical_baselines"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-nonpaper-size", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
