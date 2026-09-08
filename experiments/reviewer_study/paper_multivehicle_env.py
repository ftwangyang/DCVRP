"""Paper/released-code DCVRP generator and synchronous multi-vehicle simulator.

This module is intentionally independent from ``learner.DCVRP_Environment``.
It implements Algorithm 1 as an explicit event simulator so that classical
planners cannot accidentally collapse the fleet into a single route.

The release generator is reproduced literally where it differs from prose:
locations are integer samples in [0, 100] before normalization, demands and
service durations follow PyTorch's half-open integer ranges [5, 41) and
[10, 31), dynamic membership is Bernoulli, and one Poisson disclosure vector
is shared by every instance in a generated batch (``data.py``, lines 5-10 and
105-121).  Disclosures become visible at the next of ten interval boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np


HORIZON_MINUTES = 480.0
INTERVAL_COUNT = 10
INTERVAL_MINUTES = HORIZON_MINUTES / INTERVAL_COUNT
VEHICLE_CAPACITY = 150
# ``data.py.normalize`` divides the original 0--100 coordinates by the batch
# range and multiplies speed by 480 / range while time is divided by 480.
# Therefore one normalized distance unit still represents 100 physical travel
# minutes for the released uniform instances; using distance itself as minutes
# makes the fleet about 100 times too fast.
TRAVEL_MINUTES_PER_NORMALIZED_UNIT = 100.0


@dataclass(frozen=True)
class PaperInstance:
    coordinates: np.ndarray  # depot first, shape (n + 1, 2)
    demands: np.ndarray  # customer-only, shape (n,)
    service_minutes: np.ndarray  # customer-only, shape (n,)
    disclosure_minutes: np.ndarray  # customer-only, zero means static

    @property
    def customer_count(self) -> int:
        return int(self.demands.size)


@dataclass(frozen=True)
class PlanningState:
    current_minute: float
    next_boundary_minute: float | None
    depot: np.ndarray
    vehicle_positions: np.ndarray
    vehicle_capacities: np.ndarray
    vehicle_available_minutes: np.ndarray
    customer_ids: tuple[int, ...]
    customer_coordinates: np.ndarray
    demands: np.ndarray
    service_minutes: np.ndarray
    # Versioned environments may implement the paper prose (coordinates and
    # speed both on the [0, 1] scale) or the released generator's implicit
    # 0--100 physical scale.  Solvers must consume the environment's value
    # instead of silently assuming one of the two interpretations.
    travel_minutes_per_normalized_unit: float = (
        TRAVEL_MINUTES_PER_NORMALIZED_UNIT
    )
    # The manuscript experiment solves the complete visible static VRP at a
    # boundary and lets the environment destroy its unstarted suffix.  Older
    # simulator versions used a horizon-aware surrogate objective instead.
    optimize_complete_static_route: bool = False
    # Word-spec static comparison: prevent a nominal m-route solution from
    # collapsing into one active vehicle plus m-1 empty route containers.
    # Other versioned environments keep the standard optional-fleet behavior.
    require_active_vehicle_routes: bool = False

    @property
    def vehicle_count(self) -> int:
        return int(self.vehicle_capacities.size)


@dataclass(frozen=True)
class Plan:
    routes: tuple[tuple[int, ...], ...]
    unassigned: tuple[int, ...] = ()


@dataclass(frozen=True)
class DispatchEvent:
    interval_index: int
    vehicle: int
    customer: int | None
    start_minute: float
    arrival_minute: float
    completion_minute: float
    distance: float


@dataclass(frozen=True)
class SimulationResult:
    cost: float
    qos_percent: float
    served_customers: int
    unserved_customers: int
    final_completion_minute: float
    planning_calls: int
    events: tuple[DispatchEvent, ...]
    vehicle_customer_counts: tuple[int, ...]


Planner = Callable[[PlanningState], Plan]


def paper_vehicle_count(customer_count: int) -> int:
    if customer_count <= 0 or customer_count % 5:
        raise ValueError("paper test scales require m = n / 5")
    return customer_count // 5


def generate_release_batch(
    customer_count: int,
    instance_count: int,
    dynamic_rate: float,
    seed: int,
) -> list[PaperInstance]:
    """Reproduce ``DCVRP_Dataset.generate(...).normalize()`` for one rate.

    A local Torch RNG context makes the generated test set deterministic without
    leaking random-state changes into model training or other experiments.
    """

    if not 0.0 <= dynamic_rate <= 1.0:
        raise ValueError("dynamic_rate must lie in [0, 1]")
    if customer_count <= 0 or instance_count <= 0:
        raise ValueError("customer_count and instance_count must be positive")

    import torch

    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        coordinates = torch.randint(
            0,
            101,
            (instance_count, customer_count + 1, 2),
            dtype=torch.int64,
        ).to(torch.float64)
        demands = torch.randint(
            5, 41, (instance_count, customer_count), dtype=torch.int64
        )
        service = torch.randint(
            10, 31, (instance_count, customer_count), dtype=torch.int64
        ).to(torch.float64)
        dynamic = torch.empty(
            (instance_count, customer_count), dtype=torch.float64
        ).bernoulli_(float(dynamic_rate))
        poisson_rate = torch.linspace(
            1.0, HORIZON_MINUTES, customer_count, dtype=torch.float64
        ).view(1, customer_count)
        # The released generator calls create_logistics_distribution without
        # batch_size, so this single vector is broadcast across the batch.
        disclosure_sample = torch.poisson(poisson_rate).clamp_(
            min=1.0, max=HORIZON_MINUTES
        )
        disclosure = dynamic * disclosure_sample
    finally:
        torch.random.set_rng_state(state)

    location_min = float(coordinates.min().item())
    location_scale = float(coordinates.max().item() - location_min)
    if location_scale <= 0.0:
        raise RuntimeError("degenerate coordinate batch")
    coordinates = (coordinates - location_min) / location_scale

    instances = []
    for index in range(instance_count):
        instances.append(
            PaperInstance(
                coordinates=coordinates[index].numpy().copy(),
                demands=demands[index].numpy().copy(),
                service_minutes=service[index].numpy().copy(),
                disclosure_minutes=disclosure[index].numpy().copy(),
            )
        )
    return instances


def disclosure_boundaries(disclosure_minutes: np.ndarray) -> np.ndarray:
    """Map continuous disclosures to the next synchronized boundary."""

    values = np.asarray(disclosure_minutes, dtype=float)
    return np.where(
        values <= 0.0,
        0.0,
        np.ceil(values / INTERVAL_MINUTES - 1.0e-12) * INTERVAL_MINUTES,
    ).clip(0.0, HORIZON_MINUTES)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _validate_plan(state: PlanningState, plan: Plan) -> None:
    if len(plan.routes) != state.vehicle_count:
        raise RuntimeError(
            f"planner returned {len(plan.routes)} routes for "
            f"{state.vehicle_count} vehicles"
        )
    visible = set(state.customer_ids)
    assigned = [customer for route in plan.routes for customer in route]
    if len(assigned) != len(set(assigned)):
        raise RuntimeError("a customer was assigned to multiple vehicles")
    if not set(assigned).issubset(visible):
        raise RuntimeError("planner assigned an unrevealed customer")
    if not set(plan.unassigned).issubset(visible):
        raise RuntimeError("unassigned list contains an unrevealed customer")
    if set(assigned) & set(plan.unassigned):
        raise RuntimeError("customer is both assigned and unassigned")
    for vehicle, route in enumerate(plan.routes):
        load = sum(int(state.demands[customer]) for customer in route)
        if load > int(state.vehicle_capacities[vehicle]):
            raise RuntimeError(f"capacity violation on vehicle {vehicle}")


class PaperTimeDrivenFleet:
    """Synchronous Algorithm-1 execution for a fleet of independent vehicles."""

    def __init__(self, vehicle_count: int):
        if vehicle_count <= 0:
            raise ValueError("vehicle_count must be positive")
        self.vehicle_count = int(vehicle_count)

    def run(self, instance: PaperInstance, planner: Planner) -> SimulationResult:
        depot = np.asarray(instance.coordinates[0], dtype=float)
        coordinates = np.asarray(instance.coordinates[1:], dtype=float)
        demands = np.asarray(instance.demands, dtype=int)
        service = np.asarray(instance.service_minutes, dtype=float)
        reveal = disclosure_boundaries(instance.disclosure_minutes)
        n = instance.customer_count

        positions = np.repeat(depot[None, :], self.vehicle_count, axis=0)
        capacities = np.full(self.vehicle_count, VEHICLE_CAPACITY, dtype=int)
        available = np.zeros(self.vehicle_count, dtype=float)
        dispatched = np.zeros(n, dtype=bool)
        customer_counts = np.zeros(self.vehicle_count, dtype=int)
        total_distance = 0.0
        events: list[DispatchEvent] = []
        planning_calls = 0

        def plan_and_commit(
            interval_index: int,
            current_minute: float,
            next_boundary: float | None,
            *,
            include_cutoff_start: bool = False,
        ) -> None:
            nonlocal total_distance, planning_calls
            visible = tuple(
                int(value)
                for value in np.flatnonzero(
                    (reveal <= current_minute + 1.0e-12) & (~dispatched)
                )
            )
            state = PlanningState(
                current_minute=float(current_minute),
                next_boundary_minute=(
                    None if next_boundary is None else float(next_boundary)
                ),
                depot=depot.copy(),
                vehicle_positions=positions.copy(),
                vehicle_capacities=capacities.copy(),
                vehicle_available_minutes=available.copy(),
                customer_ids=visible,
                customer_coordinates=coordinates,
                demands=demands,
                service_minutes=service,
                travel_minutes_per_normalized_unit=(
                    TRAVEL_MINUTES_PER_NORMALIZED_UNIT
                ),
            )
            plan = planner(state)
            planning_calls += 1
            _validate_plan(state, plan)

            for vehicle, route in enumerate(plan.routes):
                route_completed = True
                for customer in route:
                    start = float(available[vehicle])
                    if next_boundary is not None:
                        too_late = (
                            start > next_boundary + 1.0e-12
                            if include_cutoff_start
                            else start >= next_boundary - 1.0e-12
                        )
                        if too_late:
                            route_completed = False
                            break
                    destination = coordinates[customer]
                    distance = _distance(positions[vehicle], destination)
                    arrival = start + distance * TRAVEL_MINUTES_PER_NORMALIZED_UNIT
                    completion = arrival + float(service[customer])
                    total_distance += distance
                    positions[vehicle] = destination
                    available[vehicle] = completion
                    capacities[vehicle] -= int(demands[customer])
                    if capacities[vehicle] < 0:
                        raise RuntimeError("negative remaining capacity")
                    if dispatched[customer]:
                        raise RuntimeError("customer dispatched twice")
                    dispatched[customer] = True
                    customer_counts[vehicle] += 1
                    events.append(
                        DispatchEvent(
                            interval_index=interval_index,
                            vehicle=vehicle,
                            customer=int(customer),
                            start_minute=start,
                            arrival_minute=arrival,
                            completion_minute=completion,
                            distance=distance,
                        )
                    )

            if next_boundary is not None:
                available[:] = np.maximum(available, next_boundary)

        for interval_index in range(INTERVAL_COUNT):
            current = interval_index * INTERVAL_MINUTES
            plan_and_commit(interval_index, current, current + INTERVAL_MINUTES)

        # Requests disclosed exactly at T are visible for one terminal
        # assignment decision.  Only a leg that can start at T is committed;
        # after serving it the same vehicle cannot start another customer
        # within the planning horizon.
        plan_and_commit(
            INTERVAL_COUNT,
            HORIZON_MINUTES,
            HORIZON_MINUTES,
            include_cutoff_start=True,
        )

        # Single-round-trip closure: an idle vehicle waits at its current
        # position between disclosure boundaries and returns to the depot only
        # once, after its last dispatched customer.  A closing depot leg may
        # finish beyond minute 480, but no customer dispatch is allowed there.
        for vehicle in range(self.vehicle_count):
            return_start = float(available[vehicle])
            distance = _distance(positions[vehicle], depot)
            arrival = return_start + distance * TRAVEL_MINUTES_PER_NORMALIZED_UNIT
            total_distance += distance
            positions[vehicle] = depot
            available[vehicle] = arrival
            events.append(
                DispatchEvent(
                    interval_index=INTERVAL_COUNT + 1,
                    vehicle=vehicle,
                    customer=None,
                    start_minute=return_start,
                    arrival_minute=arrival,
                    completion_minute=arrival,
                    distance=distance,
                )
            )
        served = int(dispatched.sum())
        return SimulationResult(
            cost=float(total_distance),
            qos_percent=100.0 * served / n,
            served_customers=served,
            unserved_customers=n - served,
            final_completion_minute=float(available.max()),
            planning_calls=planning_calls,
            events=tuple(events),
            vehicle_customer_counts=tuple(int(value) for value in customer_counts),
        )


def nearest_idle_routes(state: PlanningState) -> Plan:
    """Paper Greedy: earliest idle vehicle takes its nearest feasible task."""

    routes: list[list[int]] = [[] for _ in range(state.vehicle_count)]
    positions = state.vehicle_positions.copy()
    capacities = state.vehicle_capacities.copy()
    ready = state.vehicle_available_minutes.copy()
    remaining = set(state.customer_ids)
    disabled: set[int] = set()

    while remaining:
        candidates = []
        for vehicle in range(state.vehicle_count):
            if vehicle in disabled:
                continue
            feasible = [
                customer
                for customer in remaining
                if int(state.demands[customer]) <= int(capacities[vehicle])
            ]
            if feasible:
                candidates.append((float(ready[vehicle]), vehicle, feasible))
            else:
                disabled.add(vehicle)
        if not candidates:
            break
        _, vehicle, feasible = min(candidates, key=lambda item: (item[0], item[1]))
        customer = min(
            feasible,
            key=lambda value: (
                _distance(positions[vehicle], state.customer_coordinates[value]),
                value,
            ),
        )
        distance = _distance(positions[vehicle], state.customer_coordinates[customer])
        routes[vehicle].append(customer)
        positions[vehicle] = state.customer_coordinates[customer]
        capacities[vehicle] -= int(state.demands[customer])
        ready[vehicle] += (
            distance * state.travel_minutes_per_normalized_unit
            + float(state.service_minutes[customer])
        )
        remaining.remove(customer)

    return Plan(
        routes=tuple(tuple(route) for route in routes),
        unassigned=tuple(sorted(remaining)),
    )


def _insertion_delta(
    state: PlanningState,
    vehicle: int,
    route: list[int],
    position: int,
    customer: int,
) -> float:
    previous = (
        state.vehicle_positions[vehicle]
        if position == 0
        else state.customer_coordinates[route[position - 1]]
    )
    following = (
        state.depot
        if position == len(route)
        else state.customer_coordinates[route[position]]
    )
    point = state.customer_coordinates[customer]
    return _distance(previous, point) + _distance(point, following) - _distance(
        previous, following
    )


def regret2_routes(state: PlanningState) -> Plan:
    """Deterministic multi-vehicle regret-2 insertion."""

    routes: list[list[int]] = [[] for _ in range(state.vehicle_count)]
    loads = np.zeros(state.vehicle_count, dtype=int)
    remaining = set(state.customer_ids)
    while remaining:
        choices = []
        for customer in sorted(remaining):
            options = []
            demand = int(state.demands[customer])
            for vehicle, route in enumerate(routes):
                if loads[vehicle] + demand > int(state.vehicle_capacities[vehicle]):
                    continue
                for position in range(len(route) + 1):
                    options.append(
                        (
                            _insertion_delta(
                                state, vehicle, route, position, customer
                            ),
                            vehicle,
                            position,
                        )
                    )
            options.sort()
            if not options:
                continue
            regret = (
                math.inf
                if len(options) == 1
                else float(options[1][0] - options[0][0])
            )
            choices.append((regret, -float(options[0][0]), customer, options[0]))
        if not choices:
            break
        _, _, customer, (_, vehicle, position) = max(choices)
        routes[vehicle].insert(position, customer)
        loads[vehicle] += int(state.demands[customer])
        remaining.remove(customer)
    return Plan(
        routes=tuple(tuple(route) for route in routes),
        unassigned=tuple(sorted(remaining)),
    )


__all__ = [
    "DispatchEvent",
    "HORIZON_MINUTES",
    "INTERVAL_COUNT",
    "INTERVAL_MINUTES",
    "PaperInstance",
    "PaperTimeDrivenFleet",
    "Plan",
    "PlanningState",
    "SimulationResult",
    "TRAVEL_MINUTES_PER_NORMALIZED_UNIT",
    "disclosure_boundaries",
    "generate_release_batch",
    "nearest_idle_routes",
    "paper_vehicle_count",
    "regret2_routes",
]
