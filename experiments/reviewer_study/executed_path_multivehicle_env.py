"""Classical multi-vehicle simulator matching the DVNDA executed-path MDP.

The horizon is split into ten synchronized intervals.  A classical solver
constructs one complete static multi-vehicle plan at every boundary.  Within
the interval that plan is immutable.  Customer legs with ``start < boundary``
are dispatched and therefore committed; the unstarted suffix is destroyed and
replanned later without contributing to Cost.

Vehicles do not return merely because an interval ends.  A depot leg is
committed only when the complete static customer route has been dispatched and
the return starts before the boundary; once started, that return remains locked
under Algorithm 1 even if it reaches the depot after the boundary.  Capacity is
restored only upon that physical return.  At minute 480, requests revealed in
the last interval receive one terminal static optimization round.  Cost is
exactly the sum of committed customer and depot edges.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .paper_multivehicle_env import (
    DispatchEvent,
    HORIZON_MINUTES,
    INTERVAL_COUNT,
    INTERVAL_MINUTES,
    PaperInstance,
    Plan,
    PlanningState,
    SimulationResult,
    VEHICLE_CAPACITY,
    _distance,
    _validate_plan,
    disclosure_boundaries,
    nearest_idle_routes,
    paper_vehicle_count,
    regret2_routes,
)


Planner = Callable[[PlanningState], Plan]
ENVIRONMENT_TAG = "time_driven_executed_path_multivehicle_v9"
# data.py samples physical coordinates in [0, 100] at speed 1 before
# normalization, so one normalized distance unit represents 100 travel minutes.
TRAVEL_MINUTES_PER_NORMALIZED_UNIT = 100.0


class ExecutedPathFleet:
    """Ten-boundary periodic-static execution with auditable actual Cost."""

    def __init__(
        self,
        vehicle_count: int,
        *,
        travel_minutes_per_normalized_unit: float = (
            TRAVEL_MINUTES_PER_NORMALIZED_UNIT
        ),
        completed_route_return: str = "commit-if-started",
    ):
        if vehicle_count <= 0:
            raise ValueError("vehicle_count must be positive")
        if travel_minutes_per_normalized_unit <= 0.0:
            raise ValueError("travel time scale must be positive")
        if completed_route_return not in {
            "final-only",
            "commit-if-started",
            "complete-before-boundary",
        }:
            raise ValueError("unknown completed-route return policy")
        self.vehicle_count = int(vehicle_count)
        self.travel_minutes_per_normalized_unit = float(
            travel_minutes_per_normalized_unit
        )
        self.completed_route_return = completed_route_return

    def run(self, instance: PaperInstance, planner: Planner) -> SimulationResult:
        depot = np.asarray(instance.coordinates[0], dtype=float)
        coordinates = np.asarray(instance.coordinates[1:], dtype=float)
        demands = np.asarray(instance.demands, dtype=int)
        service = np.asarray(instance.service_minutes, dtype=float)
        reveal = disclosure_boundaries(instance.disclosure_minutes)
        customer_total = instance.customer_count

        positions = np.repeat(depot[None, :], self.vehicle_count, axis=0)
        capacities = np.full(self.vehicle_count, VEHICLE_CAPACITY, dtype=int)
        available = np.zeros(self.vehicle_count, dtype=float)
        dispatched = np.zeros(customer_total, dtype=bool)
        customer_counts = np.zeros(self.vehicle_count, dtype=int)
        total_distance = 0.0
        events: list[DispatchEvent] = []
        planning_calls = 0

        def planning_state(
            current_minute: float, next_boundary: float | None
        ) -> PlanningState:
            visible = tuple(
                int(customer)
                for customer in np.flatnonzero(
                    (reveal <= current_minute + 1.0e-12) & (~dispatched)
                )
            )
            return PlanningState(
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
                    self.travel_minutes_per_normalized_unit
                ),
                optimize_complete_static_route=True,
                # Equations (3)--(4) apply to every static CVRP solve.  With
                # fewer than m visible requests the solvers therefore use
                # min(m, |V_t|) nonempty routes; this prevents a periodic
                # multi-vehicle comparison from silently collapsing to a
                # single active route in later intervals.
                require_active_vehicle_routes=bool(visible),
            )

        def execute_plan(
            interval_index: int,
            plan: Plan,
            boundary: float | None,
        ) -> None:
            nonlocal total_distance
            for vehicle, route in enumerate(plan.routes):
                route_fully_dispatched = True
                for customer in route:
                    start = float(available[vehicle])
                    # Strict Algorithm-1 dispatch rule.  Equality belongs to
                    # the next synchronized planning round.
                    if boundary is not None and start >= boundary - 1.0e-12:
                        route_fully_dispatched = False
                        break
                    destination = coordinates[customer]
                    distance = _distance(positions[vehicle], destination)
                    arrival = (
                        start
                        + distance * self.travel_minutes_per_normalized_unit
                    )

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

                if (
                    boundary is not None
                    and route
                    and route_fully_dispatched
                    and self.completed_route_return != "final-only"
                ):
                    return_start = float(available[vehicle])
                    return_distance = _distance(positions[vehicle], depot)
                    return_arrival = (
                        return_start
                        + return_distance * self.travel_minutes_per_normalized_unit
                    )
                    should_commit = (
                        return_start < boundary - 1.0e-12
                        if self.completed_route_return == "commit-if-started"
                        else return_arrival <= boundary + 1.0e-12
                    )
                    if should_commit:
                        total_distance += return_distance
                        events.append(
                            DispatchEvent(
                                interval_index=interval_index,
                                vehicle=vehicle,
                                customer=None,
                                start_minute=return_start,
                                arrival_minute=return_arrival,
                                completion_minute=return_arrival,
                                distance=return_distance,
                            )
                        )
                        positions[vehicle] = depot
                        available[vehicle] = return_arrival
                        capacities[vehicle] = VEHICLE_CAPACITY


        for interval_index in range(INTERVAL_COUNT):
            current = interval_index * INTERVAL_MINUTES
            boundary = current + INTERVAL_MINUTES
            state = planning_state(current, boundary)
            plan = planner(state)
            planning_calls += 1
            _validate_plan(state, plan)
            execute_plan(interval_index, plan, boundary)
            # Idle vehicles wait at their actual current position.  Vehicles
            # finishing a committed customer after the boundary retain their
            # future completion time and cannot be reused prematurely.
            available[:] = np.maximum(available, boundary)

        # Terminal closure is not an eleventh disclosure interval.  It gives
        # all requests visible by T one final static solve and executes that
        # solve completely, including work that finishes after T.
        terminal_state = planning_state(HORIZON_MINUTES, None)
        terminal_plan = planner(terminal_state)
        planning_calls += 1
        _validate_plan(terminal_state, terminal_plan)
        execute_plan(INTERVAL_COUNT, terminal_plan, None)

        # One actual end-of-horizon depot closure per vehicle.  Zero-distance
        # returns are retained as events so route accounting is auditable.
        for vehicle in range(self.vehicle_count):
            start = float(available[vehicle])
            distance = _distance(positions[vehicle], depot)
            arrival = start + distance * self.travel_minutes_per_normalized_unit
            total_distance += distance
            events.append(
                DispatchEvent(
                    interval_index=INTERVAL_COUNT + 1,
                    vehicle=vehicle,
                    customer=None,
                    start_minute=start,
                    arrival_minute=arrival,
                    completion_minute=arrival,
                    distance=distance,
                )
            )
            positions[vehicle] = depot
            available[vehicle] = arrival

        served = int(dispatched.sum())
        return SimulationResult(
            cost=float(total_distance),
            qos_percent=100.0 * served / customer_total,
            served_customers=served,
            unserved_customers=customer_total - served,
            final_completion_minute=float(available.max()),
            planning_calls=planning_calls,
            events=tuple(events),
            vehicle_customer_counts=tuple(
                int(value) for value in customer_counts
            ),
        )


__all__ = [
    "ENVIRONMENT_TAG",
    "ExecutedPathFleet",
    "DispatchEvent",
    "HORIZON_MINUTES",
    "INTERVAL_COUNT",
    "INTERVAL_MINUTES",
    "PaperInstance",
    "Plan",
    "PlanningState",
    "SimulationResult",
    "TRAVEL_MINUTES_PER_NORMALIZED_UNIT",
    "VEHICLE_CAPACITY",
    "disclosure_boundaries",
    "nearest_idle_routes",
    "paper_vehicle_count",
    "regret2_routes",
]
