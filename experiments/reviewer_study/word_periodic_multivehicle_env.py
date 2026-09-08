"""Word-spec periodic-static multi-vehicle DCVRP environment.

The attached specification ``详细的需求.docx`` is treated only as an
environment specification.  The horizon is 480 minutes split into ten fixed
intervals.  At each boundary, all requests disclosed by that boundary and all
unstarted suffix customers are solved as one complete static multi-vehicle
VRP.  The plan is immutable inside the interval.  Customer legs whose dispatch
starts before the next boundary are committed; the remaining suffix is
destroyed and replanned at the next boundary.

Each active vehicle performs at most one physical trip in an interval.  After
its committed prefix, it returns to the depot exactly once.  Capacity is
restored only at that physical return, never merely because the clock crosses
an interval boundary.  Cost is the sum of committed customer edges and these
executed depot-return edges; discarded planned edges are never charged.
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
ENVIRONMENT_TAG = "word_periodic_static_multivehicle_v5"
TRAVEL_MINUTES_PER_NORMALIZED_UNIT = 1.0


class WordPeriodicStaticFleet:
    """Execute one immutable, single-trip static plan per interval."""

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
        customer_total = instance.customer_count

        # A vehicle that is still completing the previous interval's committed
        # trip is represented by its next actionable state: depot position,
        # full capacity, and the future physical return time.  It may be
        # reserved by a new static plan but cannot dispatch before that time.
        positions = np.repeat(depot[None, :], self.vehicle_count, axis=0)
        capacities = np.full(self.vehicle_count, VEHICLE_CAPACITY, dtype=int)
        available = np.zeros(self.vehicle_count, dtype=float)
        dispatched = np.zeros(customer_total, dtype=bool)
        customer_counts = np.zeros(self.vehicle_count, dtype=int)
        total_distance = 0.0
        events: list[DispatchEvent] = []
        planning_calls = 0

        def make_state(
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
                    TRAVEL_MINUTES_PER_NORMALIZED_UNIT
                ),
                # The Word protocol asks every interval solver to solve the
                # complete currently visible static VRP.  The environment, not
                # the solver objective, performs the boundary truncation.
                optimize_complete_static_route=True,
                # The Word example states that all vehicles depart together
                # from the depot in the first static stage.  Later stages use
                # the true asynchronous return/availability state and must not
                # force a vehicle that is still completing its prior trip.
                require_active_vehicle_routes=(
                    abs(current_minute) <= 1.0e-12
                ),
            )

        def validate_word_fleet_use(state: PlanningState, plan: Plan) -> None:
            _validate_plan(state, plan)
            required = (
                min(state.vehicle_count, len(state.customer_ids))
                if state.require_active_vehicle_routes
                else 0
            )
            active = sum(bool(route) for route in plan.routes)
            if active < required:
                raise RuntimeError(
                    f"Word multi-vehicle plan used {active} routes; "
                    f"{required} are required for {len(state.customer_ids)} "
                    "visible customers"
                )

        def execute_trip(
            interval_index: int,
            plan: Plan,
            next_boundary: float | None,
        ) -> None:
            nonlocal total_distance
            for vehicle, route in enumerate(plan.routes):
                committed = 0
                for customer in route:
                    start = float(available[vehicle])
                    # Equality is handled by the next synchronized interval.
                    if (
                        next_boundary is not None
                        and start >= next_boundary - 1.0e-12
                    ):
                        break
                    destination = coordinates[customer]
                    distance = _distance(positions[vehicle], destination)
                    arrival = (
                        start
                        + distance * TRAVEL_MINUTES_PER_NORMALIZED_UNIT
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
                    committed += 1
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

                if committed:
                    # Close exactly one physical trip for this vehicle in the
                    # interval.  The return may complete after the boundary;
                    # the future availability time prevents premature reuse.
                    return_start = float(available[vehicle])
                    distance = _distance(positions[vehicle], depot)
                    return_arrival = (
                        return_start
                        + distance * TRAVEL_MINUTES_PER_NORMALIZED_UNIT
                    )
                    total_distance += distance
                    events.append(
                        DispatchEvent(
                            interval_index=interval_index,
                            vehicle=vehicle,
                            customer=None,
                            start_minute=return_start,
                            arrival_minute=return_arrival,
                            completion_minute=return_arrival,
                            distance=distance,
                        )
                    )
                    positions[vehicle] = depot
                    capacities[vehicle] = VEHICLE_CAPACITY
                    available[vehicle] = return_arrival

        for interval_index in range(INTERVAL_COUNT):
            current = interval_index * INTERVAL_MINUTES
            boundary = current + INTERVAL_MINUTES
            state = make_state(current, boundary)
            plan = planner(state)
            planning_calls += 1
            validate_word_fleet_use(state, plan)
            execute_trip(interval_index, plan, boundary)
            # Idle vehicles wait.  A vehicle with a committed trip can retain
            # a later physical return time.
            available[:] = np.maximum(available, boundary)

        # Final closure at T=480 is the last static solve, not an extra
        # disclosure interval.  With no following boundary, execute the
        # complete route and close each nonempty route at the depot.
        terminal_state = make_state(HORIZON_MINUTES, None)
        terminal_plan = planner(terminal_state)
        planning_calls += 1
        validate_word_fleet_use(terminal_state, terminal_plan)
        execute_trip(INTERVAL_COUNT, terminal_plan, None)

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
    "WordPeriodicStaticFleet",
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
