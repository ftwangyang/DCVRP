"""Time-driven periodic-static multi-vehicle execution environment.

This is a versioned alternative to :mod:`paper_multivehicle_env`.  It follows
the protocol supplied for the classical-baseline comparison: one static plan
is produced at each of ten synchronized boundaries, no route changes occur
inside an interval, dispatched customer legs are committed, and the unstarted
suffix is destroyed at the next boundary.  Each vehicle closes the committed
prefix by returning to the depot.  Capacity is replenished only when that
return is completed; a vehicle cannot start a later interval's route before
its committed return time.

The original ``paper_multivehicle_v2`` simulator and its results are retained
unchanged so the two interpretations remain auditable.
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
    generate_release_batch,
    nearest_idle_routes,
    paper_vehicle_count,
    regret2_routes,
)


Planner = Callable[[PlanningState], Plan]
ENVIRONMENT_TAG = "periodic_static_multivehicle_v3"
# Manuscript prose: coordinates lie in [0, 1] and speed is 1.  Consequently,
# one normalized distance unit takes one minute in this versioned environment.
TRAVEL_MINUTES_PER_NORMALIZED_UNIT = 1.0


class PeriodicStaticFleet:
    """Execute one immutable static multi-vehicle plan per time interval.

    ``vehicle_positions`` in a planning state is the next actionable position.
    If a vehicle is still finishing a committed prefix or its depot return at
    the boundary, that position is the depot and ``vehicle_available_minutes``
    is its future depot-arrival time.  Thus a new plan may reserve that vehicle
    but cannot dispatch it before the previous trip has physically returned.
    """

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

        # The actionable state is always the end of the previously committed
        # trip.  A future availability time prevents premature reuse while a
        # vehicle is still serving or returning to the depot.
        positions = np.repeat(depot[None, :], self.vehicle_count, axis=0)
        capacities = np.full(self.vehicle_count, VEHICLE_CAPACITY, dtype=int)
        available = np.zeros(self.vehicle_count, dtype=float)
        last_physical_completion = np.zeros(self.vehicle_count, dtype=float)
        dispatched = np.zeros(customer_total, dtype=bool)
        customer_counts = np.zeros(self.vehicle_count, dtype=int)
        total_distance = 0.0
        events: list[DispatchEvent] = []
        planning_calls = 0

        for interval_index in range(INTERVAL_COUNT):
            current_minute = interval_index * INTERVAL_MINUTES
            next_boundary = current_minute + INTERVAL_MINUTES
            visible = tuple(
                int(customer)
                for customer in np.flatnonzero(
                    (reveal <= current_minute + 1.0e-12) & (~dispatched)
                )
            )
            state = PlanningState(
                current_minute=float(current_minute),
                next_boundary_minute=float(next_boundary),
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
                committed_count = 0
                for customer in route:
                    start = float(available[vehicle])
                    # A leg starting at the boundary belongs to the next
                    # interval and is therefore part of the destroyed suffix.
                    if start >= next_boundary - 1.0e-12:
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
                    committed_count += 1
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

                if committed_count:
                    # The executed/committed prefix forms exactly one trip for
                    # this interval.  Its depot return is mandatory even when
                    # it completes after the interval boundary.
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
                    # Replenishment occurs at return_arrival.  Storing full
                    # capacity now is safe because the availability constraint
                    # prevents the next route from starting before that time.
                    capacities[vehicle] = VEHICLE_CAPACITY
                    available[vehicle] = max(return_arrival, next_boundary)
                    last_physical_completion[vehicle] = return_arrival
                elif available[vehicle] < next_boundary:
                    # An idle vehicle waits at its actionable position until
                    # the next synchronized planning boundary.
                    available[vehicle] = next_boundary

        # Paper-compatible terminal closure.  This is not an eleventh regular
        # interval: no additional request stream is opened.  It is the final
        # route-construction round at T=480 used by the released paper
        # environment so that requests disclosed in the last regular interval
        # receive one assignment opportunity.  With no later boundary, the
        # selected static routes are executed completely and then closed at
        # the depot.
        terminal_visible = tuple(
            int(customer)
            for customer in np.flatnonzero(
                (reveal <= HORIZON_MINUTES + 1.0e-12) & (~dispatched)
            )
        )
        terminal_state = PlanningState(
            current_minute=HORIZON_MINUTES,
            next_boundary_minute=None,
            depot=depot.copy(),
            vehicle_positions=positions.copy(),
            vehicle_capacities=capacities.copy(),
            vehicle_available_minutes=available.copy(),
            customer_ids=terminal_visible,
            customer_coordinates=coordinates,
            demands=demands,
            service_minutes=service,
            travel_minutes_per_normalized_unit=(
                TRAVEL_MINUTES_PER_NORMALIZED_UNIT
            ),
        )
        terminal_plan = planner(terminal_state)
        planning_calls += 1
        _validate_plan(terminal_state, terminal_plan)
        for vehicle, route in enumerate(terminal_plan.routes):
            if not route:
                continue
            for customer in route:
                start = float(available[vehicle])
                destination = coordinates[customer]
                distance = _distance(positions[vehicle], destination)
                arrival = (
                    start + distance * TRAVEL_MINUTES_PER_NORMALIZED_UNIT
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
                        interval_index=INTERVAL_COUNT,
                        vehicle=vehicle,
                        customer=int(customer),
                        start_minute=start,
                        arrival_minute=arrival,
                        completion_minute=completion,
                        distance=distance,
                    )
                )
            return_start = float(available[vehicle])
            distance = _distance(positions[vehicle], depot)
            return_arrival = (
                return_start + distance * TRAVEL_MINUTES_PER_NORMALIZED_UNIT
            )
            total_distance += distance
            events.append(
                DispatchEvent(
                    interval_index=INTERVAL_COUNT,
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
            last_physical_completion[vehicle] = return_arrival

        served = int(dispatched.sum())
        return SimulationResult(
            cost=float(total_distance),
            qos_percent=100.0 * served / customer_total,
            served_customers=served,
            unserved_customers=customer_total - served,
            final_completion_minute=float(last_physical_completion.max()),
            planning_calls=planning_calls,
            events=tuple(events),
            vehicle_customer_counts=tuple(
                int(value) for value in customer_counts
            ),
        )


__all__ = [
    "ENVIRONMENT_TAG",
    "PeriodicStaticFleet",
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
    "generate_release_batch",
    "nearest_idle_routes",
    "paper_vehicle_count",
    "regret2_routes",
]
