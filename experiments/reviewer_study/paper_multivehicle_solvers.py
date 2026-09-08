"""Classical multi-vehicle planners for :mod:`paper_multivehicle_env`.

Every planner receives one state containing *m* distinct vehicle positions,
remaining capacities, and availability times, and must return exactly *m*
routes.  OR-Tools uses one artificial start node per vehicle and a common depot
end; ALNS edits a vector of routes rather than one concatenated tour.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import math
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np

try:
    from alns import ALNS
    from alns.accept import HillClimbing
    from alns.select import RouletteWheel
    from alns.stop import MaxRuntime
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except ImportError:
    # The repository keeps CPU classical dependencies in a dedicated local
    # environment.  Append (do not prepend) so the active NumPy/Pandas builds
    # remain authoritative when this module is called from the training env.
    dependency_path = (
        Path(__file__).resolve().parents[1]
        / ".venv_classical"
        / "Lib"
        / "site-packages"
    )
    if dependency_path.exists():
        sys.path.append(str(dependency_path))
    from alns import ALNS
    from alns.accept import HillClimbing
    from alns.select import RouletteWheel
    from alns.stop import MaxRuntime
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .paper_multivehicle_env import (
    HORIZON_MINUTES,
    Plan,
    PlanningState,
    TRAVEL_MINUTES_PER_NORMALIZED_UNIT,
    nearest_idle_routes,
)


DISTANCE_SCALE = 1_000_000
TIME_SCALE = 1_000
UNASSIGNED_PENALTY = 5.0
DROP_PENALTY_SCALED = int(UNASSIGNED_PENALTY * DISTANCE_SCALE)
MISSING_ACTIVE_ROUTE_PENALTY = 50.0


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _feasibly_active_vehicles(state: PlanningState) -> list[int]:
    """Return a capacity-feasible set of vehicles that can be nonempty.

    Vehicles are matched to distinct currently visible customers, considering
    lower remaining capacities first.  This avoids adding impossible nonempty
    route constraints while still preventing a feasible fleet from silently
    collapsing to one route.
    """

    remaining = list(state.customer_ids)
    active: list[int] = []
    for vehicle in sorted(
        range(state.vehicle_count),
        key=lambda value: (
            int(state.vehicle_capacities[value]),
            float(state.vehicle_available_minutes[value]),
            value,
        ),
    ):
        feasible = [
            customer
            for customer in remaining
            if int(state.demands[customer])
            <= int(state.vehicle_capacities[vehicle])
        ]
        if not feasible:
            continue
        chosen = min(feasible, key=lambda customer: (int(state.demands[customer]), customer))
        remaining.remove(chosen)
        active.append(vehicle)
        if not remaining:
            break
    return active


def route_cost(state: PlanningState, vehicle: int, route: list[int]) -> float:
    previous = state.vehicle_positions[vehicle]
    total = 0.0
    for customer in route:
        point = state.customer_coordinates[customer]
        total += _distance(previous, point)
        previous = point
    return total + _distance(previous, state.depot)


def solution_cost(state: PlanningState, routes: list[list[int]]) -> float:
    return sum(route_cost(state, vehicle, route) for vehicle, route in enumerate(routes))


def timely_prefix_count(
    state: PlanningState, vehicle: int, route: list[int]
) -> int:
    """Number of customers whose outgoing leg starts no later than T."""

    # ``None`` marks the paper-compatible terminal closure round.  It has no
    # subsequent interval whose boundary could destroy the route suffix, so
    # the complete static route is executable.
    if state.next_boundary_minute is None:
        return len(route)

    current = float(state.vehicle_available_minutes[vehicle])
    previous = state.vehicle_positions[vehicle]
    count = 0
    for customer in route:
        if current > HORIZON_MINUTES + 1.0e-12:
            break
        count += 1
        point = state.customer_coordinates[customer]
        current += (
            _distance(previous, point)
            * state.travel_minutes_per_normalized_unit
            + float(state.service_minutes[customer])
        )
        previous = point
    return count


def route_objective(state: PlanningState, vehicle: int, route: list[int]) -> float:
    if state.optimize_complete_static_route:
        return route_cost(state, vehicle, route)
    late = len(route) - timely_prefix_count(state, vehicle, route)
    return route_cost(state, vehicle, route) + UNASSIGNED_PENALTY * late


def solution_objective(
    state: PlanningState, routes: list[list[int]], unassigned: list[int]
) -> float:
    value = sum(
        route_objective(state, vehicle, route)
        for vehicle, route in enumerate(routes)
    ) + UNASSIGNED_PENALTY * len(unassigned)
    if state.require_active_vehicle_routes:
        required = len(_feasibly_active_vehicles(state))
        missing = max(0, required - sum(bool(route) for route in routes))
        value += MISSING_ACTIVE_ROUTE_PENALTY * missing
    return value


def insertion_options(
    state: PlanningState, routes: list[list[int]], customer: int
) -> list[tuple[float, int, int]]:
    demand = int(state.demands[customer])
    loads = [
        sum(int(state.demands[value]) for value in route) for route in routes
    ]
    options = []
    for vehicle, route in enumerate(routes):
        if loads[vehicle] + demand > int(state.vehicle_capacities[vehicle]):
            continue
        base = route_objective(state, vehicle, route)
        for position in range(len(route) + 1):
            trial = route.copy()
            trial.insert(position, customer)
            # The customer is removed from the unassigned set on insertion.
            delta = (
                route_objective(state, vehicle, trial)
                - base
                - UNASSIGNED_PENALTY
            )
            options.append((delta, vehicle, position))
    return sorted(options)


def _regret_reinsert(
    state: PlanningState,
    routes: list[list[int]],
    unassigned: list[int],
) -> tuple[list[list[int]], list[int]]:
    remaining = list(unassigned)
    while remaining:
        candidates = []
        for customer in remaining:
            options = insertion_options(state, routes, customer)
            if not options or options[0][0] >= -1.0e-12:
                continue
            regret = math.inf if len(options) == 1 else options[1][0] - options[0][0]
            candidates.append((regret, -options[0][0], customer, options[0]))
        if not candidates:
            break
        _, _, customer, (_, vehicle, position) = max(candidates)
        routes[vehicle].insert(position, customer)
        remaining.remove(customer)
    return routes, remaining


def time_aware_regret_routes(state: PlanningState) -> Plan:
    """Regret-2 insertion with capacity and horizon-aware fleet balancing."""

    routes: list[list[int]] = [[] for _ in range(state.vehicle_count)]
    remaining = list(state.customer_ids)
    if state.require_active_vehicle_routes and remaining:
        required = min(state.vehicle_count, len(remaining))
        vehicles = sorted(
            range(state.vehicle_count),
            key=lambda vehicle: (
                float(state.vehicle_available_minutes[vehicle]), vehicle
            ),
        )[:required]
        # Standard parallel-insertion initialization: seed active routes with
        # spatially distant feasible customers, then let regret-2 insert the
        # remaining requests.  Nearest/large-demand seeding creates avoidable
        # depot-side clusters and systematically lengthens the final routes.
        for vehicle in vehicles:
            feasible = [
                customer
                for customer in remaining
                if int(state.demands[customer])
                <= int(state.vehicle_capacities[vehicle])
            ]
            if not feasible:
                break
            customer = max(
                feasible,
                key=lambda value: (
                    _distance(
                        state.vehicle_positions[vehicle],
                        state.customer_coordinates[value],
                    )
                    + _distance(
                        state.customer_coordinates[value], state.depot
                    ),
                    -int(state.demands[value]),
                    -value,
                ),
            )
            routes[vehicle].append(customer)
            remaining.remove(customer)
    routes, remaining = _regret_reinsert(state, routes, remaining)
    return Plan(
        routes=tuple(tuple(route) for route in routes),
        unassigned=tuple(remaining),
    )


def _tabu_order_route(
    state: PlanningState,
    vehicle: int,
    initial_route: list[int],
    *,
    time_limit_ms: int,
    seed: int,
) -> list[int]:
    """Optimize one assigned route with a standalone tabu search.

    The neighbourhood contains swap, relocate, and 2-opt moves.  The search
    accepts the best admissible move even when it is temporarily worse than
    the incumbent, while aspiration permits a tabu move that improves the
    global best.  This implementation is independent of OR-Tools.
    """

    if len(initial_route) < 2:
        return list(initial_route)
    if time_limit_ms <= 0:
        raise ValueError("time_limit_ms must be positive")

    rng = np.random.default_rng(int(seed))
    current = list(initial_route)
    current_value = route_objective(state, vehicle, current)
    best = current.copy()
    best_value = current_value
    tabu_until: dict[tuple, int] = {}
    tenure = max(5, int(round(math.sqrt(len(current)))) + 3)
    maximum_iterations = max(60, 20 * len(current))
    deadline = time.perf_counter() + float(time_limit_ms) / 1_000.0

    for iteration in range(maximum_iterations):
        if time.perf_counter() >= deadline:
            break
        candidates: list[tuple[float, float, list[int], tuple]] = []
        route_size = len(current)

        for left in range(route_size - 1):
            for right in range(left + 1, route_size):
                swapped = current.copy()
                swapped[left], swapped[right] = swapped[right], swapped[left]
                swap_key = (
                    "swap",
                    min(current[left], current[right]),
                    max(current[left], current[right]),
                )
                value = route_objective(state, vehicle, swapped)
                if tabu_until.get(swap_key, -1) <= iteration or value < best_value - 1e-12:
                    candidates.append((value, float(rng.random()), swapped, swap_key))

        for source in range(route_size):
            customer = current[source]
            without = current[:source] + current[source + 1 :]
            for target in range(route_size):
                if target == source:
                    continue
                relocated = without.copy()
                relocated.insert(target, customer)
                if relocated == current:
                    continue
                relocate_key = ("relocate", customer)
                value = route_objective(state, vehicle, relocated)
                if (
                    tabu_until.get(relocate_key, -1) <= iteration
                    or value < best_value - 1e-12
                ):
                    candidates.append(
                        (value, float(rng.random()), relocated, relocate_key)
                    )

        for left in range(route_size - 2):
            for right in range(left + 2, route_size):
                reversed_route = (
                    current[:left]
                    + list(reversed(current[left : right + 1]))
                    + current[right + 1 :]
                )
                two_opt_key = (
                    "two_opt",
                    min(current[left], current[right]),
                    max(current[left], current[right]),
                )
                value = route_objective(state, vehicle, reversed_route)
                if (
                    tabu_until.get(two_opt_key, -1) <= iteration
                    or value < best_value - 1e-12
                ):
                    candidates.append(
                        (value, float(rng.random()), reversed_route, two_opt_key)
                    )

        if not candidates:
            break
        current_value, _, current, move_key = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        tabu_until[move_key] = iteration + tenure + int(rng.integers(0, 4))
        if current_value < best_value - 1e-12:
            best = current.copy()
            best_value = current_value

    return best


def standalone_tabu_routes(
    state: PlanningState, *, time_limit_ms: int, seed: int
) -> Plan:
    """Standalone tabu-search baseline, without an OR-Tools code path."""

    if not state.customer_ids:
        return Plan(tuple(() for _ in range(state.vehicle_count)))
    if time_limit_ms <= 0:
        raise ValueError("time_limit_ms must be positive")

    # Under the assignment-preserving protocol, ``customer_ids`` already carry
    # the nearest-idle construction order for this one-vehicle subproblem.  It
    # is the natural feasible starting solution for Tabu and avoids sharing
    # Regret's initializer.  The free multi-vehicle mode still uses Regret-2
    # only to construct a capacity-feasible initial assignment.
    initial = (
        Plan(routes=(tuple(state.customer_ids),), unassigned=())
        if state.vehicle_count == 1
        else time_aware_regret_routes(state)
    )
    routes = [list(route) for route in initial.routes]
    active = max(1, sum(bool(route) for route in routes))
    per_route_ms = max(1, int(time_limit_ms) // active)
    optimized = [
        _tabu_order_route(
            state,
            vehicle,
            route,
            time_limit_ms=per_route_ms,
            seed=int(seed) + 7_919 * vehicle,
        )
        for vehicle, route in enumerate(routes)
    ]
    return Plan(
        routes=tuple(tuple(route) for route in optimized),
        unassigned=initial.unassigned,
    )


def ortools_routes(
    state: PlanningState,
    *,
    metaheuristic: str,
    time_limit_ms: int,
) -> Plan:
    if not state.customer_ids:
        return Plan(tuple(() for _ in range(state.vehicle_count)))
    if time_limit_ms <= 0:
        raise ValueError("time_limit_ms must be positive")

    customer_total = len(state.customer_ids)
    vehicle_total = state.vehicle_count
    coordinates = [state.depot]
    coordinates.extend(
        state.customer_coordinates[customer] for customer in state.customer_ids
    )
    coordinates.extend(state.vehicle_positions[vehicle] for vehicle in range(vehicle_total))
    starts = list(range(customer_total + 1, customer_total + 1 + vehicle_total))
    manager = pywrapcp.RoutingIndexManager(
        len(coordinates), vehicle_total, starts, [0] * vehicle_total
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        source = manager.IndexToNode(from_index)
        target = manager.IndexToNode(to_index)
        return int(round(DISTANCE_SCALE * _distance(coordinates[source], coordinates[target])))

    transit = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    local_to_global = {
        local + 1: customer for local, customer in enumerate(state.customer_ids)
    }

    def demand_callback(index: int) -> int:
        customer = local_to_global.get(manager.IndexToNode(index))
        return 0 if customer is None else int(state.demands[customer])

    demand = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand,
        0,
        [int(value) for value in state.vehicle_capacities],
        True,
        "Capacity",
    )

    if state.require_active_vehicle_routes:
        required_vehicles = _feasibly_active_vehicles(state)
        solver = routing.solver()
        for vehicle in required_vehicles:
            solver.Add(
                routing.NextVar(routing.Start(vehicle))
                != routing.End(vehicle)
            )

    if (
        not state.optimize_complete_static_route
        and state.current_minute < HORIZON_MINUTES - 1.0e-12
    ):
        def time_callback(from_index: int, to_index: int) -> int:
            source = manager.IndexToNode(from_index)
            target = manager.IndexToNode(to_index)
            customer = local_to_global.get(source)
            service = (
                0.0
                if customer is None
                else float(state.service_minutes[customer])
            )
            travel = (
                _distance(coordinates[source], coordinates[target])
                * state.travel_minutes_per_normalized_unit
            )
            return int(round(TIME_SCALE * (service + travel)))

        time_transit = routing.RegisterTransitCallback(time_callback)
        maximum_route_minute = HORIZON_MINUTES + 200.0
        routing.AddDimension(
            time_transit,
            0,
            int(round(TIME_SCALE * maximum_route_minute)),
            False,
            "Time",
        )
        time_dimension = routing.GetDimensionOrDie("Time")
        for vehicle in range(vehicle_total):
            start_value = int(
                round(
                    TIME_SCALE
                    * float(state.vehicle_available_minutes[vehicle])
                )
            )
            start_value = min(
                start_value, int(round(TIME_SCALE * maximum_route_minute))
            )
            time_dimension.CumulVar(routing.Start(vehicle)).SetRange(
                start_value, start_value
            )
            time_dimension.CumulVar(routing.End(vehicle)).SetMax(
                int(round(TIME_SCALE * maximum_route_minute))
            )
        for local_node in range(1, customer_total + 1):
            time_dimension.CumulVar(manager.NodeToIndex(local_node)).SetMax(
                int(round(TIME_SCALE * HORIZON_MINUTES))
            )
    for local_node in range(1, customer_total + 1):
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
        raise ValueError(f"unknown OR-Tools metaheuristic: {metaheuristic}")
    parameters.time_limit.FromMilliseconds(int(time_limit_ms))
    parameters.log_search = False
    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        raise RuntimeError(
            f"OR-Tools {metaheuristic} failed to return a solution within "
            f"{time_limit_ms} ms; strict baseline mode forbids substituting "
            f"another method; current={state.current_minute}, "
            f"next={state.next_boundary_minute}, customers={state.customer_ids}, "
            f"capacities={state.vehicle_capacities.tolist()}, "
            f"available={state.vehicle_available_minutes.tolist()}"
        )

    routes: list[list[int]] = [[] for _ in range(vehicle_total)]
    assigned: set[int] = set()
    for vehicle in range(vehicle_total):
        index = routing.Start(vehicle)
        while not routing.IsEnd(index):
            customer = local_to_global.get(manager.IndexToNode(index))
            if customer is not None:
                routes[vehicle].append(customer)
                assigned.add(customer)
            index = solution.Value(routing.NextVar(index))
    return Plan(
        routes=tuple(tuple(route) for route in routes),
        unassigned=tuple(
            customer for customer in state.customer_ids if customer not in assigned
        ),
    )


class MultiRouteState:
    def __init__(
        self,
        context: PlanningState,
        routes: list[list[int]],
        unassigned: list[int],
    ):
        self.context = context
        self.routes = routes
        self.unassigned = unassigned

    def copy(self) -> "MultiRouteState":
        return MultiRouteState(
            self.context, copy.deepcopy(self.routes), self.unassigned.copy()
        )

    def objective(self) -> float:
        return solution_objective(self.context, self.routes, self.unassigned)


def _random_removal(
    route_state: MultiRouteState, rng: np.random.Generator
) -> MultiRouteState:
    destroyed = route_state.copy()
    assigned = [customer for route in destroyed.routes for customer in route]
    if not assigned:
        return destroyed
    count = max(1, int(math.ceil(0.10 * len(assigned))))
    selected = rng.choice(assigned, size=min(count, len(assigned)), replace=False)
    for value in selected:
        customer = int(value)
        for route in destroyed.routes:
            if customer in route:
                route.remove(customer)
                destroyed.unassigned.append(customer)
                break
    return destroyed


def _worst_removal(
    route_state: MultiRouteState, rng: np.random.Generator
) -> MultiRouteState:
    del rng
    destroyed = route_state.copy()
    savings = []
    for vehicle, route in enumerate(destroyed.routes):
        for position, customer in enumerate(route):
            previous = (
                destroyed.context.vehicle_positions[vehicle]
                if position == 0
                else destroyed.context.customer_coordinates[route[position - 1]]
            )
            following = (
                destroyed.context.depot
                if position == len(route) - 1
                else destroyed.context.customer_coordinates[route[position + 1]]
            )
            point = destroyed.context.customer_coordinates[customer]
            saving = _distance(previous, point) + _distance(point, following) - _distance(
                previous, following
            )
            savings.append((saving, vehicle, customer))
    count = max(1, int(math.ceil(0.10 * len(savings)))) if savings else 0
    for _, vehicle, customer in sorted(savings, reverse=True)[:count]:
        destroyed.routes[vehicle].remove(customer)
        destroyed.unassigned.append(customer)
    return destroyed


def _greedy_repair(
    route_state: MultiRouteState, rng: np.random.Generator
) -> MultiRouteState:
    repaired = route_state.copy()
    rng.shuffle(repaired.unassigned)
    remaining = []
    for customer in repaired.unassigned:
        options = insertion_options(repaired.context, repaired.routes, customer)
        if not options or options[0][0] >= -1.0e-12:
            remaining.append(customer)
            continue
        _, vehicle, position = options[0]
        repaired.routes[vehicle].insert(position, customer)
    repaired.unassigned = remaining
    return repaired


def _regret_repair(
    route_state: MultiRouteState, rng: np.random.Generator
) -> MultiRouteState:
    del rng
    repaired = route_state.copy()
    repaired.routes, repaired.unassigned = _regret_reinsert(
        repaired.context, repaired.routes, repaired.unassigned
    )
    return repaired


def alns_routes(state: PlanningState, *, time_limit_ms: int, seed: int) -> Plan:
    initial_plan = time_aware_regret_routes(state)
    initial = MultiRouteState(
        state,
        [list(route) for route in initial_plan.routes],
        list(initial_plan.unassigned),
    )
    if time_limit_ms <= 0:
        raise ValueError("time_limit_ms must be positive")
    engine = ALNS(np.random.default_rng(int(seed)))
    engine.add_destroy_operator(_random_removal)
    engine.add_destroy_operator(_worst_removal)
    engine.add_repair_operator(_greedy_repair)
    engine.add_repair_operator(_regret_repair)
    selector = RouletteWheel([25, 5, 1, 0], 0.8, 2, 2)
    result = engine.iterate(
        initial,
        selector,
        HillClimbing(),
        MaxRuntime(float(time_limit_ms) / 1_000.0),
    )
    best = result.best_state
    return Plan(
        routes=tuple(tuple(route) for route in best.routes),
        unassigned=tuple(best.unassigned),
    )


def assignment_preserving_routes(
    state: PlanningState,
    *,
    method: str,
    ortools_time_ms: int,
    alns_time_ms: int,
    seed: int,
) -> Plan:
    """Optimize route order without changing the common fleet assignment.

    The manuscript's Greedy rule supplies the capacity-feasible vehicle to
    customer allocation.  A baseline then replaces only the within-vehicle
    static route optimizer.  This isolates route-search quality from a second,
    method-specific fleet-assignment advantage and implements the requested
    comparison in which the vehicle collaboration policy remains unchanged.
    The per-update search budget is divided across active vehicle routes so a
    method never receives ``m`` times the declared computational allowance.
    """

    allocation = nearest_idle_routes(state)
    active = sum(bool(route) for route in allocation.routes)
    if active == 0 or method == "Greedy (same instances)":
        return allocation

    ortools_per_route_ms = max(1, int(ortools_time_ms) // active)
    alns_per_route_ms = max(1, int(alns_time_ms) // active)
    optimized: list[tuple[int, ...]] = []
    unassigned = list(allocation.unassigned)

    for vehicle, assigned in enumerate(allocation.routes):
        if not assigned:
            optimized.append(())
            continue
        single = replace(
            state,
            vehicle_positions=state.vehicle_positions[vehicle : vehicle + 1].copy(),
            vehicle_capacities=state.vehicle_capacities[vehicle : vehicle + 1].copy(),
            vehicle_available_minutes=(
                state.vehicle_available_minutes[vehicle : vehicle + 1].copy()
            ),
            customer_ids=tuple(assigned),
            require_active_vehicle_routes=True,
            # Every boundary subproblem is a complete static route-ordering
            # problem.  The environment, not the solver, commits the started
            # prefix and destroys the unstarted suffix at the next boundary.
            optimize_complete_static_route=True,
        )
        if method == "Regret insertion":
            result = time_aware_regret_routes(single)
        elif method == "Tabu Search":
            result = standalone_tabu_routes(
                single,
                time_limit_ms=ortools_per_route_ms,
                seed=int(seed) + 1_009 * vehicle,
            )
        elif method == "Adaptive LNS":
            result = alns_routes(
                single,
                time_limit_ms=alns_per_route_ms,
                seed=int(seed) + 1_009 * vehicle,
            )
        elif method == "OR-Tools":
            result = ortools_routes(
                single,
                metaheuristic="gls",
                time_limit_ms=ortools_per_route_ms,
            )
        else:
            raise ValueError(f"unknown method: {method}")
        optimized.append(tuple(result.routes[0]))
        unassigned.extend(result.unassigned)

    return Plan(routes=tuple(optimized), unassigned=tuple(sorted(unassigned)))


def make_planner(
    method: str,
    *,
    ortools_time_ms: int = 1_000,
    alns_time_ms: int = 1_000,
    seed: int = 314159,
    assignment_policy: str = "free",
) -> Callable[[PlanningState], Plan]:
    if assignment_policy == "fixed-greedy":
        return lambda state: assignment_preserving_routes(
            state,
            method=method,
            ortools_time_ms=ortools_time_ms,
            alns_time_ms=alns_time_ms,
            seed=seed,
        )
    if assignment_policy != "free":
        raise ValueError(f"unknown assignment policy: {assignment_policy}")
    if method == "Greedy (same instances)":
        return nearest_idle_routes
    if method == "Regret insertion":
        return time_aware_regret_routes
    if method == "Tabu Search":
        return lambda state: standalone_tabu_routes(
            state, time_limit_ms=ortools_time_ms, seed=seed
        )
    if method == "Adaptive LNS":
        return lambda state: alns_routes(
            state, time_limit_ms=alns_time_ms, seed=seed
        )
    if method == "OR-Tools":
        return lambda state: ortools_routes(
            state, metaheuristic="gls", time_limit_ms=ortools_time_ms
        )
    raise ValueError(f"unknown method: {method}")


__all__ = [
    "MultiRouteState",
    "assignment_preserving_routes",
    "alns_routes",
    "insertion_options",
    "make_planner",
    "ortools_routes",
    "route_cost",
    "solution_cost",
    "solution_objective",
    "standalone_tabu_routes",
    "time_aware_regret_routes",
    "timely_prefix_count",
]
