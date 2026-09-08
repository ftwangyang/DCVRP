"""Rolling-horizon classical baselines on controlled normalized instances."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except ImportError:  # pragma: no cover - reported by the runner
    pywrapcp = None
    routing_enums_pb2 = None


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _route_cost(start: np.ndarray, route: list[int], coordinates: np.ndarray,
                depot: np.ndarray) -> float:
    if not route:
        return _distance(start, depot)
    total = _distance(start, coordinates[route[0]])
    total += sum(
        _distance(coordinates[left], coordinates[right])
        for left, right in zip(route[:-1], route[1:])
    )
    total += _distance(coordinates[route[-1]], depot)
    return total


def _best_insertion(start, route, customer, coordinates, depot):
    best_position = 0
    best_delta = math.inf
    old = _route_cost(start, route, coordinates, depot)
    for position in range(len(route) + 1):
        candidate = route[:position] + [customer] + route[position:]
        delta = _route_cost(start, candidate, coordinates, depot) - old
        if delta < best_delta:
            best_delta = delta
            best_position = position
    return best_delta, best_position


def nearest_routes(starts, capacities, customers, coordinates, demands, depot):
    routes = [[] for _ in range(len(starts))]
    remaining = set(customers)
    current = starts.copy()
    remaining_capacity = capacities.copy()
    while remaining:
        best = None
        for vehicle in range(len(starts)):
            for customer in remaining:
                if demands[customer] > remaining_capacity[vehicle] + 1e-9:
                    continue
                candidate = (_distance(current[vehicle], coordinates[customer]), vehicle, customer)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            break
        _, vehicle, customer = best
        routes[vehicle].append(customer)
        current[vehicle] = coordinates[customer]
        remaining_capacity[vehicle] -= demands[customer]
        remaining.remove(customer)
    return routes


def regret_routes(starts, capacities, customers, coordinates, demands, depot):
    routes = [[] for _ in range(len(starts))]
    loads = np.zeros(len(starts), dtype=float)
    remaining = set(customers)
    while remaining:
        chosen = None
        for customer in remaining:
            options = []
            for vehicle in range(len(starts)):
                if loads[vehicle] + demands[customer] > capacities[vehicle] + 1e-9:
                    continue
                delta, position = _best_insertion(
                    starts[vehicle], routes[vehicle], customer, coordinates, depot
                )
                options.append((delta, vehicle, position))
            if not options:
                continue
            options.sort()
            regret = (
                options[1][0] - options[0][0]
                if len(options) > 1 else 1.0e6 - options[0][0]
            )
            candidate = (regret, -options[0][0], customer, options[0])
            if chosen is None or candidate > chosen:
                chosen = candidate
        if chosen is None:
            break
        _, _, customer, (_, vehicle, position) = chosen
        routes[vehicle].insert(position, customer)
        loads[vehicle] += demands[customer]
        remaining.remove(customer)
    return routes


def _two_opt(route, start, coordinates, depot):
    best = list(route)
    best_cost = _route_cost(start, best, coordinates, depot)
    improved = True
    while improved:
        improved = False
        for left in range(len(best) - 1):
            for right in range(left + 2, len(best) + 1):
                candidate = best[:left] + best[left:right][::-1] + best[right:]
                cost = _route_cost(start, candidate, coordinates, depot)
                if cost + 1e-10 < best_cost:
                    best, best_cost, improved = candidate, cost, True
    return best


def local_search_routes(starts, capacities, customers, coordinates, demands, depot):
    routes = regret_routes(starts, capacities, customers, coordinates, demands, depot)
    return [
        _two_opt(route, starts[vehicle], coordinates, depot)
        for vehicle, route in enumerate(routes)
    ]


def tabu_routes(starts, capacities, customers, coordinates, demands, depot,
                iterations=60):
    routes = local_search_routes(starts, capacities, customers, coordinates, demands, depot)
    loads = np.array([sum(demands[c] for c in route) for route in routes])
    tabu = {}
    best_routes = [list(route) for route in routes]
    best_cost = sum(_route_cost(starts[v], route, coordinates, depot)
                    for v, route in enumerate(routes))
    for iteration in range(iterations):
        best_move = None
        for source, source_route in enumerate(routes):
            for customer in source_route:
                for target in range(len(routes)):
                    if target == source:
                        continue
                    if loads[target] + demands[customer] > capacities[target] + 1e-9:
                        continue
                    reduced = [c for c in source_route if c != customer]
                    delta_insert, position = _best_insertion(
                        starts[target], routes[target], customer, coordinates, depot
                    )
                    old_source = _route_cost(starts[source], source_route, coordinates, depot)
                    new_source = _route_cost(starts[source], reduced, coordinates, depot)
                    delta = new_source - old_source + delta_insert
                    is_tabu = tabu.get((customer, target), -1) > iteration
                    if is_tabu:
                        continue
                    candidate = (delta, customer, source, target, position)
                    if best_move is None or candidate < best_move:
                        best_move = candidate
        if best_move is None or best_move[0] >= -1e-9:
            break
        _, customer, source, target, position = best_move
        routes[source].remove(customer)
        routes[target].insert(position, customer)
        loads[source] -= demands[customer]
        loads[target] += demands[customer]
        tabu[(customer, source)] = iteration + 7
        cost = sum(_route_cost(starts[v], route, coordinates, depot)
                   for v, route in enumerate(routes))
        if cost < best_cost:
            best_cost = cost
            best_routes = [list(route) for route in routes]
    return best_routes


def alns_routes(starts, capacities, customers, coordinates, demands, depot,
                rng, iterations=80):
    current = regret_routes(starts, capacities, customers, coordinates, demands, depot)
    best = [list(route) for route in current]
    best_cost = sum(_route_cost(starts[v], route, coordinates, depot)
                    for v, route in enumerate(best))
    for _ in range(iterations):
        candidate = [list(route) for route in best]
        assigned = [customer for route in candidate for customer in route]
        if len(assigned) < 3:
            break
        destroy_count = max(1, int(round(0.2 * len(assigned))))
        removed = set(rng.choice(assigned, size=destroy_count, replace=False).tolist())
        candidate = [[c for c in route if c not in removed] for route in candidate]
        loads = np.array([sum(demands[c] for c in route) for route in candidate])
        for customer in removed:
            options = []
            for vehicle in range(len(candidate)):
                if loads[vehicle] + demands[customer] > capacities[vehicle] + 1e-9:
                    continue
                delta, position = _best_insertion(
                    starts[vehicle], candidate[vehicle], customer, coordinates, depot
                )
                options.append((delta, vehicle, position))
            if options:
                _, vehicle, position = min(options)
                candidate[vehicle].insert(position, customer)
                loads[vehicle] += demands[customer]
        candidate = [
            _two_opt(route, starts[v], coordinates, depot)
            for v, route in enumerate(candidate)
        ]
        cost = sum(_route_cost(starts[v], route, coordinates, depot)
                   for v, route in enumerate(candidate))
        if cost < best_cost:
            best, best_cost = candidate, cost
    return best


def ortools_routes(starts, capacities, customers, coordinates, demands, depot,
                   time_limit_ms=250):
    if pywrapcp is None:
        raise RuntimeError("OR-Tools is not installed")
    customer_list = list(customers)
    vehicle_count = len(starts)
    points = [depot] + [coordinates[c] for c in customer_list] + list(starts)
    customer_offset = 1
    start_offset = 1 + len(customer_list)
    starts_index = [start_offset + vehicle for vehicle in range(vehicle_count)]
    manager = pywrapcp.RoutingIndexManager(
        len(points), vehicle_count, starts_index, [0] * vehicle_count
    )
    routing = pywrapcp.RoutingModel(manager)
    scale = 100_000

    def distance_callback(from_index, to_index):
        left = manager.IndexToNode(from_index)
        right = manager.IndexToNode(to_index)
        return int(round(scale * _distance(points[left], points[right])))

    distance_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_index)
    node_demand = [0] + [demands[c] for c in customer_list] + [0] * vehicle_count

    def demand_callback(from_index):
        return int(round(scale * node_demand[manager.IndexToNode(from_index)]))

    demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_index,
        0,
        [int(round(scale * capacity)) for capacity in capacities],
        True,
        "Capacity",
    )
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    parameters.time_limit.FromMilliseconds(time_limit_ms)
    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        return regret_routes(starts, capacities, customers, coordinates, demands, depot)
    routes = []
    for vehicle in range(vehicle_count):
        index = routing.Start(vehicle)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if customer_offset <= node < start_offset:
                route.append(customer_list[node - customer_offset])
            index = solution.Value(routing.NextVar(index))
        routes.append(route)
    return routes


PLANNERS = {
    "nearest": nearest_routes,
    "regret": regret_routes,
    "local_search": local_search_routes,
    "tabu": tabu_routes,
    "alns": alns_routes,
    "ortools": ortools_routes,
}


@dataclass
class SimulationResult:
    distance: float
    cost_with_penalty: float
    qos: float
    unserved: int
    response_time_min: float
    completion_time_min: float
    mean_completion_time_min: float
    route_balance_cv: float
    vehicle_utilization: float
    replanning_events: int
    planning_time_ms: float
    late_customers: int
    committed_not_completed: int


def simulate_instance(nodes: np.ndarray, vehicle_count: int, method: str,
                      segments: int = 10, pending_cost: float = 5.0,
                      speed: float = 4.8, seed: int = 0) -> SimulationResult:
    if method not in PLANNERS:
        raise ValueError(f"Unknown planner: {method}")
    rng = np.random.default_rng(seed)
    coordinates = nodes[:, :2]
    demands = nodes[:, 2]
    durations = nodes[:, 3]
    appearance = nodes[:, 4]
    depot = coordinates[0]
    positions = np.repeat(depot[None, :], vehicle_count, axis=0)
    capacities = np.ones(vehicle_count, dtype=float)
    served = np.zeros(len(nodes), dtype=bool)
    served[0] = True
    distance_by_vehicle = np.zeros(vehicle_count, dtype=float)
    response_times = []
    completion_times = []
    busy_time = np.zeros(vehicle_count, dtype=float)
    pending_service = np.full(vehicle_count, -1, dtype=int)
    pending_arrival = np.zeros(vehicle_count, dtype=float)
    pending_departure = np.zeros(vehicle_count, dtype=float)
    planning_time = 0.0
    boundaries = np.linspace(0.0, 1.0, segments + 1)

    planning_starts = list(boundaries[:-1])
    planning_ends = list(boundaries[1:])
    for start_time, end_time in zip(planning_starts, planning_ends):
        available = [
            customer for customer in range(1, len(nodes))
            if (
                not served[customer]
                and customer not in pending_service
                and appearance[customer] <= start_time + 1e-12
            )
        ]
        if not available:
            routes = [[] for _ in range(vehicle_count)]
        else:
            planner = PLANNERS[method]
            planning_capacities = capacities.copy()
            planning_capacities[pending_service >= 0] = 0.0
            started = time.perf_counter()
            if method == "alns":
                routes = planner(
                    positions, planning_capacities, available, coordinates, demands, depot, rng
                )
            else:
                routes = planner(
                    positions, planning_capacities, available, coordinates, demands, depot
                )
            planning_time += time.perf_counter() - started

        for vehicle, route in enumerate(routes):
            current_position = positions[vehicle].copy()
            current_time = start_time
            capacity = capacities[vehicle]
            if pending_service[vehicle] >= 0:
                customer = int(pending_service[vehicle])
                if pending_departure[vehicle] > end_time:
                    busy_time[vehicle] += end_time - start_time
                    continue
                busy_time[vehicle] += pending_departure[vehicle] - start_time
                current_time = pending_departure[vehicle]
                current_position = coordinates[customer].copy()
                capacity -= demands[customer]
                served[customer] = True
                response_times.append(
                    max(0.0, pending_arrival[vehicle] - appearance[customer]) * 480.0
                )
                completion_times.append(current_time * 480.0)
                pending_service[vehicle] = -1
            for customer in route:
                travel = _distance(current_position, coordinates[customer])
                arrival = current_time + travel / speed
                departure = arrival + durations[customer]
                if arrival > end_time + 1e-12:
                    travel_time = max(arrival - current_time, 1.0e-12)
                    fraction = np.clip((end_time - current_time) / travel_time, 0.0, 1.0)
                    distance_by_vehicle[vehicle] += travel * fraction
                    positions[vehicle] = (
                        current_position
                        + fraction * (coordinates[customer] - current_position)
                    )
                    capacities[vehicle] = max(capacity, 0.0)
                    busy_time[vehicle] += max(0.0, end_time - current_time)
                    current_time = end_time
                    break
                if departure > end_time + 1e-12:
                    distance_by_vehicle[vehicle] += travel
                    positions[vehicle] = coordinates[customer].copy()
                    capacities[vehicle] = max(capacity, 0.0)
                    pending_service[vehicle] = customer
                    pending_arrival[vehicle] = arrival
                    pending_departure[vehicle] = departure
                    busy_time[vehicle] += max(0.0, end_time - current_time)
                    current_time = end_time
                    break
                distance_by_vehicle[vehicle] += travel
                busy_time[vehicle] += departure - current_time
                current_position = coordinates[customer]
                current_time = departure
                capacity -= demands[customer]
                served[customer] = True
                response_times.append(max(0.0, arrival - appearance[customer]) * 480.0)
                completion_times.append(departure * 480.0)
            return_distance = _distance(current_position, depot)
            return_time = current_time + return_distance / speed
            if pending_service[vehicle] >= 0 or current_time >= end_time:
                continue
            if return_time <= end_time + 1e-12:
                distance_by_vehicle[vehicle] += return_distance
                busy_time[vehicle] += return_time - current_time
                positions[vehicle] = depot
                capacities[vehicle] = 1.0
            else:
                return_duration = max(return_time - current_time, 1.0e-12)
                fraction = np.clip((end_time - current_time) / return_duration, 0.0, 1.0)
                distance_by_vehicle[vehicle] += return_distance * fraction
                busy_time[vehicle] += max(0.0, end_time - current_time)
                positions[vehicle] = current_position + fraction * (depot - current_position)
                capacities[vehicle] = max(capacity, 0.0)

    unserved = int((~served[1:]).sum())
    distance = float(distance_by_vehicle.sum())
    mean_distance = distance_by_vehicle.mean()
    route_balance = float(distance_by_vehicle.std() / mean_distance) if mean_distance > 0 else 0.0
    return SimulationResult(
        distance=distance,
        cost_with_penalty=distance + pending_cost * unserved,
        qos=float(served[1:].mean()),
        unserved=unserved,
        response_time_min=float(np.mean(response_times)) if response_times else math.nan,
        completion_time_min=float(max(completion_times)) if completion_times else 0.0,
        mean_completion_time_min=(
            float(np.mean(completion_times)) if completion_times else math.nan
        ),
        route_balance_cv=route_balance,
        vehicle_utilization=float(np.clip(busy_time.sum() / vehicle_count, 0.0, 1.0)),
        replanning_events=segments,
        planning_time_ms=planning_time * 1000.0,
        late_customers=unserved,
        committed_not_completed=int((pending_service >= 0).sum()),
    )
