"""Download and prepare directed OpenStreetMap road networks."""

from __future__ import annotations

import json
import heapq
import math
import re
from pathlib import Path

import networkx as nx
import numpy as np
import requests
import torch

from data import DCVRP_Dataset
from .data_generation import _arrival_values


CITIES = {
    "Vienna": (48.2082, 16.3738),
    "London": (51.5074, -0.1278),
    "New_York": (40.7128, -74.0060),
}


DEFAULT_SPEED_KPH = {
    "motorway": 80.0,
    "motorway_link": 50.0,
    "trunk": 65.0,
    "trunk_link": 45.0,
    "primary": 50.0,
    "primary_link": 40.0,
    "secondary": 40.0,
    "secondary_link": 35.0,
    "tertiary": 35.0,
    "tertiary_link": 30.0,
    "residential": 30.0,
    "living_street": 15.0,
    "service": 20.0,
    "unclassified": 25.0,
}


DRIVABLE_HIGHWAYS = frozenset(DEFAULT_SPEED_KPH)


# Controlled peak-hour factors.  The maximum 45% delay matches the preceding
# uncertainty experiment, while road-class heterogeneity prevents a single
# scalar from being applied to every urban street.
PEAK_DELAY_BY_HIGHWAY = {
    "motorway": 0.45,
    "motorway_link": 0.40,
    "trunk": 0.45,
    "trunk_link": 0.40,
    "primary": 0.45,
    "primary_link": 0.40,
    "secondary": 0.35,
    "secondary_link": 0.32,
    "tertiary": 0.30,
    "tertiary_link": 0.28,
    "residential": 0.20,
    "living_street": 0.15,
    "service": 0.15,
    "unclassified": 0.25,
}


def _haversine_km(left, right):
    lat1, lon1 = map(math.radians, left)
    lat2, lon2 = map(math.radians, right)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(math.sqrt(value))


def _speed(tags, direction: str | None = None):
    directional_key = f"maxspeed:{direction}" if direction else None
    value = tags.get(directional_key) if directional_key else None
    if value is None:
        value = tags.get("maxspeed")
    if isinstance(value, list):
        value = value[0] if value else None
    if value:
        match = re.search(r"(\d+(?:\.\d+)?)", str(value))
        if match:
            speed = float(match.group(1))
            if "mph" in str(value).lower():
                speed *= 1.609344
            return max(speed, 5.0)
    highway = tags.get("highway", "unclassified")
    if isinstance(highway, list):
        highway = highway[0]
    return DEFAULT_SPEED_KPH.get(highway, 25.0)


def _has_explicit_speed(tags: dict, direction: str | None = None) -> bool:
    if direction and tags.get(f"maxspeed:{direction}") is not None:
        return True
    return tags.get("maxspeed") is not None


def _is_drivable(tags: dict) -> bool:
    highway = tags.get("highway")
    if isinstance(highway, list):
        highway = highway[0] if highway else None
    if highway not in DRIVABLE_HIGHWAYS:
        return False
    for key in ("motor_vehicle", "motorcar", "vehicle", "access"):
        value = str(tags.get(key, "")).lower()
        if value in {"no", "private"}:
            return False
    return True


def download_osm_json(city: str, cache_dir: Path, radius_m: int = 1200) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{city.lower()}_{radius_m}m.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    latitude, longitude = CITIES[city]
    query = (
        f"[out:json][timeout:90];"
        f"way[highway](around:{radius_m},{latitude},{longitude});"
        "(._;>;);out;"
    )
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
    ]
    response = None
    errors = []
    for endpoint in endpoints:
        try:
            candidate = requests.get(
                endpoint,
                params={"data": query},
                headers={"User-Agent": "DCVRP-reviewer-study/1.0"},
                timeout=120,
            )
            candidate.raise_for_status()
            response = candidate
            break
        except requests.RequestException as error:
            errors.append(f"{endpoint}: {error}")
    if response is None:
        raise RuntimeError("All Overpass endpoints failed: " + " | ".join(errors))
    payload = response.json()
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def build_graph(payload: dict) -> nx.DiGraph:
    coordinates = {
        element["id"]: (element["lat"], element["lon"])
        for element in payload["elements"]
        if element["type"] == "node"
    }
    graph = nx.DiGraph()
    for node, (latitude, longitude) in coordinates.items():
        graph.add_node(node, lat=latitude, lon=longitude)
    for element in payload["elements"]:
        if element["type"] != "way":
            continue
        tags = element.get("tags", {})
        if not _is_drivable(tags):
            continue
        highway = tags.get("highway")
        if isinstance(highway, list):
            highway = highway[0]
        oneway_value = str(tags.get("oneway", "")).lower()
        implied_oneway = (
            highway in {"motorway", "motorway_link"}
            or str(tags.get("junction", "")).lower() in {"roundabout", "circular"}
        )
        oneway = oneway_value in {"yes", "true", "1", "-1"} or (
            implied_oneway and oneway_value not in {"no", "false", "0"}
        )
        reverse = oneway_value == "-1"
        nodes = element.get("nodes", [])
        for left, right in zip(nodes[:-1], nodes[1:]):
            if left not in coordinates or right not in coordinates:
                continue
            length = _haversine_km(coordinates[left], coordinates[right])

            def add_edge(source: int, target: int, direction: str) -> None:
                speed = _speed(tags, direction)
                travel_minutes = 60.0 * length / speed
                peak_delay = PEAK_DELAY_BY_HIGHWAY.get(str(highway), 0.25)
                edge = {
                    "length_km": length,
                    "travel_time_min": travel_minutes,
                    "peak_travel_time_min": travel_minutes * (1.0 + peak_delay),
                    "speed_kph": speed,
                    "speed_explicit": _has_explicit_speed(tags, direction),
                    "highway": str(highway),
                    "oneway": oneway,
                    "peak_delay": peak_delay,
                }
                if (
                    not graph.has_edge(source, target)
                    or travel_minutes < graph[source][target]["travel_time_min"]
                ):
                    graph.add_edge(source, target, **edge)

            if reverse:
                add_edge(right, left, "backward")
            else:
                add_edge(left, right, "forward")
                if not oneway:
                    add_edge(right, left, "backward")
    components = list(nx.strongly_connected_components(graph))
    if not components:
        raise RuntimeError("Downloaded road graph has no strongly connected component")
    largest = max(components, key=len)
    graph = graph.subgraph(largest).copy()
    latitudes = [float(data["lat"]) for _, data in graph.nodes(data=True)]
    longitudes = [float(data["lon"]) for _, data in graph.nodes(data=True)]
    latitude_span = max(max(latitudes) - min(latitudes), 1.0e-12)
    longitude_span = max(max(longitudes) - min(longitudes), 1.0e-12)
    # The manuscript represents coordinates in a [0,1] x [0,1] square.  Keep
    # that unit system for the reported objective, but accumulate edge lengths
    # along directed road paths so that topology is not discarded.
    for source, target, edge in graph.edges(data=True):
        delta_latitude = (
            float(graph.nodes[target]["lat"])
            - float(graph.nodes[source]["lat"])
        ) / latitude_span
        delta_longitude = (
            float(graph.nodes[target]["lon"])
            - float(graph.nodes[source]["lon"])
        ) / longitude_span
        edge["normalized_length"] = math.hypot(
            delta_latitude, delta_longitude
        )
    graph.graph["latitude_span"] = latitude_span
    graph.graph["longitude_span"] = longitude_span
    return graph


def _pool_matrices(graph: nx.DiGraph, pool: list[int]):
    size = len(pool)
    length = np.full((size, size), np.inf, dtype=np.float32)
    travel = np.full((size, size), np.inf, dtype=np.float32)
    for source_index, source in enumerate(pool):
        length_paths = nx.single_source_dijkstra_path_length(
            graph, source, weight="length_km"
        )
        travel_paths = nx.single_source_dijkstra_path_length(
            graph, source, weight="travel_time_min"
        )
        for target_index, target in enumerate(pool):
            length[source_index, target_index] = length_paths[target]
            travel[source_index, target_index] = travel_paths[target]
    return length, travel


def _single_source_fastest(
    graph: nx.DiGraph,
    source: int,
    time_attribute: str,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Return fastest time plus physical and paper-scale path lengths."""

    best_time = {source: 0.0}
    path_distance_km = {source: 0.0}
    path_normalized_cost = {source: 0.0}
    queue = [(0.0, 0.0, 0.0, source)]
    while queue:
        current_time, current_cost, current_distance_km, node = heapq.heappop(
            queue
        )
        if current_time > best_time.get(node, math.inf) + 1.0e-12:
            continue
        for target, edge in graph[node].items():
            candidate_time = current_time + float(edge[time_attribute])
            candidate_distance_km = current_distance_km + float(
                edge["length_km"]
            )
            candidate_cost = current_cost + float(edge["normalized_length"])
            known_time = best_time.get(target, math.inf)
            known_cost = path_normalized_cost.get(target, math.inf)
            if candidate_time < known_time - 1.0e-12 or (
                abs(candidate_time - known_time) <= 1.0e-12
                and candidate_cost < known_cost
            ):
                best_time[target] = candidate_time
                path_distance_km[target] = candidate_distance_km
                path_normalized_cost[target] = candidate_cost
                heapq.heappush(
                    queue,
                    (
                        candidate_time,
                        candidate_cost,
                        candidate_distance_km,
                        target,
                    ),
                )
    return best_time, path_distance_km, path_normalized_cost


def _fastest_pool_matrices(graph: nx.DiGraph, pool: list[int]):
    """Build base/peak matrices on the directed fastest path for each OD pair."""

    size = len(pool)
    base_distance = np.full((size, size), np.inf, dtype=np.float32)
    base_normalized_cost = np.full((size, size), np.inf, dtype=np.float32)
    base_travel = np.full((size, size), np.inf, dtype=np.float32)
    peak_distance = np.full((size, size), np.inf, dtype=np.float32)
    peak_normalized_cost = np.full((size, size), np.inf, dtype=np.float32)
    peak_travel = np.full((size, size), np.inf, dtype=np.float32)
    for source_index, source in enumerate(pool):
        base_times, base_distances, base_costs = _single_source_fastest(
            graph, source, "travel_time_min"
        )
        peak_times, peak_distances, peak_costs = _single_source_fastest(
            graph, source, "peak_travel_time_min"
        )
        for target_index, target in enumerate(pool):
            base_travel[source_index, target_index] = base_times[target]
            base_distance[source_index, target_index] = base_distances[target]
            base_normalized_cost[source_index, target_index] = base_costs[target]
            peak_travel[source_index, target_index] = peak_times[target]
            peak_distance[source_index, target_index] = peak_distances[target]
            peak_normalized_cost[source_index, target_index] = peak_costs[target]
    return (
        base_distance,
        base_normalized_cost,
        base_travel,
        peak_distance,
        peak_normalized_cost,
        peak_travel,
    )


def generate_paired_road_datasets(
    graph: nx.DiGraph,
    batch_size: int,
    customer_count: int,
    vehicle_count: int,
    dynamic_rates: list[float] | tuple[float, ...],
    seed: int,
):
    """Generate common-random-number city instances for all dynamic rates.

    Coordinates are normalized only as neural-network features.  Objective
    distance and vehicle travel time remain the directed OSM shortest-path
    quantities returned alongside the datasets.
    """

    rates = tuple(sorted(float(rate) for rate in dynamic_rates))
    if not rates or rates[0] < 0.0 or rates[-1] > 1.0:
        raise ValueError("dynamic rates must be a non-empty subset of [0, 1]")
    rng = np.random.default_rng(seed)
    graph_nodes = np.array(list(graph.nodes), dtype=object)
    pool_size = min(
        len(graph_nodes), max(100, batch_size + customer_count * 2)
    )
    pool = rng.choice(graph_nodes, size=pool_size, replace=False).tolist()
    (
        base_distance_pool,
        base_normalized_cost_pool,
        base_travel_pool,
        peak_distance_pool,
        peak_normalized_cost_pool,
        peak_travel_pool,
    ) = _fastest_pool_matrices(graph, pool)

    graph_coordinates = np.array(
        [[graph.nodes[node]["lat"], graph.nodes[node]["lon"]] for node in graph],
        dtype=np.float32,
    )
    coordinate_minimum = graph_coordinates.min(axis=0)
    coordinate_scale = np.maximum(
        graph_coordinates.max(axis=0) - coordinate_minimum, 1.0e-9
    )

    selections: list[list[int]] = []
    normalized_locations = []
    demands = []
    service_minutes = []
    disclosure_minutes = []
    dynamic_orders = []
    all_base_distance = []
    all_base_normalized_cost = []
    all_base_travel = []
    all_peak_distance = []
    all_peak_normalized_cost = []
    all_peak_travel = []
    poisson_means = np.linspace(1.0, 480.0, customer_count)

    for _ in range(batch_size):
        selection = rng.choice(
            pool_size, size=customer_count + 1, replace=False
        )
        chosen = [pool[index] for index in selection]
        latlon = np.array(
            [[graph.nodes[node]["lat"], graph.nodes[node]["lon"]] for node in chosen],
            dtype=np.float32,
        )
        normalized_locations.append(
            ((latlon - coordinate_minimum) / coordinate_scale).astype(np.float32)
        )
        demands.append(rng.integers(5, 42, size=customer_count).astype(np.float32))
        service_minutes.append(
            rng.integers(10, 32, size=customer_count).astype(np.float32)
        )
        disclosure_minutes.append(
            np.clip(rng.poisson(poisson_means), 1, 480).astype(np.float32)
        )
        dynamic_orders.append(rng.permutation(customer_count))
        all_base_distance.append(
            base_distance_pool[np.ix_(selection, selection)]
        )
        all_base_normalized_cost.append(
            base_normalized_cost_pool[np.ix_(selection, selection)]
        )
        all_base_travel.append(base_travel_pool[np.ix_(selection, selection)])
        all_peak_distance.append(
            peak_distance_pool[np.ix_(selection, selection)]
        )
        all_peak_normalized_cost.append(
            peak_normalized_cost_pool[np.ix_(selection, selection)]
        )
        all_peak_travel.append(peak_travel_pool[np.ix_(selection, selection)])
        selections.append(chosen)

    normalized_locations = np.stack(normalized_locations)
    demands = np.stack(demands)
    service_minutes = np.stack(service_minutes)
    disclosure_minutes = np.stack(disclosure_minutes)
    dynamic_orders = np.stack(dynamic_orders)
    datasets: dict[float, DCVRP_Dataset] = {}
    for rate in rates:
        dynamic_count = int(round(customer_count * rate))
        appearance = np.zeros(
            (batch_size, customer_count), dtype=np.float32
        )
        for batch_index in range(batch_size):
            dynamic_indices = dynamic_orders[batch_index, :dynamic_count]
            appearance[batch_index, dynamic_indices] = disclosure_minutes[
                batch_index, dynamic_indices
            ]
        customer_features = np.concatenate(
            [
                normalized_locations[:, 1:, :],
                (demands / 150.0)[..., None],
                (service_minutes / 480.0)[..., None],
                (appearance / 480.0)[..., None],
            ],
            axis=2,
        ).astype(np.float32)
        depot = np.zeros((batch_size, 1, 5), dtype=np.float32)
        depot[:, 0, :2] = normalized_locations[:, 0, :]
        nodes = torch.tensor(
            np.concatenate([depot, customer_features], axis=1),
            dtype=torch.float32,
        )
        data = DCVRP_Dataset(
            vehicle_count,
            1.0,
            480.0,
            nodes,
        )
        data.paper_dynamic_rates = torch.full((batch_size,), rate)
        data.paper_dynamic_counts = torch.full(
            (batch_size,), dynamic_count, dtype=torch.int64
        )
        datasets[rate] = data

    return (
        datasets,
        torch.tensor(np.stack(all_base_distance), dtype=torch.float32),
        torch.tensor(np.stack(all_base_normalized_cost), dtype=torch.float32),
        torch.tensor(np.stack(all_base_travel) / 480.0, dtype=torch.float32),
        torch.tensor(np.stack(all_peak_distance), dtype=torch.float32),
        torch.tensor(np.stack(all_peak_normalized_cost), dtype=torch.float32),
        torch.tensor(np.stack(all_peak_travel) / 480.0, dtype=torch.float32),
        selections,
    )


def generate_road_dataset(
    graph: nx.DiGraph,
    batch_size: int,
    customer_count: int,
    vehicle_count: int,
    dynamic_ratio: float,
    seed: int,
    arrival_profile: str = "uniform",
):
    rng = np.random.default_rng(seed)
    graph_nodes = np.array(list(graph.nodes), dtype=object)
    pool_size = min(len(graph_nodes), max(100, batch_size + customer_count * 2))
    pool = rng.choice(graph_nodes, size=pool_size, replace=False).tolist()
    length_pool, travel_pool = _pool_matrices(graph, pool)
    all_nodes = []
    all_length = []
    all_travel = []
    selections = []
    dynamic_count = int(round(customer_count * dynamic_ratio))
    for _ in range(batch_size):
        selection = rng.choice(pool_size, size=customer_count + 1, replace=False)
        chosen = [pool[index] for index in selection]
        latlon = np.array([
            [graph.nodes[node]["lat"], graph.nodes[node]["lon"]] for node in chosen
        ], dtype=np.float32)
        minimum = latlon.min(axis=0)
        scale = np.maximum(latlon.max(axis=0) - minimum, 1.0e-9)
        normalized = (latlon - minimum) / scale
        demands = rng.integers(5, 41, size=(customer_count, 1)) / 150.0
        durations = rng.integers(10, 31, size=(customer_count, 1)) / 480.0
        appearance = np.zeros((customer_count, 1), dtype=np.float32)
        dynamic_index = rng.choice(customer_count, size=dynamic_count, replace=False)
        appearance[dynamic_index, 0] = _arrival_values(
            rng, dynamic_count, arrival_profile
        )
        customers = np.concatenate(
            [normalized[1:], demands, durations, appearance], axis=1
        ).astype(np.float32)
        depot = np.zeros((1, 5), dtype=np.float32)
        depot[0, :2] = normalized[0]
        all_nodes.append(np.concatenate([depot, customers], axis=0))
        all_length.append(length_pool[np.ix_(selection, selection)])
        all_travel.append(travel_pool[np.ix_(selection, selection)] / 480.0)
        selections.append(chosen)
    dataset = DCVRP_Dataset(
        vehicle_count,
        1.0,
        1.0,
        torch.tensor(np.stack(all_nodes), dtype=torch.float32),
    )
    return (
        dataset,
        torch.tensor(np.stack(all_length), dtype=torch.float32),
        torch.tensor(np.stack(all_travel), dtype=torch.float32),
        selections,
    )


def graph_metadata(city: str, graph: nx.DiGraph, payload: dict) -> dict:
    one_way_edges = sum(bool(data.get("oneway")) for _, _, data in graph.edges(data=True))
    speed_values = [data["speed_kph"] for _, _, data in graph.edges(data=True)]
    return {
        "city": city,
        "nodes": graph.number_of_nodes(),
        "directed_edges": graph.number_of_edges(),
        "oneway_edge_fraction": one_way_edges / max(graph.number_of_edges(), 1),
        "explicit_speed_edge_fraction": sum(
            bool(data.get("speed_explicit")) for _, _, data in graph.edges(data=True)
        ) / max(graph.number_of_edges(), 1),
        "speed_min_kph": min(speed_values),
        "speed_median_kph": float(np.median(speed_values)),
        "speed_max_kph": max(speed_values),
        "peak_delay_min": min(
            data["peak_delay"] for _, _, data in graph.edges(data=True)
        ),
        "peak_delay_max": max(
            data["peak_delay"] for _, _, data in graph.edges(data=True)
        ),
        "coordinate_normalization": "fixed city bounding box to [0,1]^2",
        "osm_timestamp": payload.get("osm3s", {}).get("timestamp_osm_base"),
    }
