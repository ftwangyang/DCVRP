import networkx as nx
import torch

from data import DCVRP_Dataset
from experiments.reviewer_study.paper_greedy import EarliestAvailableVehicleSelector
from experiments.reviewer_study.road_data import (
    _fastest_pool_matrices,
    build_graph,
    generate_paired_road_datasets,
)
from experiments.reviewer_study.road_environment import (
    VectorizedRoadPaperDCVRPEnvironment,
)


def test_osm_graph_enforces_oneway_speed_and_drivable_filter():
    payload = {
        "elements": [
            {"type": "node", "id": 1, "lat": 48.0, "lon": 16.0},
            {"type": "node", "id": 2, "lat": 48.0, "lon": 16.001},
            {"type": "node", "id": 3, "lat": 48.001, "lon": 16.001},
            {"type": "node", "id": 4, "lat": 48.001, "lon": 16.0},
            {
                "type": "way",
                "id": 10,
                "nodes": [1, 2],
                "tags": {
                    "highway": "primary",
                    "oneway": "yes",
                    "maxspeed": "30",
                },
            },
            {
                "type": "way",
                "id": 11,
                "nodes": [2, 3],
                "tags": {"highway": "residential", "maxspeed": "50"},
            },
            {
                "type": "way",
                "id": 12,
                "nodes": [3, 1],
                "tags": {
                    "highway": "secondary",
                    "oneway": "yes",
                    "maxspeed": "40",
                },
            },
            {
                "type": "way",
                "id": 13,
                "nodes": [3, 4],
                "tags": {"highway": "footway"},
            },
        ]
    }
    graph = build_graph(payload)
    assert set(graph.nodes) == {1, 2, 3}
    assert graph.has_edge(1, 2)
    assert not graph.has_edge(2, 1)
    assert graph.has_edge(2, 3) and graph.has_edge(3, 2)
    assert graph[1][2]["speed_kph"] == 30.0
    assert graph[1][2]["peak_travel_time_min"] > graph[1][2]["travel_time_min"]


def _ring_graph(node_count=30):
    graph = nx.DiGraph()
    for node in range(node_count):
        graph.add_node(
            node,
            lat=48.0 + 0.001 * (node // 6),
            lon=16.0 + 0.001 * (node % 6),
        )
    for node in range(node_count):
        target = (node + 1) % node_count
        for source, destination in ((node, target), (target, node)):
            graph.add_edge(
                source,
                destination,
                length_km=0.1,
                normalized_length=0.05,
                travel_time_min=0.2,
                peak_travel_time_min=0.26,
                speed_kph=30.0,
                speed_explicit=True,
                highway="residential",
                oneway=False,
                peak_delay=0.3,
            )
    return graph


def test_paired_road_data_keeps_nested_dynamic_sets_and_road_units():
    graph = _ring_graph()
    (
        datasets,
        distance,
        normalized_cost,
        travel,
        peak_distance,
        peak_normalized_cost,
        peak_travel,
        selections,
    ) = (
        generate_paired_road_datasets(
            graph,
            batch_size=2,
            customer_count=20,
            vehicle_count=4,
            dynamic_rates=(0.10, 0.25, 0.50, 0.75),
            seed=7,
        )
    )
    assert distance.shape == (2, 21, 21)
    assert normalized_cost.shape == (2, 21, 21)
    assert travel.shape == (2, 21, 21)
    assert len(selections) == 2 and len(selections[0]) == 21
    assert torch.all(peak_travel >= travel)
    assert torch.all(peak_normalized_cost >= 0)
    assert not bool(torch.all(peak_distance > 0))  # zero diagonal is preserved
    previous = torch.zeros((2, 20), dtype=torch.bool)
    for rate, expected in ((0.10, 2), (0.25, 5), (0.50, 10), (0.75, 15)):
        dynamic = datasets[rate].nodes[:, 1:, 4] > 0
        assert torch.all(dynamic.sum(dim=1) == expected)
        assert torch.all(dynamic | (~previous))
        previous = dynamic
        assert datasets[rate].veh_speed == 480.0


def test_vectorized_road_environment_uses_matrix_and_restores_road_node():
    nodes = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.2, 0.1, 0.01, 0.0],
                [0.8, 0.8, 0.1, 0.01, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    data = DCVRP_Dataset(1, 1.0, 480.0, nodes)
    distance = torch.tensor(
        [[[0.0, 2.0, 8.0], [4.0, 0.0, 3.0], [7.0, 5.0, 0.0]]]
    )
    travel = torch.tensor(
        [[[0.0, 0.01, 0.03], [0.02, 0.0, 0.01], [0.03, 0.02, 0.0]]]
    )
    environment = VectorizedRoadPaperDCVRPEnvironment(
        data,
        nodes=nodes,
        base_distance_matrix=distance,
        base_travel_time_matrix=travel,
        peak_aware=False,
        pending_cost=0.0,
    )
    environment.selector = EarliestAvailableVehicleSelector()
    environment.policy_greedy = True
    environment.reset()
    environment.step(torch.tensor([[1]]))
    environment.step(torch.tensor([[2]]))
    environment.step(torch.tensor([[0]]))
    assert torch.allclose(environment.route_distance(), torch.tensor([5.0]))
    assert torch.allclose(
        environment.route_driving_time_minutes(), torch.tensor([9.6])
    )
    assert int(environment.current_node[0, 0]) == 2
    assert abs(float(environment.vehicles[0, 0, 3]) - 0.1) < 1.0e-6


def test_vectorized_road_operational_metrics_follow_executed_events():
    nodes = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.2, 0.1, 0.01, 0.0],
                [0.8, 0.8, 0.1, 0.02, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    data = DCVRP_Dataset(1, 1.0, 480.0, nodes)
    distance = torch.tensor(
        [[[0.0, 2.0, 8.0], [4.0, 0.0, 3.0], [7.0, 5.0, 0.0]]]
    )
    physical = torch.tensor(
        [[[0.0, 1.0, 4.0], [2.0, 0.0, 1.5], [3.5, 2.5, 0.0]]]
    )
    travel = torch.tensor(
        [[[0.0, 0.01, 0.03], [0.02, 0.0, 0.01], [0.03, 0.02, 0.0]]]
    )
    environment = VectorizedRoadPaperDCVRPEnvironment(
        data,
        nodes=nodes,
        base_distance_matrix=distance,
        base_travel_time_matrix=travel,
        base_physical_distance_matrix=physical,
        peak_aware=False,
        pending_cost=0.0,
    )
    environment.selector = EarliestAvailableVehicleSelector()
    environment.policy_greedy = True
    environment.reset()
    environment.step(torch.tensor([[1]]))
    environment.step(torch.tensor([[2]]))
    environment.step(torch.tensor([[0]]))
    metrics = environment.operational_metrics()
    torch.testing.assert_close(
        metrics["response_wait_time_min"], torch.tensor([9.6])
    )
    torch.testing.assert_close(
        metrics["completion_delay_min"], torch.tensor([16.8])
    )
    torch.testing.assert_close(
        metrics["uncompleted_requests"], torch.tensor([0.0])
    )
    torch.testing.assert_close(
        metrics["route_balance_cv"], torch.tensor([0.0])
    )
    assert 0.0 < float(metrics["vehicle_utilization_percent"].item()) <= 100.0
