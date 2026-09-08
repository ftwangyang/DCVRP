import numpy as np

from experiments.reviewer_study.executed_path_multivehicle_env import (
    INTERVAL_MINUTES,
    ExecutedPathFleet,
    PaperInstance,
    Plan,
    nearest_idle_routes,
)


def instance(coordinates, demands, services, disclosures=None):
    demands = np.asarray(demands, dtype=int)
    if disclosures is None:
        disclosures = np.zeros(demands.size, dtype=float)
    return PaperInstance(
        coordinates=np.asarray(coordinates, dtype=float),
        demands=demands,
        service_minutes=np.asarray(services, dtype=float),
        disclosure_minutes=np.asarray(disclosures, dtype=float),
    )


def test_uses_ten_regular_intervals_and_one_complete_terminal_closure():
    data = instance(
        [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]],
        [10, 10],
        [60, 60],
        [480.0, 480.0],
    )
    result = ExecutedPathFleet(1).run(data, nearest_idle_routes)
    customers = [event for event in result.events if event.customer is not None]
    assert result.planning_calls == 11
    assert [event.customer for event in customers] == [0, 1]
    assert all(event.interval_index == 10 for event in customers)
    assert customers[1].start_minute > 480.0


def test_disclosure_is_not_inserted_inside_interval():
    data = instance(
        [[0.0, 0.0], [0.1, 0.0]], [10], [1], disclosures=[1.0]
    )
    result = ExecutedPathFleet(1).run(data, nearest_idle_routes)
    customer = next(event for event in result.events if event.customer is not None)
    assert customer.interval_index == 1
    assert customer.start_minute == INTERVAL_MINUTES


def test_unstarted_suffix_is_destroyed_and_not_charged_twice():
    data = instance(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]],
        [10, 10],
        [60, 1],
    )
    calls = []

    def ordered(state):
        calls.append(state.customer_ids)
        return Plan((tuple(state.customer_ids),))

    result = ExecutedPathFleet(1).run(data, ordered)
    customer_events = [
        event for event in result.events if event.customer is not None
    ]
    assert [event.customer for event in customer_events] == [0, 1]
    assert 1 in calls[1]
    assert sum(event.distance for event in result.events) == result.cost
    # Executed route: depot->0 (0.1), 0->1 (0.1), 1->depot (0.2).
    assert abs(result.cost - 0.4) < 1.0e-12


def test_committed_route_return_restores_capacity_after_physical_return():
    data = instance(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]],
        [100, 100],
        [1, 1],
    )
    result = ExecutedPathFleet(1).run(data, nearest_idle_routes)
    returns = [event for event in result.events if event.customer is None]
    assert result.served_customers == 2
    assert result.qos_percent == 100.0
    assert len(returns) == 3
    assert [event.interval_index for event in returns] == [0, 1, 11]


def test_normalized_distance_uses_paper_physical_travel_scale():
    data = instance([[0.0, 0.0], [0.5, 0.0]], [10], [10])
    result = ExecutedPathFleet(1).run(data, nearest_idle_routes)
    customer = next(event for event in result.events if event.customer is not None)
    assert customer.arrival_minute == 50.0
    assert customer.completion_minute == 60.0


def test_planner_must_return_one_route_per_vehicle():
    data = instance([[0.0, 0.0], [0.1, 0.0]], [10], [1])

    def invalid(state):
        assert state.vehicle_count == 2
        return Plan(((0,),))

    try:
        ExecutedPathFleet(2).run(data, invalid)
    except RuntimeError as error:
        assert "routes for 2 vehicles" in str(error)
    else:
        raise AssertionError("single-vehicle route was accepted")


def test_every_nonempty_static_problem_requires_real_multivehicle_routes():
    data = instance(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [0.4, 0.0],
            [0.5, 0.0],
        ],
        [10, 10, 10, 10, 10],
        [60, 60, 60, 60, 60],
    )
    flags = []

    def capture(state):
        flags.append(state.require_active_vehicle_routes)
        return nearest_idle_routes(state)

    ExecutedPathFleet(2).run(data, capture)
    assert flags[0] is True
    assert any(flags[1:])
    assert flags[-1] is False
