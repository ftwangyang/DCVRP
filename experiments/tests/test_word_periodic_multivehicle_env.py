import numpy as np

from experiments.reviewer_study.word_periodic_multivehicle_env import (
    INTERVAL_MINUTES,
    PaperInstance,
    Plan,
    WordPeriodicStaticFleet,
    nearest_idle_routes,
)


def _instance(coordinates, demands, services, disclosures=None):
    demands = np.asarray(demands, dtype=int)
    if disclosures is None:
        disclosures = np.zeros(demands.size, dtype=float)
    return PaperInstance(
        coordinates=np.asarray(coordinates, dtype=float),
        demands=demands,
        service_minutes=np.asarray(services, dtype=float),
        disclosure_minutes=np.asarray(disclosures, dtype=float),
    )


def test_word_environment_has_ten_intervals_and_terminal_static_solve():
    data = _instance(
        [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]],
        [10, 10],
        [60, 60],
        [480.0, 480.0],
    )
    result = WordPeriodicStaticFleet(1).run(data, nearest_idle_routes)
    customers = [event for event in result.events if event.customer is not None]
    assert result.planning_calls == 11
    assert [event.customer for event in customers] == [0, 1]
    assert all(event.interval_index == 10 for event in customers)


def test_disclosure_waits_for_the_next_synchronized_boundary():
    data = _instance(
        [[0.0, 0.0], [0.1, 0.0]], [10], [1], disclosures=[1.0]
    )
    result = WordPeriodicStaticFleet(1).run(data, nearest_idle_routes)
    customer = next(event for event in result.events if event.customer is not None)
    assert customer.interval_index == 1
    assert customer.start_minute == INTERVAL_MINUTES


def test_complete_static_plan_is_requested_then_suffix_is_destroyed():
    data = _instance(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]],
        [10, 10],
        [60, 1],
    )
    calls = []

    def ordered(state):
        calls.append((state.customer_ids, state.optimize_complete_static_route))
        return Plan((tuple(state.customer_ids),))

    result = WordPeriodicStaticFleet(1).run(data, ordered)
    customers = [event for event in result.events if event.customer is not None]
    assert all(complete for _, complete in calls)
    assert [event.customer for event in customers] == [0, 1]
    assert 1 in calls[1][0]


def test_one_real_return_per_active_vehicle_and_interval_is_charged():
    data = _instance(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]],
        [10, 10],
        [1, 1],
    )
    result = WordPeriodicStaticFleet(2).run(data, nearest_idle_routes)
    returns = [event for event in result.events if event.customer is None]
    keys = [(event.interval_index, event.vehicle) for event in returns]
    assert len(keys) == len(set(keys))
    assert {event.vehicle for event in returns} == {0, 1}
    assert abs(result.cost - sum(event.distance for event in result.events)) < 1e-12


def test_capacity_is_replenished_only_after_physical_depot_return():
    data = _instance(
        [[0.0, 0.0], [0.1, 0.0], [0.1, 0.0]],
        [100, 100],
        [1, 1],
    )
    result = WordPeriodicStaticFleet(1).run(data, nearest_idle_routes)
    customers = [event for event in result.events if event.customer is not None]
    returns = [event for event in result.events if event.customer is None]
    assert len(customers) == 2
    assert len(returns) == 2
    assert customers[1].start_minute >= returns[0].arrival_minute


def test_multivehicle_plan_cannot_be_collapsed_to_one_route():
    data = _instance([[0.0, 0.0], [0.1, 0.0]], [10], [1])

    def invalid(state):
        assert state.vehicle_count == 2
        return Plan(((0,),))

    try:
        WordPeriodicStaticFleet(2).run(data, invalid)
    except RuntimeError as error:
        assert "routes for 2 vehicles" in str(error)
    else:
        raise AssertionError("single-vehicle route was accepted")


def test_first_stage_requires_all_four_routes_but_later_stages_do_not():
    data = _instance(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.0, 0.1],
            [0.0, 0.2],
        ],
        [10, 10, 10, 10],
        [1, 1, 1, 1],
    )
    seen = []

    def planned(state):
        seen.append(state.require_active_vehicle_routes)
        if state.customer_ids:
            routes = tuple((customer,) for customer in state.customer_ids)
            routes += tuple(() for _ in range(state.vehicle_count - len(routes)))
            return Plan(routes)
        return Plan(tuple(() for _ in range(state.vehicle_count)))

    WordPeriodicStaticFleet(4).run(data, planned)
    assert seen[0] is True
    assert all(value is False for value in seen[1:])
