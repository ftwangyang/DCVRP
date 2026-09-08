import numpy as np

from experiments.reviewer_study.periodic_static_multivehicle_env import (
    INTERVAL_MINUTES,
    PaperInstance,
    PeriodicStaticFleet,
    Plan,
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


def test_ten_regular_intervals_plus_paper_terminal_closure():
    instance = _instance(
        [[0.0, 0.0], [0.1, 0.0]], [10], [10], disclosures=[480.0]
    )
    result = PeriodicStaticFleet(1).run(instance, nearest_idle_routes)
    assert result.planning_calls == 11
    assert result.served_customers == 1
    customer = next(event for event in result.events if event.customer is not None)
    assert customer.interval_index == 10
    assert customer.start_minute == 480.0


def test_new_customer_is_not_inserted_inside_an_interval():
    instance = _instance(
        [[0.0, 0.0], [0.1, 0.0]], [10], [1], disclosures=[1.0]
    )
    result = PeriodicStaticFleet(1).run(instance, nearest_idle_routes)
    customer = next(event for event in result.events if event.customer is not None)
    assert customer.interval_index == 1
    assert customer.start_minute == INTERVAL_MINUTES


def test_periodic_environment_uses_manuscript_coordinate_speed_units():
    instance = _instance([[0.0, 0.0], [0.5, 0.0]], [10], [10])
    result = PeriodicStaticFleet(1).run(instance, nearest_idle_routes)
    customer = next(event for event in result.events if event.customer is not None)
    assert customer.arrival_minute == 0.5
    assert customer.completion_minute == 10.5


def test_dispatched_prefix_is_kept_and_unstarted_suffix_is_replanned():
    instance = _instance(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], [10, 10], [60, 1]
    )
    calls = []

    def ordered(state):
        calls.append(state.customer_ids)
        return Plan((tuple(state.customer_ids),))

    result = PeriodicStaticFleet(1).run(instance, ordered)
    customers = [event for event in result.events if event.customer is not None]
    assert [event.customer for event in customers] == [0, 1]
    assert customers[0].start_minute == 0.0
    assert customers[0].completion_minute > INTERVAL_MINUTES
    assert 1 in calls[1]
    assert customers[1].start_minute >= customers[0].completion_minute


def test_capacity_is_restored_only_after_committed_depot_return():
    instance = _instance(
        [[0.0, 0.0], [0.1, 0.0], [0.1, 0.0]], [100, 100], [1, 1]
    )
    result = PeriodicStaticFleet(1).run(instance, nearest_idle_routes)
    customers = [event for event in result.events if event.customer is not None]
    returns = [event for event in result.events if event.customer is None]
    assert len(customers) == 2
    assert len(returns) == 2
    assert customers[1].start_minute >= returns[0].arrival_minute


def test_each_active_vehicle_has_at_most_one_return_per_interval():
    instance = _instance(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], [10, 10], [1, 1]
    )
    result = PeriodicStaticFleet(2).run(instance, nearest_idle_routes)
    returns = [event for event in result.events if event.customer is None]
    keys = [(event.interval_index, event.vehicle) for event in returns]
    assert len(keys) == len(set(keys))
    assert {event.vehicle for event in returns} == {0, 1}


def test_multivehicle_plan_cannot_be_collapsed_to_one_route():
    instance = _instance([[0.0, 0.0], [0.1, 0.0]], [10], [1])

    def invalid(state):
        assert state.vehicle_count == 2
        return Plan(((0,),))

    try:
        PeriodicStaticFleet(2).run(instance, invalid)
    except RuntimeError as error:
        assert "routes for 2 vehicles" in str(error)
    else:
        raise AssertionError("single-route plan was accepted")
