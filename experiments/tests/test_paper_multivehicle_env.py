import numpy as np

from experiments.reviewer_study.paper_multivehicle_env import (
    PaperInstance,
    PaperTimeDrivenFleet,
    Plan,
    generate_release_batch,
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


def test_release_generator_uses_batch_shared_poisson_schedule():
    instances = generate_release_batch(20, 8, 1.0, 1234)
    reference = instances[0].disclosure_minutes
    assert all(
        np.array_equal(reference, instance.disclosure_minutes)
        for instance in instances[1:]
    )
    coordinates = np.stack([instance.coordinates for instance in instances])
    assert coordinates.min() == 0.0
    assert coordinates.max() == 1.0


def test_two_vehicle_dispatch_keeps_independent_routes_and_capacities():
    instance = _instance(
        coordinates=[
            [0.5, 0.5],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        demands=[70, 70, 70, 70],
        services=[10, 10, 10, 10],
    )
    result = PaperTimeDrivenFleet(2).run(instance, nearest_idle_routes)
    assert result.served_customers == 4
    assert len(result.vehicle_customer_counts) == 2
    assert all(count == 2 for count in result.vehicle_customer_counts)
    customer_events = [event for event in result.events if event.customer is not None]
    assert {event.vehicle for event in customer_events} == {0, 1}


def test_customer_leg_started_before_boundary_is_frozen_once():
    instance = _instance(
        coordinates=[[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]],
        demands=[10, 10],
        services=[60, 10],
    )

    def ordered(state):
        return Plan((tuple(state.customer_ids),))

    result = PaperTimeDrivenFleet(1).run(instance, ordered)
    customer_events = [event for event in result.events if event.customer is not None]
    assert [event.customer for event in customer_events] == [0, 1]
    assert customer_events[0].start_minute == 0.0
    assert customer_events[0].completion_minute > 48.0
    assert 48.0 < customer_events[1].start_minute < 96.0


def test_normalized_distance_uses_released_physical_travel_scale():
    instance = _instance(
        coordinates=[[0.0, 0.0], [0.5, 0.0]],
        demands=[10],
        services=[10],
    )
    result = PaperTimeDrivenFleet(1).run(instance, nearest_idle_routes)
    customer = next(event for event in result.events if event.customer is not None)
    assert customer.arrival_minute == 50.0
    assert customer.completion_minute == 60.0


def test_no_customer_is_dispatched_at_or_after_horizon():
    instance = _instance(
        coordinates=[[0.0, 0.0], [0.1, 0.0]],
        demands=[10],
        services=[10],
        disclosures=[480.0],
    )
    result = PaperTimeDrivenFleet(1).run(instance, nearest_idle_routes)
    customer_events = [event for event in result.events if event.customer is not None]
    assert result.served_customers == 1
    assert result.planning_calls == 11
    assert len(customer_events) == 1
    assert customer_events[0].start_minute == 480.0


def test_terminal_boundary_does_not_dispatch_a_second_customer_per_vehicle():
    instance = _instance(
        coordinates=[[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]],
        demands=[10, 10],
        services=[10, 10],
        disclosures=[480.0, 480.0],
    )
    result = PaperTimeDrivenFleet(1).run(instance, nearest_idle_routes)
    customer_events = [event for event in result.events if event.customer is not None]
    assert result.served_customers == 1
    assert len(customer_events) == 1
    assert customer_events[0].start_minute == 480.0


def test_each_vehicle_has_exactly_one_final_depot_return():
    instance = _instance(
        coordinates=[[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]],
        demands=[10, 10],
        services=[10, 10],
    )
    result = PaperTimeDrivenFleet(2).run(instance, nearest_idle_routes)
    returns = [event for event in result.events if event.customer is None]
    assert len(returns) == 2
    assert {event.vehicle for event in returns} == {0, 1}
    assert all(event.interval_index == 11 for event in returns)


def test_depot_return_does_not_reset_single_trip_capacity():
    instance = _instance(
        coordinates=[[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]],
        demands=[100, 100],
        services=[10, 10],
    )
    result = PaperTimeDrivenFleet(1).run(instance, nearest_idle_routes)
    assert result.served_customers == 1
    assert result.unserved_customers == 1
    assert result.qos_percent == 50.0


def test_planner_cannot_collapse_a_multivehicle_state_to_one_route():
    instance = _instance(
        coordinates=[[0.0, 0.0], [0.2, 0.0]],
        demands=[10],
        services=[10],
    )

    def invalid_single_route(state):
        assert state.vehicle_count == 2
        return Plan(((0,),))

    try:
        PaperTimeDrivenFleet(2).run(instance, invalid_single_route)
    except RuntimeError as error:
        assert "routes for 2 vehicles" in str(error)
    else:
        raise AssertionError("single-route plan was accepted for a two-vehicle fleet")
