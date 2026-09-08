import torch

from data import DCVRP_Dataset
from experiments.reviewer_study.periodic_static_dvnda import (
    MANUSCRIPT_NORMALIZED_SPEED,
    VectorizedPeriodicStaticDVNDAEnvironment,
)


class FirstVehicleSelector:
    def select(self, vehicles, customers, vehicle_done, customer_mask, greedy):
        del customers, customer_mask, greedy
        scores = vehicles.new_zeros(vehicles.shape[:2])
        scores = scores.masked_fill(vehicle_done, -torch.inf)
        all_done = vehicle_done.all(dim=1)
        scores[all_done, 0] = 0.0
        index = scores.argmax(dim=1, keepdim=True)
        return index, scores, vehicles.new_zeros((vehicles.size(0), 1))


def make_environment() -> VectorizedPeriodicStaticDVNDAEnvironment:
    nodes = torch.tensor(
        [[
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.4, 0.02, 0.0],
            [0.5, 0.0, 0.2, 0.02, 0.0],
        ]],
        dtype=torch.float32,
    )
    data = DCVRP_Dataset(
        veh_count=1,
        veh_capa=1.0,
        veh_speed=MANUSCRIPT_NORMALIZED_SPEED,
        nodes=nodes,
    )
    environment = VectorizedPeriodicStaticDVNDAEnvironment(data)
    environment.selector = FirstVehicleSelector()
    environment.policy_greedy = True
    environment.reset()
    return environment


def append_event(
    environment: VectorizedPeriodicStaticDVNDAEnvironment,
    *,
    customer: int,
    start: float,
    distance: float,
    state: list[float],
) -> None:
    environment._event_vehicle.append(torch.tensor([[0]]))
    environment._event_customer.append(torch.tensor([[customer]]))
    environment._event_start.append(torch.tensor([[start]]))
    environment._event_distance.append(torch.tensor([[distance]]))
    environment._event_vehicle_state.append(
        torch.tensor([[state]], dtype=torch.float32)
    )


def test_committed_prefix_is_returned_to_depot_and_capacity_is_restored():
    environment = make_environment()
    append_event(
        environment,
        customer=1,
        start=0.02,
        distance=0.3,
        state=[0.3, 0.0, 0.6, 0.05],
    )
    environment.served[0, 1] = True
    environment.total_distance[0] = 0.3

    correction = environment._commit_prefix(0.1)

    torch.testing.assert_close(
        environment.vehicles[0, 0],
        torch.tensor([0.0, 0.0, 1.0, 0.1]),
    )
    torch.testing.assert_close(environment.total_distance, torch.tensor([0.6]))
    torch.testing.assert_close(correction, torch.tensor([[-0.3]]))
    assert int(environment.mandatory_return_count[0, 0]) == 1


def test_unstarted_suffix_is_released_and_never_charged():
    environment = make_environment()
    append_event(
        environment,
        customer=1,
        start=0.02,
        distance=0.3,
        state=[0.3, 0.0, 0.6, 0.05],
    )
    append_event(
        environment,
        customer=2,
        start=0.12,
        distance=0.2,
        state=[0.5, 0.0, 0.4, 0.15],
    )
    environment.served[0, 1:] = True
    environment.total_distance[0] = 0.5

    correction = environment._commit_prefix(0.1)

    assert bool(environment.served[0, 1])
    assert not bool(environment.served[0, 2])
    # 0.3 outbound + 0.3 mandatory return; the 0.2 suffix is refunded.
    torch.testing.assert_close(environment.total_distance, torch.tensor([0.6]))
    torch.testing.assert_close(correction, torch.tensor([[-0.1]]))


def test_normalized_speed_guard_prevents_unit_mismatch():
    nodes = torch.zeros((1, 2, 5), dtype=torch.float32)
    data = DCVRP_Dataset(1, 1.0, 1.0, nodes)
    try:
        VectorizedPeriodicStaticDVNDAEnvironment(data)
    except ValueError as error:
        assert "normalized vehicle speed" in str(error)
    else:
        raise AssertionError("unit-mismatched environment was accepted")

