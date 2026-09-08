import torch
from types import SimpleNamespace

from experiments.reviewer_study.selectors import (
    AttentionAggregationSelector,
    CentralizedSelector,
    EarliestAvailableSelector,
    IndependentSelector,
    OperationalIndependentSelector,
    RoundRobinSelector,
    SingleObjectiveIndependentSelector,
)


def selector_inputs(batch_size=2, vehicle_count=4, node_count=6):
    vehicles = torch.zeros(batch_size, vehicle_count, 4)
    customers = torch.zeros(batch_size, node_count, 5)
    vehicle_done = torch.zeros(batch_size, vehicle_count, dtype=torch.bool)
    customer_mask = torch.zeros(
        batch_size, vehicle_count, node_count, dtype=torch.bool
    )
    return vehicles, customers, vehicle_done, customer_mask


def test_round_robin_resets_for_each_new_rollout():
    selector = RoundRobinSelector(4)
    inputs = selector_inputs()
    sequence = [
        selector.select(*inputs, greedy=True)[0][:, 0].tolist()
        for _ in range(4)
    ]
    assert sequence == [[0, 0], [1, 1], [2, 2], [3, 3]]
    selector.reset_state()
    assert selector.select(*inputs, greedy=True)[0][:, 0].tolist() == [0, 0]


def test_earliest_available_selects_minimum_vehicle_timeline():
    selector = EarliestAvailableSelector(4)
    vehicles, customers, vehicle_done, customer_mask = selector_inputs()
    vehicles[:, :, 3] = torch.tensor([4.0, 2.0, 3.0, 1.0])
    index, _, _ = selector.select(
        vehicles, customers, vehicle_done, customer_mask, greedy=True
    )
    assert index[:, 0].tolist() == [3, 3]


def test_learned_selectors_accept_per_vehicle_customer_masks():
    inputs = selector_inputs()
    for selector in (CentralizedSelector(4), AttentionAggregationSelector(4)):
        scores = selector.scores(*inputs)
        assert scores.shape == (2, 4)
        assert torch.isfinite(scores).all()


def test_operational_selector_uses_timeline_and_cumulative_distance():
    selector = OperationalIndependentSelector(
        4, timeline_weight=10.0, balance_weight=2.0
    )
    for parameter in selector.parameters():
        parameter.data.zero_()
    vehicles, customers, vehicle_done, customer_mask = selector_inputs()
    vehicles[:, :, 3] = torch.tensor([0.4, 0.1, 0.1, 0.2])
    selector.set_operational_environment(
        SimpleNamespace(
            _committed_vehicle_physical_distance=torch.tensor(
                [[0.0, 2.0, 0.5, 0.0], [0.0, 2.0, 0.5, 0.0]]
            ),
            _event_vehicle=[],
            _event_physical_distance=[],
        )
    )
    index, _, _ = selector.select(
        vehicles, customers, vehicle_done, customer_mask, greedy=True
    )
    # Vehicles 1 and 2 have the same earliest timeline; vehicle 2 has the
    # smaller cumulative route and must therefore be selected.
    assert index[:, 0].tolist() == [2, 2]
    selector.clear_operational_environment()


def test_single_objective_selector_preserves_checkpoint_parameter_layout():
    original = IndependentSelector(4)
    refined = SingleObjectiveIndependentSelector(4)
    refined.load_state_dict(original.state_dict(), strict=True)
    assert set(refined.state_dict()) == set(original.state_dict())


def test_single_objective_selector_randomizes_only_training_module_mapping():
    selector = SingleObjectiveIndependentSelector(4)
    selector.train()
    selector.reset_state()
    assert sorted(selector._network_permutation) == [0, 1, 2, 3]
    selector.eval()
    selector.reset_state()
    assert selector._network_permutation == [0, 1, 2, 3]


def test_single_objective_near_tie_uses_earliest_vehicle_but_zero_is_argmax():
    selector = SingleObjectiveIndependentSelector(4, tie_tolerance=1.0)
    selector.scores = lambda *args: torch.tensor(
        [[1.00, 0.95, 0.10, -0.10], [1.00, 0.95, 0.10, -0.10]]
    )
    vehicles, customers, vehicle_done, customer_mask = selector_inputs()
    vehicles[:, :, 3] = torch.tensor([0.4, 0.1, 0.2, 0.3])
    index, _, _ = selector.select(
        vehicles, customers, vehicle_done, customer_mask, greedy=True
    )
    assert index[:, 0].tolist() == [1, 1]

    selector.tie_tolerance = 0.0
    index, _, _ = selector.select(
        vehicles, customers, vehicle_done, customer_mask, greedy=True
    )
    assert index[:, 0].tolist() == [0, 0]
