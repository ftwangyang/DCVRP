from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import DCVRP_Dataset
from experiments.reviewer_study.model import ExperimentalAttentionLearner
from experiments.reviewer_study.paper_dcvrp import (
    PAPER_POISSON_RATE,
    PaperDCVRPEnvironment,
    VectorizedPaperDCVRPEnvironment,
    generate_paired_paper_datasets,
    generate_paper_dataset,
)
from experiments.reviewer_study.paper_greedy import run_paper_greedy
from experiments.reviewer_study.executed_path_dvnda import (
    VectorizedExecutedPathDVNDAEnvironment,
)
from experiments.reviewer_study.selectors import IndependentSelector, SharedSelector


class _FirstVehicleSelector:
    def select(self, vehicles, customers, vehicle_done, customer_mask, greedy):
        del customers, vehicle_done, customer_mask, greedy
        batch_size, vehicle_count, _ = vehicles.shape
        index = torch.zeros((batch_size, 1), dtype=torch.long, device=vehicles.device)
        scores = torch.zeros((batch_size, vehicle_count), device=vehicles.device)
        log_probability = torch.zeros((batch_size, 1), device=vehicles.device)
        return index, scores, log_probability


def _manual_time_driven_environment(nodes: torch.Tensor) -> PaperDCVRPEnvironment:
    data = DCVRP_Dataset(
        veh_count=1,
        veh_capa=1.0,
        veh_speed=1.0,
        nodes=nodes,
    )
    environment = PaperDCVRPEnvironment(data, pending_cost=5.0)
    environment.selector = _FirstVehicleSelector()
    environment.reset()
    return environment


def small_model(greedy: bool = True) -> ExperimentalAttentionLearner:
    model = ExperimentalAttentionLearner(
        selector=IndependentSelector(4, model_size=16, head_count=4),
        model_size=32,
        layer_count=1,
        head_count=4,
        ff_size=64,
    )
    model.greedy = greedy
    return model


def test_exact_dynamic_counts_and_paper_ranges():
    torch.manual_seed(11)
    for rate, expected in [(0.10, 2), (0.25, 5), (0.50, 10), (0.75, 15)]:
        data = generate_paper_dataset(32, rate)
        counts = (data.nodes[:, 1:, 4] > 0).sum(dim=1)
        assert torch.equal(counts, torch.full_like(counts, expected))
        assert 0.0 <= float(data.nodes[:, :, :2].min())
        assert float(data.nodes[:, :, :2].max()) <= 1.0
        demands = data.nodes[:, 1:, 2] * 150.0
        services = data.nodes[:, 1:, 3] * 480.0
        assert float(demands.min()) >= 5.0 and float(demands.max()) <= 41.0
        assert float(services.min()) >= 10.0 and float(services.max()) <= 31.0


def test_poisson_disclosure_mean_matches_stated_rate():
    torch.manual_seed(12)
    data = generate_paper_dataset(1000, 0.75)
    values = data.nodes[:, 1:, 4]
    reveal_minutes = values[values > 0] * 480.0
    assert abs(float(reveal_minutes.mean()) - PAPER_POISSON_RATE) < 3.0


def test_all_customer_slots_use_the_same_scalar_poisson_rate():
    torch.manual_seed(122)
    data = generate_paper_dataset(2000, 1.0)
    reveal_minutes = data.nodes[:, 1:, 4] * 480.0
    first_mean = float(reveal_minutes[:, :5].mean())
    last_mean = float(reveal_minutes[:, -5:].mean())
    assert abs(first_mean - PAPER_POISSON_RATE) < 3.0
    assert abs(last_mean - PAPER_POISSON_RATE) < 3.0
    assert abs(first_mean - last_mean) < 3.0


def test_paired_dynamic_rate_datasets_are_nested_and_physically_identical():
    torch.manual_seed(121)
    paired = generate_paired_paper_datasets(12, (0.10, 0.25, 0.50, 0.75))
    previous_mask = None
    reference = paired[0.75].nodes
    for rate, expected_count in ((0.10, 2), (0.25, 5), (0.50, 10), (0.75, 15)):
        nodes = paired[rate].nodes
        torch.testing.assert_close(nodes[:, :, :4], reference[:, :, :4])
        dynamic_mask = nodes[:, 1:, 4] > 0.0
        assert torch.equal(
            dynamic_mask.sum(dim=1),
            torch.full((12,), expected_count, dtype=torch.int64),
        )
        if previous_mask is not None:
            assert (previous_mask & ~dynamic_mask).sum() == 0
        previous_mask = dynamic_mask


def test_requests_are_revealed_only_at_interval_boundaries():
    nodes = torch.tensor(
        [[
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.1, 0.0, 0.0],
            [0.4, 0.0, 0.1, 0.0, 0.25],
            [0.6, 0.0, 0.1, 0.0, 0.75],
        ]],
        dtype=torch.float32,
    )
    environment = _manual_time_driven_environment(nodes)

    assert environment.cust_mask[0].tolist() == [False, False, True, True]
    environment._segment_transition(0.20)
    assert environment.cust_mask[0].tolist() == [False, False, True, True]
    environment._segment_transition(0.30)
    assert environment.cust_mask[0].tolist() == [False, False, False, True]


def test_depot_is_masked_while_selected_vehicle_has_a_feasible_customer():
    torch.manual_seed(123)
    data = generate_paper_dataset(8, 0.50)
    environment = PaperDCVRPEnvironment(data)
    environment.selector = _FirstVehicleSelector()
    environment.reset()
    has_customer = (~environment.cur_veh_mask[:, :, 1:]).any(dim=2)
    assert torch.equal(environment.cur_veh_mask[:, :, 0], has_customer)


def test_dispatched_prefix_is_committed_and_unstarted_suffix_is_replanned():
    nodes = torch.tensor(
        [[
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.1, 0.0, 0.0],
            [0.6, 0.0, 0.1, 0.0, 0.0],
            [0.9, 0.0, 0.1, 0.0, 0.0],
        ]],
        dtype=torch.float32,
    )
    environment = _manual_time_driven_environment(nodes)
    environment.route_logs[0][0] = [
        {
            "customer": 1,
            "start": 0.00,
            "arrival": 0.10,
            "departure": 0.20,
            "distance": 0.20,
        },
        {
            # Dispatched before the boundary, but still en route/in service at
            # it: Algorithm 1 keeps this customer fixed in C_done.
            "customer": 2,
            "start": 0.20,
            "arrival": 0.60,
            "departure": 0.70,
            "distance": 0.40,
        },
        {
            # start == boundary is not dispatched under the strict '<' rule.
            "customer": 3,
            "start": 0.50,
            "arrival": 0.80,
            "departure": 0.90,
            "distance": 0.30,
        },
    ]
    environment.served[0, 1:] = True

    environment._segment_transition(0.50)

    assert [event["customer"] for event in environment.route_logs[0][0]] == [1, 2]
    assert environment.served[0].tolist() == [False, True, True, False]
    torch.testing.assert_close(
        environment._transition_refund,
        torch.tensor([[0.30]], dtype=environment.nodes.dtype),
    )
    torch.testing.assert_close(environment.vehicles[0, 0, :2], nodes[0, 2, :2])
    torch.testing.assert_close(
        environment.vehicles[0, 0, 3],
        torch.tensor(0.70, dtype=environment.nodes.dtype),
    )


def test_reward_equals_final_route_distance_plus_unserved_penalty():
    torch.manual_seed(13)
    data = generate_paper_dataset(8, 0.75)
    model = small_model(greedy=True)
    environment = PaperDCVRPEnvironment(data, pending_cost=5.0)
    with torch.no_grad():
        _, _, reward_steps = model(environment)
    cost = -torch.stack(reward_steps).sum(dim=0).squeeze(-1)
    pending = (1.0 - environment.qos()) * 20.0
    expected = environment.route_distance() + 5.0 * pending
    torch.testing.assert_close(cost, expected, atol=2.0e-5, rtol=1.0e-5)


def test_vehicle_selector_receives_policy_gradient():
    torch.manual_seed(14)
    data = generate_paper_dataset(8, (0.10, 0.25, 0.50, 0.75))
    model = small_model(greedy=False)
    environment = PaperDCVRPEnvironment(data, pending_cost=5.0)
    _, log_probabilities, reward_steps = model(environment)
    reward = torch.stack(reward_steps).sum(dim=0).squeeze(-1)
    log_probability = torch.stack(log_probabilities).sum(dim=0).squeeze(-1)
    advantage = (reward - reward.mean()).detach()
    loss = -(advantage * log_probability).mean()
    loss.backward()
    gradient = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.selector.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0.0


def test_shared_selector_accepts_latest_per_vehicle_customer_mask():
    torch.manual_seed(140)
    data = generate_paper_dataset(4, (0.10, 0.25, 0.50, 0.75))
    model = ExperimentalAttentionLearner(
        selector=SharedSelector(4, model_size=16, head_count=4),
        model_size=32,
        layer_count=1,
        head_count=4,
        ff_size=64,
    )
    model.greedy = True
    model.vehicle_greedy = True
    environment = VectorizedPaperDCVRPEnvironment(data, pending_cost=5.0)
    with torch.no_grad():
        model(environment)
    assert environment.mask.shape == (4, 4, 21)
    assert torch.isfinite(environment.route_distance()).all()


def test_customer_sampling_can_keep_vehicle_selection_greedy():
    torch.manual_seed(141)
    data = generate_paper_dataset(8, 0.50)
    model = small_model(greedy=False)
    model.vehicle_greedy = True
    environment = PaperDCVRPEnvironment(data, pending_cost=5.0)
    with torch.no_grad():
        model(environment)
    assert environment.policy_greedy is True


def test_paper_greedy_selects_nearest_feasible_customer_first():
    nodes = torch.tensor(
        [[
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.0, 0.1, 0.0, 0.0],
            [0.2, 0.0, 0.1, 0.0, 0.0],
        ]],
        dtype=torch.float32,
    )
    data = DCVRP_Dataset(
        veh_count=1,
        veh_capa=1.0,
        veh_speed=1.0,
        nodes=nodes,
    )
    cost, qos, actions = run_paper_greedy(data, torch.device("cpu"))
    assert actions[0][0][1] == 2
    assert torch.isfinite(cost).all()
    assert 0.0 <= float(qos.item()) <= 1.0


def test_paper_greedy_uses_same_time_driven_cost_accounting():
    torch.manual_seed(142)
    data = generate_paper_dataset(8, (0.10, 0.25, 0.50, 0.75))
    cost, qos, _ = run_paper_greedy(data, torch.device("cpu"))
    assert cost.shape == (8,)
    assert qos.shape == (8,)
    assert torch.isfinite(cost).all()
    assert ((qos >= 0.0) & (qos <= 1.0)).all()


def test_zero_noise_is_seed_invariant():
    torch.manual_seed(15)
    data = generate_paper_dataset(8, 0.50)
    model = small_model(greedy=True)
    model.eval()
    values = []
    for stochastic_seed in (1, 999):
        environment = PaperDCVRPEnvironment(
            data,
            pending_cost=0.0,
            stochastic_seed=stochastic_seed,
        )
        with torch.no_grad():
            _, _, rewards = model(environment)
        values.append(-torch.stack(rewards).sum(dim=0).squeeze(-1))
    torch.testing.assert_close(values[0], values[1], atol=0.0, rtol=0.0)


def test_vectorized_environment_matches_reference_environment():
    torch.manual_seed(16)
    data = generate_paper_dataset(12, (0.10, 0.25, 0.50, 0.75))
    model = small_model(greedy=True)
    model.eval()
    outcomes = []
    for environment_type in (
        PaperDCVRPEnvironment,
        VectorizedPaperDCVRPEnvironment,
    ):
        environment = environment_type(data, pending_cost=5.0)
        with torch.no_grad():
            _, _, rewards = model(environment)
        outcomes.append(
            (
                -torch.stack(rewards).sum(dim=0).squeeze(-1),
                environment.route_distance(),
                environment.qos(),
            )
        )
    for reference, vectorized in zip(outcomes[0], outcomes[1]):
        torch.testing.assert_close(reference, vectorized, atol=2.0e-5, rtol=1.0e-5)


def test_executed_path_protocol_matches_equation_14_and_tracks_destroyed_edges():
    torch.manual_seed(160)
    data = generate_paper_dataset(12, (0.10, 0.25, 0.50, 0.75))
    model = small_model(greedy=True)
    model.eval()
    environment = VectorizedExecutedPathDVNDAEnvironment(
        data, pending_cost=5.0
    )
    with torch.no_grad():
        _, _, rewards = model(environment)

    penalized_cost = -torch.stack(rewards).sum(dim=0).squeeze(-1)
    pending = (1.0 - environment.qos()) * 20.0
    torch.testing.assert_close(
        penalized_cost,
        environment.route_distance() + 5.0 * pending,
        atol=2.0e-5,
        rtol=1.0e-5,
    )
    torch.testing.assert_close(
        environment.proposed_route_distance(),
        environment.route_distance() + environment.discarded_planned_distance,
    )
    assert (environment.discarded_planned_distance >= 0.0).all()
