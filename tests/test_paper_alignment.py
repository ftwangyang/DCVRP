"""Checks that the learner matches the manuscript decision rules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import DCVRPEnvironment, generate_dataset, set_seed
from env.dataset import DCVRPDataset
from models import AttentionLearner, build_selector
from train import repeat_rollout_batch


class _FixedSelector:
    def select(self, vehicles, customers, vehicle_done, customer_mask, greedy):
        del customers, vehicle_done, customer_mask, greedy
        index = torch.zeros(vehicles.size(0), 1, dtype=torch.long, device=vehicles.device)
        return index, vehicles.new_zeros(vehicles.size(0), vehicles.size(1)), vehicles.new_zeros(vehicles.size(0), 1)


def test_parallel_rollout_batch_preserves_instance_order_and_metadata():
    data = generate_dataset(batch_size=3, customer_count=20, dynamic_rate=0.5, seed=9)
    repeated = repeat_rollout_batch(data, 3)
    assert repeated.batch_size == 9
    assert repeated.veh_count == data.veh_count
    assert repeated.veh_capa == data.veh_capa
    assert repeated.veh_speed == data.veh_speed
    for row in range(3):
        expected = data.nodes[row : row + 1].expand(3, -1, -1)
        torch.testing.assert_close(repeated.nodes[row * 3 : (row + 1) * 3], expected)


def _build_model(method: str = "DVNDA", vehicle_count: int = 4, greedy: bool = True):
    selector = build_selector(method, vehicle_count=vehicle_count)
    model = AttentionLearner(selector)
    model.eval()
    model.greedy = greedy
    model.vehicle_greedy = True
    return model


def test_vehicle_selector_is_argmax():
    set_seed(0)
    selector = build_selector("DVNDA", vehicle_count=4)
    vehicles = torch.rand(6, 4, 4)
    customers = torch.rand(6, 21, 5)
    vehicle_done = torch.zeros(6, 4, dtype=torch.bool)
    customer_mask = torch.zeros(6, 21, dtype=torch.bool)
    idx_a, logits, _ = selector.select(
        vehicles, customers, vehicle_done, customer_mask, greedy=False
    )
    idx_b, _, _ = selector.select(
        vehicles, customers, vehicle_done, customer_mask, greedy=False
    )
    assert torch.equal(idx_a, idx_b)
    assert torch.equal(idx_a.squeeze(1), logits.argmax(dim=1))


def test_reward_matches_equation_14():
    set_seed(1)
    data = generate_dataset(batch_size=8, customer_count=20, vehicle_count=4, dynamic_rate=0.50)
    model = _build_model(greedy=True)
    env = DCVRPEnvironment(data, pending_cost=5.0)
    _, _, rewards = model(env)
    total_reward = torch.stack(rewards).sum(0).squeeze(-1)
    pending = (~env.served).float().sum(-1) - 1.0
    target = -(env.route_distance() + env.pending_cost * pending)
    err = (total_reward - target).abs().max().item()
    assert err < 1e-5, err
    assert env._finalized


def test_final_routes_return_to_depot():
    set_seed(2)
    data = generate_dataset(batch_size=4, customer_count=20, vehicle_count=4, dynamic_rate=0.25)
    model = _build_model(greedy=True)
    env = DCVRPEnvironment(data, pending_cost=0.0)
    model(env)
    extra = env._depot_return_distance()
    assert float(extra.max()) < 1e-5


def test_vehicle_selector_has_gradient():
    set_seed(3)
    data = generate_dataset(batch_size=4, customer_count=20, vehicle_count=4, dynamic_rate=0.25)
    model = _build_model(greedy=False)
    model.train()
    env = DCVRPEnvironment(data, pending_cost=5.0)
    _, logps, rewards = model(env)
    loss = -(torch.stack(logps).sum(0) * torch.stack(rewards).sum(0).detach()).mean()
    loss.backward()
    grads = [
        p.grad.abs().sum().item()
        for n, p in model.named_parameters()
        if p.grad is not None and "selector" in n
    ]
    assert grads and max(grads) > 0.0


def test_algorithm1_boundary_clocks_are_synchronized():
    """After Algorithm 1, every vehicle clock equals T_{r+1}, not the planning time."""
    set_seed(5)
    data = generate_dataset(batch_size=8, customer_count=20, vehicle_count=4, dynamic_rate=0.50)
    model = _build_model(greedy=True)
    env = DCVRPEnvironment(data, pending_cost=0.0)
    model(env)
    assert env.boundary_clocks, "interval execution never recorded a boundary"
    for i, clocks in enumerate(env.boundary_clocks):
        expected = (i + 1) * env.segment_duration
        assert float(clocks.min()) + 1e-6 >= expected, (
            f"boundary {i + 1}: min clock {clocks.min().item():.4f} < {expected:.4f}"
        )


def test_algorithm1_interpolates_en_route_position():
    """In-progress work at T_i is kept and the vehicle is placed at that customer."""
    nodes = torch.zeros(1, 2, 5)
    nodes[0, 1, 0] = 0.5
    nodes[0, 1, 2] = 0.2
    nodes[0, 1, 3] = 0.01
    data = DCVRPDataset(vehicle_count=1, vehicle_capacity=1.0, vehicle_speed=1.0, nodes=nodes)
    env = DCVRPEnvironment(data, pending_cost=0.0)
    env.selector = _FixedSelector()
    env.reset()
    env.step(torch.tensor([[1]]))
    env.step(torch.tensor([[0]]))
    assert env.current_segment == 1
    assert float(env.vehicles[0, 0, 3]) + 1e-6 >= 0.1
    assert abs(float(env.vehicles[0, 0, 0]) - 0.5) < 1e-5
    assert bool(env.served[0, 1])
    assert not bool(env._residual_active[0, 0])


def test_algorithm1_does_not_keep_planning_clock():
    """The first boundary is T=0.1 even if planned routes run past that time."""
    set_seed(6)
    data = generate_dataset(batch_size=4, customer_count=20, vehicle_count=4, dynamic_rate=0.75)
    model = _build_model(greedy=True)
    env = DCVRPEnvironment(data, pending_cost=0.0)
    model(env)
    first = env.boundary_clocks[0]
    assert float(first.min()) >= 0.1 - 1e-6


def test_five_environment_knobs():
    """The shared generator used by all 5 methods follows the Table I protocol."""
    set_seed(11)
    data = generate_dataset(batch_size=64, customer_count=20, vehicle_count=4, dynamic_rate=0.50)
    assert abs(data.veh_speed - 480.0) < 1e-4, data.veh_speed
    assert float(data.nodes[:, :, :2].min()) >= -1e-6
    assert float(data.nodes[:, :, :2].max()) <= 1.0 + 1e-6
    raw_demand = data.nodes[:, 1:, 2] * 150.0
    raw_service = data.nodes[:, 1:, 3] * 480.0
    assert float(raw_demand.min()) >= 5.0 - 1e-4
    assert float(raw_demand.max()) <= 41.0 + 1e-4
    assert float(raw_service.min()) >= 10.0 - 1e-4
    assert float(raw_service.max()) <= 31.0 + 1e-4
    dynamic_frac = float((data.nodes[:, 1:, 4] > 0).float().mean())
    assert abs(dynamic_frac - 0.50) < 1e-6, dynamic_frac


def test_interval_uses_the_fleet_not_one_vehicle():
    """A vehicle that cannot start before T_{r+1} must release leftover customers."""

    class _FirstFreeSelector:
        def select(self, vehicles, customers, vehicle_done, customer_mask, greedy):
            del customers, customer_mask, greedy
            scores = (~vehicle_done).float()
            index = scores.argmax(dim=1, keepdim=True)
            logp = vehicles.new_zeros(vehicles.size(0), 1)
            return index, scores, logp

    nodes = torch.zeros(1, 9, 5)
    nodes[0, 1:, 0] = torch.linspace(0.1, 0.8, 8)
    nodes[0, 1:, 2] = 0.4
    nodes[0, 1:, 3] = 0.04
    data = DCVRPDataset(
        vehicle_count=4, vehicle_capacity=1.0, vehicle_speed=480.0, nodes=nodes
    )
    env = DCVRPEnvironment(data, pending_cost=0.0)
    env.selector = _FirstFreeSelector()
    env.reset()
    guard = 0
    while env.current_segment == 0:
        mask = env.cur_veh_mask[0, 0]
        available = (~mask).nonzero(as_tuple=False).flatten()
        customers = available[available > 0]
        choice = customers[0] if customers.numel() else available.new_zeros(1)
        env.step(choice.view(1, 1))
        guard += 1
        assert guard < 40
    used = int(env._committed_customer_vehicle_mask[0, :, 1:].any(dim=1).sum().item())
    assert used >= 2, used


def test_dynamic_revelation_is_spread_across_horizon():
    """HPP arrivals occupy the whole day, not a Poisson(240.5) mid-horizon clump."""
    set_seed(12)
    data = generate_dataset(batch_size=256, customer_count=20, vehicle_count=4, dynamic_rate=1.0)
    minutes = data.nodes[:, 1:, 4] * 480.0
    mean = float(minutes.mean())
    std = float(minutes.std())
    assert 220.0 < mean < 260.0, mean
    assert 120.0 < std < 160.0, std
    interval = torch.clamp((minutes / 48.0).long(), max=9)
    counts = torch.bincount(interval.flatten(), minlength=10).float()
    shares = counts / counts.sum()
    assert float(shares.min()) > 0.06
    assert float(shares.max()) < 0.14


def test_last_interval_reveals_tail_arrivals():
    """HPP customers in (T_9, T] become visible when the last interval starts."""
    nodes = torch.zeros(1, 3, 5)
    nodes[0, 1] = torch.tensor([0.1, 0.0, 0.2, 0.01, 0.0])
    nodes[0, 2] = torch.tensor([0.2, 0.0, 0.2, 0.01, 0.95])
    env = DCVRPEnvironment(
        DCVRPDataset(1, 1.0, 480.0, nodes), pending_cost=0.0, record_trace=False
    )
    env.selector = _FixedSelector()
    env.reset()
    assert bool(env.cust_mask[0, 2])
    for i in range(8):
        env._segment_transition((i + 1) * 0.1)
    assert env.current_segment == 8
    assert bool(env.cust_mask[0, 2])
    env._segment_transition(0.9)
    assert env.current_segment == 9
    assert not bool(env.cust_mask[0, 2])


def test_first_interval_hides_dynamic_customers():
    set_seed(8)
    data = generate_dataset(batch_size=4, customer_count=20, vehicle_count=4, dynamic_rate=0.25)
    env = DCVRPEnvironment(data, pending_cost=0.0)
    model = _build_model(greedy=True)
    env.selector = model.selector
    env.reset()
    hidden = env.cust_mask[0, 1:]
    dynamic = data.nodes[0, 1:, 4] > 0
    assert bool(hidden[dynamic].all())
    assert bool((~hidden[~dynamic]).all())


def test_service_waits_until_appearance_time():
    """Visible last-window customers cannot be served before a_i."""
    nodes = torch.zeros(1, 2, 5)
    nodes[0, 1] = torch.tensor([0.1, 0.0, 0.2, 0.05, 0.5])
    env = DCVRPEnvironment(
        DCVRPDataset(1, 1.0, 480.0, nodes), pending_cost=0.0, record_trace=False
    )
    env.selector = _FixedSelector()
    env.reset()
    env.cust_mask[0, 1] = False
    env.total_cust_mask[:, :, 1] = False
    env._rebuild_mask()
    env._update_cur_veh()
    env.step(torch.tensor([[1]]))
    assert abs(float(env.vehicles[0, 0, 3]) - 0.55) < 1e-4


def test_high_dynamism_keeps_max_demand_slack_on_another_vehicle():
    """No-refill: do not empty a vehicle if an emptier one can take the request."""
    nodes = torch.zeros(1, 4, 5)
    nodes[0, 1] = torch.tensor([0.2, 0.0, 0.20, 0.01, 0.0])
    nodes[0, 2] = torch.tensor([0.4, 0.0, 0.20, 0.01, 0.4])
    nodes[0, 3] = torch.tensor([0.6, 0.0, 0.20, 0.01, 0.8])
    env = DCVRPEnvironment(
        DCVRPDataset(2, 1.0, 480.0, nodes), pending_cost=0.0, record_trace=False
    )
    env.selector = _FixedSelector()
    env.reset()
    assert int(env._init_hidden[0].item()) == 2
    env.vehicles[0, 0, 2] = 0.30
    env.vehicles[0, 1, 2] = 1.00
    env._rebuild_mask()
    assert bool(env.mask[0, 0, 1]), env.mask[0, :, 1]
    assert not bool(env.mask[0, 1, 1]), env.mask[0, :, 1]


def test_low_dynamism_does_not_reserve_capacity():
    """A single hidden customer must not turn on the high-dynamism slack rule."""
    nodes = torch.zeros(1, 4, 5)
    nodes[0, 1] = torch.tensor([0.2, 0.0, 0.20, 0.01, 0.0])
    nodes[0, 2] = torch.tensor([0.4, 0.0, 0.20, 0.01, 0.0])
    nodes[0, 3] = torch.tensor([0.6, 0.0, 0.20, 0.01, 0.5])
    env = DCVRPEnvironment(
        DCVRPDataset(2, 1.0, 480.0, nodes), pending_cost=0.0, record_trace=False
    )
    env.selector = _FixedSelector()
    env.reset()
    assert int(env._init_hidden[0].item()) == 1
    env.vehicles[0, 0, 2] = 0.30
    env.vehicles[0, 1, 2] = 1.00
    env._rebuild_mask()
    assert not bool(env.mask[0, 0, 1])
    assert not bool(env.mask[0, 1, 1])


def test_high_dynamism_first_idle_is_a_real_depot():
    """High-phi first interval terminator returns; a later one stays put."""
    nodes = torch.zeros(1, 4, 5)
    nodes[0, 1] = torch.tensor([0.2, 0.0, 0.2, 0.01, 0.0])
    nodes[0, 2] = torch.tensor([0.4, 0.0, 0.2, 0.01, 0.4])
    nodes[0, 3] = torch.tensor([0.6, 0.0, 0.2, 0.01, 0.8])
    env = DCVRPEnvironment(
        DCVRPDataset(1, 1.0, 480.0, nodes), pending_cost=0.0, record_trace=True
    )
    env.selector = _FixedSelector()
    env.reset()
    assert int(env._init_hidden[0].item()) == 2
    env.step(torch.tensor([[1]]))
    env.step(torch.tensor([[0]]))
    assert abs(float(env.vehicles[0, 0, 0])) < 1e-5
    assert abs(float(env.route_distance().item()) - 0.4) < 1e-3
    assert env.current_segment >= 1
    before = float(env.route_distance().item())
    env.step(torch.tensor([[0]]))
    assert abs(float(env.vehicles[0, 0, 0])) < 1e-5
    assert abs(float(env.route_distance().item()) - before) < 1e-5


def test_low_dynamism_first_idle_still_stays_in_field():
    """phi=10%-style instances keep the hidden-customer stay-in-field terminator."""
    nodes = torch.zeros(1, 3, 5)
    nodes[0, 1] = torch.tensor([0.2, 0.0, 0.2, 0.01, 0.0])
    nodes[0, 2] = torch.tensor([0.8, 0.0, 0.2, 0.01, 0.5])
    env = DCVRPEnvironment(
        DCVRPDataset(1, 1.0, 480.0, nodes), pending_cost=0.0, record_trace=True
    )
    env.selector = _FixedSelector()
    env.reset()
    env.step(torch.tensor([[1]]))
    env.step(torch.tensor([[0]]))
    assert env.current_segment == 1
    assert abs(float(env.vehicles[0, 0, 0]) - 0.2) < 1e-5
    assert abs(float(env.route_distance().item()) - 0.2) < 1e-4


def test_depot_stays_closed_while_customers_remain():
    """Middle intervals must finish the static VRP before a depot return."""
    nodes = torch.zeros(1, 4, 5)
    nodes[0, 1] = torch.tensor([0.2, 0.0, 0.2, 0.01, 0.0])
    nodes[0, 2] = torch.tensor([0.4, 0.0, 0.2, 0.01, 0.0])
    nodes[0, 3] = torch.tensor([0.6, 0.0, 0.2, 0.01, 0.0])
    env = DCVRPEnvironment(
        DCVRPDataset(1, 1.0, 480.0, nodes), pending_cost=0.0, record_trace=False
    )
    env.selector = _FixedSelector()
    env.reset()
    assert bool(env.mask[0, 0, 0])
    env.step(torch.tensor([[1]]))
    assert bool(env.mask[0, 0, 0])
    env.step(torch.tensor([[2]]))
    assert bool(env.mask[0, 0, 0])
    env.step(torch.tensor([[3]]))
    assert not bool(env.mask[0, 0, 0])


def test_middle_interval_cannot_dispatch_after_boundary_clock():
    """Once the planning clock reaches T_i, leftover visible customers wait."""
    nodes = torch.zeros(1, 3, 5)
    nodes[0, 1] = torch.tensor([0.2, 0.0, 0.2, 0.15, 0.0])
    nodes[0, 2] = torch.tensor([0.4, 0.0, 0.2, 0.01, 0.0])
    env = DCVRPEnvironment(
        DCVRPDataset(1, 1.0, 480.0, nodes), pending_cost=0.0, record_trace=False
    )
    env.selector = _FixedSelector()
    env.reset()
    env.step(torch.tensor([[1]]))
    assert float(env.vehicles[0, 0, 3]) >= 0.1
    assert bool(env.mask[0, 0, 2])


def test_middle_interval_depot_stays_if_hidden_customers_remain():
    """Unrevealed customers must not trigger a committed speed-480 depot trip."""
    nodes = torch.zeros(1, 3, 5)
    nodes[0, 1] = torch.tensor([0.2, 0.0, 0.2, 0.01, 0.0])
    nodes[0, 2] = torch.tensor([0.8, 0.0, 0.2, 0.01, 0.5])
    env = DCVRPEnvironment(
        DCVRPDataset(1, 1.0, 480.0, nodes), pending_cost=0.0, record_trace=True
    )
    env.selector = _FixedSelector()
    env.reset()
    env.step(torch.tensor([[1]]))
    env.step(torch.tensor([[0]]))
    assert env.current_segment == 1
    assert abs(float(env.vehicles[0, 0, 0]) - 0.2) < 1e-5
    assert abs(float(env.route_distance().item()) - 0.2) < 1e-4


def test_keep_started_destroy_unstarted():
    nodes = torch.zeros(1, 3, 5)
    nodes[0, 1, 0] = 0.5
    nodes[0, 1, 2] = 0.2
    nodes[0, 1, 3] = 0.01
    nodes[0, 2, 0] = 0.8
    nodes[0, 2, 2] = 0.2
    nodes[0, 2, 3] = 0.01
    data = DCVRPDataset(vehicle_count=1, vehicle_capacity=1.0, vehicle_speed=1.0, nodes=nodes)
    env = DCVRPEnvironment(data, pending_cost=0.0)
    env.selector = _FixedSelector()
    env.reset()
    env.step(torch.tensor([[1]]))
    env.step(torch.tensor([[2]]))
    env.step(torch.tensor([[0]]))
    assert bool(env.served[0, 1])
    assert not bool(env.served[0, 2])
    assert abs(float(env.vehicles[0, 0, 0]) - 0.5) < 1e-5
    assert env.interval_logs and env.interval_logs[0]["kept"] == [1]
    assert env.interval_logs[0]["destroyed"] == [2]


def test_compile_forward_all_methods():
    set_seed(4)
    data = generate_dataset(batch_size=2, customer_count=20, vehicle_count=4, dynamic_rate=0.10)
    for method in ("DVNDA", "AMCVN", "LiDRL", "MAAM", "MARDAM"):
        model = _build_model(method=method, greedy=True)
        env = DCVRPEnvironment(data, pending_cost=0.0)
        actions, logps, rewards = model(env)
        assert actions and logps and rewards


def test_service_rate_is_complete():
    """Last-interval disclose plus Constraint (8) keep mean QoS near Table I."""
    set_seed(21)
    model = _build_model(greedy=True)
    for rate in (0.10, 0.25, 0.50, 0.75):
        data = generate_dataset(batch_size=32, customer_count=20, vehicle_count=4, dynamic_rate=rate)
        env = DCVRPEnvironment(data, pending_cost=0.0)
        model(env)
        mean_qos = float(env.qos().mean())
        assert mean_qos >= 0.98, (rate, mean_qos, float(env.qos().min()))


def test_route_cost_increases_with_dynamism():
    """Paired nested splits: higher phi raises distance while QoS stays comparable."""
    from env import generate_evaluation_split
    from models import build_selector

    split = generate_evaluation_split(
        instances=100, customer_count=20, vehicle_count=4, seed=20260821
    )
    means = []
    qoss = []
    for rate in (0.10, 0.25, 0.50, 0.75):
        selector = build_selector("DVNDA", vehicle_count=4)
        env = DCVRPEnvironment(split[rate], pending_cost=0.0)
        env.selector = selector
        env.policy_greedy = True
        env.reset()
        guard = 0
        while not env.done:
            dist = torch.norm(env.nodes[:, :, :2] - env.cur_veh[:, :, :2], dim=-1)
            dist = dist.masked_fill(env.cur_veh_mask.squeeze(1), float("inf"))
            idx = dist.argmin(dim=1, keepdim=True)
            all_inf = ~torch.isfinite(dist).any(dim=1, keepdim=True)
            idx = torch.where(all_inf, torch.zeros_like(idx), idx)
            env.step(idx)
            guard += 1
            assert guard < 2000
        means.append(float(env.route_distance().mean()))
        qoss.append(float(env.qos().mean()))
    assert all(np.isfinite(means))
    assert means[0] < means[1] < means[2] < means[3], (means, qoss)


if __name__ == "__main__":
    test_vehicle_selector_is_argmax()
    test_reward_matches_equation_14()
    test_final_routes_return_to_depot()
    test_vehicle_selector_has_gradient()
    test_algorithm1_boundary_clocks_are_synchronized()
    test_algorithm1_interpolates_en_route_position()
    test_algorithm1_does_not_keep_planning_clock()
    test_five_environment_knobs()
    test_interval_uses_the_fleet_not_one_vehicle()
    test_dynamic_revelation_is_spread_across_horizon()
    test_last_interval_reveals_tail_arrivals()
    test_first_interval_hides_dynamic_customers()
    test_service_waits_until_appearance_time()
    test_high_dynamism_keeps_max_demand_slack_on_another_vehicle()
    test_low_dynamism_does_not_reserve_capacity()
    test_high_dynamism_first_idle_is_a_real_depot()
    test_low_dynamism_first_idle_still_stays_in_field()
    test_depot_stays_closed_while_customers_remain()
    test_middle_interval_cannot_dispatch_after_boundary_clock()
    test_middle_interval_depot_stays_if_hidden_customers_remain()
    test_keep_started_destroy_unstarted()
    test_compile_forward_all_methods()
    test_service_rate_is_complete()
    test_route_cost_increases_with_dynamism()
    print("all paper-alignment tests passed")
