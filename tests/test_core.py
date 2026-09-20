"""Core checks for the time-driven DCVRP environment and vehicle policy."""

from __future__ import annotations

import torch

from env import DCVRPEnvironment, generate_dataset, generate_evaluation_split
from models.selectors import BaseSelector, IndependentSelector


class DummySelector(BaseSelector):
    def scores(self, vehicles, customers, vehicle_done, customer_mask):
        del customers, vehicle_done, customer_mask
        batch, fleet, _ = vehicles.shape
        scores = vehicles.new_zeros((batch, fleet))
        scores[:, 0] = 1.0
        self.last_hidden = vehicles.new_zeros(batch, fleet, self.representation_size)
        return scores


def _attach_selector(env: DCVRPEnvironment) -> None:
    env.selector = DummySelector(vehicle_count=env.veh_count)
    env.policy_greedy = True
    env.reset()
    env._update_cur_veh()


def test_nested_dynamic_customers():
    split = generate_evaluation_split(instances=16, seed=0, customer_count=20)
    static = {
        rate: int((data.nodes[0, 1:, 4] == 0).sum().item())
        for rate, data in split.items()
    }
    assert static[0.10] == 18
    assert static[0.25] == 15
    assert static[0.50] == 10
    assert static[0.75] == 5
    appear_10 = split[0.10].nodes[:, 1:, 4]
    appear_25 = split[0.25].nodes[:, 1:, 4]
    shared = appear_10 > 0
    assert torch.equal(appear_10[shared], appear_25[shared])


def test_first_interval_shows_only_static_customers():
    data = generate_dataset(batch_size=4, customer_count=20, dynamic_rate=0.75, seed=1)
    env = DCVRPEnvironment(data, pending_cost=0.0)
    _attach_selector(env)
    cutoff = env._visibility_cutoff()
    assert abs(cutoff) < 1e-6
    appear = env.nodes[:, 1:, 4]
    visible = ~env.cust_mask[:, 1:]
    hidden = env.cust_mask[:, 1:]
    assert torch.all(appear[visible] <= 1e-6)
    assert torch.all(appear[hidden] > 1e-6)


def test_lookahead_discloses_through_next_boundary():
    data = generate_dataset(batch_size=4, customer_count=20, dynamic_rate=0.75, seed=1)
    env = DCVRPEnvironment(data, pending_cost=0.0, disclose_horizon_tail=True)
    _attach_selector(env)
    cutoff = env._visibility_cutoff()
    assert abs(cutoff - env.segment_duration) < 1e-6
    appear = env.nodes[:, 1:, 4]
    visible = ~env.cust_mask[:, 1:]
    hidden = env.cust_mask[:, 1:]
    assert torch.all(appear[visible] <= cutoff + 1e-6)
    assert torch.all(appear[hidden] > cutoff - 1e-6)


def test_closed_horizon_hides_future_arrivals():
    data = generate_dataset(batch_size=4, customer_count=20, dynamic_rate=0.75, seed=2)
    env = DCVRPEnvironment(data, pending_cost=0.0, disclose_horizon_tail=False)
    _attach_selector(env)
    cutoff = env._visibility_cutoff()
    assert abs(cutoff) < 1e-6
    appear = env.nodes[:, 1:, 4]
    visible = ~env.cust_mask[:, 1:]
    hidden = env.cust_mask[:, 1:]
    assert torch.all(appear[visible] <= 1e-6)
    assert torch.all(appear[hidden] > 1e-6)


def test_depot_return_ends_the_interval_tour():
    data = generate_dataset(batch_size=1, customer_count=20, dynamic_rate=0.0, seed=3)
    env = DCVRPEnvironment(data, pending_cost=0.0)
    _attach_selector(env)
    env._update_done(torch.zeros((1, 1), dtype=torch.long))
    env._rebuild_mask()
    env._sync_vehicle_completion()
    vehicle = int(env.cur_veh_idx[0, 0].item())
    assert bool(env.veh_done[0, vehicle])


def test_unstarted_suffix_is_destroyed_at_boundary():
    data = generate_dataset(batch_size=1, customer_count=20, dynamic_rate=0.0, seed=6)
    env = DCVRPEnvironment(data, pending_cost=0.0)
    _attach_selector(env)
    visible = (~env.mask[0, 0, 1:]).nonzero(as_tuple=False)
    assert visible.numel() > 0
    cid = int(visible[0, 0].item()) + 1
    env.step(torch.tensor([[cid]]))
    env._event_start[-1] = torch.full_like(env._event_start[-1], 0.2)
    env._execute_until_boundary(0.10)
    assert not bool(env.served[0, cid])


def test_qos_counts_service_started_by_horizon():
    data = generate_dataset(batch_size=1, customer_count=20, dynamic_rate=0.0, seed=4)
    env = DCVRPEnvironment(data, pending_cost=0.0)
    _attach_selector(env)
    env.served[0, 1:] = True
    env._start_time[0, 1:11] = 0.5
    env._start_time[0, 11:] = 1.1
    qos = float(env.qos()[0])
    assert abs(qos - 0.5) < 1e-6


def test_objective_penalizes_unserved_customers():
    data = generate_dataset(batch_size=1, customer_count=20, dynamic_rate=0.0, seed=5)
    env = DCVRPEnvironment(data, pending_cost=5.0)
    _attach_selector(env)
    env.total_distance[:] = 0.0
    reward = env._finalize_routes(torch.zeros((1, 1)))
    pending = float(env.unserved_count()[0, 0])
    assert pending == 20.0
    assert abs(float(reward[0, 0]) + 5.0 * 20.0) < 1e-4


def test_vehicle_assignment_is_argmax():
    selector = DummySelector(vehicle_count=4)
    vehicles = torch.zeros(2, 4, 4)
    customers = torch.zeros(2, 5, 5)
    done = torch.zeros(2, 4, dtype=torch.bool)
    mask = torch.zeros(2, 4, 5, dtype=torch.bool)
    index, logits, logp = selector.select(vehicles, customers, done, mask, greedy=True)
    assert torch.equal(index, torch.zeros((2, 1), dtype=torch.long))
    expected = torch.log_softmax(logits, dim=1).gather(1, index)
    assert torch.allclose(logp, expected)


def test_vehicle_score_head_receives_policy_gradient():
    selector = IndependentSelector(vehicle_count=4)
    vehicles = torch.zeros(3, 4, 4)
    customers = torch.zeros(3, 6, 5)
    done = torch.zeros(3, 4, dtype=torch.bool)
    mask = torch.zeros(3, 4, 6, dtype=torch.bool)
    _, _, logp = selector.select(vehicles, customers, done, mask, greedy=False)
    logp.mean().backward()
    score_layer = selector.vehicle_networks[0].decision_layer2
    assert score_layer.weight.grad is not None
    assert float(score_layer.weight.grad.abs().sum()) > 0.0
