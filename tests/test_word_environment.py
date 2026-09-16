import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from env.dataset import DCVRPDataset
from env.environment import DCVRPEnvironment

class FixedVehicle:
    def select(self, vehicles, customers, vehicle_done, customer_mask, greedy):
        return torch.zeros((len(vehicles), 1), dtype=torch.long), vehicles.new_zeros(len(vehicles), 1), vehicles.new_zeros(len(vehicles), 1)

def make(nodes, speed=1.):
    env = DCVRPEnvironment(DCVRPDataset(1, 1., speed, nodes), record_trace=True)
    env.selector = FixedVehicle()
    env.reset()
    return env

def step(env, customer):
    return env.step(torch.tensor([[customer]]))

def test_depot_is_real_return_without_refill():
    nodes = torch.zeros(1, 2, 5)
    nodes[0, 1] = torch.tensor([.02, 0, .3, .01, 0])
    env = make(nodes)
    reward = step(env, 1) + step(env, 0)
    assert abs(env.route_distance().item() - .04) < 1e-6
    assert abs(env.vehicles[0, 0, 2].item() - .7) < 1e-6
    assert abs(env.vehicles[0, 0, 0].item()) < 1e-6
    assert env.last_node.item() == 0
    assert abs(reward.item() + .04) < 1e-6

def test_commitment_availability_never_rewinds():
    nodes = torch.zeros(1, 2, 5)
    nodes[0, 1] = torch.tensor([.3, 0, .7, .01, 0])
    env = make(nodes)
    step(env, 1)
    step(env, 0)
    assert abs(env.vehicles[0, 0, 0].item() - .3) < 1e-6
    assert abs(env.vehicles[0, 0, 3].item() - .31) < 1e-6
    step(env, 0)
    assert abs(env.vehicles[0, 0, 3].item() - .31) < 1e-6
    while not env.done:
        step(env, 0)
    assert abs(env.route_distance().item() - .6) < 1e-5
    assert abs(env.vehicles[0, 0, 2].item() - .3) < 1e-6
    assert len(env.interval_logs) == 10

def test_visibility_and_destroyed_suffix():
    nodes = torch.zeros(1, 4, 5)
    nodes[0, 1] = torch.tensor([.02, 0, .2, .09, 0])
    nodes[0, 2] = torch.tensor([.03, 0, .2, .01, 0])
    nodes[0, 3] = torch.tensor([.04, 0, .2, .01, .1])
    env = make(nodes)
    step(env, 1)
    assert env.cust_mask[0, 3]  # Planning time has crossed .1, but no disclosure.
    step(env, 2)
    step(env, 0)
    assert env.served[0, 1] and not env.served[0, 2]
    assert not env.cust_mask[0, 3]
    log = env.interval_logs[0]
    assert log['kept'] == [1] and log['destroyed'] == [2]
    assert log['revealed'] == [3]
    assert abs(env.vehicles[0, 0, 2].item() - .8) < 1e-6

def test_global_horizon_feasibility_mask_is_enforced():
    nodes = torch.zeros(1, 2, 5)
    nodes[0, 1] = torch.tensor([.3, 0, .2, .01, 0])
    env = make(nodes)
    env.vehicles[0, 0, 3] = .99
    env._rebuild_mask()
    assert bool(env.mask[0, 0, 1])
