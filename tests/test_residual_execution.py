import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from env.dataset import DCVRPDataset
from env.environment import DCVRPEnvironment

class FirstVehicle:
    def select(self, vehicles, customers, vehicle_done, customer_mask, greedy):
        return torch.zeros((vehicles.size(0), 1), dtype=torch.long), vehicles.new_zeros(vehicles.size(0), 1), vehicles.new_zeros(vehicles.size(0), 1)

def build():
    nodes = torch.zeros(1, 3, 5)
    nodes[0, 1] = torch.tensor([.3, 0, .7, .01, 0])
    nodes[0, 2] = torch.tensor([.31, 0, .4, .01, .1])
    env = DCVRPEnvironment(DCVRPDataset(1, 1., 1., nodes))
    env.selector = FirstVehicle()
    env.reset()
    env.step(torch.tensor([[1]]))
    env.step(torch.tensor([[0]]))
    return env

def test_residual_capacity_is_reserved_before_next_action():
    env = build()
    assert env.current_segment == 1
    assert env.cur_veh_mask[0, 0, 2], 'residual consumes .7, leaving only .3 for demand .4'

def test_committed_work_spanning_boundaries_preserves_available_state():
    env = build()
    env.step(torch.tensor([[0]]))
    # Word requirement: use the committed destination with a future available
    # time, not the physical interpolation used by the earlier experiment.
    assert abs(env.vehicles[0, 0, 0].item() - .3) < 1e-6
    assert abs(env.route_distance().item() - .3) < 1e-6
    assert abs(env.vehicles[0, 0, 3].item() - .31) < 1e-6
    while not env.done:
        env.step(torch.tensor([[0]]))
    assert abs(env.route_distance().item() - .6) < 1e-5
    assert env.served[0, 1] and not env.served[0, 2]

def test_pending_service_finish_is_used_in_horizon_mask():
    env = build()
    env.vehicles[0, 0, 3] = .99
    env.vehicles[0, 0, 2] = 1.
    env.nodes[0, 2, 0] = 0.8
    env._rebuild_mask()
    assert env.mask[0, 0, 2], 'service cannot start after the horizon'

def test_unrevealed_customer_remains_masked():
    env = build()
    env.nodes[0, 2, 4] = .75
    env.cust_mask[0, 2] = True
    env.total_cust_mask[:, :, 2] = True
    env._rebuild_mask()
    assert env.mask[0, 0, 2]

def test_high_dynamism_routes_have_independent_closed_cost_and_capacity():
    from scripts.dvnda_bounded import dataset
    from models import AttentionLearner, build_selector

    class AuditedEnvironment(DCVRPEnvironment):
        def reset(self):
            self.audit_pending = []
            self.audit_committed = []
            super().reset()

        def _record_tensor_event(self, customer_index, distance, valid=None):
            super()._record_tensor_event(customer_index, distance, valid)
            keep = valid is None or bool(valid.any())
            if keep:
                self.audit_pending.append((self.current_segment, self.cur_veh_idx.clone(),
                    customer_index.clone(), self._last_start.clone(), self.cur_veh.clone()))

        def commit_audit(self, boundary):
            for segment, vehicles, customers, starts, states in self.audit_pending:
                for b in range(self.minibatch_size):
                    c = int(customers[b, 0])
                    if float(starts[b, 0]) < boundary:
                        self.audit_committed.append((b, int(vehicles[b, 0]), c,
                            float(starts[b, 0]), float(states[b, 0, 3]), segment))
            self.audit_pending = []

        def _execute_until_boundary(self, boundary):
            self.commit_audit(boundary)
            return super()._execute_until_boundary(boundary)

        def _finalize_routes(self, reward):
            self.commit_audit(float('inf'))
            return super()._finalize_routes(reward)

    torch.set_num_threads(2)
    torch.manual_seed(17)
    for rate in (.5, .75):
        data = dataset(1, rate, torch.Generator().manual_seed(91))
        model = AttentionLearner(build_selector('DVNDA', vehicle_count=4)).eval()
        model.greedy = True
        env = AuditedEnvironment(data)
        with torch.no_grad():
            model(env)
        seen = set()
        positions = data.nodes[:, :1, :2].expand(-1, 4, -1).clone()
        totals = torch.zeros(1)
        loads = torch.zeros(1, 4)
        finishes = torch.zeros(1, 4)
        for b, v, c, start, finish, segment in env.audit_committed:
            if c:
                assert (b, c) not in seen
                seen.add((b, c))
            cutoff = 1.0 if segment >= 9 else segment / 10.0
            assert data.nodes[b, c, 4] <= cutoff + 1e-6
            assert start + 1e-6 >= finishes[b, v]
            totals[b] += torch.norm(data.nodes[b, c, :2] - positions[b, v])
            positions[b, v] = data.nodes[b, c, :2]
            loads[b, v] += data.nodes[b, c, 2]
            finishes[b, v] = finish
        totals += torch.norm(positions - data.nodes[:, :1, :2], dim=-1).sum(1)
        assert torch.all(loads <= 1 + 1e-6)
        assert torch.allclose(totals, env.route_distance(), atol=2e-4)
        assert len(seen) == int(env.served[:, 1:].sum())
