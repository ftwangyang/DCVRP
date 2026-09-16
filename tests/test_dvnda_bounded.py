import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.dvnda_bounded import dataset, gate_fails


def test_gate_boundary_and_nonfinite():
    assert not gate_fails(9, 99.)
    assert not gate_fails(10, 8.)
    assert gate_fails(10, 8.001)
    assert gate_fails(11, 9.)
    assert gate_fails(10, float('nan'))


def test_literal_distribution_and_independent_rng():
    data = dataset(2000, .75, torch.Generator().manual_seed(42))
    again = dataset(2000, .75, torch.Generator().manual_seed(42))
    assert torch.equal(data.nodes, again.nodes)
    release = data.nodes[:, 1:, 4] * 480
    assert ((release > 0).sum(1) == 15).all()
    for column in range(20):
        samples = release[:, column][release[:, column] > 0]
        assert abs(samples.mean().item() - 240.5) < 2
    assert data.nodes[:, :, :2].min() >= 0
    assert data.nodes[:, :, :2].max() <= 1
    assert data.veh_speed == 480
    assert data.nodes[:, 1:, 2].min() >= 5/150
    assert data.nodes[:, 1:, 2].max() <= 41/150
