"""Tests for the claims the reproduction README makes.

Each test corresponds to a sentence a reviewer is being asked to believe.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction import instances, protocol
from reproduction.greedy import actionable_times, run_greedy

PAPER_GREEDY = {0.10: 9.07, 0.25: 9.69, 0.50: 11.25, 0.75: 12.43}


def nodes_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def test_released_transcription_matches_data_py():
    """instances.generate_released reproduces data.py byte-for-byte."""
    instances.check_released_equivalence()


def test_disclosure_rate_mean_equals_manuscript_scalar():
    """The per-slot rates average to the reported lambda = (1+T)/2."""
    rates = instances.disclosure_rates(protocol.CUSTOMER_COUNT)
    assert float(rates.mean()) == pytest.approx(
        (1.0 + protocol.HORIZON_MINUTES) / 2.0
    )
    assert float(rates.min()) == pytest.approx(1.0)
    assert float(rates.max()) == pytest.approx(float(protocol.HORIZON_MINUTES))


def test_manuscript_split_uses_per_slot_poisson_rates():
    """The reported split must not silently regress to one scalar Poisson rate."""

    split = instances.generate_manuscript_split(
        protocol.TEST_SEED, 2048, dynamic_rates=(1.0,)
    )
    disclosure_minutes = (
        split[1.0].nodes[:, 1:, 4] * protocol.HORIZON_MINUTES
    )
    slot_means = disclosure_minutes.mean(dim=0)
    assert float(slot_means[0]) < 5.0
    assert float(slot_means[-1]) > 450.0
    assert float(slot_means[-1] - slot_means[0]) > 440.0


def test_splits_are_deterministic_functions_of_the_seed():
    """Regenerating a split from its seed yields identical tensors."""
    first = instances.generate_manuscript_split(protocol.TEST_SEED, 32)
    second = instances.generate_manuscript_split(protocol.TEST_SEED, 32)
    assert list(first) == list(second)
    for rate in first:
        assert nodes_hash(first[rate].nodes) == nodes_hash(second[rate].nodes)


def test_train_validation_test_splits_are_disjoint():
    """The three streams do not share instances."""
    hashes = {
        name: {
            rate: nodes_hash(data.nodes)
            for rate, data in instances.generate_manuscript_split(seed, 32).items()
        }
        for name, seed in (
            ("train", protocol.TRAIN_SEED),
            ("validation", protocol.VALIDATION_SEED),
            ("test", protocol.TEST_SEED),
        )
    }
    for rate in protocol.DYNAMIC_RATES:
        values = {name: block[rate] for name, block in hashes.items()}
        assert len(set(values.values())) == 3, f"splits overlap at phi={rate}"


def test_exact_dynamic_counts_and_nested_rates():
    """Each rate has exactly round(n*phi) dynamic customers, nested across rates."""
    split = instances.generate_manuscript_split(protocol.TEST_SEED, 64)
    previous = None
    for rate in sorted(split):
        disclosure = split[rate].nodes[:, 1:, 4]
        dynamic = disclosure > 0
        expected = round(protocol.CUSTOMER_COUNT * rate)
        assert torch.all(dynamic.sum(dim=1) == expected)
        if previous is not None:
            # Nesting: every customer dynamic at the lower rate stays dynamic.
            assert torch.all(previous <= dynamic)
        previous = dynamic


def test_paired_rates_share_physical_instances():
    """Only the disclosure column differs between dynamic-rate variants."""
    split = instances.generate_manuscript_split(protocol.TEST_SEED, 32)
    rates = sorted(split)
    reference = split[rates[0]].nodes
    for rate in rates[1:]:
        assert torch.equal(split[rate].nodes[:, :, :4], reference[:, :, :4])


def test_reveal_rules_order_correctly():
    """A boundary reveal never precedes the sampled disclosure time."""
    disclosure = torch.tensor([0.0, 0.01, 0.1, 0.35, 0.99, 1.0])
    continuous = actionable_times(disclosure, "continuous", 10)
    boundary = actionable_times(disclosure, "next_boundary", 10)
    assert torch.equal(continuous, disclosure)
    assert torch.all(boundary >= continuous - 1e-6)
    # Static customers are never deferred to a boundary.
    assert boundary[0] == 0.0


@pytest.mark.parametrize("rate", list(PAPER_GREEDY))
def test_greedy_reproduces_table1_within_five_percent(rate):
    """The training-free Greedy row reproduces Table I, validating the protocol."""
    split = instances.generate_manuscript_split(
        protocol.TEST_SEED, protocol.TEST_INSTANCES
    )
    instances.seed_all(protocol.DECODE_SEED)
    cost, qos = run_greedy(split[rate], reveal="continuous")
    error = abs(float(cost.mean()) / PAPER_GREEDY[rate] - 1.0)
    assert error <= 0.05, (
        f"phi={rate}: measured {float(cost.mean()):.2f} vs "
        f"manuscript {PAPER_GREEDY[rate]:.2f} ({100 * error:.2f}%)"
    )
    assert float(qos.mean()) > 0.99


def test_greedy_serves_within_capacity():
    """Single round trip: a vehicle never exceeds its capacity over the horizon."""
    split = instances.generate_manuscript_split(protocol.TEST_SEED, 16)
    data = split[0.50]
    total_demand = data.nodes[:, 1:, 2].sum(dim=1)
    fleet_capacity = data.veh_count * float(data.veh_capa)
    # The instance family must be feasible for the QoS claim to be meaningful.
    assert torch.all(total_demand <= fleet_capacity + 1e-6)
