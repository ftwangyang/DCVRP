import torch

from experiments.dvnda_detailed_requirements_v21.instances import (
    generate_training_dataset,
)


def _make(mode: str):
    torch.manual_seed(900)
    return generate_training_dataset(128, (0.10, 0.25, 0.50, 0.75), training_data_mode=mode)


def test_shared_exact_is_matched_except_disclosure_correlation() -> None:
    independent = _make("paper_independent_exact")
    shared = _make("released_shared_exact")
    assert torch.equal(independent.nodes[:, :, :4], shared.nodes[:, :, :4])
    assert torch.equal(independent.nodes[:, :, 4] > 0, shared.nodes[:, :, 4] > 0)
    assert torch.equal(independent.detailed_dynamic_counts, shared.detailed_dynamic_counts)
    assert not torch.equal(independent.nodes[:, :, 4], shared.nodes[:, :, 4])


def test_bernoulli_and_full_reconstruct_released_support() -> None:
    bernoulli = _make("released_shared_bernoulli")
    full = _make("released_full")
    assert torch.equal(bernoulli.nodes[:, :, 4], full.nodes[:, :, 4])
    assert torch.equal(bernoulli.detailed_dynamic_counts, full.detailed_dynamic_counts)
    assert torch.allclose(full.nodes[:, :, :2] * 100, (full.nodes[:, :, :2] * 100).round())
    assert float((full.nodes[:, 1:, 2] * 150).max()) <= 40.0 + 1e-5
    assert float((full.nodes[:, 1:, 3] * 480).max()) <= 30.0 + 1e-5
