from pathlib import Path

import pytest
from experiments.dvnda_detailed_requirements_v19.diagnose_greedy_timing import evaluate
from experiments.reproducibility_n20.run import load_paired_instances


FROZEN = Path("experiments/reproducibility_n20/data/test_n20_seed20260821_paired.pt")
HISTORICAL = {
    0.10: 9.032598,
    0.25: 9.708289,
    0.50: 11.312253,
    0.75: 12.208043,
}


@pytest.mark.parametrize("rate,expected", HISTORICAL.items())
def test_exact_event_reconstructs_historical_greedy(rate: float, expected: float) -> None:
    instances = load_paired_instances(FROZEN)[rate]
    costs, qos = evaluate(instances, mode="exact_event")
    assert costs.mean() == pytest.approx(expected, abs=2e-6)
    assert 0.0 <= qos.min() <= qos.max() <= 1.0
