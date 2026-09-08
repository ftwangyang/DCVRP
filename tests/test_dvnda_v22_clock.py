import pytest
import torch

from experiments.dvnda_detailed_requirements_v15.instances import generate_dataset
from experiments.dvnda_detailed_requirements_v22.diagnose_neural_clock import (
    variable_interval_environment,
)


@pytest.mark.parametrize("intervals", [5, 10, 20, 480])
def test_variable_clock_changes_only_clock_fields(intervals: int) -> None:
    torch.manual_seed(4)
    data = generate_dataset(2, 0.5)
    environment = variable_interval_environment(data, data.nodes, intervals)
    assert environment.segment_count == intervals
    assert environment.segment_duration == pytest.approx(1.0 / intervals)
    assert environment.nodes.data_ptr() == data.nodes.data_ptr()
    assert environment.veh_count == 4
