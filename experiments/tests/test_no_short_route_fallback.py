import numpy as np
import pytest

pytest.importorskip("alns", reason="classical-baseline dependencies are isolated")

from experiments.reviewer_study import paper_multivehicle_solvers as solvers
from experiments.reviewer_study.paper_multivehicle_env import Plan, PlanningState


def one_customer_state() -> PlanningState:
    return PlanningState(
        current_minute=0.0,
        next_boundary_minute=48.0,
        depot=np.array([0.0, 0.0]),
        vehicle_positions=np.array([[0.0, 0.0]]),
        vehicle_capacities=np.array([150]),
        vehicle_available_minutes=np.array([0.0]),
        customer_ids=(0,),
        customer_coordinates=np.array([[0.1, 0.0]]),
        demands=np.array([10]),
        service_minutes=np.array([10.0]),
        travel_minutes_per_normalized_unit=100.0,
        optimize_complete_static_route=True,
        require_active_vehicle_routes=True,
    )


@pytest.mark.parametrize(
    ("method", "target"),
    [
        ("Regret insertion", "time_aware_regret_routes"),
        ("Tabu Search", "standalone_tabu_routes"),
        ("Adaptive LNS", "alns_routes"),
        ("OR-Tools", "ortools_routes"),
    ],
)
def test_named_solver_is_called_even_for_one_customer(monkeypatch, method, target):
    calls = []

    def stub(state, *args, **kwargs):
        calls.append((state.customer_ids, args, kwargs))
        return Plan((state.customer_ids,))

    monkeypatch.setattr(solvers, target, stub)
    result = solvers.assignment_preserving_routes(
        one_customer_state(),
        method=method,
        ortools_time_ms=1_000,
        alns_time_ms=1_000,
        seed=314159,
    )

    assert result.routes == ((0,),)
    assert len(calls) == 1
    assert calls[0][0] == (0,)
