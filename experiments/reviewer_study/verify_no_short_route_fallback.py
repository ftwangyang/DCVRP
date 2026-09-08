"""Executable audit that every named solver is called on a one-customer route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import paper_multivehicle_solvers as solvers
from .paper_multivehicle_env import Plan, PlanningState


def _one_customer_state() -> PlanningState:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    targets = {
        "Regret insertion": "time_aware_regret_routes",
        "Tabu Search": "standalone_tabu_routes",
        "Adaptive LNS": "alns_routes",
        "OR-Tools": "ortools_routes",
    }
    report = {}
    for method, target in targets.items():
        calls = []

        def stub(state, *args, **kwargs):
            calls.append(tuple(state.customer_ids))
            return Plan((tuple(state.customer_ids),))

        with patch.object(solvers, target, side_effect=stub):
            plan = solvers.assignment_preserving_routes(
                _one_customer_state(),
                method=method,
                ortools_time_ms=1_000,
                alns_time_ms=1_000,
                seed=314159,
            )
        report[method] = {
            "named_solver_calls": len(calls),
            "customers_seen": calls,
            "route": plan.routes,
            "passed": calls == [(0,)] and plan.routes == ((0,),),
        }

    passed = all(value["passed"] for value in report.values())
    payload = {"passed": passed, "methods": report}
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.resolve().write_text(rendered + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
