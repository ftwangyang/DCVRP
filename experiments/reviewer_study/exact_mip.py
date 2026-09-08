"""Small-instance CVRP MIP reference for optimality-gap experiments.

The formulation uses one binary variable per directed arc and continuous
single-commodity load variables.  Removing vehicle indices avoids the large
amount of symmetry in an otherwise equivalent multi-vehicle formulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


@dataclass
class MipResult:
    objective: float
    success: bool
    status: int
    message: str
    mip_gap: float


def solve_cvrp_mip(
    coordinates: np.ndarray,
    demands: np.ndarray,
    vehicle_count: int,
    capacity: float = 1.0,
    time_limit_s: float = 30.0,
) -> MipResult:
    """Solve a small homogeneous CVRP with a compact directed-arc MIP."""
    node_count = len(coordinates)
    customers = range(1, node_count)
    arcs = [
        (left, right)
        for left in range(node_count)
        for right in range(node_count)
        if left != right
    ]
    arc_index = {arc: index for index, arc in enumerate(arcs)}
    load_offset = len(arcs)
    load_index = {
        customer: load_offset + customer - 1 for customer in customers
    }
    variable_count = load_offset + node_count - 1

    objective = np.zeros(variable_count, dtype=float)
    for (left, right), index in arc_index.items():
        objective[index] = np.linalg.norm(coordinates[left] - coordinates[right])

    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coefficients: dict[int, float], lb: float, ub: float) -> None:
        rows.append(coefficients)
        lower.append(lb)
        upper.append(ub)

    # Each customer has exactly one predecessor and one successor.
    for customer in customers:
        add(
            {
                arc_index[origin, customer]: 1.0
                for origin in range(node_count)
                if origin != customer
            },
            1.0,
            1.0,
        )
        add(
            {
                arc_index[customer, destination]: 1.0
                for destination in range(node_count)
                if destination != customer
            },
            1.0,
            1.0,
        )

    # Depot balance and fleet-size limit. Unused vehicles are permitted.
    depot_departures = {
        arc_index[0, destination]: 1.0 for destination in customers
    }
    depot_balance = dict(depot_departures)
    for origin in customers:
        depot_balance[arc_index[origin, 0]] = -1.0
    add(depot_balance, 0.0, 0.0)
    add(depot_departures, 0.0, float(vehicle_count))

    # Load propagation simultaneously enforces route capacity and removes
    # customer-only subtours (Miller-Tucker-Zemlin CVRP constraints).
    for left in customers:
        for right in customers:
            if left == right:
                continue
            add(
                {
                    load_index[left]: 1.0,
                    load_index[right]: -1.0,
                    arc_index[left, right]: capacity,
                },
                -np.inf,
                capacity - float(demands[right]),
            )

    matrix = lil_matrix((len(rows), variable_count), dtype=float)
    for row_index, coefficients in enumerate(rows):
        for column, value in coefficients.items():
            matrix[row_index, column] = value

    integrality = np.zeros(variable_count, dtype=int)
    integrality[:load_offset] = 1
    variable_lower = np.concatenate(
        [np.zeros(load_offset), np.asarray(demands[1:], dtype=float)]
    )
    variable_upper = np.concatenate(
        [np.ones(load_offset), np.full(node_count - 1, capacity)]
    )
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": time_limit_s, "mip_rel_gap": 0.0},
    )
    gap_value = getattr(result, "mip_gap", np.nan)
    return MipResult(
        objective=float(result.fun) if result.fun is not None else np.nan,
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        mip_gap=float(gap_value) if gap_value is not None else np.nan,
    )
