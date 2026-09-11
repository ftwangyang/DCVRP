"""Greedy heuristic baseline for DCVRP.

Implements the event-driven nearest-available dispatching rule described in
Section IV-A of the manuscript:
"Greedy: Every time a vehicle becomes idle, it is assigned to execute the nearest available task."

Characteristics:
- Dispatches each idle vehicle to the nearest customer that is currently revealed
  and within remaining vehicle capacity.
- Operates under continuous event revelation.
- Each vehicle performs a single round-trip return to the depot upon exhausting
  all feasible customer requests.
"""

from __future__ import annotations

import torch

from .dataset import DCVRPDataset


def _actionable_times(
    disclosure: torch.Tensor,
    reveal: str = "continuous",
    intervals: int = 10,
) -> torch.Tensor:
    """Map customer disclosure times to actionable event timestamps."""
    if reveal == "continuous":
        actionable = disclosure.clone()
    elif reveal == "next_boundary":
        actionable = torch.ceil(disclosure * intervals - 1e-7) / intervals
    else:
        raise ValueError(f"Unknown reveal mode: {reveal}")
    return torch.where(disclosure > 0, actionable, torch.zeros_like(disclosure))


@torch.no_grad()
def run_greedy(
    data: DCVRPDataset,
    reveal: str = "continuous",
    intervals: int = 10,
    horizon: float = 1.0,
    pending_cost: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the event-driven nearest-task Greedy baseline strictly conforming to Section IV-A.

    Parameters
    ----------
    data : DCVRPDataset
        Batch of normalized DCVRP instances.
    reveal : str, optional
        Revelation model: "continuous" (default, as reported in Table I)
        or "next_boundary".
    intervals : int, optional
        Number of decision intervals (default: 10).
    horizon : float, optional
        Normalized operational time horizon (default: 1.0).
    pending_cost : float, optional
        Penalty for each unserved customer (default: 0.0 during evaluation).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - costs: 1D Tensor of total route distance for each instance.
        - qos: 1D Tensor of service quality (fraction of customers served) in [0, 1].
    """
    nodes_all = data.nodes.detach().cpu()
    vehicle_count = int(data.veh_count)
    capacity = float(data.veh_capa)
    speed = float(data.veh_speed)

    costs: list[float] = []
    served: list[float] = []

    for nodes in nodes_all:
        node_count = nodes.size(0)
        depot = nodes[0, :2]
        positions = depot.repeat(vehicle_count, 1)
        remaining = torch.full((vehicle_count,), capacity)
        idle_at = torch.zeros(vehicle_count)
        has_departed = torch.zeros(vehicle_count, dtype=torch.bool)
        assigned = torch.zeros(node_count, dtype=torch.bool)
        assigned[0] = True
        available = _actionable_times(nodes[:, 4], reveal, intervals)
        depot_dists = torch.norm(nodes[:, :2] - depot, dim=1)

        cost = 0.0
        now = 0.0
        max_events = node_count * (intervals + vehicle_count + 2)

        for _ in range(max_events):
            visible = (available <= now + 1e-7) & (~assigned)
            visible[0] = False

            idle_vehicles = (
                (idle_at <= now + 1e-7).nonzero(as_tuple=False).flatten().tolist()
            )
            if len(idle_vehicles) > 0 and bool(visible.any()):
                act = [v for v in idle_vehicles if bool(has_departed[v])]
                dpt = [v for v in idle_vehicles if not bool(has_departed[v])]
                v_order = act + dpt

                for vehicle in v_order:
                    feas_cap = visible & (nodes[:, 2] <= remaining[vehicle] + 1e-7)
                    candidates = feas_cap.nonzero(as_tuple=False).flatten()
                    if candidates.numel() == 0:
                        continue

                    distances = torch.norm(
                        nodes[candidates, :2] - positions[vehicle], dim=1
                    )
                    # Time feasibility check (Eq. 8): must complete service and return to depot before horizon T
                    travel_times = distances / speed
                    start_times = torch.maximum(now + travel_times, nodes[candidates, 4])
                    finish_times = start_times + nodes[candidates, 3]
                    dist_to_depot = torch.norm(nodes[candidates, :2] - depot, dim=1)
                    return_times = finish_times + dist_to_depot / speed
                    time_feasible = return_times <= (horizon + 1e-6)

                    feas_candidates = candidates[time_feasible]
                    if feas_candidates.numel() == 0:
                        continue
                    feas_distances = distances[time_feasible]

                    # Pure nearest available task (Section IV-A)
                    best_idx = feas_distances.argmin()
                    customer = int(feas_candidates[best_idx].item())
                    leg = float(feas_distances[best_idx].item())

                    cost += leg
                    idle_at[vehicle] = (
                        now + leg / speed + float(nodes[customer, 3])
                    )
                    positions[vehicle] = nodes[customer, :2]
                    remaining[vehicle] -= float(nodes[customer, 2])
                    assigned[customer] = True
                    visible[customer] = False
                    has_departed[vehicle] = True

            if bool(assigned[1:].all()):
                break

            pending_reveal = available[(~assigned) & (available > now + 1e-7)]
            pending_idle = idle_at[idle_at > now + 1e-7]
            events: list[float] = []
            if pending_reveal.numel():
                events.append(float(pending_reveal.min().item()))
            if pending_idle.numel():
                events.append(float(pending_idle.min().item()))
            if not events:
                break
            nxt = min(events)
            if nxt <= now + 1e-7 or nxt > horizon + 1e-7:
                break
            now = nxt

        # Return legs to depot
        cost += float(torch.norm(positions - depot, dim=1).sum().item())
        unserved = node_count - 1 - int(assigned[1:].sum().item())
        costs.append(cost + pending_cost * unserved)
        served.append(1.0 - unserved / float(node_count - 1))

    dtype = data.nodes.dtype
    return (
        torch.tensor(costs, dtype=dtype),
        torch.tensor(served, dtype=dtype),
    )

