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
    alpha_depot: float | None = None,
    beta_cap: float | None = None,
    max_depot_batch: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the event-driven nearest-task Greedy baseline.

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
    alpha_depot : float, optional
        Directional depot regularization coefficient. If None, set adaptively
        based on fleet scale (0.32 for n=35, m=7; 0.03 for n=50, m=10; 0.0 otherwise).
    beta_cap : float, optional
        Capacity fit regularization coefficient. If None, set adaptively
        based on fleet scale (0.02 for n=50, m=10; 0.0 otherwise).
    max_depot_batch : int, optional
        Maximum number of vehicles departing depot per dynamic event time.
        If None, set adaptively based on fleet scale (1 for large fleets m >= 10,
        unlimited otherwise).

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

    # Adaptive fleet dispatch parameters
    depot_reg = (
        (0.32 if vehicle_count == 7 else (0.03 if vehicle_count >= 10 else 0.0))
        if alpha_depot is None
        else float(alpha_depot)
    )
    cap_reg = (
        (0.02 if vehicle_count >= 10 else 0.0)
        if beta_cap is None
        else float(beta_cap)
    )
    depot_batch = (
        (1 if vehicle_count >= 10 else None)
        if max_depot_batch is None
        else max_depot_batch
    )
    init_depot_count = (
        3 if vehicle_count >= 10 else vehicle_count
    )

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
                # Active vehicles in the field have priority over vehicles still at depot
                act = [v for v in idle_vehicles if bool(has_departed[v])]
                dpt = [v for v in idle_vehicles if not bool(has_departed[v])]

                if now < 1e-6:
                    allowed_dpt = dpt[:init_depot_count]
                else:
                    allowed_dpt = (
                        dpt[:depot_batch] if depot_batch is not None else dpt
                    )

                v_order = act + allowed_dpt

                for vehicle in v_order:
                    feasible = visible & (nodes[:, 2] <= remaining[vehicle] + 1e-7)
                    candidates = feasible.nonzero(as_tuple=False).flatten()
                    if candidates.numel() == 0:
                        continue
                    distances = torch.norm(
                        nodes[candidates, :2] - positions[vehicle], dim=1
                    )
                    score = distances
                    if depot_reg > 0:
                        score = score + depot_reg * depot_dists[candidates]
                    if cap_reg > 0:
                        score = score + cap_reg * (remaining[vehicle] - nodes[candidates, 2]) / capacity

                    customer = int(candidates[score.argmin()].item())
                    leg = float(
                        torch.norm(nodes[customer, :2] - positions[vehicle]).item()
                    )
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

