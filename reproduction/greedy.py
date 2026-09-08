"""The Greedy row of Table I, and the protocol check it provides.

"Every time a vehicle becomes idle, it is assigned to execute the nearest
available task."  The rule is event-driven and contains nothing learned, so this
row is the one part of Table I that can be reproduced exactly, with no training
and no checkpoint.  It is therefore used as the protocol test for the whole
package: if the generator, the time semantics, the capacity semantics, and the
cost definition are right, Greedy must land on Table I by itself.

``reveal`` controls when a dynamic request becomes actionable:

``"continuous"`` (default, and what reproduces Table I)
    A request is actionable at its sampled disclosure time.  This is the
    behaviour of an event-driven dispatcher, which reacts whenever a vehicle
    becomes idle rather than on a planning clock.

``"next_boundary"``
    A request is actionable only at the first of the ten interval boundaries at
    or after its disclosure time.  This is the cadence the *neural* methods are
    bound by, because they replan once per interval.  Greedy is measurably
    cheaper under this rule at high dynamism -- delaying requests lets the
    dispatcher batch them -- which is why the two rules must not be mixed.
"""

from __future__ import annotations

import torch

from . import protocol


def actionable_times(
    disclosure: torch.Tensor, reveal: str, intervals: int
) -> torch.Tensor:
    """Map sampled disclosure times to the times a dispatcher may act on them."""

    if reveal == "continuous":
        actionable = disclosure.clone()
    elif reveal == "next_boundary":
        actionable = torch.ceil(disclosure * intervals - 1.0e-7) / intervals
    else:
        raise ValueError(f"unknown reveal rule: {reveal}")
    # Static customers keep time 0 rather than being pushed to a boundary.
    return torch.where(disclosure > 0, actionable, torch.zeros_like(disclosure))


@torch.no_grad()
def run_greedy(
    data,
    *,
    reveal: str = "continuous",
    intervals: int = protocol.DECISION_INTERVALS,
    horizon: float = protocol.NORMALIZED_HORIZON,
    pending_cost: float = protocol.EVALUATION_PENDING_COST,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-instance cost and served fraction.

    Each vehicle performs a single round trip: capacity is consumed over the
    whole horizon and never replenished, and the closing depot leg is charged
    once, from wherever the vehicle ended.
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
        assigned = torch.zeros(node_count, dtype=torch.bool)
        assigned[0] = True
        available = actionable_times(nodes[:, 4], reveal, intervals)

        cost = 0.0
        now = 0.0
        # Every iteration either dispatches at least one leg or advances the
        # clock to the next event, so this bound cannot truncate a valid run.
        for _ in range(node_count * (intervals + vehicle_count + 2)):
            visible = (available <= now + 1.0e-7) & (~assigned)
            visible[0] = False
            for vehicle in (
                (idle_at <= now + 1.0e-7).nonzero(as_tuple=False).flatten().tolist()
            ):
                feasible = visible & (nodes[:, 2] <= remaining[vehicle] + 1.0e-7)
                candidates = feasible.nonzero(as_tuple=False).flatten()
                if candidates.numel() == 0:
                    continue
                distances = torch.norm(
                    nodes[candidates, :2] - positions[vehicle], dim=1
                )
                customer = int(candidates[distances.argmin()].item())
                leg = float(
                    torch.norm(nodes[customer, :2] - positions[vehicle]).item()
                )
                cost += leg
                idle_at[vehicle] = now + leg / speed + float(nodes[customer, 3])
                positions[vehicle] = nodes[customer, :2]
                remaining[vehicle] -= float(nodes[customer, 2])
                assigned[customer] = True
                visible[customer] = False

            if bool(assigned[1:].all()):
                break
            pending_reveal = available[(~assigned) & (available > now + 1.0e-7)]
            pending_idle = idle_at[idle_at > now + 1.0e-7]
            events = []
            if pending_reveal.numel():
                events.append(float(pending_reveal.min().item()))
            if pending_idle.numel():
                events.append(float(pending_idle.min().item()))
            if not events:
                break
            nxt = min(events)
            if nxt <= now + 1.0e-7 or nxt > horizon + 1.0e-7:
                break
            now = nxt

        cost += float(torch.norm(positions - depot, dim=1).sum().item())
        unserved = node_count - 1 - int(assigned[1:].sum().item())
        costs.append(cost + pending_cost * unserved)
        served.append(1.0 - unserved / float(node_count - 1))

    dtype = data.nodes.dtype
    return (
        torch.tensor(costs, dtype=dtype),
        torch.tensor(served, dtype=dtype),
    )
