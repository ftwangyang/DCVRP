"""Event-driven nearest-task Greedy baseline from the manuscript."""

from __future__ import annotations

import torch

from data import DCVRP_Dataset

from .paper_dcvrp import PAPER_INTERVAL_COUNT


class EarliestAvailableVehicleSelector:
    """Choose the feasible vehicle with the earliest completion time."""

    def select(
        self,
        vehicles: torch.Tensor,
        customers: torch.Tensor,
        vehicle_done: torch.Tensor,
        customer_mask: torch.Tensor,
        greedy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del customers, customer_mask, greedy
        safe_done = vehicle_done.clone()
        all_done = safe_done.all(dim=1)
        safe_done[all_done, 0] = False
        scores = -vehicles[:, :, 3]
        scores = scores.masked_fill(safe_done, -torch.inf)
        index = scores.argmax(dim=1, keepdim=True)
        log_probability = vehicles.new_zeros((vehicles.size(0), 1))
        return index, scores, log_probability


@torch.no_grad()
def run_paper_greedy(
    data: DCVRP_Dataset,
    device: torch.device,
    *,
    pending_cost: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, list[list[tuple[int, int]]]]:
    """Assign the nearest revealed task whenever a vehicle becomes idle.

    An idle vehicle waits at its current position if no task is available; it
    is not forced back to the depot at every disclosure interval.  Customer
    information is still disclosed only at the ten interval boundaries.  Each
    vehicle closes its single round trip by returning to the depot once.
    """

    nodes_cpu = data.nodes.detach().cpu()
    batch_costs: list[float] = []
    batch_qos: list[float] = []
    batch_actions: list[list[tuple[int, int]]] = []
    horizon = 1.0

    for nodes in nodes_cpu:
        node_count = nodes.size(0)
        depot = nodes[0, :2]
        positions = depot.repeat(data.veh_count, 1)
        capacities = torch.full((data.veh_count,), float(data.veh_capa))
        available_at = torch.zeros(data.veh_count)
        assigned = torch.zeros(node_count, dtype=torch.bool)
        assigned[0] = True
        disclosures = (
            torch.ceil(nodes[:, 4] * PAPER_INTERVAL_COUNT - 1.0e-7)
            / PAPER_INTERVAL_COUNT
        )
        route_cost = 0.0
        current_time = 0.0
        instance_actions: list[tuple[int, int]] = []

        for _ in range(node_count * (PAPER_INTERVAL_COUNT + data.veh_count + 2)):
            visible = (disclosures <= current_time + 1.0e-7) & (~assigned)
            visible[0] = False
            idle_vehicles = (
                available_at <= current_time + 1.0e-7
            ).nonzero(as_tuple=False).flatten()

            for vehicle_index_tensor in idle_vehicles:
                vehicle_index = int(vehicle_index_tensor.item())
                feasible = visible & (
                    nodes[:, 2] <= capacities[vehicle_index] + 1.0e-7
                )
                candidates = feasible.nonzero(as_tuple=False).flatten()
                if candidates.numel() == 0:
                    continue
                distances = torch.norm(
                    nodes[candidates, :2] - positions[vehicle_index], dim=1
                )
                customer_index = int(candidates[distances.argmin()].item())
                distance = float(
                    torch.norm(
                        nodes[customer_index, :2] - positions[vehicle_index]
                    ).item()
                )
                route_cost += distance
                available_at[vehicle_index] = (
                    current_time
                    + distance / float(data.veh_speed)
                    + float(nodes[customer_index, 3].item())
                )
                positions[vehicle_index] = nodes[customer_index, :2]
                capacities[vehicle_index] -= nodes[customer_index, 2]
                assigned[customer_index] = True
                visible[customer_index] = False
                instance_actions.append((vehicle_index, customer_index))

            if bool(assigned[1:].all()):
                break
            future_disclosures = disclosures[(~assigned) & (disclosures > current_time + 1.0e-7)]
            future_completions = available_at[available_at > current_time + 1.0e-7]
            next_events: list[float] = []
            if future_disclosures.numel():
                next_events.append(float(future_disclosures.min().item()))
            if future_completions.numel():
                next_events.append(float(future_completions.min().item()))
            if not next_events:
                break
            next_time = min(next_events)
            if next_time <= current_time + 1.0e-7 or next_time > horizon + 1.0e-7:
                break
            current_time = next_time

        route_cost += float(torch.norm(positions - depot, dim=1).sum().item())
        served_count = int(assigned[1:].sum().item())
        pending = node_count - 1 - served_count
        batch_costs.append(route_cost + float(pending_cost) * pending)
        batch_qos.append(served_count / float(node_count - 1))
        batch_actions.append(instance_actions)

    return (
        torch.tensor(batch_costs, dtype=data.nodes.dtype, device=device),
        torch.tensor(batch_qos, dtype=data.nodes.dtype, device=device),
        batch_actions,
    )


__all__ = ["EarliestAvailableVehicleSelector", "run_paper_greedy"]
