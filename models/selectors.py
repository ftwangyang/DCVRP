"""Vehicle selection policies for DCVRP.

Includes the proposed DVNDA independent dual-attention selector along with
baseline vehicle dispatching and selection strategies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.func import functional_call, vmap

from .transformer import VehicleSelectionNetwork


class BaseSelector(nn.Module):
    """Base class for vehicle selectors."""

    def __init__(self, vehicle_count: int, evaluation_rule: str = "argmax"):
        super().__init__()
        self.vehicle_count = vehicle_count
        self.evaluation_rule = evaluation_rule

    def scores(
        self,
        vehicles: torch.Tensor,
        customers: torch.Tensor,
        vehicle_done: torch.Tensor,
        customer_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def select(
        self,
        vehicles: torch.Tensor,
        customers: torch.Tensor,
        vehicle_done: torch.Tensor,
        customer_mask: torch.Tensor,
        greedy: bool,
    ):
        safe_done = vehicle_done.clone()
        all_done = safe_done.all(dim=1)
        safe_done[all_done, 0] = False
        logits = self.scores(vehicles, customers, safe_done, customer_mask)
        logits = logits.masked_fill(safe_done, -torch.inf)
        distribution = Categorical(logits=logits)
        if greedy and self.evaluation_rule == "argmax":
            index = logits.argmax(dim=1)
        else:
            index = distribution.sample()
        log_probability = distribution.log_prob(index).unsqueeze(1)
        return index.unsqueeze(1), logits, log_probability


class IndependentSelector(BaseSelector):
    """DVNDA: Independent Dual-Attention Sub-Networks for Vehicle Selection.

    Each vehicle possesses a dedicated dual-attention sub-network (fleet
    attention across vehicles and customer attention across demand points).
    Evaluated efficiently via batched functional calls (vmap).
    """

    def __init__(
        self,
        vehicle_count: int,
        vehicle_state_size: int = 4,
        customer_feature_size: int = 5,
        model_size: int = 64,
        head_count: int = 4,
        evaluation_rule: str = "argmax",
    ):
        super().__init__(vehicle_count, evaluation_rule)
        self.vehicle_networks = nn.ModuleList([
            VehicleSelectionNetwork(
                vehicle_state_size, customer_feature_size, model_size, head_count
            )
            for _ in range(vehicle_count)
        ])

    def scores(
        self,
        vehicles: torch.Tensor,
        customers: torch.Tensor,
        vehicle_done: torch.Tensor,
        customer_mask: torch.Tensor,
    ) -> torch.Tensor:
        parameter_maps = [
            dict(net.named_parameters()) for net in self.vehicle_networks
        ]
        stacked_parameters = {
            name: torch.stack([pm[name] for pm in parameter_maps], dim=0)
            for name in parameter_maps[0]
        }
        own_states = vehicles.transpose(0, 1).unsqueeze(2)
        template = self.vehicle_networks[0]

        if customer_mask.dim() == 3:
            per_vehicle_customer_masks = customer_mask.transpose(0, 1)
        else:
            per_vehicle_customer_masks = customer_mask.unsqueeze(0).expand(
                self.vehicle_count, -1, -1
            )

        def evaluate_one(params, own_state, own_customer_mask):
            return functional_call(
                template,
                params,
                (
                    own_state,
                    vehicles,
                    customers,
                    vehicle_done,
                    own_customer_mask,
                ),
                strict=True,
            )

        values = vmap(evaluate_one, in_dims=(0, 0, 0))(
            stacked_parameters, own_states, per_vehicle_customer_masks
        )
        return values.squeeze(2).transpose(0, 1)


class CentralizedSelector(BaseSelector):
    """AMCVN: Centralized Attention Vehicle Selector."""

    def __init__(
        self,
        vehicle_count: int,
        vehicle_state_size: int = 4,
        hidden_size: int = 128,
        evaluation_rule: str = "argmax",
    ):
        super().__init__(vehicle_count, evaluation_rule)
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(vehicle_state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.customer_encoder = nn.Sequential(
            nn.Linear(5, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.decision = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def scores(
        self,
        vehicles: torch.Tensor,
        customers: torch.Tensor,
        vehicle_done: torch.Tensor,
        customer_mask: torch.Tensor,
    ) -> torch.Tensor:
        vehicle_features = self.vehicle_encoder(vehicles)
        valid_vehicles = (~vehicle_done).float().unsqueeze(2)
        fleet = (vehicle_features * valid_vehicles).sum(1, keepdim=True)
        fleet = fleet / valid_vehicles.sum(1, keepdim=True).clamp_min(1.0)
        customer_features = self.customer_encoder(customers)
        if customer_mask.dim() == 3:
            valid_customers = (~customer_mask).float().unsqueeze(3)
            customer_context = (
                customer_features[:, None, :, :] * valid_customers
            ).sum(2)
            customer_context = customer_context / valid_customers.sum(2).clamp_min(1.0)
        else:
            valid_customers = (~customer_mask).float().unsqueeze(2)
            customer_context = (
                customer_features * valid_customers
            ).sum(1, keepdim=True)
            customer_context = customer_context / valid_customers.sum(
                1, keepdim=True
            ).clamp_min(1.0)
            customer_context = customer_context.expand_as(vehicle_features)
        context = torch.cat(
            [
                vehicle_features,
                fleet.expand_as(vehicle_features),
                customer_context,
            ],
            dim=2,
        )
        return self.decision(context).squeeze(2)


class LiDRLTourHistorySelector(BaseSelector):
    """LiDRL: Tour History Recurrent Vehicle Selector."""

    def __init__(
        self,
        vehicle_count: int,
        model_size: int = 64,
        hidden_size: int = 512,
        evaluation_rule: str = "argmax",
    ):
        super().__init__(vehicle_count, evaluation_rule)
        self.model_size = model_size
        self.node_embedding = nn.Linear(5, model_size)
        self.vehicle_context = nn.Sequential(
            nn.Linear(vehicle_count * 3, model_size),
            nn.Linear(model_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, model_size),
        )
        self.tour_context = nn.Sequential(
            nn.Linear(vehicle_count * model_size, model_size),
            nn.Linear(model_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, model_size),
        )
        self.decision = nn.Linear(model_size * 2, vehicle_count)
        self._operational_environment = None

    def set_operational_environment(self, environment) -> None:
        self._operational_environment = environment

    def clear_operational_environment(self) -> None:
        self._operational_environment = None

    def _planned_distance(self, vehicles: torch.Tensor) -> torch.Tensor:
        environment = self._operational_environment
        if environment is None:
            return torch.zeros_like(vehicles[:, :, 3])
        committed = getattr(
            environment, "_committed_vehicle_physical_distance", None
        )
        distance = (
            torch.zeros_like(vehicles[:, :, 3])
            if committed is None
            else committed.detach().clone()
        )
        event_vehicles = getattr(environment, "_event_vehicle", None)
        event_distances = getattr(environment, "_event_distance", None)
        if event_vehicles and event_distances:
            indices = torch.stack(event_vehicles, dim=1).squeeze(-1)
            values = torch.stack(event_distances, dim=1).squeeze(-1)
            distance.scatter_add_(1, indices, values)
        return distance

    def _tour_visit_mask(
        self, vehicles: torch.Tensor, node_count: int
    ) -> torch.Tensor:
        batch_size = vehicles.size(0)
        environment = self._operational_environment
        committed = (
            None
            if environment is None
            else getattr(environment, "_committed_customer_vehicle_mask", None)
        )
        if committed is None:
            visits = torch.zeros(
                batch_size,
                self.vehicle_count,
                node_count,
                dtype=torch.bool,
                device=vehicles.device,
            )
        else:
            visits = committed.detach().clone()
        visits[:, :, 0] = True
        if environment is None:
            return visits
        event_vehicles = getattr(environment, "_event_vehicle", None)
        event_customers = getattr(environment, "_event_customer", None)
        if event_vehicles and event_customers:
            vehicle_index = torch.stack(event_vehicles, dim=1).squeeze(-1)
            customer_index = torch.stack(event_customers, dim=1).squeeze(-1)
            pair_index = vehicle_index * node_count + customer_index
            counts = torch.zeros(
                batch_size,
                self.vehicle_count * node_count,
                dtype=torch.int32,
                device=vehicles.device,
            )
            counts.scatter_add_(
                1, pair_index, torch.ones_like(pair_index, dtype=torch.int32)
            )
            visits |= counts.view(batch_size, self.vehicle_count, node_count) > 0
        return visits

    def scores(
        self,
        vehicles: torch.Tensor,
        customers: torch.Tensor,
        vehicle_done: torch.Tensor,
        customer_mask: torch.Tensor,
    ) -> torch.Tensor:
        del vehicle_done, customer_mask
        batch_size = vehicles.size(0)
        node_embedding = self.node_embedding(customers)
        visits = self._tour_visit_mask(vehicles, customers.size(1))
        expanded_nodes = node_embedding[:, None, :, :].expand(
            -1, self.vehicle_count, -1, -1
        )
        tour = expanded_nodes.masked_fill(~visits.unsqueeze(-1), -torch.inf)
        tour = tour.amax(dim=2)

        distance = self._planned_distance(vehicles)
        vehicle_state = torch.cat(
            [distance.unsqueeze(-1), vehicles[:, :, :2]], dim=2
        ).reshape(batch_size, self.vehicle_count * 3)
        vehicle_context = self.vehicle_context(vehicle_state)
        tour_context = self.tour_context(
            tour.reshape(batch_size, self.vehicle_count * self.model_size)
        )
        return self.decision(
            torch.cat([vehicle_context, tour_context], dim=1)
        )


class EarliestAvailableSelector(BaseSelector):
    """MARDAM: Deterministic earliest-idle vehicle selection rule."""

    def scores(
        self,
        vehicles: torch.Tensor,
        customers: torch.Tensor,
        vehicle_done: torch.Tensor,
        customer_mask: torch.Tensor,
    ) -> torch.Tensor:
        del customers, customer_mask
        return (-vehicles[:, :, 3]).masked_fill(vehicle_done, -torch.inf)

    def select(
        self,
        vehicles: torch.Tensor,
        customers: torch.Tensor,
        vehicle_done: torch.Tensor,
        customer_mask: torch.Tensor,
        greedy: bool,
    ):
        del greedy
        safe_done = vehicle_done.clone()
        all_done = safe_done.all(dim=1)
        safe_done[all_done, 0] = False
        logits = self.scores(vehicles, customers, safe_done, customer_mask)
        index = logits.argmax(dim=1, keepdim=True)
        return index, logits, vehicles.new_zeros((vehicles.size(0), 1))


class RoundRobinSelector(BaseSelector):
    """MAAM: Deterministic round-robin vehicle dispatch rule."""

    def __init__(self, vehicle_count: int):
        super().__init__(vehicle_count, evaluation_rule="argmax")
        self._next_vehicle: torch.Tensor | None = None

    def reset_state(self) -> None:
        self._next_vehicle = None

    def scores(
        self,
        vehicles: torch.Tensor,
        customers: torch.Tensor,
        vehicle_done: torch.Tensor,
        customer_mask: torch.Tensor,
    ) -> torch.Tensor:
        del customers, customer_mask
        return vehicles.new_zeros((vehicles.size(0), self.vehicle_count)).masked_fill(
            vehicle_done, -torch.inf
        )

    def select(
        self,
        vehicles: torch.Tensor,
        customers: torch.Tensor,
        vehicle_done: torch.Tensor,
        customer_mask: torch.Tensor,
        greedy: bool,
    ):
        del greedy
        safe_done = vehicle_done.clone()
        all_done = safe_done.all(dim=1)
        safe_done[all_done, 0] = False
        batch_size = vehicles.size(0)
        if (
            self._next_vehicle is None
            or self._next_vehicle.size(0) != batch_size
            or self._next_vehicle.device != vehicles.device
        ):
            self._next_vehicle = torch.zeros(
                batch_size, dtype=torch.long, device=vehicles.device
            )

        offsets = torch.arange(self.vehicle_count, device=vehicles.device)
        candidates = (
            self._next_vehicle[:, None] + offsets[None, :]
        ) % self.vehicle_count
        candidate_done = safe_done.gather(1, candidates)
        first_feasible = (~candidate_done).to(torch.int64).argmax(dim=1)
        index = candidates.gather(1, first_feasible[:, None])
        self._next_vehicle = (index[:, 0] + 1) % self.vehicle_count
        logits = vehicles.new_full((batch_size, self.vehicle_count), -torch.inf)
        logits.scatter_(1, index, 0.0)
        return index, logits, vehicles.new_zeros((batch_size, 1))


class RandomSelector(BaseSelector):
    """Random vehicle selection policy."""

    def scores(
        self,
        vehicles: torch.Tensor,
        customers: torch.Tensor,
        vehicle_done: torch.Tensor,
        customer_mask: torch.Tensor,
    ) -> torch.Tensor:
        return vehicles.new_zeros((vehicles.size(0), self.vehicle_count))


def build_selector(
    name: str,
    vehicle_count: int,
    *,
    vehicle_state_size: int = 4,
    customer_feature_size: int = 5,
    selector_size: int = 64,
    selector_heads: int = 4,
    evaluation_rule: str = "argmax",
) -> BaseSelector:
    """Instantiate a vehicle selector by strategy or method name."""
    norm = name.strip().lower()
    if norm in ("dvnda", "independent"):
        return IndependentSelector(
            vehicle_count,
            vehicle_state_size,
            customer_feature_size,
            selector_size,
            selector_heads,
            evaluation_rule,
        )
    if norm in ("amcvn", "centralized"):
        return CentralizedSelector(
            vehicle_count,
            vehicle_state_size,
            hidden_size=selector_size * 2,
            evaluation_rule=evaluation_rule,
        )
    if norm in ("lidrl", "lidrl_tour_history"):
        return LiDRLTourHistorySelector(
            vehicle_count,
            model_size=selector_size,
            evaluation_rule=evaluation_rule,
        )
    if norm in ("mardam", "earliest_available"):
        return EarliestAvailableSelector(vehicle_count)
    if norm in ("maam", "round_robin"):
        return RoundRobinSelector(vehicle_count)
    if norm == "random":
        return RandomSelector(vehicle_count, evaluation_rule="softmax")
    raise ValueError(f"Unknown selector name: {name}")
