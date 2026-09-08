"""Vehicle selectors used in controlled architecture ablations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.func import functional_call, vmap

from learner import VehicleSelectionNetwork


class BaseSelector(nn.Module):
    def __init__(self, vehicle_count: int, evaluation_rule: str = "argmax"):
        super().__init__()
        self.vehicle_count = vehicle_count
        self.evaluation_rule = evaluation_rule

    def scores(self, vehicles, customers, vehicle_done, customer_mask):
        raise NotImplementedError

    def select(self, vehicles, customers, vehicle_done, customer_mask, greedy: bool):
        # The supplied environment advances a minibatch synchronously.  A row
        # can therefore finish before the remaining rows.  Keep one harmless
        # depot action available for such rows instead of constructing an
        # all-masked categorical distribution (which yields NaNs).
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
    def __init__(self, vehicle_count, vehicle_state_size=4, customer_feature_size=5,
                 model_size=64, head_count=4, evaluation_rule="argmax"):
        super().__init__(vehicle_count, evaluation_rule)
        self.vehicle_networks = nn.ModuleList([
            VehicleSelectionNetwork(
                vehicle_state_size, customer_feature_size, model_size, head_count
            )
            for _ in range(vehicle_count)
        ])

    def scores(self, vehicles, customers, vehicle_done, customer_mask):
        # Evaluate the genuinely independent parameter sets as one batched
        # functional call.  This is mathematically identical to the former
        # Python loop, but removes its severe m=100 interpreter overhead.
        parameter_maps = [dict(network.named_parameters()) for network in self.vehicle_networks]
        stacked_parameters = {
            name: torch.stack([parameters[name] for parameters in parameter_maps], dim=0)
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

        def evaluate_one(parameters, own_state, own_customer_mask):
            return functional_call(
                template,
                parameters,
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


class SingleObjectiveIndependentSelector(IndependentSelector):
    """Independent DVNDA with symmetry augmentation and a near-tie decoder.

    The training reward and all trainable layers are identical to the original
    independent selector.  During training, a randomly permuted independent
    module processes each physical vehicle state; this discourages a module
    from memorising a fixed vehicle slot.  During greedy evaluation, the
    original neural argmax is preserved unless multiple feasible vehicle
    scores lie within ``tie_tolerance`` standardized score units.  Only within
    that near-tie set is the earlier-available, then less-travelled, vehicle
    selected.  No operational metric enters the loss or checkpoint selection.
    """

    def __init__(
        self,
        vehicle_count,
        vehicle_state_size=4,
        customer_feature_size=5,
        model_size=64,
        head_count=4,
        evaluation_rule="argmax",
        tie_tolerance=0.0,
    ):
        super().__init__(
            vehicle_count,
            vehicle_state_size,
            customer_feature_size,
            model_size,
            head_count,
            evaluation_rule,
        )
        self.tie_tolerance = float(tie_tolerance)
        self._network_permutation: list[int] | None = None
        self._operational_environment = None

    def reset_state(self) -> None:
        if self.training:
            self._network_permutation = torch.randperm(
                self.vehicle_count
            ).tolist()
        else:
            self._network_permutation = list(range(self.vehicle_count))

    def set_operational_environment(self, environment) -> None:
        self._operational_environment = environment

    def clear_operational_environment(self) -> None:
        self._operational_environment = None

    def _ordered_parameter_maps(self):
        order = self._network_permutation
        if order is None:
            order = list(range(self.vehicle_count))
        return [
            dict(self.vehicle_networks[network_index].named_parameters())
            for network_index in order
        ]

    def scores(self, vehicles, customers, vehicle_done, customer_mask):
        parameter_maps = self._ordered_parameter_maps()
        stacked_parameters = {
            name: torch.stack([parameters[name] for parameters in parameter_maps], dim=0)
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

        def evaluate_one(parameters, own_state, own_customer_mask):
            return functional_call(
                template,
                parameters,
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

    def _cumulative_vehicle_distance(self, vehicles: torch.Tensor) -> torch.Tensor:
        environment = self._operational_environment
        if environment is None:
            return torch.zeros_like(vehicles[:, :, 3])
        committed = getattr(
            environment, "_committed_vehicle_physical_distance", None
        )
        if committed is None:
            distance = getattr(environment, "vehicle_distance", None)
            return (
                torch.zeros_like(vehicles[:, :, 3])
                if distance is None
                else distance.detach().clone()
            )
        distance = committed.detach().clone()
        event_vehicles = getattr(environment, "_event_vehicle", None)
        event_distances = getattr(environment, "_event_physical_distance", None)
        if event_vehicles and event_distances:
            vehicle_index = torch.stack(event_vehicles, dim=1).squeeze(-1)
            physical_distance = torch.stack(event_distances, dim=1).squeeze(-1)
            distance.scatter_add_(1, vehicle_index, physical_distance)
        return distance

    def select(self, vehicles, customers, vehicle_done, customer_mask, greedy):
        if not greedy or self.evaluation_rule != "argmax":
            return super().select(
                vehicles, customers, vehicle_done, customer_mask, greedy
            )

        safe_done = vehicle_done.clone()
        all_done = safe_done.all(dim=1)
        safe_done[all_done, 0] = False
        logits = self.scores(vehicles, customers, safe_done, customer_mask)
        logits = logits.masked_fill(safe_done, -torch.inf)
        distribution = Categorical(logits=logits)

        feasible = ~safe_done
        feasible_count = feasible.sum(dim=1, keepdim=True).clamp_min(1)
        finite_logits = torch.where(feasible, logits, torch.zeros_like(logits))
        mean = finite_logits.sum(dim=1, keepdim=True) / feasible_count
        variance = torch.where(
            feasible, (logits - mean).square(), torch.zeros_like(logits)
        ).sum(dim=1, keepdim=True) / feasible_count
        scale = variance.sqrt().clamp_min(1.0e-6)
        maximum = logits.max(dim=1, keepdim=True).values
        near_tie = feasible & (
            (maximum - logits) <= self.tie_tolerance * scale
        )

        timeline = vehicles[:, :, 3]
        earliest_time = timeline.masked_fill(~near_tie, torch.inf).min(
            dim=1, keepdim=True
        ).values
        earliest = near_tie & (timeline <= earliest_time + 1.0e-7)

        distance = self._cumulative_vehicle_distance(vehicles)
        least_distance = distance.masked_fill(~earliest, torch.inf).min(
            dim=1, keepdim=True
        ).values
        finalists = earliest & (distance <= least_distance + 1.0e-7)
        choice_logits = logits.masked_fill(~finalists, -torch.inf)
        index = choice_logits.argmax(dim=1)
        log_probability = distribution.log_prob(index).unsqueeze(1)
        return index.unsqueeze(1), logits, log_probability


class OperationalIndependentSelector(IndependentSelector):
    """Independent DVNDA selector calibrated for operational objectives.

    The original independently parameterized vehicle scores are retained. A
    deterministic calibration term favors a vehicle with an earlier timeline
    and a smaller cumulative executed/planned distance. The latter is read
    from the active environment and is used only by the explicitly labelled
    DVNDA-MO variant; the original DVNDA implementation is unchanged.

    Times are normalized to the paper horizon and synthetic route distances
    are expressed in normalized coordinate units. The default weights were
    selected on a validation set disjoint from the reviewer test set.
    """

    def __init__(
        self,
        vehicle_count,
        vehicle_state_size=4,
        customer_feature_size=5,
        model_size=64,
        head_count=4,
        evaluation_rule="argmax",
        timeline_weight=10_000.0,
        balance_weight=50.0,
    ):
        super().__init__(
            vehicle_count,
            vehicle_state_size,
            customer_feature_size,
            model_size,
            head_count,
            evaluation_rule,
        )
        self.timeline_weight = float(timeline_weight)
        self.balance_weight = float(balance_weight)
        self._operational_environment = None

    def set_operational_environment(self, environment) -> None:
        self._operational_environment = environment

    def clear_operational_environment(self) -> None:
        self._operational_environment = None

    def _cumulative_vehicle_distance(self, vehicles: torch.Tensor) -> torch.Tensor:
        environment = self._operational_environment
        if environment is None:
            return torch.zeros_like(vehicles[:, :, 3])

        committed = getattr(
            environment, "_committed_vehicle_physical_distance", None
        )
        if committed is None:
            distance = getattr(environment, "vehicle_distance", None)
            return (
                torch.zeros_like(vehicles[:, :, 3])
                if distance is None
                else distance.detach().clone()
            )

        distance = committed.detach().clone()
        event_vehicles = getattr(environment, "_event_vehicle", None)
        event_distances = getattr(environment, "_event_physical_distance", None)
        if event_vehicles and event_distances:
            vehicle_index = torch.stack(event_vehicles, dim=1).squeeze(-1)
            physical_distance = torch.stack(event_distances, dim=1).squeeze(-1)
            distance.scatter_add_(1, vehicle_index, physical_distance)
        return distance

    def scores(self, vehicles, customers, vehicle_done, customer_mask):
        neural_scores = super().scores(
            vehicles, customers, vehicle_done, customer_mask
        )
        timeline = vehicles[:, :, 3]
        distance = self._cumulative_vehicle_distance(vehicles)
        return (
            neural_scores
            - self.timeline_weight * timeline
            - self.balance_weight * distance
        )


class SharedSelector(BaseSelector):
    def __init__(self, vehicle_count, vehicle_state_size=4, customer_feature_size=5,
                 model_size=64, head_count=4, evaluation_rule="argmax"):
        super().__init__(vehicle_count, evaluation_rule)
        self.network = VehicleSelectionNetwork(
            vehicle_state_size, customer_feature_size, model_size, head_count
        )

    def scores(self, vehicles, customers, vehicle_done, customer_mask):
        batch_size, vehicle_count, state_size = vehicles.shape
        node_count, customer_size = customers.shape[1:]
        own_state = vehicles.reshape(batch_size * vehicle_count, 1, state_size)
        fleet = vehicles[:, None, :, :].expand(
            batch_size, vehicle_count, vehicle_count, state_size
        ).reshape(batch_size * vehicle_count, vehicle_count, state_size)
        demand = customers[:, None, :, :].expand(
            batch_size, vehicle_count, node_count, customer_size
        ).reshape(batch_size * vehicle_count, node_count, customer_size)
        fleet_mask = vehicle_done[:, None, :].expand(
            batch_size, vehicle_count, vehicle_count
        ).reshape(batch_size * vehicle_count, vehicle_count)
        if customer_mask.dim() == 3:
            # The latest time-driven environment maintains a capacity-aware
            # customer mask for every candidate vehicle.  Preserve that same
            # per-vehicle information when applying the shared scorer.
            demand_mask = customer_mask.reshape(
                batch_size * vehicle_count, node_count
            )
        else:
            demand_mask = customer_mask[:, None, :].expand(
                batch_size, vehicle_count, node_count
            ).reshape(batch_size * vehicle_count, node_count)
        return self.network(
            own_state, fleet, demand, fleet_mask, demand_mask
        ).reshape(batch_size, vehicle_count)


class SharedIdSelector(BaseSelector):
    """Shared selector augmented with a learned vehicle-identity embedding."""

    def __init__(self, vehicle_count, vehicle_state_size=4, customer_feature_size=5,
                 model_size=64, head_count=4, id_size=8, evaluation_rule="argmax"):
        super().__init__(vehicle_count, evaluation_rule)
        self.identity = nn.Embedding(vehicle_count, id_size)
        self.network = VehicleSelectionNetwork(
            vehicle_state_size + id_size,
            customer_feature_size,
            model_size,
            head_count,
        )

    def scores(self, vehicles, customers, vehicle_done, customer_mask):
        batch_size = vehicles.size(0)
        identities = self.identity.weight.unsqueeze(0).expand(batch_size, -1, -1)
        augmented = torch.cat([vehicles, identities], dim=2)
        values = []
        for index in range(self.vehicle_count):
            own_state = augmented[:, index:index + 1, :]
            value = self.network(
                own_state, augmented, customers, vehicle_done, customer_mask
            ).squeeze(1)
            values.append(value)
        return torch.stack(values, dim=1)


class CentralizedSelector(BaseSelector):
    """Feature-based centralized selector with shared parameters."""

    def __init__(self, vehicle_count, vehicle_state_size=4, hidden_size=128,
                 evaluation_rule="argmax"):
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

    def scores(self, vehicles, customers, vehicle_done, customer_mask):
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
        context = torch.cat([
            vehicle_features,
            fleet.expand_as(vehicle_features),
            customer_context,
        ], dim=2)
        return self.decision(context).squeeze(2)


class AttentionAggregationSelector(BaseSelector):
    """Learned attention aggregation over the full fleet and visible demand."""

    def __init__(self, vehicle_count, vehicle_state_size=4, model_size=32,
                 head_count=4, evaluation_rule="argmax"):
        super().__init__(vehicle_count, evaluation_rule)
        self.vehicle_embedding = nn.Linear(vehicle_state_size, model_size)
        self.customer_embedding = nn.Linear(5, model_size)
        self.fleet_attention = nn.MultiheadAttention(
            model_size, head_count, batch_first=True
        )
        self.demand_attention = nn.MultiheadAttention(
            model_size, head_count, batch_first=True
        )
        self.decision = nn.Sequential(
            nn.Linear(model_size * 3, model_size),
            nn.ReLU(),
            nn.Linear(model_size, 1),
        )

    def scores(self, vehicles, customers, vehicle_done, customer_mask):
        vehicle = self.vehicle_embedding(vehicles)
        fleet, _ = self.fleet_attention(
            vehicle, vehicle, vehicle, key_padding_mask=vehicle_done
        )
        customer = self.customer_embedding(customers)
        if customer_mask.dim() == 3:
            # MultiheadAttention accepts one key-padding mask per batch row,
            # not a different mask for every query vehicle.  The collaboration
            # context therefore retains a customer whenever at least one
            # feasible vehicle can serve it; the environment's candidate-
            # vehicle mask still enforces capacity feasibility at selection.
            customer_mask = customer_mask.all(dim=1)
        demand, _ = self.demand_attention(
            fleet,
            customer,
            customer,
            key_padding_mask=customer_mask,
        )
        return self.decision(torch.cat([vehicle, fleet, demand], dim=2)).squeeze(2)


class LiDRLTourHistorySelector(BaseSelector):
    """LiDRL/HCVRP-style centralized vehicle-selection head.

    The official HCVRP implementation scores all vehicles from two contexts:
    current route length/location for the complete fleet, and a concatenation
    of max-pooled node embeddings for the tour built by every vehicle.  This
    adapter generalizes the original fixed three-vehicle head to ``m`` vehicles
    and reads executed/provisional tour state from the unified environment.
    """

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
        self,
        vehicles: torch.Tensor,
        node_count: int,
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
        # The depot is the initial element of every tour, ensuring max pooling
        # is well-defined before the first customer is selected.
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
                1,
                pair_index,
                torch.ones_like(pair_index, dtype=torch.int32),
            )
            visits |= counts.view(
                batch_size, self.vehicle_count, node_count
            ) > 0
        return visits

    def scores(self, vehicles, customers, vehicle_done, customer_mask):
        del vehicle_done, customer_mask
        batch_size, _, _ = vehicles.shape
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


class RandomSelector(BaseSelector):
    def scores(self, vehicles, customers, vehicle_done, customer_mask):
        return vehicles.new_zeros((vehicles.size(0), self.vehicle_count))


class EarliestAvailableSelector(BaseSelector):
    """Deterministic earliest-idle vehicle rule used by the MARDAM proxy.

    The fourth vehicle-state component is the vehicle timeline.  Selecting its
    minimum among feasible vehicles implements the paper-level rule that a new
    neural customer decision is requested whenever the next vehicle becomes
    available.  The rule has no trainable vehicle-selection parameters; the
    customer decoder remains trainable.
    """

    def scores(self, vehicles, customers, vehicle_done, customer_mask):
        del customers, customer_mask
        return (-vehicles[:, :, 3]).masked_fill(vehicle_done, -torch.inf)

    def select(self, vehicles, customers, vehicle_done, customer_mask, greedy):
        del greedy
        safe_done = vehicle_done.clone()
        all_done = safe_done.all(dim=1)
        safe_done[all_done, 0] = False
        logits = self.scores(vehicles, customers, safe_done, customer_mask)
        index = logits.argmax(dim=1, keepdim=True)
        return index, logits, vehicles.new_zeros((vehicles.size(0), 1))


class RoundRobinSelector(BaseSelector):
    """Deterministic feasible round-robin rule used by the MAAM proxy."""

    def __init__(self, vehicle_count: int):
        super().__init__(vehicle_count, evaluation_rule="argmax")
        self._next_vehicle: torch.Tensor | None = None

    def reset_state(self) -> None:
        self._next_vehicle = None

    def scores(self, vehicles, customers, vehicle_done, customer_mask):
        del customers, customer_mask
        return vehicles.new_zeros((vehicles.size(0), self.vehicle_count)).masked_fill(
            vehicle_done, -torch.inf
        )

    def select(self, vehicles, customers, vehicle_done, customer_mask, greedy):
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
        logits = vehicles.new_full(
            (batch_size, self.vehicle_count), -torch.inf
        )
        logits.scatter_(1, index, 0.0)
        return index, logits, vehicles.new_zeros((batch_size, 1))


def build_selector(
    name: str,
    vehicle_count: int,
    selector_size: int = 64,
    selector_heads: int = 4,
    evaluation_rule: str = "argmax",
) -> BaseSelector:
    normalized = name.lower()
    if normalized == "independent":
        return IndependentSelector(
            vehicle_count, model_size=selector_size, head_count=selector_heads,
            evaluation_rule=evaluation_rule,
        )
    if normalized == "operational_independent":
        return OperationalIndependentSelector(
            vehicle_count, model_size=selector_size, head_count=selector_heads,
            evaluation_rule=evaluation_rule,
        )
    if normalized == "single_objective_independent":
        return SingleObjectiveIndependentSelector(
            vehicle_count, model_size=selector_size, head_count=selector_heads,
            evaluation_rule=evaluation_rule,
        )
    if normalized == "independent_narrow":
        return IndependentSelector(
            vehicle_count, model_size=max(16, selector_size // 2),
            head_count=max(1, selector_heads // 2), evaluation_rule=evaluation_rule,
        )
    if normalized == "shared":
        return SharedSelector(
            vehicle_count, model_size=selector_size, head_count=selector_heads,
            evaluation_rule=evaluation_rule,
        )
    if normalized == "shared_wide":
        return SharedSelector(
            vehicle_count,
            model_size=int(math.ceil(selector_size * math.sqrt(vehicle_count) / selector_heads) * selector_heads),
            head_count=selector_heads,
            evaluation_rule=evaluation_rule,
        )
    if normalized == "shared_id":
        return SharedIdSelector(
            vehicle_count, model_size=selector_size, head_count=selector_heads,
            evaluation_rule=evaluation_rule,
        )
    if normalized == "centralized":
        return CentralizedSelector(
            vehicle_count, hidden_size=selector_size * 2,
            evaluation_rule=evaluation_rule,
        )
    if normalized == "attention_aggregation":
        return AttentionAggregationSelector(
            vehicle_count,
            model_size=selector_size,
            head_count=selector_heads,
            evaluation_rule=evaluation_rule,
        )
    if normalized == "lidrl_tour_history":
        return LiDRLTourHistorySelector(
            vehicle_count,
            model_size=selector_size,
            evaluation_rule=evaluation_rule,
        )
    if normalized == "random":
        return RandomSelector(vehicle_count, evaluation_rule="softmax")
    if normalized == "earliest_available":
        return EarliestAvailableSelector(vehicle_count)
    if normalized == "round_robin":
        return RoundRobinSelector(vehicle_count)
    raise ValueError(f"Unknown selector: {name}")
