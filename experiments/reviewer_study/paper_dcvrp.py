"""Manuscript-faithful DCVRP data and interval execution semantics.

The public repository mixes normalized node features with an unnormalized
480-minute environment horizon and charges distance for route suffixes that are
discarded at an interval boundary.  This module keeps all neural-network inputs
normalized while preserving a single physical time scale, exact dynamic-node
ratios, the *scalar, horizon-truncated* Poisson disclosure process stated in
the revised manuscript, and Eq. (14)'s final executed route cost.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from data import DCVRP_Dataset
from learner import DCVRP_Environment


PAPER_HORIZON_MINUTES = 480.0
PAPER_INTERVAL_COUNT = 10
PAPER_VEHICLE_CAPACITY = 150.0
PAPER_VEHICLE_SPEED = 1.0
PAPER_POISSON_RATE = (1.0 + PAPER_HORIZON_MINUTES) / 2.0


def _sample_truncated_poisson(
    shape: tuple[int, ...],
    *,
    rate: float = PAPER_POISSON_RATE,
    lower: int = 1,
    upper: int = int(PAPER_HORIZON_MINUTES),
) -> torch.Tensor:
    """Sample ``Poisson(rate)`` conditional on ``lower <= X <= upper``.

    Rejection sampling implements the normalized probability mass function
    printed in the manuscript exactly.  Clamping is deliberately avoided: it
    would move all out-of-range probability mass to the two boundaries and is
    therefore not a sample from the stated truncated distribution.
    """

    if lower > upper:
        raise ValueError("lower truncation bound cannot exceed upper bound")
    samples = torch.poisson(torch.full(shape, float(rate)))
    invalid = (samples < float(lower)) | (samples > float(upper))
    while invalid.any():
        samples[invalid] = torch.poisson(
            torch.full((int(invalid.sum().item()),), float(rate))
        )
        invalid = (samples < float(lower)) | (samples > float(upper))
    return samples


def _select_instance_rates(
    batch_size: int,
    dynamic_rate: float | Iterable[float],
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(dynamic_rate, (float, int)):
        rates = torch.full((batch_size,), float(dynamic_rate), device=device)
    else:
        choices = torch.tensor(tuple(dynamic_rate), dtype=torch.float32, device=device)
        if choices.numel() == 0:
            raise ValueError("dynamic_rate choices cannot be empty")
        indices = torch.randint(0, choices.numel(), (batch_size,), device=device)
        rates = choices[indices]
    if ((rates < 0.0) | (rates > 1.0)).any():
        raise ValueError("dynamic rates must lie in [0, 1]")
    return rates


def generate_paper_dataset(
    batch_size: int,
    dynamic_rate: float | Iterable[float],
    *,
    customer_count: int = 20,
    vehicle_count: int = 4,
) -> DCVRP_Dataset:
    """Generate the synthetic distribution stated in Section IV-A.

    Physical values are generated first and only feature inputs are normalized:
    coordinates stay in [0,1], demand is divided by 150, and all temporal
    features are divided by 480.  Consequently the normalized vehicle speed is
    480 coordinate units per normalized horizon.
    """

    rates = _select_instance_rates(batch_size, dynamic_rate)
    coordinates = torch.rand(batch_size, customer_count + 1, 2)
    demands = torch.randint(
        5, 42, (batch_size, customer_count, 1), dtype=torch.int64
    ).float()
    service_minutes = torch.randint(
        10, 32, (batch_size, customer_count, 1), dtype=torch.int64
    ).float()

    dynamic_mask = torch.zeros(
        batch_size, customer_count, dtype=torch.bool
    )
    dynamic_counts = torch.round(rates * customer_count).to(torch.int64)
    for batch_index, count in enumerate(dynamic_counts.tolist()):
        if count > 0:
            selected = torch.randperm(customer_count)[:count]
            dynamic_mask[batch_index, selected] = True

    disclosure_minutes = torch.zeros(batch_size, customer_count, 1)
    # Revised manuscript, Node Revelation Process:
    #   a_i ~ Poisson(lambda) truncated to {1, ..., T},
    #   lambda = (1 + T) / 2.
    # Every dynamic customer uses the same scalar rate.  Customer indices do
    # not encode different arrival-rate parameters.
    sampled = _sample_truncated_poisson(
        (batch_size, customer_count, 1),
    )
    disclosure_minutes[dynamic_mask.unsqueeze(-1)] = sampled[
        dynamic_mask.unsqueeze(-1)
    ]

    customer_features = torch.cat(
        [
            coordinates[:, 1:, :],
            demands / PAPER_VEHICLE_CAPACITY,
            service_minutes / PAPER_HORIZON_MINUTES,
            disclosure_minutes / PAPER_HORIZON_MINUTES,
        ],
        dim=2,
    )
    depot = torch.zeros(batch_size, 1, DCVRP_Dataset.CUST_FEAT_SIZE)
    depot[:, :, :2] = coordinates[:, :1, :]
    nodes = torch.cat([depot, customer_features], dim=1)
    dataset = DCVRP_Dataset(
        vehicle_count,
        veh_capa=1.0,
        veh_speed=PAPER_VEHICLE_SPEED * PAPER_HORIZON_MINUTES,
        nodes=nodes,
        cust_mask=None,
    )
    dataset.paper_dynamic_rates = rates
    dataset.paper_dynamic_counts = dynamic_counts
    return dataset


def generate_paired_paper_datasets(
    batch_size: int,
    dynamic_rates: Iterable[float],
    *,
    customer_count: int = 20,
    vehicle_count: int = 4,
) -> dict[float, DCVRP_Dataset]:
    """Create nested dynamic-rate variants of the same physical instances.

    Coordinates, demands, service times, and disclosure samples are identical
    across variants.  The dynamic sets are nested, so increasing the rate only
    changes additional requests from static to dynamic.  This common-random-
    numbers design isolates the effect of dynamic rate from instance difficulty.
    """

    rates = tuple(sorted(float(rate) for rate in dynamic_rates))
    if not rates:
        raise ValueError("dynamic_rates cannot be empty")
    if rates[0] < 0.0 or rates[-1] > 1.0:
        raise ValueError("dynamic rates must lie in [0, 1]")
    maximum = generate_paper_dataset(
        batch_size,
        rates[-1],
        customer_count=customer_count,
        vehicle_count=vehicle_count,
    )
    dynamic_order = torch.full(
        (batch_size, customer_count),
        customer_count,
        dtype=torch.int64,
    )
    maximum_dynamic = maximum.nodes[:, 1:, 4] > 0.0
    for batch_index in range(batch_size):
        indices = maximum_dynamic[batch_index].nonzero(as_tuple=False).flatten()
        if indices.numel():
            indices = indices[torch.randperm(indices.numel())]
            dynamic_order[batch_index, indices] = torch.arange(indices.numel())

    paired: dict[float, DCVRP_Dataset] = {}
    for rate in rates:
        count = int(round(rate * customer_count))
        nodes = maximum.nodes.clone()
        active_dynamic = dynamic_order < count
        nodes[:, 1:, 4] = torch.where(
            active_dynamic,
            maximum.nodes[:, 1:, 4],
            torch.zeros_like(maximum.nodes[:, 1:, 4]),
        )
        data = DCVRP_Dataset(
            vehicle_count,
            maximum.veh_capa,
            maximum.veh_speed,
            nodes,
            None,
        )
        data.paper_dynamic_rates = torch.full((batch_size,), rate)
        data.paper_dynamic_counts = torch.full(
            (batch_size,), count, dtype=torch.int64
        )
        paired[rate] = data
    return paired


class PaperDCVRPEnvironment(DCVRP_Environment):
    """Synchronous interval environment implementing Eqs. (11)-(14).

    Complete routes are planned at every interval.  At its boundary, every leg
    already dispatched (leg start < boundary) is committed; the unstarted suffix
    is discarded and replanned.  Discarded suffix distance is refunded because
    Eq. (14) evaluates the final executed/committed route, not abandoned plans.
    """

    def __init__(
        self,
        data: DCVRP_Dataset,
        nodes: torch.Tensor | None = None,
        *,
        pending_cost: float = 5.0,
        segment_count: int = PAPER_INTERVAL_COUNT,
        travel_sigma: float = 0.0,
        service_sigma: float = 0.0,
        congestion_amplitude: float = 0.0,
        stochastic_seed: int = 0,
    ) -> None:
        self.selector = None
        self.policy_greedy = False
        super().__init__(
            data,
            nodes=nodes,
            pending_cost=pending_cost,
            segment_count=int(segment_count),
            horizon=1.0,
        )
        self.travel_sigma = float(travel_sigma)
        self.service_sigma = float(service_sigma)
        self.congestion_amplitude = float(congestion_amplitude)
        self._noise_generator = torch.Generator(device=self.nodes.device)
        self._noise_generator.manual_seed(int(stochastic_seed))
        self._transition_refund = self.nodes.new_zeros(
            (self.minibatch_size, 1)
        )
        self._finalized = False

    def _update_cur_veh(self):
        if self.selector is None:
            return super()._update_cur_veh()
        feasible_customer = (~self.mask[:, :, 1:]).any(dim=2)
        feasible_vehicle = feasible_customer & (~self.veh_done)
        row_has_feasible_vehicle = feasible_vehicle.any(dim=1)
        effective_vehicle_done = torch.where(
            row_has_feasible_vehicle[:, None],
            self.veh_done | (~feasible_vehicle),
            self.veh_done,
        )
        index, scores, log_probability = self.selector.select(
            self.vehicles,
            self.nodes,
            effective_vehicle_done,
            self.mask,
            greedy=self.policy_greedy,
        )
        self.cur_veh_idx = index
        self.cur_vehicle_scores = scores
        self.cur_vehicle_logp = log_probability
        index_state = index[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE)
        self.cur_veh = self.vehicles.gather(1, index_state)
        index_nodes = index[:, :, None].expand(-1, -1, self.nodes_count)
        self.cur_veh_mask = self.mask.gather(1, index_nodes)

    def _rebuild_mask(self):
        super()._rebuild_mask()
        # Eq. (31) is a distribution over unserved customer nodes.  A vehicle
        # may close its current plan at the depot only when it has no feasible
        # revealed customer; otherwise premature depot choices create invalid
        # low-distance/low-QoS solutions.
        has_feasible_customer = (~self.mask[:, :, 1:]).any(dim=2)
        self.mask[:, :, 0] = has_feasible_customer

    def reset(self):
        super().reset()
        self._transition_refund = self.nodes.new_zeros(
            (self.minibatch_size, 1)
        )
        self._finalized = False
        self._terminal_dispatch = False

    def _mean_one_lognormal(
        self, reference: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        if sigma <= 0.0:
            return torch.ones_like(reference)
        standard_normal = torch.randn(
            reference.shape,
            generator=self._noise_generator,
            device=reference.device,
            dtype=reference.dtype,
        )
        return torch.exp(sigma * standard_normal - 0.5 * sigma * sigma)

    def _peak_multiplier(self) -> torch.Tensor:
        current_time = self.cur_veh[:, :, 3]
        if self.congestion_amplitude <= 0.0:
            return torch.ones_like(current_time)
        first_peak = torch.exp(-0.5 * ((current_time - 0.30) / 0.09) ** 2)
        second_peak = torch.exp(-0.5 * ((current_time - 0.75) / 0.11) ** 2)
        return 1.0 + self.congestion_amplitude * (first_peak + second_peak)

    def _update_vehicles(self, dest: torch.Tensor):
        self._last_start = self.cur_veh[:, :, 3].detach().clone()
        wait_in_place = getattr(self, "_wait_in_place", None)
        if wait_in_place is None:
            wait_in_place = torch.zeros(
                (self.minibatch_size, 1), dtype=torch.bool, device=self.device
            )
        distance = torch.norm(
            self.cur_veh[:, 0, :2] - dest[:, 0, :2], dim=1, keepdim=True
        )
        travel_multiplier = self._mean_one_lognormal(
            distance, self.travel_sigma
        ) * self._peak_multiplier()
        service_multiplier = self._mean_one_lognormal(
            distance, self.service_sigma
        )
        travel_time = distance * travel_multiplier / self.veh_speed
        arrival_time = self.cur_veh[:, :, 3] + travel_time
        departure_time = arrival_time + dest[:, :, 3] * service_multiplier

        new_cur_veh = self.cur_veh.clone()
        new_cur_veh[:, :, :2] = dest[:, :, :2]
        new_cur_veh[:, :, 2] = (
            new_cur_veh[:, :, 2] - dest[:, :, 2]
        ).clamp(min=0.0)
        new_cur_veh[:, :, 3] = departure_time
        distance = torch.where(wait_in_place, torch.zeros_like(distance), distance)
        arrival_time = torch.where(
            wait_in_place, self.cur_veh[:, :, 3], arrival_time
        )
        departure_time = torch.where(
            wait_in_place, self.cur_veh[:, :, 3], departure_time
        )
        new_cur_veh = torch.where(
            wait_in_place.unsqueeze(2), self.cur_veh, new_cur_veh
        )
        updated = self.vehicles.clone()
        updated.scatter_(
            1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE),
            new_cur_veh,
        )
        self.vehicles = updated
        self.cur_veh = new_cur_veh
        self._last_distance = distance.detach().clone()
        self._last_arrival = arrival_time.detach().clone()
        self._last_departure = departure_time.detach().clone()
        return distance, arrival_time

    def _log_route_step(
        self,
        batch_idx: int,
        veh_idx: int,
        cust_idx: int,
        arrival_time: float,
        departure_time: float,
    ) -> None:
        del arrival_time, departure_time
        self.route_logs[batch_idx][veh_idx].append(
            {
                "customer": int(cust_idx),
                "start": float(self._last_start[batch_idx, 0].item()),
                "arrival": float(self._last_arrival[batch_idx, 0].item()),
                "departure": float(self._last_departure[batch_idx, 0].item()),
                "distance": float(self._last_distance[batch_idx, 0].item()),
                "vehicle_state": self.cur_veh[batch_idx, 0].detach().clone(),
            }
        )

    def _segment_transition(self, seg_time: float):
        batch_size, vehicle_count = self.minibatch_size, self.veh_count
        for batch_index in range(batch_size):
            for vehicle_index in range(vehicle_count):
                route = self.route_logs[batch_index][vehicle_index]
                cut_index = len(route)
                for event_index, event in enumerate(route):
                    if event["start"] >= seg_time:
                        cut_index = event_index
                        break
                removed = route[cut_index:]
                if removed:
                    route[:] = route[:cut_index]
                    self._transition_refund[batch_index, 0] += sum(
                        event["distance"] for event in removed
                    )
                    for event in removed:
                        customer = event["customer"]
                        if customer != 0:
                            self.served[batch_index, customer] = False

                if route:
                    last = route[-1]
                    if "vehicle_state" in last:
                        self.vehicles[batch_index, vehicle_index] = last[
                            "vehicle_state"
                        ]
                    else:
                        delivered = sum(
                            float(
                                self.nodes[
                                    batch_index, event["customer"], 2
                                ].item()
                            )
                            for event in route
                            if event["customer"] != 0
                        )
                        self.vehicles[batch_index, vehicle_index, :2] = self.nodes[
                            batch_index, last["customer"], :2
                        ]
                        self.vehicles[batch_index, vehicle_index, 2] = max(
                            float(self.veh_capa) - delivered, 0.0
                        )
                    self.vehicles[batch_index, vehicle_index, 3] = max(
                        float(last["departure"]), float(seg_time)
                    )
                else:
                    self.vehicles[batch_index, vehicle_index, :2] = self.nodes[
                        batch_index, 0, :2
                    ]
                    self.vehicles[batch_index, vehicle_index, 2] = float(
                        self.veh_capa
                    )
                    self.vehicles[batch_index, vehicle_index, 3] = float(seg_time)

        reveal = (self.nodes[:, :, 4] <= seg_time) & (~self.served)
        old_mask = self.cust_mask.clone()
        self.cust_mask = self.cust_mask & ~reveal
        reveal_expanded = reveal[:, None, :].expand(-1, vehicle_count, -1)
        self.total_cust_mask = self.total_cust_mask & ~reveal_expanded
        self.new_customers = (old_mask != self.cust_mask).any().item()
        self.current_segment += 1
        self.pending_customers = (~self.served).float().sum(-1, keepdim=True) - 1
        self.done = False
        self.veh_done[:] = False
        self.returned_to_depot[:] = False
        self._rebuild_mask()

    def _finalize_horizon(self) -> torch.Tensor:
        refund = self.nodes.new_zeros((self.minibatch_size, 1))
        for batch_index in range(self.minibatch_size):
            for vehicle_index in range(self.veh_count):
                route = self.route_logs[batch_index][vehicle_index]
                cut_index = len(route)
                for event_index, event in enumerate(route):
                    if event["start"] >= self.horizon:
                        cut_index = event_index
                        break
                removed = route[cut_index:]
                if removed:
                    route[:] = route[:cut_index]
                    refund[batch_index, 0] += sum(
                        event["distance"] for event in removed
                    )
                    for event in removed:
                        customer = event["customer"]
                        if customer != 0:
                            self.served[batch_index, customer] = False
        return refund

    def step(self, cust_idx: torch.Tensor):
        self._transition_refund.zero_()
        self._wait_in_place = (
            (cust_idx.to(self.device) == 0)
            & (self.current_segment < self.segment_count)
        )
        reward = super().step(cust_idx)
        self._wait_in_place = None
        reward = reward + self._transition_refund
        if (
            self.done
            and self.current_segment == self.segment_count - 1
            and not self._terminal_dispatch
        ):
            # Undo the base class's premature pending penalty.  Requests
            # disclosed during the last regular interval are inserted at the
            # horizon boundary and receive one final route-construction round.
            pending = (~self.served).float().sum(-1, keepdim=True) - 1
            reward = reward + self.pending_cost * pending
            self._segment_transition(self.horizon)
            reward = reward + self._transition_refund
            self._terminal_dispatch = True
            self._update_cur_veh()
            return reward
        if self.done and not self._finalized:
            pending = (~self.served).float().sum(-1, keepdim=True) - 1
            self.pending_customers = pending
            self._finalized = True
        return reward

    def route_distance(self) -> torch.Tensor:
        values = self.nodes.new_zeros(self.minibatch_size)
        for batch_index in range(self.minibatch_size):
            values[batch_index] = sum(
                event["distance"]
                for route in self.route_logs[batch_index]
                for event in route
            )
        return values

    def qos(self) -> torch.Tensor:
        pending = (~self.served).sum(dim=-1).float() - 1.0
        return 1.0 - pending / float(self.nodes_count - 1)


class VectorizedPaperDCVRPEnvironment(PaperDCVRPEnvironment):
    """GPU-vectorized implementation equivalent to PaperDCVRPEnvironment.

    It stores the current interval's planned events as tensors and commits the
    dispatched prefix for every batch/vehicle in one operation.  This removes
    per-instance Python route logging from the training hot path without
    changing the interval policy or reward definition.
    """

    def reset(self):
        self._event_vehicle: list[torch.Tensor] = []
        self._event_customer: list[torch.Tensor] = []
        self._event_start: list[torch.Tensor] = []
        self._event_distance: list[torch.Tensor] = []
        self._event_vehicle_state: list[torch.Tensor] = []
        self.total_distance = self.nodes.new_zeros(self.minibatch_size)
        self._committed_vehicle_physical_distance = self.nodes.new_zeros(
            self.minibatch_size, self.veh_count
        )
        self._committed_customer_vehicle_mask = torch.zeros(
            self.minibatch_size,
            self.veh_count,
            self.nodes_count,
            dtype=torch.bool,
            device=self.device,
        )
        self._committed_customer_vehicle_mask[:, :, 0] = True
        super().reset()
        self._segment_vehicle_snapshot = self.vehicles.detach().clone()

    def _record_tensor_event(
        self, customer_index: torch.Tensor, distance: torch.Tensor
    ) -> None:
        self._event_vehicle.append(self.cur_veh_idx.detach().clone())
        self._event_customer.append(customer_index.detach().clone())
        self._event_start.append(self._last_start.detach().clone())
        self._event_distance.append(distance.detach().clone())
        self._event_vehicle_state.append(self.cur_veh.detach().clone())

    def _commit_prefix(self, boundary: float) -> torch.Tensor:
        if not self._event_start:
            self.vehicles = self._segment_vehicle_snapshot.clone()
            self.vehicles[:, :, 3] = torch.maximum(
                self.vehicles[:, :, 3],
                self.vehicles.new_full(self.vehicles[:, :, 3].shape, boundary),
            )
            return self.nodes.new_zeros((self.minibatch_size, 1))

        vehicles = torch.stack(self._event_vehicle, dim=1).squeeze(-1)
        customers = torch.stack(self._event_customer, dim=1).squeeze(-1)
        starts = torch.stack(self._event_start, dim=1).squeeze(-1)
        distances = torch.stack(self._event_distance, dim=1).squeeze(-1)
        states = torch.stack(self._event_vehicle_state, dim=1).squeeze(2)
        committed = starts < float(boundary)
        removed = ~committed
        refund = (distances * removed).sum(dim=1, keepdim=True)

        self._committed_vehicle_physical_distance.scatter_add_(
            1, vehicles, distances * committed
        )
        pair_index = vehicles * self.nodes_count + customers
        committed_visit_counts = torch.zeros(
            self.minibatch_size,
            self.veh_count * self.nodes_count,
            dtype=torch.int32,
            device=self.device,
        )
        committed_visit_counts.scatter_add_(
            1,
            pair_index,
            committed.to(torch.int32),
        )
        self._committed_customer_vehicle_mask |= committed_visit_counts.view(
            self.minibatch_size, self.veh_count, self.nodes_count
        ) > 0

        valid_removed_customer = removed & (customers > 0)
        removal_counts = torch.zeros_like(self.served, dtype=torch.int32)
        removal_counts.scatter_add_(
            1, customers, valid_removed_customer.to(torch.int32)
        )
        self.served = self.served & (removal_counts == 0)

        event_count = starts.size(1)
        event_order = torch.arange(
            event_count, device=self.device, dtype=torch.int64
        ).view(1, event_count, 1)
        vehicle_one_hot = torch.nn.functional.one_hot(
            vehicles, num_classes=self.veh_count
        ).to(torch.bool)
        valid_event = committed.unsqueeze(-1) & vehicle_one_hot
        latest_event = torch.where(
            valid_event,
            event_order,
            event_order.new_full((1, 1, 1), -1),
        ).amax(dim=1)
        gather_index = latest_event.clamp_min(0).unsqueeze(-1).expand(
            -1, -1, self.VEH_STATE_SIZE
        )
        latest_state = states.gather(1, gather_index)
        has_committed_event = latest_event >= 0
        restored = torch.where(
            has_committed_event.unsqueeze(-1),
            latest_state,
            self._segment_vehicle_snapshot,
        )
        restored[:, :, 3] = torch.maximum(
            restored[:, :, 3],
            restored.new_full(restored[:, :, 3].shape, float(boundary)),
        )
        self.vehicles = restored
        self.total_distance = self.total_distance - refund.squeeze(1)
        return refund

    def _clear_interval_events(self) -> None:
        self._event_vehicle.clear()
        self._event_customer.clear()
        self._event_start.clear()
        self._event_distance.clear()
        self._event_vehicle_state.clear()
        self._segment_vehicle_snapshot = self.vehicles.detach().clone()

    def _segment_transition(self, seg_time: float):
        self._transition_refund += self._commit_prefix(seg_time)
        vehicle_count = self.veh_count
        reveal = (self.nodes[:, :, 4] <= seg_time) & (~self.served)
        old_mask = self.cust_mask.clone()
        self.cust_mask = self.cust_mask & ~reveal
        reveal_expanded = reveal[:, None, :].expand(-1, vehicle_count, -1)
        self.total_cust_mask = self.total_cust_mask & ~reveal_expanded
        self.new_customers = (old_mask != self.cust_mask).any().item()
        self.current_segment += 1
        self.pending_customers = (~self.served).float().sum(-1, keepdim=True) - 1
        self.done = False
        self.veh_done[:] = False
        self.returned_to_depot[:] = False
        self._rebuild_mask()
        self._clear_interval_events()

    def _finalize_vectorized_horizon(self) -> torch.Tensor:
        refund = self._commit_prefix(self.horizon)
        self._clear_interval_events()
        return refund

    def step(self, cust_idx: torch.Tensor):
        self._transition_refund.zero_()
        cust_idx = cust_idx.to(self.device)
        self._wait_in_place = (
            (cust_idx == 0) & (self.current_segment < self.segment_count)
        )
        destination = self.nodes.gather(
            1, cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE)
        )
        distance, _ = self._update_vehicles(destination)
        self._wait_in_place = None
        self._record_tensor_event(cust_idx, distance)
        self.total_distance += distance.squeeze(1)
        self._update_done(cust_idx)
        self._update_mask_after_action(cust_idx)
        reward = -distance

        if (
            self.done
            and self.current_segment == self.segment_count - 1
            and not self._terminal_dispatch
        ):
            self._segment_transition(self.horizon)
            self._terminal_dispatch = True
            reward = reward + self._transition_refund
        elif self.done and self.current_segment >= self.segment_count:
            pending = (~self.served).float().sum(-1, keepdim=True) - 1
            reward = reward - self.pending_cost * pending
            self.pending_customers = pending
            self._finalized = True
        else:
            self._check_segment_transition()
            reward = reward + self._transition_refund

        if not self.done:
            self._update_cur_veh()
        return reward

    def route_distance(self) -> torch.Tensor:
        return self.total_distance.clone()


__all__ = [
    "PAPER_HORIZON_MINUTES",
    "PAPER_INTERVAL_COUNT",
    "PAPER_POISSON_RATE",
    "generate_paper_dataset",
    "generate_paired_paper_datasets",
    "PaperDCVRPEnvironment",
    "VectorizedPaperDCVRPEnvironment",
]
