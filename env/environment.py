"""Dynamic Capacitated Vehicle Routing Problem (DCVRP) Environment.

Implements the time-driven synchronized decision interval framework described in
Section III & IV-A of the manuscript:
- Planning Horizon T = 480 minutes, divided into beta = 10 equal synchronized intervals (48 min each).
- At each interval boundary, dispatched legs (start time < boundary) are committed.
- Any unstarted provisional route suffix is refunded, releasing remaining customers for dynamic replanning.
- The evaluated route distance is the cumulative physical distance of the final executed routes (Eq. 14).
"""

from __future__ import annotations

import torch

from .dataset import DCVRPDataset


class DCVRPEnvironment:
    """Vectorized time-driven synchronized environment for DCVRP."""

    VEH_STATE_SIZE = 4  # [x, y, remaining_capacity, current_time]
    CUST_FEAT_SIZE = 5  # [x, y, demand, service_duration, appearance_time]

    def __init__(
        self,
        data: DCVRPDataset,
        nodes: torch.Tensor | None = None,
        pending_cost: float = 5.0,
        segment_count: int = 10,
        horizon: float = 1.0,
    ):
        self.veh_count = data.veh_count
        self.veh_capa = data.veh_capa
        self.veh_speed = data.veh_speed
        self.nodes = data.nodes if nodes is None else nodes
        self.minibatch_size, self.nodes_count, _ = self.nodes.size()
        self.pending_cost = float(pending_cost)
        self.device = self.nodes.device
        self.horizon = float(horizon)
        self.segment_count = int(segment_count)
        self.segment_duration = self.horizon / self.segment_count
        self.current_segment = 0
        self.selector = None
        self.policy_greedy = False
        self.new_customers = False

        # Event ledger for interval prefix commitment
        self._event_vehicle: list[torch.Tensor] = []
        self._event_customer: list[torch.Tensor] = []
        self._event_start: list[torch.Tensor] = []
        self._event_distance: list[torch.Tensor] = []
        self._event_vehicle_state: list[torch.Tensor] = []
        self._transition_refund = self.nodes.new_zeros((self.minibatch_size, 1))
        self._finalized = False

    def reset(self):
        B, V = self.minibatch_size, self.veh_count
        self.vehicles = self.nodes.new_zeros((B, V, self.VEH_STATE_SIZE))
        self.vehicles[:, :, :2] = self.nodes[:, 0:1, :2]
        self.vehicles[:, :, 2] = self.veh_capa
        self.vehicles[:, :, 3] = 0.0

        self.veh_done = torch.zeros((B, V), dtype=torch.bool, device=self.device)
        self.returned_to_depot = torch.zeros(
            (B, V), dtype=torch.bool, device=self.device
        )
        self.done = False
        self.cust_mask = self.nodes[:, :, 4] > 0
        self.total_cust_mask = (
            self.cust_mask[:, None, :].expand(-1, V, -1).clone()
        )
        self.served = torch.zeros(
            (B, self.nodes_count), dtype=torch.bool, device=self.device
        )
        self.pending_customers = (~self.served).float().sum(-1, keepdim=True) - 1
        self.current_segment = 0
        self.new_customers = False
        self.total_distance = self.nodes.new_zeros(B)
        self._committed_vehicle_physical_distance = self.nodes.new_zeros(B, V)
        self._committed_customer_vehicle_mask = torch.zeros(
            B, V, self.nodes_count, dtype=torch.bool, device=self.device
        )
        self._committed_customer_vehicle_mask[:, :, 0] = True
        self._clear_interval_events()
        self._transition_refund = self.nodes.new_zeros((B, 1))
        self._finalized = False
        self._rebuild_mask()
        self._update_cur_veh()

    def _rebuild_mask(self):
        self.mask = self.total_cust_mask | self.served[:, None, :]
        cap_mask = self.vehicles[:, :, 2].unsqueeze(-1) < self.nodes[:, None, :, 2]
        self.mask = self.mask | cap_mask
        self.mask = self.mask | self.veh_done[:, :, None]

        # Time feasibility constraint (Eq. 8): customer must be serviceable and vehicle able to return before horizon T
        dist_to_cust = torch.norm(
            self.vehicles[:, :, None, :2] - self.nodes[:, None, :, :2], dim=-1
        )
        travel_time = dist_to_cust / self.veh_speed
        arrival_time = self.vehicles[:, :, None, 3] + travel_time
        service_start = torch.maximum(arrival_time, self.nodes[:, None, :, 4])
        service_finish = service_start + self.nodes[:, None, :, 3]
        dist_to_depot = torch.norm(
            self.nodes[:, None, :, :2] - self.nodes[:, None, 0:1, :2], dim=-1
        )
        return_time = service_finish + dist_to_depot / self.veh_speed
        time_mask = return_time > (self.horizon + 1e-6)
        time_mask[:, :, 0] = False
        self.mask = self.mask | time_mask

        # Allow returning to depot only when all other dynamic customers cannot be served
        has_feasible_customer = (~self.mask[:, :, 1:]).any(dim=2)
        self.mask[:, :, 0] = has_feasible_customer

    def _update_mask_after_action(self, cust_idx: torch.Tensor):
        new_served = self.served.clone()
        new_served.scatter_(1, cust_idx, cust_idx > 0)
        self.served = new_served
        self.pending_customers = (~self.served).float().sum(-1, keepdim=True) - 1
        self._rebuild_mask()

    def _update_vehicles(self, dest: torch.Tensor):
        self._last_start = self.cur_veh[:, :, 3].detach().clone()
        wait_in_place = getattr(self, "_wait_in_place", None)
        if wait_in_place is None:
            wait_in_place = torch.zeros(
                (self.minibatch_size, 1), dtype=torch.bool, device=self.device
            )

        dist = torch.norm(
            self.cur_veh[:, 0, :2] - dest[:, 0, :2], dim=1, keepdim=True
        )
        travel_time = dist / self.veh_speed
        arrival_time = self.cur_veh[:, :, 3] + travel_time
        departure_time = arrival_time + dest[:, :, 3]

        new_cur_veh = self.cur_veh.clone()
        new_cur_veh[:, :, :2] = dest[:, :, :2]
        new_cur_veh[:, :, 2] = (
            new_cur_veh[:, :, 2] - dest[:, :, 2]
        ).clamp(min=0.0)
        new_cur_veh[:, :, 3] = departure_time

        dist = torch.where(wait_in_place, torch.zeros_like(dist), dist)
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
        return dist, arrival_time

    def _update_done(self, cust_idx: torch.Tensor):
        is_depot = cust_idx == 0
        self.returned_to_depot = self.returned_to_depot.clone().scatter_(
            1, self.cur_veh_idx, is_depot
        )
        self.veh_done = self.veh_done.clone().scatter_(
            1, self.cur_veh_idx, is_depot
        )
        if self.current_segment >= self.segment_count - 1:
            self.done = self.veh_done.all(dim=1).all().item()

    def _update_cur_veh(self):
        if self.selector is None:
            raise RuntimeError("Selector is not configured on environment.")
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
        committed_counts = torch.zeros(
            self.minibatch_size,
            self.veh_count * self.nodes_count,
            dtype=torch.int32,
            device=self.device,
        )
        committed_counts.scatter_add_(1, pair_index, committed.to(torch.int32))
        self._committed_customer_vehicle_mask |= (
            committed_counts.view(
                self.minibatch_size, self.veh_count, self.nodes_count
            )
            > 0
        )

        valid_removed = removed & (customers > 0)
        removal_counts = torch.zeros_like(self.served, dtype=torch.int32)
        removal_counts.scatter_add_(1, customers, valid_removed.to(torch.int32))
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

    def _check_segment_transition(self):
        if self.current_segment >= self.segment_count - 1:
            return
        if not self.returned_to_depot.all().item():
            return
        next_seg_time = (self.current_segment + 1) * self.segment_duration
        self._segment_transition(next_seg_time)

    def _segment_transition(self, seg_time: float):
        self._transition_refund += self._commit_prefix(seg_time)
        reveal = (self.nodes[:, :, 4] <= seg_time) & (~self.served)
        old_mask = self.cust_mask.clone()
        self.cust_mask = self.cust_mask & ~reveal
        reveal_expanded = reveal[:, None, :].expand(-1, self.veh_count, -1)
        self.total_cust_mask = self.total_cust_mask & ~reveal_expanded
        self.new_customers = (old_mask != self.cust_mask).any().item()
        self.current_segment += 1
        self.pending_customers = (~self.served).float().sum(-1, keepdim=True) - 1
        self.done = False
        self.veh_done[:] = False
        self.returned_to_depot[:] = False
        self._rebuild_mask()
        self._clear_interval_events()

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

        if self.done and self.current_segment >= self.segment_count - 1:
            # Reached planning horizon T: apply penalty for any unserved customers (Eq. 14)
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
        """Cumulative physical distance of executed routes (Eq. 14)."""
        return self.total_distance.clone()

    def qos(self) -> torch.Tensor:
        """Quality of Service: percentage of customers successfully fulfilled."""
        pending = (~self.served).sum(dim=-1).float() - 1.0
        return 1.0 - pending / float(self.nodes_count - 1)
