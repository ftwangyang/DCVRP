"""Time-driven DCVRP environment (Algorithm 1).

The horizon T is split into ten equal intervals. Each interval is a static
VRP: the customer set is fixed at the start of the interval and no new
requests appear until the next boundary.

At the start of interval r the controller observes customers with
appearance time a_i <= T_{r+1} and builds a route plan. At T_{r+1}:

- keep work that already started (finished, en route, or in service);
- discard the unstarted suffix and return those customers to the pool;
- idle vehicles wait in place until the next decision epoch;
- vehicles still in service cannot start a new task before they finish.

Returning to the depot restores capacity so a vehicle may continue if
leftover demand is capacity-blocked. A request counts as served if
service starts by T.
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
        record_trace: bool = True,
        disclose_horizon_tail: bool = True,
    ):
        self.veh_count = data.veh_count
        self.veh_capa = data.veh_capa
        self.veh_speed = data.veh_speed
        self.nodes = data.nodes if nodes is None else nodes
        self.minibatch_size, self.nodes_count, _ = self.nodes.size()
        self.pending_cost = float(pending_cost)
        self.device = self.nodes.device
        self.horizon = float(horizon)
        self.record_trace = record_trace
        self.disclose_horizon_tail = bool(disclose_horizon_tail)
        self.segment_count = int(segment_count)
        self.segment_duration = self.horizon / self.segment_count
        self.current_segment = 0
        self.selector = None
        self.policy_greedy = False
        self.new_customers = False
        self.interval_advanced = False

        self._event_vehicle: list[torch.Tensor] = []
        self._event_valid: list[torch.Tensor] = []
        self._event_customer: list[torch.Tensor] = []
        self._event_start: list[torch.Tensor] = []
        self._event_arrival: list[torch.Tensor] = []
        self._event_distance: list[torch.Tensor] = []
        self._event_origin: list[torch.Tensor] = []
        self._event_vehicle_state: list[torch.Tensor] = []
        self._transition_refund = self.nodes.new_zeros((self.minibatch_size, 1))
        self._finalized = False
        self.boundary_clocks: list[torch.Tensor] = []

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
        self.served = torch.zeros(
            (B, self.nodes_count), dtype=torch.bool, device=self.device
        )
        self.current_segment = 0
        self.cust_mask = torch.ones(
            (B, self.nodes_count), dtype=torch.bool, device=self.device
        )
        self.cust_mask[:, 0] = False
        self.total_cust_mask = (
            self.cust_mask[:, None, :].expand(-1, V, -1).clone()
        )
        self._sync_visibility()
        self.pending_customers = (~self.served).float().sum(-1, keepdim=True) - 1
        self.new_customers = False
        self.interval_advanced = False
        self.total_distance = self.nodes.new_zeros(B)
        self._committed_vehicle_physical_distance = self.nodes.new_zeros(B, V)
        self._committed_customer_vehicle_mask = torch.zeros(
            B, V, self.nodes_count, dtype=torch.bool, device=self.device
        )
        self._committed_customer_vehicle_mask[:, :, 0] = True
        self.last_node = torch.zeros((B, V), dtype=torch.long, device=self.device)
        self._clear_interval_events()
        self._transition_refund = self.nodes.new_zeros((B, 1))
        self._finalized = False
        self.boundary_clocks = []
        self._interval_assigned = torch.zeros(
            (B, V), dtype=torch.bool, device=self.device
        )
        self._finish_time = self.nodes.new_full(
            (B, self.nodes_count), float("inf")
        )
        self._finish_time[:, 0] = 0.0
        self._start_time = self.nodes.new_full(
            (B, self.nodes_count), float("inf")
        )
        self._start_time[:, 0] = 0.0
        self.interval_logs: list[dict] = []
        self._construct_steps = 0
        self._needs_refill = torch.zeros((B,), dtype=torch.bool, device=self.device)
        self._rebuild_mask()
        self._update_cur_veh()

    def _visibility_cutoff_for_segment(self, segment: int) -> float:
        """Customers visible at the start of interval r.

        Algorithm 1 discloses a_i <= T_{r+1} (one-interval lookahead) so the
        first interval already sees arrivals in (0, T_1]. If lookahead is
        disabled, interval r sees a_i <= T_r and the last interval uses T.
        """
        if self.disclose_horizon_tail:
            return min(self.horizon, (segment + 1) * self.segment_duration)
        if segment >= self.segment_count - 1:
            return self.horizon
        return segment * self.segment_duration

    def _visibility_cutoff(self) -> float:
        return self._visibility_cutoff_for_segment(self.current_segment)

    def _sync_visibility(self) -> None:
        cutoff = self._visibility_cutoff()
        reveal = (self.nodes[:, :, 4] <= cutoff) & (~self.served)
        self.cust_mask = self.cust_mask & ~reveal
        reveal_expanded = reveal[:, None, :].expand(-1, self.veh_count, -1)
        self.total_cust_mask = self.total_cust_mask & ~reveal_expanded
        self.new_customers = bool(reveal.any())

    def _planning_vehicles(self):
        """Current physical state of the vehicles."""
        return self.vehicles

    def _rebuild_mask(self):
        planning = self._planning_vehicles()
        self.mask = self.total_cust_mask | self.served[:, None, :]
        cap_mask = planning[:, :, 2].unsqueeze(-1) < self.nodes[:, None, :, 2]
        self.mask = self.mask | cap_mask
        self.mask = self.mask | self.veh_done[:, :, None]

        dist_to_customer = torch.norm(
            planning[:, :, None, :2] - self.nodes[:, None, :, :2], dim=-1
        )
        arrival_time = planning[:, :, None, 3] + dist_to_customer / self.veh_speed
        service_start = torch.maximum(arrival_time, self.nodes[:, None, :, 4])
        # Constraint (8): service must start by T. The static plan may extend
        # past the next boundary; unstarted stops are destroyed at T_{r+1}.
        time_mask = service_start > (self.horizon + 1e-6)
        time_mask[:, :, 0] = False
        self.mask = self.mask | time_mask

        # Keep the depot closed while this vehicle still has a feasible
        # customer. Open it when remaining demand cannot be served with
        # leftover capacity, so the fleet can refill and continue.
        has_feasible_customer = (~self.mask[:, :, 1:]).any(dim=2)
        any_vehicle_can_serve = (~self.mask[:, :, 1:]).any(dim=1)
        visible_unserved = (~self.cust_mask) & (~self.served)
        visible_unserved[:, 0] = False
        stranded = visible_unserved.clone()
        stranded[:, 1:] = visible_unserved[:, 1:] & ~any_vehicle_can_serve
        row_needs_refill = stranded.any(dim=1, keepdim=True)
        self.mask[:, :, 0] = has_feasible_customer & ~row_needs_refill
        self._needs_refill = row_needs_refill.squeeze(1)

    def _update_mask_after_action(self, cust_idx: torch.Tensor):
        new_served = self.served.clone()
        new_served.scatter_(1, cust_idx, cust_idx > 0)
        self.served = new_served
        assigned = (cust_idx > 0).squeeze(-1)
        batch = torch.arange(self.minibatch_size, device=self.device)
        customer = cust_idx.squeeze(-1)
        finish = self.cur_veh[:, 0, 3]
        service = self.nodes[batch, customer, 3]
        start = finish - service
        self._finish_time = self._finish_time.clone()
        self._start_time = self._start_time.clone()
        self._finish_time[batch, customer] = torch.where(
            assigned, finish, self._finish_time[batch, customer]
        )
        self._start_time[batch, customer] = torch.where(
            assigned, start, self._start_time[batch, customer]
        )
        self._interval_assigned = self._interval_assigned.clone()
        self._interval_assigned.scatter_(
            1, self.cur_veh_idx, assigned.unsqueeze(-1) | self._interval_assigned.gather(1, self.cur_veh_idx)
        )
        self.pending_customers = (~self.served).float().sum(-1, keepdim=True) - 1
        self._rebuild_mask()

    def _update_vehicles(self, dest: torch.Tensor):
        self._last_start = self.cur_veh[:, :, 3].detach().clone()
        self._last_origin = self.cur_veh[:, :, :2].detach().clone()
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
        service_start = torch.maximum(arrival_time, dest[:, :, 4])
        departure_time = service_start + dest[:, :, 3]

        new_cur_veh = self.cur_veh.clone()
        new_cur_veh[:, :, :2] = dest[:, :, :2]
        new_cur_veh[:, :, 2] = (
            new_cur_veh[:, :, 2] - dest[:, :, 2]
        ).clamp(min=0.0)
        new_cur_veh[:, :, 3] = departure_time

        dist = torch.where(wait_in_place, torch.zeros_like(dist), dist)
        arrival_time = torch.where(wait_in_place, self.cur_veh[:, :, 3], arrival_time)
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
        self._last_arrival = arrival_time.detach().clone()
        return dist, arrival_time

    def _update_done(self, cust_idx: torch.Tensor):
        is_depot = cust_idx == 0
        wait_in_place = getattr(self, "_wait_in_place", None)
        if wait_in_place is not None:
            curr_ret = self.returned_to_depot.gather(1, self.cur_veh_idx)
            new_ret = torch.where(wait_in_place, curr_ret, is_depot)
            self.returned_to_depot = self.returned_to_depot.clone().scatter_(
                1, self.cur_veh_idx, new_ret
            )
        else:
            self.returned_to_depot = self.returned_to_depot.clone().scatter_(
                1, self.cur_veh_idx, is_depot
            )
        # veh_done is resolved after the mask rebuild.

    def _refill_depot_capacity(self, cust_idx: torch.Tensor) -> None:
        """Restore Q when the acting vehicle reaches the depot."""
        is_depot = (cust_idx == 0).squeeze(-1)
        wait_in_place = getattr(self, "_wait_in_place", None)
        if wait_in_place is not None:
            is_depot = is_depot & (~wait_in_place.squeeze(-1))
        if not bool(is_depot.any()):
            return
        batch = torch.arange(self.minibatch_size, device=self.device)
        vehicle = self.cur_veh_idx.squeeze(-1)
        capa = self.vehicles.new_full((self.minibatch_size,), float(self.veh_capa))
        cap = self.vehicles[:, :, 2].clone()
        cap[batch, vehicle] = torch.where(is_depot, capa, cap[batch, vehicle])
        self.vehicles = self.vehicles.clone()
        self.vehicles[:, :, 2] = cap
        self.cur_veh = self.cur_veh.clone()
        self.cur_veh[:, 0, 2] = torch.where(is_depot, capa, self.cur_veh[:, 0, 2])

    def _sync_vehicle_completion(self) -> None:
        feasible_customer = (~self.mask[:, :, 1:]).any(dim=2)
        self.veh_done = self.returned_to_depot & ~feasible_customer
        if self.current_segment >= self.segment_count - 1:
            self.done = bool(self.veh_done.all().item()) and not bool(
                feasible_customer.any().item()
            )

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
            self._planning_vehicles(),
            self.nodes,
            effective_vehicle_done,
            self.mask,
            greedy=self.policy_greedy,
        )
        self.cur_veh_idx = index
        self.cur_vehicle_scores = scores
        self.cur_vehicle_logp = log_probability
        hidden = getattr(self.selector, "last_hidden", None)
        if hidden is None:
            hidden = self.vehicles.new_zeros(
                self.minibatch_size,
                self.veh_count,
                int(getattr(self.selector, "representation_size", 64)),
            )
        self.cur_vehicle_hidden = hidden.gather(
            1, index[:, :, None].expand(-1, -1, hidden.size(-1))
        )
        index_state = index[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE)
        self.cur_veh = self.vehicles.gather(1, index_state)
        index_nodes = index[:, :, None].expand(-1, -1, self.nodes_count)
        self.cur_veh_mask = self.mask.gather(1, index_nodes)

    def _record_tensor_event(
        self, customer_index: torch.Tensor, distance: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> None:
        self._event_vehicle.append(self.cur_veh_idx.detach().clone())
        self._event_valid.append(torch.ones_like(customer_index, dtype=torch.bool) if valid is None else valid.detach().clone())
        self._event_customer.append(customer_index.detach().clone())
        self._event_start.append(self._last_start.detach().clone())
        self._event_arrival.append(self._last_arrival.detach().clone())
        self._event_distance.append(distance.detach().clone())
        self._event_origin.append(self._last_origin.detach().clone())
        self._event_vehicle_state.append(self.cur_veh.detach().clone())

    def _latest_per_vehicle(
        self,
        vehicles: torch.Tensor,
        valid: torch.Tensor,
        values: torch.Tensor,
        empty: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the latest valid event payload per vehicle."""
        event_count = vehicles.size(1)
        event_order = torch.arange(
            event_count, device=self.device, dtype=torch.int64
        ).view(1, event_count, 1)
        vehicle_one_hot = torch.nn.functional.one_hot(
            vehicles, num_classes=self.veh_count
        ).to(torch.bool)
        valid_event = valid.unsqueeze(-1) & vehicle_one_hot
        latest_event = torch.where(
            valid_event,
            event_order,
            event_order.new_full((1, 1, 1), -1),
        ).amax(dim=1)
        gather_index = latest_event.clamp_min(0).unsqueeze(-1).expand(
            -1, -1, values.size(-1)
        )
        latest = values.gather(1, gather_index)
        has_event = latest_event >= 0
        latest = torch.where(has_event.unsqueeze(-1), latest, empty)
        return latest, has_event

    def _execute_until_boundary(self, boundary: float) -> torch.Tensor:
        """Execute planned routes up to T_{r+1} (Algorithm 1)."""
        B, V = self.minibatch_size, self.veh_count
        T = float(boundary)
        if not self._event_start:
            self.vehicles = self._segment_vehicle_snapshot.clone()
            self.vehicles[:, :, 3] = T
            self.boundary_clocks.append(self.vehicles[:, :, 3].detach().clone())
            return self.nodes.new_zeros((B, 1))

        vehicles = torch.stack(self._event_vehicle, dim=1).squeeze(-1)
        valid = torch.stack(self._event_valid, dim=1).squeeze(-1)
        customers = torch.stack(self._event_customer, dim=1).squeeze(-1)
        starts = torch.stack(self._event_start, dim=1).squeeze(-1)
        arrivals = torch.stack(self._event_arrival, dim=1).squeeze(-1)
        distances = torch.stack(self._event_distance, dim=1).squeeze(-1)
        origins = torch.stack(self._event_origin, dim=1)
        if origins.dim() == 4:
            origins = origins.squeeze(2)
        states = torch.stack(self._event_vehicle_state, dim=1)
        if states.dim() == 4:
            states = states.squeeze(2)
        finishes = states[:, :, 3]

        not_started = (starts > (T + 1e-6)) & valid
        committed = (starts <= (T + 1e-6)) & valid
        executed = torch.where(committed, distances, torch.zeros_like(distances))
        refund = torch.where(not_started, distances, torch.zeros_like(distances)).sum(
            dim=1, keepdim=True
        )

        self._committed_vehicle_physical_distance.scatter_add_(
            1, vehicles, executed
        )
        pair_index = vehicles * self.nodes_count + customers
        committed_counts = torch.zeros(
            B, V * self.nodes_count, dtype=torch.int32, device=self.device
        )
        committed_counts.scatter_add_(1, pair_index, committed.to(torch.int32))
        self._committed_customer_vehicle_mask |= (
            committed_counts.view(B, V, self.nodes_count) > 0
        )

        valid_removed = not_started & (customers > 0)
        removal_counts = torch.zeros_like(self.served, dtype=torch.int32)
        removal_counts.scatter_add_(1, customers, valid_removed.to(torch.int32))
        removed = removal_counts > 0
        self.served = self.served & ~removed
        self._finish_time = torch.where(
            removed, self._finish_time.new_full(self._finish_time.shape, float("inf")), self._finish_time
        )
        self._start_time = torch.where(
            removed, self._start_time.new_full(self._start_time.shape, float("inf")), self._start_time
        )
        self._finish_time[:, 0] = 0.0
        self._start_time[:, 0] = 0.0

        dest_xy = states[:, :, :2]
        event_cap = states[:, :, 2]
        event_time = torch.maximum(finishes, finishes.new_full(finishes.shape, T))
        event_payload = torch.cat(
            [dest_xy, event_cap.unsqueeze(-1), event_time.unsqueeze(-1)], dim=-1
        )
        snapshot_payload = torch.cat(
            [
                self._segment_vehicle_snapshot[:, :, :2],
                self._segment_vehicle_snapshot[:, :, 2:3],
                self._segment_vehicle_snapshot[:, :, 3:4],
            ],
            dim=-1,
        )
        restored, has_committed = self._latest_per_vehicle(
            vehicles, committed, event_payload, snapshot_payload
        )
        restored = restored.clone()
        unused_time = torch.maximum(restored[:, :, 3], restored.new_full((B, V), T))
        restored[:, :, 3] = torch.where(
            has_committed, restored[:, :, 3], unused_time
        )

        last_payload = customers.unsqueeze(-1).to(restored.dtype)
        empty_last = last_payload.new_zeros((B, V, 1))
        last_node, has_last = self._latest_per_vehicle(
            vehicles, committed, last_payload, empty_last
        )
        self.last_node = torch.where(
            has_last, last_node.squeeze(-1).long(), self.last_node
        )

        unused = ~has_committed
        restored[:, :, :2] = torch.where(
            unused.unsqueeze(-1),
            self._segment_vehicle_snapshot[:, :, :2],
            restored[:, :, :2],
        )
        restored[:, :, 2] = torch.where(
            unused,
            self._segment_vehicle_snapshot[:, :, 2],
            restored[:, :, 2],
        )
        self.vehicles = restored
        self.total_distance = self.total_distance - refund.squeeze(1)
        self.boundary_clocks.append(self.vehicles[:, :, 3].detach().clone())
        return refund


    def _clear_interval_events(self) -> None:
        self._event_vehicle.clear()
        self._event_valid.clear()
        self._event_customer.clear()
        self._event_start.clear()
        self._event_arrival.clear()
        self._event_distance.clear()
        self._event_origin.clear()
        self._event_vehicle_state.clear()
        self._segment_vehicle_snapshot = self.vehicles.detach().clone()

    def _planning_complete(self) -> bool:
        """True when this interval's static VRP has no remaining work.

        Algorithm 1 does not require every vehicle to return to the depot
        before T_{r+1}. Idle vehicles wait in place until the next epoch.
        Keep the interval open only while leftover visible demand still
        needs a depot refill to continue.
        """
        no_customer = not bool((~self.mask[:, :, 1:]).any().item())
        if not no_customer:
            return False
        needs_refill = getattr(self, "_needs_refill", None)
        if needs_refill is not None and bool(needs_refill.any().item()):
            return False
        return True

    def _check_segment_transition(self):
        if self.current_segment >= self.segment_count - 1:
            return
        if not self._planning_complete():
            return
        next_seg_time = (self.current_segment + 1) * self.segment_duration
        self._segment_transition(next_seg_time)

    def _log_interval_cut(self, boundary: float) -> None:
        """Record keep / destroy / reveal for the just-finished static plan."""
        if not self.record_trace:
            return
        planned: list[int] = []
        kept: list[int] = []
        destroyed: list[int] = []
        if self._event_customer:
            customers = torch.stack(self._event_customer, dim=1).squeeze(-1)
            starts = torch.stack(self._event_start, dim=1).squeeze(-1)
            valid = torch.stack(self._event_valid, dim=1).squeeze(-1)
            cust0 = customers[0]
            start0 = starts[0]
            valid0 = valid[0]
            for event in range(cust0.size(0)):
                cid = int(cust0[event].item())
                if cid <= 0 or not bool(valid0[event]):
                    continue
                planned.append(cid)
                if float(start0[event].item()) < float(boundary):
                    kept.append(cid)
                else:
                    destroyed.append(cid)
        visible_before = (~self.cust_mask[0]).clone()
        next_cutoff = self._visibility_cutoff_for_segment(self.current_segment + 1)
        reveal = (self.nodes[0, :, 4] <= float(next_cutoff)) & (~self.served[0])
        newly = reveal & self.cust_mask[0]
        events = []
        for i in range(len(self._event_customer)):
            if not bool(self._event_valid[i][0, 0]):
                continue
            start = float(self._event_start[i][0, 0])
            arrival = float(self._event_arrival[i][0, 0])
            state = self._event_vehicle_state[i][0, 0]
            finish = float(state[3])
            events.append(dict(vehicle=int(self._event_vehicle[i][0, 0]),
                customer=int(self._event_customer[i][0, 0]), start=start,
                arrival=arrival, finish=finish,
                origin=self._event_origin[i][0, 0].detach().cpu().tolist(),
                destination=state[:2].detach().cpu().tolist(), capacity=float(state[2]),
                decision='destroyed' if start >= boundary else ('completed' if finish <= boundary else 'committed_in_progress')))
        self.interval_logs.append(
            {
                "interval": int(self.current_segment),
                "events": events,
                "boundary": float(boundary),
                "planned": planned,
                "kept": kept,
                "destroyed": destroyed,
                "revealed": [int(i) for i in newly.nonzero(as_tuple=False).flatten().tolist() if i > 0],
                "visible_before": int(visible_before[1:].sum().item()),
                "vehicle_xy": self.vehicles[0, :, :2].detach().cpu().tolist(),
                "vehicle_time": self.vehicles[0, :, 3].detach().cpu().tolist(),
                "vehicle_cap": self.vehicles[0, :, 2].detach().cpu().tolist(),
            }
        )

    def _segment_transition(self, seg_time: float):
        self._transition_refund += self._execute_until_boundary(seg_time)
        self._log_interval_cut(seg_time)
        self.current_segment += 1
        old_mask = self.cust_mask.clone()
        self._sync_visibility()
        self.new_customers = (old_mask != self.cust_mask).any().item()
        self.interval_advanced = True
        self.pending_customers = (~self.served).float().sum(-1, keepdim=True) - 1
        self.done = False
        self.veh_done[:] = False
        self.returned_to_depot[:] = False
        self._interval_assigned[:] = False
        self._rebuild_mask()
        self._clear_interval_events()

    def _depot_return_distance(self) -> torch.Tensor:
        depot = self.nodes[:, 0, :2]
        dist = torch.norm(self.vehicles[:, :, :2] - depot[:, None, :], dim=-1)
        return dist.sum(dim=1, keepdim=True)

    def _finalize_routes(self, reward: torch.Tensor) -> torch.Tensor:
        if self._finalized:
            return reward
        if self._event_start:
            reward = reward + self._execute_until_boundary(self.horizon)
        extra = self._depot_return_distance()
        extra = torch.where(extra <= 1e-6, torch.zeros_like(extra), extra)
        self.total_distance = self.total_distance + extra.squeeze(1)
        depot = self.nodes[:, 0, :2]
        away = torch.norm(self.vehicles[:, :, :2] - depot[:, None, :], dim=-1) > 1e-6
        self.vehicles = self.vehicles.clone()
        self.vehicles[:, :, :2] = torch.where(
            away.unsqueeze(-1), depot[:, None, :], self.vehicles[:, :, :2]
        )
        self.last_node = torch.where(away, torch.zeros_like(self.last_node), self.last_node)
        pending = self.unserved_count()
        self.pending_customers = pending
        self._log_interval_cut(self.horizon)
        self._finalized = True
        return reward - extra - self.pending_cost * pending

    def step(self, cust_idx: torch.Tensor):
        self._transition_refund.zero_()
        self._construct_steps += 1
        cust_idx = cust_idx.to(self.device)
        last_interval = self.current_segment >= self.segment_count - 1
        all_veh_done = self.veh_done.all(dim=1, keepdim=True)
        has_feasible_customer = (~self.mask[:, :, 1:]).any(dim=2).any(dim=1, keepdim=True)
        needs_refill = (
            self._needs_refill.view(-1, 1)
            if getattr(self, "_needs_refill", None) is not None
            else torch.zeros_like(all_veh_done)
        )
        has_feasible_work = has_feasible_customer | needs_refill
        wait_in_place = (
            ((~has_feasible_work) & (cust_idx == 0) & (not last_interval))
            | (all_veh_done & (cust_idx == 0))
        )
        self._wait_in_place = wait_in_place
        destination = self.nodes.gather(
            1, cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE)
        )
        distance, _ = self._update_vehicles(destination)
        self._refill_depot_capacity(cust_idx)
        self._record_tensor_event(cust_idx, distance, valid=~wait_in_place)
        self.total_distance += distance.squeeze(1)
        self.last_node.scatter_(
            1,
            self.cur_veh_idx,
            torch.where(
                wait_in_place,
                self.last_node.gather(1, self.cur_veh_idx),
                cust_idx,
            ),
        )
        self._update_done(cust_idx)
        self._wait_in_place = None
        self._update_mask_after_action(cust_idx)
        self._sync_vehicle_completion()
        reward = -distance

        max_steps = self.nodes_count * self.veh_count * (self.segment_count + 5)
        if self._construct_steps > max_steps and not self.done:
            self.done = True
            last_interval = True

        if self.done and last_interval:
            reward = self._finalize_routes(reward)
        else:
            self._check_segment_transition()
            reward = reward + self._transition_refund

        if not self.done:
            self._update_cur_veh()
        return reward

    def route_distance(self) -> torch.Tensor:
        """Cumulative physical distance of executed routes (Eq. 14)."""
        return self.total_distance.clone()

    def physically_completed(self) -> torch.Tensor:
        """Constraint (8): assigned and service started at or before T."""
        done = self.served.clone()
        done[:, 1:] = self.served[:, 1:] & (
            self._start_time[:, 1:] <= (self.horizon + 1e-6)
        )
        return done

    def unserved_count(self) -> torch.Tensor:
        """|C_unserved| in Eq. 14: not started by the planning horizon."""
        return (~self.physically_completed()).float().sum(-1, keepdim=True) - 1.0

    def qos(self) -> torch.Tensor:
        """Quality of Service: fraction whose service started by T."""
        pending = self.unserved_count().squeeze(-1)
        return 1.0 - pending / float(self.nodes_count - 1)
