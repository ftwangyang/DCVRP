"""Time-driven DCVRP environment (10 equal intervals over T = 480).

Each interval is one static VRP on the customers already visible at its
start. A vehicle may only dispatch a customer if its planning clock is
still before T_{r+1}; in-progress work that started before the boundary is
kept (Algorithm 1). Planning the rest of the horizon and cutting the
suffix leaks lookahead and underprices high dynamism.

- Keep: service already finished, or already started / in progress.
  The vehicle is placed at that customer (requirement doc P20 / P51).
- Destroy: planned start after T_i. Those customers return to the pool.
- Reveal: customers with disclosure time <= T_i become visible.
  Entering the last interval also discloses remaining (T_9, T] arrivals;
  otherwise a homogeneous Poisson process permanently hides ~phi/10 of
  customers and QoS cannot reach Table I's 100%. Constraint (8) still
  waits until a_i and refuses starts after T.

Capacity is never refilled. Each interval plan includes at most one depot
return per vehicle. Mid-interval depot is a real trip only when no hidden
customers remain; otherwise it ends planning in place so Algorithm 1 cannot
commit an instant speed-480 return. A committed leg is represented by its
destination and its actual availability time, which may be later than the
interval boundary.

When more than half the customers start unrevealed, a vehicle may not take
a request that would drop its leftover capacity below one max-size demand
if another feasible vehicle can keep that slack. Without that no-refill
coupling, phi=75% fills two or three vehicles early; the last interval then
drops late HPP arrivals that were static at phi=50%, so the 50-to-75
distance jump stays too small.

On those same high-dynamism instances, the first depot action while hidden
customers remain is a real return (Algorithm 1 keeps it: start < T_i). Later
interval terminators stay in the field so speed 480 cannot commit a round
trip every interval. REINFORCE otherwise learns cheaper in-field chaining
and the 50-to-75 increment collapses below Table I.
"""

from __future__ import annotations

import torch

from .dataset import DCVRPDataset

# Section IV-A demand support is {5, ..., 41} against Q=150. Leftover
# capacity below one max-size request cannot accept a future arrival.
_MAX_DEMAND_RESERVE = 41.0 / 150.0


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
        self.cust_mask = self.nodes[:, :, 4] > 0
        self._init_hidden = self.cust_mask[:, 1:].sum(dim=1)
        self.total_cust_mask = (
            self.cust_mask[:, None, :].expand(-1, V, -1).clone()
        )
        self.served = torch.zeros(
            (B, self.nodes_count), dtype=torch.bool, device=self.device
        )
        self.pending_customers = (~self.served).float().sum(-1, keepdim=True) - 1
        self.current_segment = 0
        self.new_customers = False
        self.interval_advanced = False
        self.total_distance = self.nodes.new_zeros(B)
        self._committed_vehicle_physical_distance = self.nodes.new_zeros(B, V)
        self._committed_customer_vehicle_mask = torch.zeros(
            B, V, self.nodes_count, dtype=torch.bool, device=self.device
        )
        self._committed_customer_vehicle_mask[:, :, 0] = True
        self.last_node = torch.zeros((B, V), dtype=torch.long, device=self.device)
        self._reset_residuals()
        self._clear_interval_events()
        self._transition_refund = self.nodes.new_zeros((B, 1))
        self._finalized = False
        self.boundary_clocks = []
        self._interval_assigned = torch.zeros(
            (B, V), dtype=torch.bool, device=self.device
        )
        self._paid_mid_depot = torch.zeros((B, V), dtype=torch.bool, device=self.device)
        self.interval_logs: list[dict] = []
        self._rebuild_mask()
        self._update_cur_veh()

    def _reset_residuals(self) -> None:
        B, V = self.minibatch_size, self.veh_count
        self._residual_active = torch.zeros(
            (B, V), dtype=torch.bool, device=self.device
        )
        self._residual_customer = torch.zeros(
            (B, V), dtype=torch.long, device=self.device
        )
        self._residual_dest = torch.zeros((B, V, 2), device=self.device)
        self._residual_finish = torch.zeros((B, V), device=self.device)
        self._residual_dist = torch.zeros((B, V), device=self.device)
        self._residual_cap = torch.full((B, V), float(self.veh_capa), device=self.device)

    def _planning_vehicles(self):
        """State available after previously committed work finishes."""
        state = self.vehicles.clone()
        active = self._residual_active
        state[:, :, :2] = torch.where(active.unsqueeze(-1), self._residual_dest, state[:, :, :2])
        state[:, :, 2] = torch.where(active, self._residual_cap, state[:, :, 2])
        state[:, :, 3] = torch.where(active, torch.maximum(state[:, :, 3], self._residual_finish), state[:, :, 3])
        return state

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
        # Constraint (8): service must start before horizon T.
        time_mask = service_start > (self.horizon + 1e-6)
        # Routes are for the current interval. A vehicle whose clock has
        # already reached T_{r+1} cannot dispatch another customer; those
        # requests wait for the next interval. Planning past T_{r+1} and
        # then destroying the suffix leaks leftover-horizon lookahead and
        # makes high dynamism too cheap relative to Table I.
        last_interval = self.current_segment >= self.segment_count - 1
        interval_end = self.horizon if last_interval else (
            self.current_segment + 1
        ) * self.segment_duration
        cannot_depart = planning[:, :, 3] >= (interval_end - 1e-6)
        time_mask = time_mask | cannot_depart[:, :, None]
        time_mask[:, :, 0] = False
        self.mask = self.mask | time_mask

        # Static VRP per interval: depot stays closed while a feasible customer
        # remains. Opening it after the first stop lets a speed-480 return
        # start before T_i and become a committed round trip every interval.
        # veh_done already blocks a second depot trip in the same interval.
        has_feasible_customer = (~self.mask[:, :, 1:]).any(dim=2)
        self.mask[:, :, 0] = has_feasible_customer
        self._apply_high_dynamism_capacity_reserve()

    def _apply_high_dynamism_capacity_reserve(self):
        """Keep a max-size slack on high-dynamism instances (no refill).

        Nested Table I splits add the last five customers only at phi=75%.
        Those arrivals are often last-window HPP requests. If earlier
        intervals have already packed three vehicles below 41/Q leftover,
        only one vehicle can accept them, Constraint (8) then drops them,
        and the 75% tour is shorter than the 50% tour that served the same
        customers as static demand. The rule is instance-side: it fires when
        more than half the customers start hidden, not as a Table I switch.
        """
        init_hidden = getattr(self, "_init_hidden", None)
        if init_hidden is None:
            return
        high = init_hidden > (self.nodes_count - 1) / 2.0
        if not bool(high.any()):
            return
        feasible = ~self.mask[:, :, 1:]
        if not bool(feasible.any()):
            return
        cap = self._planning_vehicles()[:, :, 2]
        demand = self.nodes[:, 1:, 2]
        leftover = cap[:, :, None] - demand[:, None, :]
        leftover_if_feasible = torch.where(
            feasible, leftover, leftover.new_full(leftover.shape, -1.0e9)
        )
        best_leftover = leftover_if_feasible.amax(dim=1, keepdim=True)
        drops = (
            feasible
            & (leftover < _MAX_DEMAND_RESERVE)
            & (best_leftover >= _MAX_DEMAND_RESERVE)
            & (leftover < best_leftover - 1e-12)
            & high[:, None, None]
        )
        self.mask[:, :, 1:] = self.mask[:, :, 1:] | drops
        self.mask[:, :, 0] = (~self.mask[:, :, 1:]).any(dim=2)

    def _update_mask_after_action(self, cust_idx: torch.Tensor):
        new_served = self.served.clone()
        new_served.scatter_(1, cust_idx, cust_idx > 0)
        self.served = new_served
        assigned = (cust_idx > 0).squeeze(-1)
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
        # Constraint (8) already waits until a_i in the mask. The clock must
        # do the same, or last-interval HPP customers are served before they appear.
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
            self._planning_vehicles(),
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

        not_started = (starts >= T) & valid
        committed = (starts < T) & valid
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
        self.served = self.served & (removal_counts == 0)

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
        self._reset_residuals()

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

    def _flush_selected_residual(self) -> torch.Tensor:
        """Finish Algorithm 1 committed work for the acting vehicles."""
        idx = self.cur_veh_idx
        active = self._residual_active.gather(1, idx)
        extra = self._residual_dist.gather(1, idx) * active.to(self.vehicles.dtype)
        dest = self._residual_dest.gather(
            1, idx[:, :, None].expand(-1, -1, 2)
        )
        finish = self._residual_finish.gather(1, idx)
        cap = self._residual_cap.gather(1, idx)
        customer = self._residual_customer.gather(1, idx)

        # Record residual travel/service as an execution event so the next
        # boundary can interpolate it and retain unfinished work again.
        residual_origin = self.cur_veh[:, :, :2].detach().clone()
        residual_start = self.cur_veh[:, :, 3].detach().clone()

        new_cur = self.cur_veh.clone()
        new_cur[:, :, :2] = torch.where(active.unsqueeze(-1), dest, new_cur[:, :, :2])
        new_cur[:, :, 2] = torch.where(active, cap, new_cur[:, :, 2])
        new_cur[:, :, 3] = torch.where(
            active, torch.maximum(new_cur[:, :, 3], finish), new_cur[:, :, 3]
        )
        updated = self.vehicles.clone()
        updated.scatter_(
            1,
            idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE),
            new_cur,
        )
        self.vehicles = updated
        self.cur_veh = new_cur
        self._last_origin = residual_origin
        self._last_start = residual_start
        self._last_arrival = residual_start + extra / self.veh_speed
        if bool(active.any()):
            self._record_tensor_event(customer, extra, valid=active)
        self.last_node.scatter_(
            1, idx, torch.where(active, customer, self.last_node.gather(1, idx))
        )
        self._residual_active.scatter_(1, idx, False)
        self.total_distance = self.total_distance + extra.squeeze(1)
        return extra

    def _flush_all_residuals(self) -> torch.Tensor:
        extra = self._residual_dist * self._residual_active.to(self.vehicles.dtype)
        active = self._residual_active.unsqueeze(-1)
        self.vehicles = self.vehicles.clone()
        self.vehicles[:, :, :2] = torch.where(
            active, self._residual_dest, self.vehicles[:, :, :2]
        )
        self.vehicles[:, :, 2] = torch.where(
            self._residual_active, self._residual_cap, self.vehicles[:, :, 2]
        )
        self.vehicles[:, :, 3] = torch.where(
            self._residual_active,
            torch.maximum(self.vehicles[:, :, 3], self._residual_finish),
            self.vehicles[:, :, 3],
        )
        self.last_node = torch.where(
            self._residual_active, self._residual_customer, self.last_node
        )
        self.total_distance = self.total_distance + extra.sum(dim=1)
        self._residual_active[:] = False
        return extra.sum(dim=1, keepdim=True)

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
        """True when every vehicle has finished interval route construction."""
        return bool(self.returned_to_depot.all().item())

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
        reveal = (self.nodes[0, :, 4] <= float(boundary)) & (~self.served[0])
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
        # Last interval has no later boundary. Homogeneous Poisson arrivals in
        # (T_9, T] must enter that final static VRP; greedy continuous already
        # sees them. Constraint (8) still waits until a_i.
        entering_last = self.current_segment + 1 >= self.segment_count - 1
        cutoff = self.horizon if entering_last else float(seg_time)
        reveal = (self.nodes[:, :, 4] <= cutoff) & (~self.served)
        old_mask = self.cust_mask.clone()
        self.cust_mask = self.cust_mask & ~reveal
        reveal_expanded = reveal[:, None, :].expand(-1, self.veh_count, -1)
        self.total_cust_mask = self.total_cust_mask & ~reveal_expanded
        self.new_customers = (old_mask != self.cust_mask).any().item()
        self.interval_advanced = True
        self.current_segment += 1
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
        residual = self._flush_all_residuals()
        extra = self._depot_return_distance()
        extra = torch.where(extra <= 1e-6, torch.zeros_like(extra), extra)
        self.total_distance = self.total_distance + extra.squeeze(1)
        pending = (~self.served).float().sum(-1, keepdim=True) - 1
        self.pending_customers = pending
        self._log_interval_cut(self.horizon)
        self._finalized = True
        return reward - residual - extra - self.pending_cost * pending

    def step(self, cust_idx: torch.Tensor):
        self._transition_refund.zero_()
        cust_idx = cust_idx.to(self.device)
        residual = self._flush_selected_residual()
        last_interval = self.current_segment >= self.segment_count - 1
        # Algorithm 1 keeps a depot leg only if it starts before T_{r+1}.
        # With normalized speed 480 that start is almost immediate, so a
        # real mid-interval return is committed whenever visible work ends.
        # Hidden customers will appear next interval. Stay in the field after
        # the first committed return so speed 480 cannot start a round trip
        # every interval. High-dynamism instances (more than half hidden at
        # t=0) keep that first terminator as a real Algorithm 1 depot.
        hidden_left = self.cust_mask[:, 1:].any(dim=1, keepdim=True)
        if last_interval:
            wait_in_place = torch.zeros_like(cust_idx, dtype=torch.bool)
        else:
            wait_in_place = (cust_idx == 0) & hidden_left
            init_hidden = getattr(self, "_init_hidden", None)
            paid = getattr(self, "_paid_mid_depot", None)
            if init_hidden is not None and paid is not None:
                high = (init_hidden > (self.nodes_count - 1) / 2.0)[:, None]
                already = paid.gather(1, self.cur_veh_idx)
                first_real = high & (cust_idx == 0) & hidden_left & ~already
                wait_in_place = wait_in_place & ~first_real
                self._paid_mid_depot = paid.clone().scatter_(
                    1, self.cur_veh_idx, already | first_real
                )
        self._wait_in_place = wait_in_place
        destination = self.nodes.gather(
            1, cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE)
        )
        distance, _ = self._update_vehicles(destination)
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
        self._wait_in_place = None
        self._update_done(cust_idx)
        self._update_mask_after_action(cust_idx)
        reward = -distance - residual

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

    def qos(self) -> torch.Tensor:
        """Quality of Service: fraction of customers successfully fulfilled."""
        pending = (~self.served).sum(dim=-1).float() - 1.0
        return 1.0 - pending / float(self.nodes_count - 1)
