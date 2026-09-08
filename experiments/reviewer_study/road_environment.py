"""Road-matrix environment preserving directed shortest-path travel."""

from __future__ import annotations

import torch

from data import DCVRP_Dataset

from .environment import ExperimentalEnvironment
from .paper_dcvrp import PAPER_HORIZON_MINUTES, VectorizedPaperDCVRPEnvironment


class RoadExperimentalEnvironment(ExperimentalEnvironment):
    def __init__(self, *args, distance_matrix, travel_time_matrix, **kwargs):
        self.distance_matrix = distance_matrix
        self.travel_time_matrix = travel_time_matrix
        self._destination_index = None
        super().__init__(*args, **kwargs)

    def reset(self):
        self.current_node = torch.zeros(
            (self.minibatch_size, self.veh_count),
            dtype=torch.long,
            device=self.nodes.device,
        )
        super().reset()

    def step(self, customer_index):
        self._destination_index = customer_index.to(self.nodes.device)
        return super().step(customer_index)

    def _update_vehicles(self, dest):
        batch = torch.arange(self.minibatch_size, device=self.nodes.device)
        vehicle = self.cur_veh_idx[:, 0]
        origin = self.current_node[batch, vehicle]
        destination = self._destination_index[:, 0]
        self._last_action_origin = self.cur_veh[:, 0, :2].detach().clone()
        self._last_action_start_time = self.cur_veh[:, 0, 3].detach().clone()
        self._last_action_origin_node = origin.detach().clone()
        distance = self.distance_matrix[batch, origin, destination].unsqueeze(1)
        travel_time = self.travel_time_matrix[batch, origin, destination].unsqueeze(1)
        if self.congestion > 0:
            current_time = self.cur_veh[:, :, 3]
            peak = (
                torch.exp(-0.5 * ((current_time - 0.30) / 0.09) ** 2)
                + torch.exp(-0.5 * ((current_time - 0.75) / 0.11) ** 2)
            )
            travel_time = travel_time * (1.0 + self.congestion * peak)
        if self.travel_noise > 0:
            noise = torch.randn(
                travel_time.shape,
                generator=self._generator,
                device=travel_time.device,
                dtype=travel_time.dtype,
            )
            travel_time = travel_time * torch.exp(
                self.travel_noise * noise - 0.5 * self.travel_noise ** 2
            )
        service_multiplier = torch.ones_like(travel_time)
        if self.service_noise > 0:
            noise = torch.randn(
                travel_time.shape,
                generator=self._generator,
                device=travel_time.device,
                dtype=travel_time.dtype,
            )
            service_multiplier = torch.exp(
                self.service_noise * noise - 0.5 * self.service_noise ** 2
            )
        arrival_time = self.cur_veh[:, :, 3] + travel_time
        updated_vehicle = self.cur_veh.clone()
        updated_vehicle[:, :, :2] = dest[:, :, :2]
        updated_vehicle[:, :, 2] = (
            updated_vehicle[:, :, 2] - dest[:, :, 2]
        ).clamp(min=0.0)
        updated_vehicle[:, :, 3] = arrival_time + dest[:, :, 3] * service_multiplier
        updated = self.vehicles.clone()
        updated.scatter_(
            1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE),
            updated_vehicle,
        )
        self.vehicles = updated
        self.cur_veh = updated_vehicle
        self.current_node[batch, vehicle] = destination
        self.total_distance += distance.squeeze(1)
        self.vehicle_distance.scatter_add_(1, self.cur_veh_idx, distance)
        self._last_action_distance = distance.detach().clone()
        return distance, arrival_time

    def _log_route_step(
        self, batch_idx, veh_idx, cust_idx, arrival_time, departure_time
    ):
        super()._log_route_step(
            batch_idx, veh_idx, cust_idx, arrival_time, departure_time
        )
        self.route_logs[batch_idx][veh_idx][-1]["origin_node"] = int(
            self._last_action_origin_node[batch_idx].item()
        )

    def _segment_transition(self, segment_time):
        fallback = self.current_node.clone()
        for batch_index in range(self.minibatch_size):
            for vehicle_index in range(self.veh_count):
                route = self.route_logs[batch_index][vehicle_index]
                offset = self._route_offsets[batch_index][vehicle_index]
                for event in route[offset:]:
                    if event["arrival"] > segment_time:
                        fallback[batch_index, vehicle_index] = int(
                            event.get("origin_node", fallback[batch_index, vehicle_index])
                        )
                        break
        super()._segment_transition(segment_time)
        for batch_index in range(self.minibatch_size):
            for vehicle_index in range(self.veh_count):
                route = self.route_logs[batch_index][vehicle_index]
                self.current_node[batch_index, vehicle_index] = (
                    route[-1]["customer"]
                    if route else fallback[batch_index, vehicle_index]
                )


class VectorizedRoadPaperDCVRPEnvironment(VectorizedPaperDCVRPEnvironment):
    """Paper-aligned interval execution on a directed road-time matrix.

    Node coordinates remain model features.  Every executed leg instead uses
    the fastest directed OSM path.  In the peak-aware condition, dispatches in
    the two declared peak windows use road-class-adjusted peak shortest paths.
    """

    PEAK_WINDOWS = ((0.20, 0.40), (0.65, 0.85))

    def __init__(
        self,
        *args,
        base_distance_matrix: torch.Tensor,
        base_travel_time_matrix: torch.Tensor,
        base_physical_distance_matrix: torch.Tensor | None = None,
        peak_distance_matrix: torch.Tensor | None = None,
        peak_travel_time_matrix: torch.Tensor | None = None,
        peak_physical_distance_matrix: torch.Tensor | None = None,
        peak_aware: bool = False,
        **kwargs,
    ):
        self.base_distance_matrix = base_distance_matrix
        self.base_travel_time_matrix = base_travel_time_matrix
        self.base_physical_distance_matrix = (
            base_distance_matrix
            if base_physical_distance_matrix is None
            else base_physical_distance_matrix
        )
        self.peak_distance_matrix = (
            base_distance_matrix
            if peak_distance_matrix is None
            else peak_distance_matrix
        )
        self.peak_travel_time_matrix = (
            base_travel_time_matrix
            if peak_travel_time_matrix is None
            else peak_travel_time_matrix
        )
        self.peak_physical_distance_matrix = (
            self.base_physical_distance_matrix
            if peak_physical_distance_matrix is None
            else peak_physical_distance_matrix
        )
        self.peak_aware = bool(peak_aware)
        self._destination_index: torch.Tensor | None = None
        super().__init__(*args, **kwargs)
        expected = (self.minibatch_size, self.nodes_count, self.nodes_count)
        for name in (
            "base_distance_matrix",
            "base_travel_time_matrix",
            "base_physical_distance_matrix",
            "peak_distance_matrix",
            "peak_travel_time_matrix",
            "peak_physical_distance_matrix",
        ):
            matrix = getattr(self, name).to(self.device)
            if tuple(matrix.shape) != expected:
                raise ValueError(
                    f"{name} shape {tuple(matrix.shape)} does not match {expected}"
                )
            if not torch.isfinite(matrix).all():
                raise ValueError(f"{name} contains non-finite road paths")
            setattr(self, name, matrix)

    def reset(self):
        self.current_node = torch.zeros(
            (self.minibatch_size, self.veh_count),
            dtype=torch.long,
            device=self.device,
        )
        self._event_node_state: list[torch.Tensor] = []
        self._event_travel_time: list[torch.Tensor] = []
        self._event_physical_distance: list[torch.Tensor] = []
        self.total_travel_time = self.nodes.new_zeros(self.minibatch_size)
        self.total_physical_distance = self.nodes.new_zeros(self.minibatch_size)
        super().reset()
        self._event_arrival: list[torch.Tensor] = []
        self._event_departure: list[torch.Tensor] = []
        self._committed_response_sum_min = self.nodes.new_zeros(
            self.minibatch_size
        )
        self._committed_completion_delay_sum_min = self.nodes.new_zeros(
            self.minibatch_size
        )
        self._committed_customer_count = self.nodes.new_zeros(
            self.minibatch_size
        )
        self._committed_busy_time = self.nodes.new_zeros(
            (self.minibatch_size, self.veh_count)
        )
        self._committed_vehicle_physical_distance = self.nodes.new_zeros(
            (self.minibatch_size, self.veh_count)
        )
        self._segment_node_snapshot = self.current_node.detach().clone()

    def _peak_departure_mask(self) -> torch.Tensor:
        current_time = self.cur_veh[:, :, 3]
        peak = torch.zeros_like(current_time, dtype=torch.bool)
        if not self.peak_aware:
            return peak
        for lower, upper in self.PEAK_WINDOWS:
            peak |= (current_time >= lower) & (current_time < upper)
        return peak

    def step(self, cust_idx: torch.Tensor):
        self._destination_index = cust_idx.to(self.device)
        return super().step(cust_idx)

    def _update_vehicles(self, dest: torch.Tensor):
        if self._destination_index is None:
            raise RuntimeError("road destination index was not set before dispatch")
        self._last_start = self.cur_veh[:, :, 3].detach().clone()
        wait_in_place = getattr(self, "_wait_in_place", None)
        if wait_in_place is None:
            wait_in_place = torch.zeros(
                (self.minibatch_size, 1), dtype=torch.bool, device=self.device
            )
        batch = torch.arange(self.minibatch_size, device=self.device)
        vehicle = self.cur_veh_idx[:, 0]
        origin = self.current_node[batch, vehicle]
        destination = self._destination_index[:, 0]
        base_distance = self.base_distance_matrix[batch, origin, destination]
        base_travel = self.base_travel_time_matrix[batch, origin, destination]
        peak_distance = self.peak_distance_matrix[batch, origin, destination]
        base_physical_distance = self.base_physical_distance_matrix[
            batch, origin, destination
        ]
        peak_physical_distance = self.peak_physical_distance_matrix[
            batch, origin, destination
        ]
        peak_travel = self.peak_travel_time_matrix[batch, origin, destination]
        peak_dispatch = self._peak_departure_mask()[:, 0]
        distance = torch.where(
            peak_dispatch, peak_distance, base_distance
        ).unsqueeze(1)
        physical_distance = torch.where(
            peak_dispatch, peak_physical_distance, base_physical_distance
        ).unsqueeze(1)
        travel_time = torch.where(
            peak_dispatch, peak_travel, base_travel
        ).unsqueeze(1)
        travel_time = travel_time * self._mean_one_lognormal(
            travel_time, self.travel_sigma
        )
        service_multiplier = self._mean_one_lognormal(
            travel_time, self.service_sigma
        )
        arrival_time = self.cur_veh[:, :, 3] + travel_time
        departure_time = arrival_time + dest[:, :, 3] * service_multiplier

        new_cur_veh = self.cur_veh.clone()
        new_cur_veh[:, :, :2] = dest[:, :, :2]
        new_cur_veh[:, :, 2] = (
            new_cur_veh[:, :, 2] - dest[:, :, 2]
        ).clamp(min=0.0)
        new_cur_veh[:, :, 3] = departure_time
        distance = torch.where(wait_in_place, torch.zeros_like(distance), distance)
        physical_distance = torch.where(
            wait_in_place, torch.zeros_like(physical_distance), physical_distance
        )
        executed_travel_time = torch.where(
            wait_in_place, torch.zeros_like(travel_time), travel_time
        )
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
        self.current_node[batch, vehicle] = torch.where(
            wait_in_place[:, 0], origin, destination
        )
        self._last_distance = distance.detach().clone()
        self._last_physical_distance = physical_distance.detach().clone()
        self._last_travel_time = executed_travel_time.detach().clone()
        self._last_arrival = arrival_time.detach().clone()
        self._last_departure = departure_time.detach().clone()
        return distance, arrival_time

    def _record_tensor_event(
        self, customer_index: torch.Tensor, distance: torch.Tensor
    ) -> None:
        super()._record_tensor_event(customer_index, distance)
        batch = torch.arange(self.minibatch_size, device=self.device)
        self._event_node_state.append(
            self.current_node[batch, self.cur_veh_idx[:, 0]].detach().clone()
        )
        self._event_travel_time.append(self._last_travel_time.detach().clone())
        self._event_physical_distance.append(
            self._last_physical_distance.detach().clone()
        )
        self._event_arrival.append(self._last_arrival.detach().clone())
        self._event_departure.append(self._last_departure.detach().clone())
        self.total_travel_time += self._last_travel_time[:, 0]
        self.total_physical_distance += self._last_physical_distance[:, 0]

    def _event_operational_contributions(
        self, included: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return trace metrics for the selected current-interval events.

        ``included`` has shape ``(batch, event)``.  It is normally the strict
        dispatched-prefix mask used by Algorithm 1.  The final terminal route
        uses an all-true mask because no later interval can destroy its suffix.
        """

        response_sum = self.nodes.new_zeros(self.minibatch_size)
        completion_sum = self.nodes.new_zeros(self.minibatch_size)
        customer_count = self.nodes.new_zeros(self.minibatch_size)
        busy_time = self.nodes.new_zeros(
            (self.minibatch_size, self.veh_count)
        )
        vehicle_distance = self.nodes.new_zeros(
            (self.minibatch_size, self.veh_count)
        )
        if not self._event_start:
            return (
                response_sum,
                completion_sum,
                customer_count,
                busy_time,
                vehicle_distance,
            )

        vehicles = torch.stack(self._event_vehicle, dim=1).squeeze(-1)
        customers = torch.stack(self._event_customer, dim=1).squeeze(-1)
        starts = torch.stack(self._event_start, dim=1).squeeze(-1)
        arrivals = torch.stack(self._event_arrival, dim=1).squeeze(-1)
        departures = torch.stack(self._event_departure, dim=1).squeeze(-1)
        physical_distances = torch.stack(
            self._event_physical_distance, dim=1
        ).squeeze(-1)

        busy_time.scatter_add_(
            1,
            vehicles,
            (departures - starts).clamp_min(0.0) * included,
        )
        vehicle_distance.scatter_add_(
            1, vehicles, physical_distances * included
        )

        customer_event = included & (customers > 0)
        effective_disclosure = (
            torch.ceil(
                self.nodes[:, :, 4] * float(self.segment_count) - 1.0e-7
            )
            / float(self.segment_count)
        )
        disclosure = effective_disclosure.gather(1, customers)
        response_sum = (
            (arrivals - disclosure).clamp_min(0.0)
            * customer_event
        ).sum(dim=1) * PAPER_HORIZON_MINUTES
        completion_sum = (
            (departures - disclosure).clamp_min(0.0)
            * customer_event
        ).sum(dim=1) * PAPER_HORIZON_MINUTES
        customer_count = customer_event.sum(dim=1).to(self.nodes.dtype)
        return (
            response_sum,
            completion_sum,
            customer_count,
            busy_time,
            vehicle_distance,
        )

    def _accumulate_committed_operational_metrics(self, boundary: float) -> None:
        if not self._event_start:
            return
        starts = torch.stack(self._event_start, dim=1).squeeze(-1)
        contributions = self._event_operational_contributions(
            starts < float(boundary)
        )
        (
            response_sum,
            completion_sum,
            customer_count,
            busy_time,
            vehicle_distance,
        ) = contributions
        self._committed_response_sum_min += response_sum
        self._committed_completion_delay_sum_min += completion_sum
        self._committed_customer_count += customer_count
        self._committed_busy_time += busy_time
        self._committed_vehicle_physical_distance += vehicle_distance

    def _commit_prefix(self, boundary: float) -> torch.Tensor:
        if not self._event_start:
            refund = super()._commit_prefix(boundary)
            self.current_node = self._segment_node_snapshot.clone()
            return refund

        self._accumulate_committed_operational_metrics(boundary)

        vehicles = torch.stack(self._event_vehicle, dim=1).squeeze(-1)
        starts = torch.stack(self._event_start, dim=1).squeeze(-1)
        node_states = torch.stack(self._event_node_state, dim=1)
        travel_times = torch.stack(self._event_travel_time, dim=1).squeeze(-1)
        physical_distances = torch.stack(
            self._event_physical_distance, dim=1
        ).squeeze(-1)
        committed = starts < float(boundary)
        self.total_travel_time -= (travel_times * (~committed)).sum(dim=1)
        self.total_physical_distance -= (
            physical_distances * (~committed)
        ).sum(dim=1)
        event_count = starts.size(1)
        event_order = torch.arange(
            event_count, device=self.device, dtype=torch.int64
        ).view(1, event_count, 1)
        vehicle_one_hot = torch.nn.functional.one_hot(
            vehicles, num_classes=self.veh_count
        ).to(torch.bool)
        latest_event = torch.where(
            committed.unsqueeze(-1) & vehicle_one_hot,
            event_order,
            event_order.new_full((1, 1, 1), -1),
        ).amax(dim=1)
        latest_node = node_states.gather(1, latest_event.clamp_min(0))
        restored_node = torch.where(
            latest_event >= 0, latest_node, self._segment_node_snapshot
        )
        refund = super()._commit_prefix(boundary)
        self.current_node = restored_node
        return refund

    def _clear_interval_events(self) -> None:
        super()._clear_interval_events()
        self._event_node_state.clear()
        self._event_travel_time.clear()
        self._event_physical_distance.clear()
        self._event_arrival.clear()
        self._event_departure.clear()
        self._segment_node_snapshot = self.current_node.detach().clone()

    def route_driving_time_minutes(self) -> torch.Tensor:
        return self.total_travel_time.clone() * PAPER_HORIZON_MINUTES

    def route_physical_distance_km(self) -> torch.Tensor:
        return self.total_physical_distance.clone()

    def operational_metrics(self) -> dict[str, torch.Tensor]:
        """Metrics derived from committed/actually executed road events.

        The current event buffer is the terminal route after the tenth
        periodic update.  It cannot be destroyed by a later replan and is
        therefore included in full without mutating the environment.
        """

        response_sum = self._committed_response_sum_min.clone()
        completion_sum = self._committed_completion_delay_sum_min.clone()
        customer_count = self._committed_customer_count.clone()
        busy_time = self._committed_busy_time.clone()
        vehicle_distance = self._committed_vehicle_physical_distance.clone()
        if self._event_start:
            starts = torch.stack(self._event_start, dim=1).squeeze(-1)
            contributions = self._event_operational_contributions(
                torch.ones_like(starts, dtype=torch.bool)
            )
            response_sum += contributions[0]
            completion_sum += contributions[1]
            customer_count += contributions[2]
            busy_time += contributions[3]
            vehicle_distance += contributions[4]

        divisor = customer_count.clamp_min(1.0)
        response_wait = response_sum / divisor
        completion_delay = completion_sum / divisor
        response_wait = torch.where(
            customer_count > 0, response_wait, torch.zeros_like(response_wait)
        )
        completion_delay = torch.where(
            customer_count > 0,
            completion_delay,
            torch.zeros_like(completion_delay),
        )

        makespan = self.vehicles[:, :, 3].max(dim=1).values.clamp_min(1.0e-12)
        utilization = (
            busy_time.sum(dim=1) / (float(self.veh_count) * makespan)
        ).clamp(0.0, 1.0)
        mean_distance = vehicle_distance.mean(dim=1)
        route_balance = vehicle_distance.std(dim=1, unbiased=False) / mean_distance.clamp_min(
            1.0e-12
        )
        route_balance = torch.where(
            mean_distance > 0,
            route_balance,
            torch.zeros_like(route_balance),
        )
        uncompleted = (~self.served[:, 1:]).sum(dim=1).to(self.nodes.dtype)
        return {
            "response_wait_time_min": response_wait,
            "completion_delay_min": completion_delay,
            "uncompleted_requests": uncompleted,
            "vehicle_utilization_percent": 100.0 * utilization,
            "route_balance_cv": route_balance,
        }


@torch.no_grad()
def run_road_greedy(
    data: DCVRP_Dataset,
    base_distance_matrix: torch.Tensor,
    base_travel_time_matrix: torch.Tensor,
    *,
    base_physical_distance_matrix: torch.Tensor | None = None,
    peak_distance_matrix: torch.Tensor | None = None,
    peak_travel_time_matrix: torch.Tensor | None = None,
    peak_physical_distance_matrix: torch.Tensor | None = None,
    peak_aware: bool = False,
    pending_cost: float = 0.0,
    return_operational_metrics: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Event-driven nearest-travel-time baseline on the same directed graph."""

    nodes = data.nodes.detach().cpu()
    base_distance = base_distance_matrix.detach().cpu()
    base_travel = base_travel_time_matrix.detach().cpu()
    base_physical_distance = (
        base_distance
        if base_physical_distance_matrix is None
        else base_physical_distance_matrix.detach().cpu()
    )
    peak_distance = (
        base_distance
        if peak_distance_matrix is None
        else peak_distance_matrix.detach().cpu()
    )
    peak_travel = (
        base_travel
        if peak_travel_time_matrix is None
        else peak_travel_time_matrix.detach().cpu()
    )
    peak_physical_distance = (
        base_physical_distance
        if peak_physical_distance_matrix is None
        else peak_physical_distance_matrix.detach().cpu()
    )
    costs, physical_distances, qualities, completions, driving_times = (
        [],
        [],
        [],
        [],
        [],
    )
    response_wait_times = []
    completion_delays = []
    uncompleted_requests = []
    vehicle_utilizations = []
    route_balances = []

    def in_peak(value: float) -> bool:
        return peak_aware and any(
            lower <= value < upper
            for lower, upper in VectorizedRoadPaperDCVRPEnvironment.PEAK_WINDOWS
        )

    for batch_index, instance in enumerate(nodes):
        node_count = instance.size(0)
        current_node = torch.zeros(data.veh_count, dtype=torch.long)
        capacity = torch.full((data.veh_count,), float(data.veh_capa))
        available = torch.zeros(data.veh_count)
        assigned = torch.zeros(node_count, dtype=torch.bool)
        assigned[0] = True
        disclosures = (
            torch.ceil(instance[:, 4] * 10.0 - 1.0e-7) / 10.0
        )
        route_distance = 0.0
        route_physical_distance = 0.0
        driving_time = 0.0
        current_time = 0.0
        response_times = []
        customer_completion_delays = []
        vehicle_busy_time = torch.zeros(data.veh_count)
        vehicle_physical_distance = torch.zeros(data.veh_count)

        for _ in range(node_count * (10 + data.veh_count + 2)):
            visible = (disclosures <= current_time + 1.0e-7) & (~assigned)
            visible[0] = False
            idle = (available <= current_time + 1.0e-7).nonzero(
                as_tuple=False
            ).flatten()
            for vehicle_tensor in idle:
                vehicle = int(vehicle_tensor.item())
                feasible = visible & (
                    instance[:, 2] <= capacity[vehicle] + 1.0e-7
                )
                candidates = feasible.nonzero(as_tuple=False).flatten()
                if not candidates.numel():
                    continue
                time_matrix = peak_travel if in_peak(current_time) else base_travel
                distance_matrix = (
                    peak_distance if in_peak(current_time) else base_distance
                )
                physical_distance_matrix = (
                    peak_physical_distance
                    if in_peak(current_time)
                    else base_physical_distance
                )
                travel_values = time_matrix[
                    batch_index, current_node[vehicle], candidates
                ]
                customer = int(candidates[travel_values.argmin()].item())
                travel_time = float(
                    time_matrix[batch_index, current_node[vehicle], customer].item()
                )
                route_distance += float(
                    distance_matrix[
                        batch_index, current_node[vehicle], customer
                    ].item()
                )
                route_physical_distance += float(
                    physical_distance_matrix[
                        batch_index, current_node[vehicle], customer
                    ].item()
                )
                driving_time += travel_time
                service_time = float(instance[customer, 3].item())
                arrival_time = current_time + travel_time
                completion_time = arrival_time + service_time
                response_times.append(
                    max(0.0, arrival_time - float(disclosures[customer].item()))
                )
                customer_completion_delays.append(
                    max(0.0, completion_time - float(disclosures[customer].item()))
                )
                vehicle_busy_time[vehicle] += travel_time + service_time
                vehicle_physical_distance[vehicle] += float(
                    physical_distance_matrix[
                        batch_index, current_node[vehicle], customer
                    ].item()
                )
                available[vehicle] = completion_time
                current_node[vehicle] = customer
                capacity[vehicle] -= instance[customer, 2]
                assigned[customer] = True
                visible[customer] = False

            if bool(assigned[1:].all()):
                break
            future_disclosures = disclosures[
                (~assigned) & (disclosures > current_time + 1.0e-7)
            ]
            future_completions = available[available > current_time + 1.0e-7]
            next_events = []
            if future_disclosures.numel():
                next_events.append(float(future_disclosures.min().item()))
            if future_completions.numel():
                next_events.append(float(future_completions.min().item()))
            if not next_events:
                break
            next_time = min(next_events)
            if next_time <= current_time + 1.0e-7 or next_time > 1.0 + 1.0e-7:
                break
            current_time = next_time

        for vehicle in range(data.veh_count):
            departure = float(available[vehicle].item())
            distance_matrix = peak_distance if in_peak(departure) else base_distance
            physical_distance_matrix = (
                peak_physical_distance
                if in_peak(departure)
                else base_physical_distance
            )
            time_matrix = peak_travel if in_peak(departure) else base_travel
            route_distance += float(
                distance_matrix[batch_index, current_node[vehicle], 0].item()
            )
            route_physical_distance += float(
                physical_distance_matrix[
                    batch_index, current_node[vehicle], 0
                ].item()
            )
            available[vehicle] += time_matrix[
                batch_index, current_node[vehicle], 0
            ]
            depot_travel = float(
                time_matrix[batch_index, current_node[vehicle], 0].item()
            )
            depot_physical_distance = float(
                physical_distance_matrix[
                    batch_index, current_node[vehicle], 0
                ].item()
            )
            driving_time += depot_travel
            vehicle_busy_time[vehicle] += depot_travel
            vehicle_physical_distance[vehicle] += depot_physical_distance
        served = int(assigned[1:].sum().item())
        pending = node_count - 1 - served
        costs.append(route_distance + float(pending_cost) * pending)
        physical_distances.append(route_physical_distance)
        qualities.append(served / float(node_count - 1))
        makespan = float(available.max().item())
        completions.append(makespan * PAPER_HORIZON_MINUTES)
        driving_times.append(driving_time * PAPER_HORIZON_MINUTES)
        response_wait_times.append(
            float(torch.tensor(response_times).mean().item())
            * PAPER_HORIZON_MINUTES
            if response_times
            else 0.0
        )
        completion_delays.append(
            float(torch.tensor(customer_completion_delays).mean().item())
            * PAPER_HORIZON_MINUTES
            if customer_completion_delays
            else 0.0
        )
        uncompleted_requests.append(float(pending))
        vehicle_utilizations.append(
            100.0
            * float(vehicle_busy_time.sum().item())
            / max(float(data.veh_count) * makespan, 1.0e-12)
        )
        mean_vehicle_distance = float(vehicle_physical_distance.mean().item())
        route_balances.append(
            float(vehicle_physical_distance.std(unbiased=False).item())
            / max(mean_vehicle_distance, 1.0e-12)
            if mean_vehicle_distance > 0.0
            else 0.0
        )

    base_result = (
        torch.tensor(costs, dtype=torch.float32),
        torch.tensor(physical_distances, dtype=torch.float32),
        torch.tensor(qualities, dtype=torch.float32),
        torch.tensor(completions, dtype=torch.float32),
        torch.tensor(driving_times, dtype=torch.float32),
    )
    if not return_operational_metrics:
        return base_result
    return (
        *base_result,
        torch.tensor(response_wait_times, dtype=torch.float32),
        torch.tensor(completion_delays, dtype=torch.float32),
        torch.tensor(uncompleted_requests, dtype=torch.float32),
        torch.tensor(vehicle_utilizations, dtype=torch.float32),
        torch.tensor(route_balances, dtype=torch.float32),
    )
