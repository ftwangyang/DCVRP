"""Instrumented normalized-time environment for reviewer experiments."""

from __future__ import annotations

import time

import numpy as np
import torch

from learner import DCVRP_Environment


class ExperimentalEnvironment(DCVRP_Environment):
    """A compatibility-preserving environment with traceable metrics.

    The route construction logic is inherited from the supplied implementation.
    The isolated class fixes the normalized horizon and exposes selector policy
    log-probabilities to REINFORCE.
    """

    def __init__(
        self,
        data,
        nodes=None,
        pending_cost=5.0,
        segment_count=10,
        horizon=1.0,
        travel_noise=0.0,
        service_noise=0.0,
        congestion=0.0,
        stochastic_seed=0,
        segment_boundaries=None,
    ):
        self.segment_boundaries = (
            [float(value) for value in segment_boundaries]
            if segment_boundaries is not None
            else None
        )
        if self.segment_boundaries is not None:
            if len(self.segment_boundaries) < 2:
                raise ValueError("segment_boundaries must include start and end")
            if self.segment_boundaries[0] != 0.0:
                raise ValueError("segment_boundaries must start at 0.0")
            segment_count = len(self.segment_boundaries) - 1
        super().__init__(
            data,
            nodes=nodes,
            pending_cost=pending_cost,
            segment_count=segment_count,
            horizon=horizon,
        )
        self.selector = None
        self.policy_greedy = False
        self.travel_noise = float(travel_noise)
        self.service_noise = float(service_noise)
        self.congestion = float(congestion)
        self._generator = torch.Generator(device=self.nodes.device)
        self._generator.manual_seed(int(stochastic_seed))

    def _check_segment_transition(self):
        if self.segment_boundaries is None:
            return super()._check_segment_transition()
        if self.current_segment >= self.segment_count - 1:
            return
        if not self.returned_to_depot.all().item():
            return
        next_segment_time = self.segment_boundaries[self.current_segment + 1]
        self._segment_transition(next_segment_time)

    def reset(self):
        self.total_distance = self.nodes.new_zeros(self.minibatch_size)
        self.vehicle_distance = self.nodes.new_zeros(
            (self.minibatch_size, self.veh_count)
        )
        self.idle_boundary_wait = self.nodes.new_zeros(self.minibatch_size)
        self.decision_epochs = 0
        self.selector_wall_time = 0.0
        super().reset()
        self.committed = torch.zeros_like(self.served)
        self.completed = torch.zeros_like(self.served)
        self.executed_busy_time = self.nodes.new_zeros(
            (self.minibatch_size, self.veh_count)
        )
        self.executed_partial_distance = self.nodes.new_zeros(
            (self.minibatch_size, self.veh_count)
        )
        self._route_offsets = [
            [0 for _ in range(self.veh_count)]
            for _ in range(self.minibatch_size)
        ]
        self._segment_start_vehicles = self.vehicles.clone()
        self._current_boundary = 0.0
        self._last_action_distance = None
        self._last_action_origin = None
        self._last_action_start_time = None
        self._transition_distance_refund = self.nodes.new_zeros(
            (self.minibatch_size, 1)
        )
        self._finalized = False

    def _update_mask_after_action(self, cust_idx):
        super()._update_mask_after_action(cust_idx)
        updated = self.committed.clone()
        updated.scatter_(1, cust_idx, cust_idx > 0)
        self.committed = updated

    def _update_cur_veh(self):
        if self.selector is None:
            raise RuntimeError("selector not set before environment reset")
        started = time.perf_counter()
        index, scores, log_probability = self.selector.select(
            self.vehicles,
            self.nodes,
            self.veh_done,
            self.cust_mask,
            greedy=self.policy_greedy,
        )
        self.selector_wall_time += time.perf_counter() - started
        self.cur_veh_idx = index
        self.cur_vehicle_scores = scores
        self.cur_vehicle_logp = log_probability
        idx_exp = index[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE)
        self.cur_veh = self.vehicles.gather(1, idx_exp)
        idx_exp_n = index[:, :, None].expand(-1, -1, self.nodes_count)
        self.cur_veh_mask = self.mask.gather(1, idx_exp_n)
        self.decision_epochs += 1

    def _update_vehicles(self, dest):
        self._last_action_origin = self.cur_veh[:, 0, :2].detach().clone()
        self._last_action_start_time = self.cur_veh[:, 0, 3].detach().clone()
        distance = torch.norm(
            self.cur_veh[:, 0, :2] - dest[:, 0, :2], dim=1, keepdim=True
        )
        travel_multiplier = torch.ones_like(distance)
        if self.travel_noise > 0:
            noise = torch.randn(
                distance.shape,
                generator=self._generator,
                device=distance.device,
                dtype=distance.dtype,
            )
            travel_multiplier = torch.exp(
                self.travel_noise * noise - 0.5 * self.travel_noise ** 2
            )
        if self.congestion > 0:
            current_time = self.cur_veh[:, :, 3]
            peak = (
                torch.exp(-0.5 * ((current_time - 0.30) / 0.09) ** 2)
                + torch.exp(-0.5 * ((current_time - 0.75) / 0.11) ** 2)
            )
            travel_multiplier = travel_multiplier * (1.0 + self.congestion * peak)

        service_multiplier = torch.ones_like(distance)
        if self.service_noise > 0:
            noise = torch.randn(
                distance.shape,
                generator=self._generator,
                device=distance.device,
                dtype=distance.dtype,
            )
            service_multiplier = torch.exp(
                self.service_noise * noise - 0.5 * self.service_noise ** 2
            )

        travel_time = distance * travel_multiplier / self.veh_speed
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
        event = self.route_logs[batch_idx][veh_idx][-1]
        event["distance"] = float(self._last_action_distance[batch_idx, 0].item())
        event["origin"] = [
            float(value)
            for value in self._last_action_origin[batch_idx].detach().cpu().tolist()
        ]
        event["start_time"] = float(
            self._last_action_start_time[batch_idx].item()
        )

    @staticmethod
    def _event_executed_fraction(event, boundary: float) -> float:
        start = float(event.get("start_time", event["arrival"]))
        arrival = float(event["arrival"])
        if arrival <= boundary:
            return 1.0
        if boundary <= start or arrival <= start:
            return 0.0
        return float(np.clip((boundary - start) / (arrival - start), 0.0, 1.0))

    def _executed_distance_at(self, boundary: float) -> torch.Tensor:
        distances = self.executed_partial_distance.clone()
        for batch_index in range(self.minibatch_size):
            for vehicle_index in range(self.veh_count):
                for event in self.route_logs[batch_index][vehicle_index]:
                    fraction = self._event_executed_fraction(event, boundary)
                    distances[batch_index, vehicle_index] += (
                        float(event.get("distance", 0.0)) * fraction
                    )
                    if fraction < 1.0:
                        break
        return distances

    def step(self, cust_idx):
        self._transition_distance_refund.zero_()
        reward = super().step(cust_idx)
        reward = reward + self._transition_distance_refund
        if self.done and self.current_segment >= self.segment_count - 1 and not self._finalized:
            executed = self._executed_distance_at(self.horizon)
            refund = self.total_distance - executed.sum(1)
            original_pending = (
                (~self.served).float().sum(-1, keepdim=True) - 1
            )
            completed = torch.zeros_like(self.served)
            for batch_index in range(self.minibatch_size):
                for route in self.route_logs[batch_index]:
                    for event in route:
                        customer = int(event["customer"])
                        if customer != 0 and event["departure"] <= self.horizon:
                            completed[batch_index, customer] = True
            actual_pending = (
                (~completed[:, 1:]).float().sum(-1, keepdim=True)
            )
            reward = (
                reward
                + refund.unsqueeze(1)
                + self.pending_cost * original_pending
                - self.pending_cost * actual_pending
            )
            self.vehicle_distance = executed
            self.total_distance = executed.sum(1)
            self._finalized = True
        return reward

    def _segment_transition(self, seg_time):
        """Advance vehicles continuously to a replanning boundary.

        The supplied implementation discards every action whose arrival lies
        after the boundary and places the vehicle at its previous customer.
        Here, completed work is retained, an in-service vehicle stays occupied
        until service completion, and an en-route vehicle is interpolated to
        its physical position before the remaining plan is released.
        """
        start_boundary = float(self._current_boundary)
        for batch_index in range(self.minibatch_size):
            for vehicle_index in range(self.veh_count):
                route = self.route_logs[batch_index][vehicle_index]
                offset = self._route_offsets[batch_index][vehicle_index]
                prefix = route[:offset]
                planned = route[offset:]
                for prior in prefix:
                    customer = int(prior["customer"])
                    if customer != 0 and prior["departure"] <= seg_time:
                        self.completed[batch_index, customer] = True
                        self.committed[batch_index, customer] = False
                anchor = self._segment_start_vehicles[batch_index, vehicle_index]
                position = anchor[:2].clone()
                capacity = float(anchor[2].item())
                available_time = float(anchor[3].item())
                busy = max(
                    0.0,
                    min(available_time, float(seg_time)) - start_boundary,
                )
                kept = []
                removed = []
                partially_refunded = None

                for event_index, event in enumerate(planned):
                    customer = int(event["customer"])
                    arrival = float(event["arrival"])
                    departure = float(event["departure"])
                    destination = self.nodes[batch_index, customer, :2]
                    action_start = available_time
                    busy += max(
                        0.0,
                        min(departure, float(seg_time))
                        - max(action_start, start_boundary),
                    )

                    if departure <= seg_time:
                        kept.append(event)
                        position = destination.clone()
                        available_time = departure
                        if customer == 0:
                            capacity = float(self.veh_capa)
                        else:
                            capacity = max(
                                capacity
                                - float(self.nodes[batch_index, customer, 2].item()),
                                0.0,
                            )
                            self.completed[batch_index, customer] = True
                            self.committed[batch_index, customer] = False
                        continue

                    if arrival <= seg_time < departure:
                        # The vehicle has reached the customer and is in
                        # service.  It remains unavailable until departure.
                        kept.append(event)
                        position = destination.clone()
                        available_time = departure
                        if customer != 0:
                            capacity = max(
                                capacity
                                - float(self.nodes[batch_index, customer, 2].item()),
                                0.0,
                            )
                            self.committed[batch_index, customer] = True
                        removed = planned[event_index + 1:]
                    else:
                        # The next stop has not been reached.  Compute the
                        # continuous en-route position and release the
                        # unexecuted assignments for replanning.
                        fraction = 0.0
                        if arrival > action_start and seg_time > action_start:
                            fraction = min(
                                1.0,
                                max(0.0, (seg_time - action_start) / (arrival - action_start)),
                            )
                            position = position + fraction * (destination - position)
                        unexecuted = float(event.get("distance", 0.0)) * (1.0 - fraction)
                        self.total_distance[batch_index] -= unexecuted
                        self.vehicle_distance[batch_index, vehicle_index] -= unexecuted
                        self._transition_distance_refund[batch_index, 0] += unexecuted
                        self.executed_partial_distance[
                            batch_index, vehicle_index
                        ] += float(event.get("distance", 0.0)) * fraction
                        partially_refunded = event
                        available_time = float(seg_time)
                        removed = planned[event_index:]
                    break
                else:
                    if available_time < seg_time:
                        if kept and int(kept[-1]["customer"]) == 0:
                            self.idle_boundary_wait[batch_index] += (
                                seg_time - available_time
                            )
                        available_time = float(seg_time)

                for event in removed:
                    customer = int(event["customer"])
                    if event is not partially_refunded:
                        unexecuted = float(event.get("distance", 0.0))
                        self.total_distance[batch_index] -= unexecuted
                        self.vehicle_distance[batch_index, vehicle_index] -= unexecuted
                        self._transition_distance_refund[batch_index, 0] += unexecuted
                    if customer != 0 and not self.completed[batch_index, customer]:
                        self.served[batch_index, customer] = False
                        self.committed[batch_index, customer] = False

                route[:] = prefix + kept
                self._route_offsets[batch_index][vehicle_index] = len(route)
                self.executed_busy_time[batch_index, vehicle_index] += min(
                    busy, float(seg_time) - start_boundary
                )
                self.vehicles[batch_index, vehicle_index, :2] = position
                self.vehicles[batch_index, vehicle_index, 2] = capacity
                self.vehicles[batch_index, vehicle_index, 3] = available_time

        reveal = (self.nodes[:, :, 4] <= seg_time) & (~self.served)
        old_customer_mask = self.cust_mask.clone()
        self.cust_mask = self.cust_mask & ~reveal
        reveal_expanded = reveal[:, None, :].expand(-1, self.veh_count, -1)
        self.total_cust_mask = self.total_cust_mask & ~reveal_expanded
        self.new_customers = (old_customer_mask != self.cust_mask).any().item()
        self.current_segment += 1
        self.pending_customers = (~self.served).float().sum(-1, keepdim=True) - 1
        self._current_boundary = float(seg_time)
        self._segment_start_vehicles = self.vehicles.clone()

        if self.current_segment >= self.segment_count:
            self.done = True
        else:
            self.done = False
            self.veh_done[:] = False
            self.returned_to_depot[:] = False
            self._rebuild_mask()

    def metrics(self) -> dict[str, np.ndarray]:
        completed = torch.zeros_like(self.served)
        late = torch.zeros_like(self.served)
        response = []
        completion = []
        mean_completion = []
        utilization = []
        for batch_index in range(self.minibatch_size):
            response_times = []
            departures = []
            per_vehicle_last_departure = []
            for route in self.route_logs[batch_index]:
                last_departure = 0.0
                for event in route:
                    customer = event["customer"]
                    if customer == 0:
                        continue
                    if event["departure"] <= self.horizon:
                        completed[batch_index, customer] = True
                    else:
                        late[batch_index, customer] = True
                        continue
                    reveal = float(self.nodes[batch_index, customer, 4].item())
                    response_times.append(max(0.0, event["arrival"] - reveal) * 480.0)
                    departures.append(event["departure"] * 480.0)
                    last_departure = max(last_departure, float(event["departure"]))
                per_vehicle_last_departure.append(last_departure)
            response.append(float(np.mean(response_times)) if response_times else np.nan)
            completion.append(float(max(departures)) if departures else 0.0)
            mean_completion.append(float(np.mean(departures)) if departures else np.nan)
            utilization.append(
                float(np.clip(
                    (
                        self.executed_busy_time[batch_index].sum().item()
                        + sum(
                            max(0.0, min(value, self.horizon) - self._current_boundary)
                            for value in per_vehicle_last_departure
                        )
                    ) / max(self.veh_count * self.horizon, 1.0e-12),
                    0.0,
                    1.0,
                ))
            )
        completed_customer = completed[:, 1:]
        unserved = (~completed_customer).sum(1).detach().cpu().numpy()
        qos = completed_customer.float().mean(1).detach().cpu().numpy()
        late_count = late[:, 1:].sum(1).detach().cpu().numpy()
        committed_count = (
            self.served[:, 1:] & ~completed_customer
        ).sum(1).detach().cpu().numpy()
        vehicle_distance = self.vehicle_distance.detach().cpu().numpy()
        balance = vehicle_distance.std(1) / np.maximum(vehicle_distance.mean(1), 1e-12)
        return {
            "distance": self.total_distance.detach().cpu().numpy(),
            "qos": qos,
            "unserved": unserved,
            "committed_not_completed": committed_count,
            "late": late_count,
            "response_time_min": np.asarray(response),
            "completion_time_min": np.asarray(completion),
            "mean_completion_time_min": np.asarray(mean_completion),
            "vehicle_utilization": np.asarray(utilization),
            "route_balance_cv": balance,
            "idle_boundary_wait_min": (
                self.idle_boundary_wait.detach().cpu().numpy() * 480.0
            ),
            "replanning_events": np.full(
                self.minibatch_size, self.current_segment + 1
            ),
            "decision_epochs": np.full(self.minibatch_size, self.decision_epochs),
            "selector_time_ms_per_instance": np.full(
                self.minibatch_size,
                1000.0 * self.selector_wall_time / max(self.minibatch_size, 1),
            ),
        }
