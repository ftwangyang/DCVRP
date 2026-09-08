"""GPU-vectorized DVNDA environment for periodic static re-optimization.

This module intentionally lives beside, rather than replacing, the historical
paper environment.  It implements the revised execution protocol used for the
reviewer experiment:

* ten synchronized planning intervals;
* a fixed route within an interval;
* every customer leg whose *start* precedes the boundary is committed;
* the unstarted suffix is released for the next optimization;
* every committed prefix ends with a mandatory depot return; and
* vehicle capacity is replenished only when that depot return is completed.

The neural code uses a unit horizon.  Since the manuscript coordinates are in
``[0, 1]`` and one coordinate unit takes one minute, the normalized vehicle
speed is 480 coordinate units per normalized horizon.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .paper_dcvrp import (
    PAPER_HORIZON_MINUTES,
    VectorizedPaperDCVRPEnvironment,
)


ENVIRONMENT_TAG = "periodic_static_dvnda_v3"
MANUSCRIPT_NORMALIZED_SPEED = PAPER_HORIZON_MINUTES


class VectorizedPeriodicStaticDVNDAEnvironment(
    VectorizedPaperDCVRPEnvironment
):
    """Periodic-static DCVRP execution semantics for DVNDA training.

    Depot choices during the ten regular planning rounds terminate a proposed
    route without moving the vehicle.  At the boundary, this class commits the
    dispatched customer prefix and explicitly charges/schedules its mandatory
    return to the depot.  The final closure round executes depot actions
    physically, as in the historical vectorized environment.
    """

    environment_tag = ENVIRONMENT_TAG

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        expected = float(MANUSCRIPT_NORMALIZED_SPEED)
        actual = float(self.veh_speed)
        if abs(actual - expected) > 1.0e-6:
            raise ValueError(
                "periodic-static DVNDA expects normalized vehicle speed "
                f"{expected:g} (one coordinate unit per minute), got {actual:g}"
            )

    def reset(self):
        super().reset()
        self.mandatory_return_distance = self.nodes.new_zeros(
            self.minibatch_size
        )
        self.mandatory_return_count = torch.zeros(
            self.minibatch_size,
            self.veh_count,
            dtype=torch.int32,
            device=self.device,
        )

    def _update_vehicles(self, dest: torch.Tensor):
        """Restore capacity only for an actually executed depot arrival."""

        distance, arrival = super()._update_vehicles(dest)
        wait_in_place = getattr(self, "_wait_in_place", None)
        if wait_in_place is None:
            wait_in_place = torch.zeros(
                (self.minibatch_size, 1),
                dtype=torch.bool,
                device=self.device,
            )
        depot = self.nodes[:, 0:1, :]
        is_depot = torch.isclose(dest, depot, atol=1.0e-7, rtol=0.0).all(dim=2)
        physical_return = is_depot & (~wait_in_place)
        if physical_return.any():
            self.cur_veh[:, :, 2] = torch.where(
                physical_return,
                self.cur_veh.new_full(
                    self.cur_veh[:, :, 2].shape, float(self.veh_capa)
                ),
                self.cur_veh[:, :, 2],
            )
            self.vehicles.scatter_(
                1,
                self.cur_veh_idx[:, :, None].expand(
                    -1, -1, self.VEH_STATE_SIZE
                ),
                self.cur_veh,
            )
        return distance, arrival

    def _commit_prefix(self, boundary: float) -> torch.Tensor:
        """Commit dispatched customer legs and append one depot return.

        The returned tensor is a reward correction: abandoned planned distance
        is refunded and the newly appended depot-return distance is charged.
        """

        if not self._event_start:
            restored = self._segment_vehicle_snapshot.clone()
            restored[:, :, 3] = torch.maximum(
                restored[:, :, 3],
                restored.new_full(restored[:, :, 3].shape, float(boundary)),
            )
            self.vehicles = restored
            return self.nodes.new_zeros((self.minibatch_size, 1))

        event_vehicles = torch.stack(self._event_vehicle, dim=1).squeeze(-1)
        customers = torch.stack(self._event_customer, dim=1).squeeze(-1)
        starts = torch.stack(self._event_start, dim=1).squeeze(-1)
        distances = torch.stack(self._event_distance, dim=1).squeeze(-1)
        states = torch.stack(self._event_vehicle_state, dim=1).squeeze(2)

        committed = starts < float(boundary)
        removed = ~committed
        refund = (distances * removed).sum(dim=1, keepdim=True)

        # Planned-but-unstarted customers become eligible again.  Depot events
        # are planning terminators and never represent a served customer.
        removed_customer = removed & (customers > 0)
        removal_counts = torch.zeros_like(self.served, dtype=torch.int32)
        removal_counts.scatter_add_(
            1, customers, removed_customer.to(torch.int32)
        )
        self.served = self.served & (removal_counts == 0)

        # A zero-distance depot planning action must not hide the final
        # committed customer state.  Locate the latest committed customer for
        # every vehicle independently.
        event_count = starts.size(1)
        event_order = torch.arange(
            event_count, device=self.device, dtype=torch.int64
        ).view(1, event_count, 1)
        vehicle_one_hot = F.one_hot(
            event_vehicles, num_classes=self.veh_count
        ).to(torch.bool)
        committed_customer = committed & (customers > 0)
        valid = committed_customer.unsqueeze(-1) & vehicle_one_hot
        latest_event = torch.where(
            valid,
            event_order,
            event_order.new_full((1, 1, 1), -1),
        ).amax(dim=1)
        has_committed_customer = latest_event >= 0
        gather_index = latest_event.clamp_min(0).unsqueeze(-1).expand(
            -1, -1, self.VEH_STATE_SIZE
        )
        latest_state = states.gather(1, gather_index)

        # Vehicles without work keep the actionable state inherited from the
        # preceding interval.  Vehicles with a committed prefix must finish
        # that service, return to the depot, and only then regain full capacity.
        restored = self._segment_vehicle_snapshot.clone()
        depot_xy = self.nodes[:, 0:1, :2].expand(-1, self.veh_count, -1)
        return_distance = torch.norm(
            latest_state[:, :, :2] - depot_xy, dim=2
        )
        return_distance = torch.where(
            has_committed_customer,
            return_distance,
            torch.zeros_like(return_distance),
        )
        return_completion = (
            latest_state[:, :, 3] + return_distance / float(self.veh_speed)
        )
        active_state = latest_state.clone()
        active_state[:, :, :2] = depot_xy
        active_state[:, :, 2] = float(self.veh_capa)
        active_state[:, :, 3] = torch.maximum(
            return_completion,
            return_completion.new_full(
                return_completion.shape, float(boundary)
            ),
        )
        restored = torch.where(
            has_committed_customer.unsqueeze(-1), active_state, restored
        )
        restored[:, :, 3] = torch.maximum(
            restored[:, :, 3],
            restored.new_full(restored[:, :, 3].shape, float(boundary)),
        )
        self.vehicles = restored

        return_total = return_distance.sum(dim=1)
        self.total_distance = (
            self.total_distance - refund.squeeze(1) + return_total
        )
        self.mandatory_return_distance += return_total
        self.mandatory_return_count += has_committed_customer.to(torch.int32)

        return refund - return_total.unsqueeze(1)


__all__ = [
    "ENVIRONMENT_TAG",
    "MANUSCRIPT_NORMALIZED_SPEED",
    "VectorizedPeriodicStaticDVNDAEnvironment",
]
