"""Environment adapter for the reproduction package.

The released ``learner.DCVRP_Environment`` is used unchanged.  It expects the
attribute ``dyna.dvnda`` to be a module mapping
``(vehicles, nodes, veh_done, cust_mask) -> (vehicle_index, scores)``; that hook
is the single point where the five methods differ, which is exactly the
controlled comparison described in Section IV-A.

Two conventions of the released code are load-bearing for Table I and are
therefore restated here:

* an interval advances only once **every** vehicle has returned to the depot,
  so each interval is a set of complete depot-to-depot trips;
* vehicle capacity is consumed across the whole horizon and never replenished,
  which is the "single round trip" premise behind the manuscript's QoS.
"""

from __future__ import annotations

import torch

from learner import DCVRP_Environment

from . import protocol
from .instances import InstanceBatch


class _DatasetView:
    """Minimal stand-in for ``DCVRP_Dataset`` accepted by the environment."""

    def __init__(self, batch: InstanceBatch) -> None:
        self.veh_count = batch.vehicle_count
        self.veh_capa = batch.vehicle_capacity
        self.veh_speed = batch.vehicle_speed
        self.nodes = batch.nodes


def _as_dataset(batch):
    """Accept either an :class:`InstanceBatch` or a released ``DCVRP_Dataset``.

    Both generators of :mod:`reproduction.instances` must be evaluable in this
    environment, and only one of them returns ``InstanceBatch``.
    """

    if hasattr(batch, "veh_count"):
        return batch
    return _DatasetView(batch)


def make_environment(
    batch: InstanceBatch,
    device: torch.device,
    *,
    pending_cost: float,
    decision_intervals: int = protocol.DECISION_INTERVALS,
    horizon: float = protocol.NORMALIZED_HORIZON,
) -> DCVRP_Environment:
    """Instantiate the released environment on ``device``.

    ``horizon`` selects the time unit of the interval boundaries:

    ``1.0`` (default)
        The normalized unit horizon, consistent with the node features that
        ``normalize()`` divided by 480.  Interval boundaries then fall at
        0.1, ..., 0.9 and disclosures are spread across the working day.

    ``480``
        The literal default of the released ``DCVRP_Environment``.  Because the
        node times were already divided by 480, every dynamic request is
        already visible at the first boundary, which collapses the process into
        a static wave followed by a single dynamic wave.  Retained so the
        discrepancy can be measured rather than argued about.
    """

    environment = DCVRP_Environment(
        _as_dataset(batch),
        nodes=batch.nodes.to(device),
        pending_cost=pending_cost,
        segment_count=decision_intervals,
        horizon=horizon,
    )
    return environment


def cost_and_qos(
    environment: DCVRP_Environment, rewards: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Table I's two quality columns, as computed by the released ``eval.py``.

    Cost is the negated sum of per-step rewards.  With ``pending_cost=0`` that
    is the total executed distance with unserved-customer penalties excluded,
    as stated under "Performance Metrics".
    """

    cost = -torch.stack(rewards).sum(dim=0).squeeze(-1)
    pending = (~environment.served).float().sum(-1) - 1.0
    qos = 1.0 - pending / float(environment.nodes_count - 1)
    return cost, qos
