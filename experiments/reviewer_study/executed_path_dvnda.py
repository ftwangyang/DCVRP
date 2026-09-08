"""Auditable executed-path cost protocol for the DVNDA manuscript.

The model may construct a complete candidate route at every synchronized
interval.  Only the prefix that has actually been dispatched is irrevocable:
an edge is executed when its start time is strictly earlier than the next
boundary.  All later candidate edges are destroyed, their provisional reward
is refunded, and their customers are released for replanning.

No artificial depot return is inserted at an interval boundary.  A depot edge
is charged only when it is physically executed during terminal route closure.
Consequently ``route_distance()`` is the cumulative distance of the final
executed routes in Eq. (14), not the cumulative distance of all plans proposed
during re-optimization.
"""

from __future__ import annotations

import torch

from .paper_dcvrp import VectorizedPaperDCVRPEnvironment


ENVIRONMENT_TAG = "time_driven_executed_path_v4"


class VectorizedExecutedPathDVNDAEnvironment(
    VectorizedPaperDCVRPEnvironment
):
    """Paper environment with explicit discarded-plan accounting."""

    environment_tag = ENVIRONMENT_TAG

    def reset(self):
        super().reset()
        self.discarded_planned_distance = self.nodes.new_zeros(
            self.minibatch_size
        )

    def _commit_prefix(self, boundary: float) -> torch.Tensor:
        refund = super()._commit_prefix(boundary)
        self.discarded_planned_distance += refund.squeeze(1)
        return refund

    def proposed_route_distance(self) -> torch.Tensor:
        """Executed distance plus every provisional edge later destroyed."""

        return self.route_distance() + self.discarded_planned_distance


__all__ = ["ENVIRONMENT_TAG", "VectorizedExecutedPathDVNDAEnvironment"]

