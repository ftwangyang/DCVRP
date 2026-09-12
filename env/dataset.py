"""Dataset and instance generation for Dynamic Capacitated Vehicle Routing Problem (DCVRP).

Generates synthetic instances strictly according to Section IV-A of the manuscript:
- Locations: Uniformly distributed in [0, 100] (normalized to unit square [0, 1])
- Demands: Uniformly distributed integers in [5, 41] (normalized by capacity Q=150)
- Service durations: Uniformly distributed integers in [10, 31] minutes (normalized by horizon T=480)
- Dynamic revelation times: Poisson process with mean rate lambda = (1 + T) / 2 = 240.5 min
- Dynamic rates: phi in {0.10, 0.25, 0.50, 0.75}
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

# Default manuscript parameters
DEFAULT_CUSTOMER_COUNT = 20
DEFAULT_VEHICLE_COUNT = 4
DEFAULT_VEHICLE_CAPACITY = 150.0
DEFAULT_VEHICLE_SPEED = 1.0
DEFAULT_HORIZON = 480.0
DEFAULT_DECISION_INTERVALS = 10
DEFAULT_DYNAMIC_RATES = (0.10, 0.25, 0.50, 0.75)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DCVRPDataset(Dataset):
    """Dataset container for normalized DCVRP instances."""

    def __init__(
        self,
        vehicle_count: int,
        vehicle_capacity: float,
        vehicle_speed: float,
        nodes: torch.Tensor,
        customer_mask: torch.Tensor | None = None,
    ):
        self.veh_count = vehicle_count
        self.veh_capa = vehicle_capacity
        self.veh_speed = vehicle_speed
        self.nodes = nodes
        self.cust_mask = customer_mask
        self.batch_size, self.nodes_count, _ = nodes.size()

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, index: int):
        if self.cust_mask is None:
            return self.nodes[index]
        return self.nodes[index], self.cust_mask[index]


def generate_dataset(
    batch_size: int,
    dynamic_rate: float = 0.5,
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
    vehicle_count: int = DEFAULT_VEHICLE_COUNT,
    vehicle_capacity: float = DEFAULT_VEHICLE_CAPACITY,
    vehicle_speed: float = DEFAULT_VEHICLE_SPEED,
    horizon: float = DEFAULT_HORIZON,
    seed: int | None = None,
) -> DCVRPDataset:
    """Generate a single batch of normalized DCVRP instances at a given dynamic rate."""
    if seed is not None:
        set_seed(seed)

    # 1. Continuous coordinates in [0, 1]
    coordinates = torch.rand(batch_size, customer_count + 1, 2)

    # 2. Integer demands in [5, 41]
    demands = torch.randint(
        5, 42, (batch_size, customer_count, 1), dtype=torch.int64
    ).float()

    # 3. Integer service durations in [10, 31] minutes
    service_minutes = torch.randint(
        10, 32, (batch_size, customer_count, 1), dtype=torch.int64
    ).float()

    # 4. Poisson revelation process for dynamic customers (Eq. 34: lambda = (1 + T) / 2 = 240.5)
    poisson_rate = float((1.0 + horizon) / 2.0)
    disclosure_minutes = torch.poisson(
        torch.full((batch_size, customer_count, 1), poisson_rate)
    ).clamp_(min=1.0, max=float(horizon))

    # 5. Dynamic customer assignment
    dynamic_count = int(round(customer_count * dynamic_rate))
    dynamic_order = torch.argsort(
        torch.rand(batch_size, customer_count), dim=1
    )
    ranks = torch.empty_like(dynamic_order)
    ranks.scatter_(
        1,
        dynamic_order,
        torch.arange(customer_count).view(1, -1).expand(batch_size, -1),
    )
    dynamic = ranks < dynamic_count
    release = torch.where(
        dynamic.unsqueeze(-1),
        disclosure_minutes,
        torch.zeros_like(disclosure_minutes),
    )

    # 6. Normalized customer features: [x, y, normalized_demand, normalized_duration, normalized_arrival]
    customer_features = torch.cat(
        [
            coordinates[:, 1:, :],
            demands / float(vehicle_capacity),
            service_minutes / float(horizon),
            release / float(horizon),
        ],
        dim=2,
    )

    # 7. Depot node at index 0: [x, y, 0, 0, 0]
    depot = torch.zeros(batch_size, 1, 5)
    depot[:, :, :2] = coordinates[:, :1, :]

    nodes = torch.cat([depot, customer_features], dim=1)

    return DCVRPDataset(
        vehicle_count=vehicle_count,
        vehicle_capacity=1.0,
        vehicle_speed=vehicle_speed * horizon,
        nodes=nodes,
    )


def generate_evaluation_split(
    instances: int = 100,
    dynamic_rates: Sequence[float] = DEFAULT_DYNAMIC_RATES,
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
    vehicle_count: int = DEFAULT_VEHICLE_COUNT,
    vehicle_capacity: float = DEFAULT_VEHICLE_CAPACITY,
    vehicle_speed: float = DEFAULT_VEHICLE_SPEED,
    horizon: float = DEFAULT_HORIZON,
    seed: int = 20260821,
) -> dict[float, DCVRPDataset]:
    """Generate paired evaluation sets across multiple dynamic rates.

    Coordinates, demands, service durations, and revelation draws are shared
    across the dynamic rate variants to isolate the impact of dynamic rates.
    """
    set_seed(seed)
    rates = tuple(sorted(float(r) for r in dynamic_rates))

    coordinates = torch.rand(instances, customer_count + 1, 2)
    demands = torch.randint(
        5, 42, (instances, customer_count, 1), dtype=torch.int64
    ).float()
    service_minutes = torch.randint(
        10, 32, (instances, customer_count, 1), dtype=torch.int64
    ).float()

    poisson_rate = float((1.0 + horizon) / 2.0)
    disclosure_minutes = torch.poisson(
        torch.full((instances, customer_count, 1), poisson_rate)
    ).clamp_(min=1.0, max=float(horizon))

    dynamic_order = torch.argsort(torch.rand(instances, customer_count), dim=1)
    ranks = torch.empty_like(dynamic_order)
    ranks.scatter_(
        1,
        dynamic_order,
        torch.arange(customer_count).view(1, -1).expand(instances, -1),
    )

    split: dict[float, DCVRPDataset] = {}
    for rate in rates:
        dynamic_count = int(round(customer_count * rate))
        dynamic = ranks < dynamic_count
        release = torch.where(
            dynamic.unsqueeze(-1),
            disclosure_minutes,
            torch.zeros_like(disclosure_minutes),
        )
        customer_features = torch.cat(
            [
                coordinates[:, 1:, :],
                demands / float(vehicle_capacity),
                service_minutes / float(horizon),
                release / float(horizon),
            ],
            dim=2,
        )
        depot = torch.zeros(instances, 1, 5)
        depot[:, :, :2] = coordinates[:, :1, :]
        nodes = torch.cat([depot, customer_features], dim=1)

        dataset = DCVRPDataset(
            vehicle_count=vehicle_count,
            vehicle_capacity=1.0,
            vehicle_speed=vehicle_speed * horizon,
            nodes=nodes,
        )
        split[rate] = dataset

    return split
